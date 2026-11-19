"""PSD Prep - Qt user interface (PySide6, falls back to PySide2)."""
from __future__ import annotations

import math
import os
import sys
import traceback

import numpy as np

try:
    from PySide6 import QtCore, QtGui, QtWidgets
    from PySide6.QtCore import Signal
    QT6 = True
except ImportError:  # pragma: no cover
    from PySide2 import QtCore, QtGui, QtWidgets
    from PySide2.QtCore import Signal
    QT6 = False

from . import ops, __version__
from .pipeline import Settings, process, use_saved_composite, make_converter
from .psdfile import MODE_CMYK, MODE_NAMES, MODE_RGB, read_psd

Qt = QtCore.Qt


def E(scope_cls, scope, name):
    """Enum lookup that works on PySide2 and PySide6."""
    s = getattr(scope_cls, scope, None)
    if s is not None and hasattr(s, name):
        return getattr(s, name)
    return getattr(scope_cls, name)


CHECKED = E(Qt, "CheckState", "Checked")
UNCHECKED = E(Qt, "CheckState", "Unchecked")
PREVIEW_MAX = 2200          # longest side of the preview, in pixels
ASPECTS = [("Free", None), ("Original", "orig"), ("16:9", 16 / 9), ("9:16", 9 / 16),
           ("1:1", 1.0), ("4:5", 4 / 5), ("5:4", 5 / 4), ("4:3", 4 / 3), ("3:2", 3 / 2),
           ("2:3", 2 / 3), ("2.39:1", 2.39), ("1.85:1", 1.85)]


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------
class Worker(QtCore.QObject):
    log = Signal(str)
    progress = Signal(int, int)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs

    def run(self):
        try:
            res = self.fn(*self.args, log=self.log.emit,
                          progress=lambda a, b: self.progress.emit(a, b), **self.kwargs)
            self.done.emit(res)
        except Exception as e:
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")


def _load(path, log=None, progress=None):
    return read_psd(path)


# ---------------------------------------------------------------------------
# Preview with crop box
# ---------------------------------------------------------------------------
class PreviewView(QtWidgets.QGraphicsView):
    cropChanged = Signal(object)      # (x, y, w, h) in document pixels, or None

    HANDLE = 8

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.setRenderHint(E(QtGui.QPainter, "RenderHint", "SmoothPixmapTransform"), True)
        self.setDragMode(E(QtWidgets.QGraphicsView, "DragMode", "NoDrag"))
        self.setTransformationAnchor(E(QtWidgets.QGraphicsView, "ViewportAnchor", "AnchorUnderMouse"))
        self.setBackgroundBrush(QtGui.QColor(38, 38, 40))
        self.pix = QtWidgets.QGraphicsPixmapItem()
        self.pix.setTransformationMode(E(Qt, "TransformationMode", "SmoothTransformation"))
        self.scene().addItem(self.pix)
        self.shade = QtWidgets.QGraphicsPathItem()
        self.shade.setBrush(QtGui.QColor(0, 0, 0, 150))
        self.shade.setPen(QtGui.QPen(E(Qt, "PenStyle", "NoPen")))
        self.shade.setZValue(5)
        self.scene().addItem(self.shade)
        self.frame = QtWidgets.QGraphicsRectItem()
        pen = QtGui.QPen(QtGui.QColor(255, 200, 40))
        pen.setCosmetic(True)
        pen.setWidth(2)
        self.frame.setPen(pen)
        self.frame.setZValue(6)
        self.scene().addItem(self.frame)
        self.doc_w = self.doc_h = 0
        self.crop = None
        self.aspect = None
        self._drag = None
        self._pan = None
        self.setMouseTracking(True)

    # -- image ---------------------------------------------------------
    def set_image(self, qimg, doc_w, doc_h, keep_view=False):
        self.doc_w, self.doc_h = doc_w, doc_h
        pm = QtGui.QPixmap.fromImage(qimg)
        self.pix.setPixmap(pm)
        self.pix.setScale(doc_w / max(1, pm.width()))
        self.scene().setSceneRect(QtCore.QRectF(-doc_w * 0.05, -doc_h * 0.05, doc_w * 1.1, doc_h * 1.1))
        if not keep_view:
            self.fit()
        self._update_overlay()

    def fit(self):
        if self.doc_w:
            self.fitInView(QtCore.QRectF(0, 0, self.doc_w, self.doc_h),
                           E(Qt, "AspectRatioMode", "KeepAspectRatio"))
            self.scale(0.95, 0.95)

    def wheelEvent(self, ev):
        f = 1.25 if ev.angleDelta().y() > 0 else 0.8
        self.scale(f, f)

    # -- crop ----------------------------------------------------------
    def set_crop(self, rect, emit=False):
        if rect is not None:
            x, y, w, h = [int(round(v)) for v in rect]
            x = max(0, min(x, self.doc_w - 1)); y = max(0, min(y, self.doc_h - 1))
            w = max(1, min(w, self.doc_w - x)); h = max(1, min(h, self.doc_h - y))
            rect = (x, y, w, h)
            if rect == (0, 0, self.doc_w, self.doc_h):
                rect = None
        self.crop = rect
        self._update_overlay()
        if emit:
            self.cropChanged.emit(rect)

    def _update_overlay(self):
        full = QtGui.QPainterPath()
        full.addRect(QtCore.QRectF(0, 0, self.doc_w, self.doc_h))
        if self.crop:
            x, y, w, h = self.crop
            inner = QtGui.QPainterPath()
            inner.addRect(QtCore.QRectF(x, y, w, h))
            self.shade.setPath(full.subtracted(inner))
            self.frame.setRect(QtCore.QRectF(x, y, w, h))
            self.frame.setVisible(True)
        else:
            self.shade.setPath(QtGui.QPainterPath())
            self.frame.setVisible(False)

    def _hit(self, pos):
        """Which part of the crop box is under the mouse (view coords)."""
        if not self.crop:
            return None
        x, y, w, h = self.crop
        tl = self.mapFromScene(QtCore.QPointF(x, y))
        br = self.mapFromScene(QtCore.QPointF(x + w, y + h))
        px, py, H = pos.x(), pos.y(), self.HANDLE
        if not (tl.x() - H <= px <= br.x() + H and tl.y() - H <= py <= br.y() + H):
            return None
        edges = ""
        if abs(py - tl.y()) <= H: edges += "t"
        if abs(py - br.y()) <= H: edges += "b"
        if abs(px - tl.x()) <= H: edges += "l"
        if abs(px - br.x()) <= H: edges += "r"
        if edges:
            return edges
        if tl.x() < px < br.x() and tl.y() < py < br.y():
            return "move"
        return None

    def _cursor_for(self, hit):
        shapes = {"t": "SizeVerCursor", "b": "SizeVerCursor", "l": "SizeHorCursor",
                  "r": "SizeHorCursor", "tl": "SizeFDiagCursor", "br": "SizeFDiagCursor",
                  "tr": "SizeBDiagCursor", "bl": "SizeBDiagCursor", "move": "SizeAllCursor"}
        name = shapes.get(hit, "CrossCursor")
        return QtGui.QCursor(E(Qt, "CursorShape", name))

    def _scene_pt(self, ev):
        p = ev.position().toPoint() if hasattr(ev, "position") else ev.pos()
        return p, self.mapToScene(p)

    def mousePressEvent(self, ev):
        p, sp = self._scene_pt(ev)
        btn = ev.button()
        if btn == E(Qt, "MouseButton", "MiddleButton") or (
                btn == E(Qt, "MouseButton", "LeftButton")
                and ev.modifiers() & E(Qt, "KeyboardModifier", "AltModifier")):
            self._pan = p
            return
        if btn != E(Qt, "MouseButton", "LeftButton") or not self.doc_w:
            return
        hit = self._hit(p)
        if hit is None:
            self._drag = ("new", sp, None)
        else:
            self._drag = (hit, sp, self.crop)

    def mouseMoveEvent(self, ev):
        p, sp = self._scene_pt(ev)
        if self._pan is not None:
            d = p - self._pan
            self._pan = p
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - d.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - d.y())
            return
        if self._drag is None:
            self.viewport().setCursor(self._cursor_for(self._hit(p)))
            return
        kind, start, orig = self._drag
        if kind == "new":
            x0, y0 = start.x(), start.y()
            x1, y1 = sp.x(), sp.y()
            w, h = abs(x1 - x0), abs(y1 - y0)
            if self.aspect:
                h = w / self.aspect
                y1 = y0 + (h if y1 >= y0 else -h)
            rect = (min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
            if rect[2] >= 2 and rect[3] >= 2:
                self.set_crop(rect, emit=True)
            return
        x, y, w, h = orig
        dx, dy = sp.x() - start.x(), sp.y() - start.y()
        if kind == "move":
            x = min(max(0, x + dx), self.doc_w - w)
            y = min(max(0, y + dy), self.doc_h - h)
        else:
            l, t, r, b = x, y, x + w, y + h
            if "l" in kind: l = min(max(0, l + dx), r - 1)
            if "r" in kind: r = max(min(self.doc_w, r + dx), l + 1)
            if "t" in kind: t = min(max(0, t + dy), b - 1)
            if "b" in kind: b = max(min(self.doc_h, b + dy), t + 1)
            x, y, w, h = l, t, r - l, b - t
            if self.aspect:
                if kind in ("t", "b"):
                    w = h * self.aspect
                else:
                    h = w / self.aspect
                    if "t" in kind:
                        y = b - h
        self.set_crop((x, y, w, h), emit=True)

    def mouseReleaseEvent(self, ev):
        self._drag = None
        self._pan = None


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
KIND_LABEL = {"text": "Text → raster", "smart": "Smart obj → raster",
              "adjustment": "Adjustment (ignored)", "fill": "Fill layer", "group": "Group",
              "pixel": ""}


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"PSD Prep {__version__}")
        self.resize(1500, 900)
        self.setAcceptDrops(True)
        self.settings = QtCore.QSettings("psdprep", "psdprep")
        self.doc = None
        self.path = None
        self.items = {}            # index -> QTreeWidgetItem
        self.rgb_cache = {}
        self.scale = 1
        self.thread = None
        self._building = False
        self._preview_timer = QtCore.QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(80)
        self._preview_timer.timeout.connect(self.refresh_preview)
        self._build_ui()
        self._set_enabled(False)

    # -- UI -----------------------------------------------------------
    def _build_ui(self):
        tb = self.addToolBar("Main")
        tb.setMovable(False)
        self.act_open = tb.addAction("Open PSD…", self.open_dialog)
        self.act_open.setShortcut(QtGui.QKeySequence.StandardKey.Open if QT6 else QtGui.QKeySequence.Open)
        tb.addSeparator()
        self.act_psd = tb.addAction("Save PSD…", lambda: self.export(psd=True, flat=False))
        self.act_flat = tb.addAction("Export flattened…", lambda: self.export(psd=False, flat=True))
        self.act_both = tb.addAction("Save both…", lambda: self.export(psd=True, flat=True))
        tb.addSeparator()
        tb.addAction("Fit view", lambda: self.view.fit())

        split = QtWidgets.QSplitter()
        self.setCentralWidget(split)

        # layers
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(6, 6, 6, 6)
        lv.addWidget(QtWidgets.QLabel("<b>Layers</b> — untick to remove, eye = visible"))
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Layer", "Eye", "Blend", "Note"])
        self.tree.setColumnWidth(0, 220)
        self.tree.setColumnWidth(1, 40)
        self.tree.setColumnWidth(2, 90)
        self.tree.itemChanged.connect(self._item_changed)
        lv.addWidget(self.tree, 1)
        row = QtWidgets.QHBoxLayout()
        b1 = QtWidgets.QPushButton("Remove hidden")
        b1.clicked.connect(self.remove_hidden)
        b2 = QtWidgets.QPushButton("Keep all")
        b2.clicked.connect(self.keep_all)
        row.addWidget(b1)
        row.addWidget(b2)
        lv.addLayout(row)
        split.addWidget(left)

        # preview
        mid = QtWidgets.QWidget()
        mv = QtWidgets.QVBoxLayout(mid)
        mv.setContentsMargins(0, 6, 0, 6)
        self.view = PreviewView()
        self.view.cropChanged.connect(self._crop_from_view)
        mv.addWidget(self.view, 1)
        self.status = QtWidgets.QLabel("Open or drop a PSD file.  Drag on the image to crop · "
                                       "wheel zooms · middle/Alt-drag pans")
        mv.addWidget(self.status)
        split.addWidget(mid)

        # settings
        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        rv.setContentsMargins(6, 6, 6, 6)

        self.info = QtWidgets.QLabel("—")
        self.info.setWordWrap(True)
        g = QtWidgets.QGroupBox("File")
        QtWidgets.QVBoxLayout(g).addWidget(self.info)
        rv.addWidget(g)

        # colour
        g = QtWidgets.QGroupBox("Colour")
        f = QtWidgets.QFormLayout(g)
        self.chk_convert = QtWidgets.QCheckBox("Convert to sRGB")
        self.chk_convert.setChecked(True)
        f.addRow(self.chk_convert)
        self.chk_embedded = QtWidgets.QCheckBox("Use embedded profile")
        self.chk_embedded.setChecked(True)
        f.addRow(self.chk_embedded)
        prow = QtWidgets.QHBoxLayout()
        self.cmb_profile = QtWidgets.QComboBox()
        self.cmb_profile.setMinimumWidth(160)
        bb = QtWidgets.QToolButton()
        bb.setText("…")
        bb.clicked.connect(self.browse_profile)
        prow.addWidget(self.cmb_profile, 1)
        prow.addWidget(bb)
        f.addRow("CMYK profile", prow)
        self.cmb_intent = QtWidgets.QComboBox()
        self.cmb_intent.addItems(list(ops.INTENTS))
        self.cmb_intent.setCurrentText(self.settings.value("intent", "Relative Colorimetric"))
        f.addRow("Intent", self.cmb_intent)
        self.chk_bpc = QtWidgets.QCheckBox("Black point compensation")
        self.chk_bpc.setChecked(True)
        f.addRow(self.chk_bpc)
        self.lbl_src = QtWidgets.QLabel("")
        self.lbl_src.setWordWrap(True)
        self.lbl_src.setStyleSheet("color: #999;")
        f.addRow(self.lbl_src)
        rv.addWidget(g)
        self._fill_profiles()

        # crop
        g = QtWidgets.QGroupBox("Crop")
        f = QtWidgets.QFormLayout(g)
        self.cmb_aspect = QtWidgets.QComboBox()
        for name, _ in ASPECTS:
            self.cmb_aspect.addItem(name)
        self.cmb_aspect.currentIndexChanged.connect(self._aspect_changed)
        f.addRow("Aspect", self.cmb_aspect)
        self.spins = []
        grid = QtWidgets.QGridLayout()
        for i, lab in enumerate(("X", "Y", "W", "H")):
            sp = QtWidgets.QSpinBox()
            sp.setRange(0, 300000)
            sp.valueChanged.connect(self._crop_from_spins)
            grid.addWidget(QtWidgets.QLabel(lab), i // 2, (i % 2) * 2)
            grid.addWidget(sp, i // 2, (i % 2) * 2 + 1)
            self.spins.append(sp)
        f.addRow(grid)
        self.chk_delete = QtWidgets.QCheckBox("Delete cropped pixels")
        self.chk_delete.setChecked(True)
        f.addRow(self.chk_delete)
        br = QtWidgets.QPushButton("Reset crop")
        br.clicked.connect(lambda: self.view.set_crop(None, emit=True))
        f.addRow(br)
        rv.addWidget(g)

        # output
        g = QtWidgets.QGroupBox("Flattened export")
        f = QtWidgets.QFormLayout(g)
        self.cmb_fmt = QtWidgets.QComboBox()
        self.cmb_fmt.addItems(["PNG", "TIFF"])
        self.cmb_fmt.setCurrentText(self.settings.value("fmt", "PNG"))
        f.addRow("Format", self.cmb_fmt)
        self.cmb_depth = QtWidgets.QComboBox()
        self.cmb_depth.addItems(["Same as PSD", "8-bit", "16-bit"])
        f.addRow("Bit depth", self.cmb_depth)
        self.chk_alpha = QtWidgets.QCheckBox("Include alpha")
        self.chk_alpha.setChecked(True)
        f.addRow(self.chk_alpha)
        self.cmb_source = QtWidgets.QComboBox()
        self.cmb_source.addItems(["Auto", "Rebuild from layers", "Photoshop composite"])
        self.cmb_source.setToolTip(
            "Auto: use Photoshop's saved composite (includes layer styles and adjustments) "
            "when no layers were removed or hidden, otherwise rebuild from the layers.")
        f.addRow("Image from", self.cmb_source)
        rv.addWidget(g)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setVisible(False)
        rv.addWidget(self.progress)
        self.logbox = QtWidgets.QPlainTextEdit()
        self.logbox.setReadOnly(True)
        self.logbox.setMaximumBlockCount(500)
        rv.addWidget(self.logbox, 1)
        split.addWidget(right)
        split.setSizes([330, 850, 320])

        for w in (self.chk_convert, self.chk_embedded, self.chk_bpc):
            w.toggled.connect(self._colour_changed)
        self.cmb_profile.currentIndexChanged.connect(self._colour_changed)
        self.cmb_intent.currentIndexChanged.connect(self._colour_changed)
        self.cmb_source.currentIndexChanged.connect(self.schedule_preview)

    def _set_enabled(self, on):
        for a in (self.act_psd, self.act_flat, self.act_both):
            a.setEnabled(on)

    def log(self, msg):
        self.logbox.appendPlainText(msg)

    # -- profiles -----------------------------------------------------
    def _fill_profiles(self):
        self.cmb_profile.blockSignals(True)
        self.cmb_profile.clear()
        self.cmb_profile.addItem("(none – simple formula)", None)
        found = ops.find_cmyk_profiles()
        extra = self.settings.value("extra_profiles", []) or []
        if isinstance(extra, str):
            extra = [extra]
        for p in extra:
            if os.path.isfile(p) and p not in [f[0] for f in found]:
                with open(p, "rb") as fh:
                    found.append((p, ops.profile_info(fh.read())[1] or os.path.basename(p)))
        for p, desc in found:
            self.cmb_profile.addItem(desc, p)
            self.cmb_profile.setItemData(self.cmb_profile.count() - 1, p, E(Qt, "ItemDataRole", "ToolTipRole"))
        last = self.settings.value("cmyk_profile", "")
        idx = self.cmb_profile.findData(last) if last else -1
        if idx < 0 and self.cmb_profile.count() > 1:
            idx = 1
        self.cmb_profile.setCurrentIndex(max(0, idx))
        self.cmb_profile.blockSignals(False)

    def browse_profile(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "CMYK ICC profile", "",
                                                     "ICC profiles (*.icc *.icm);;All files (*)")
        if not p:
            return
        extra = self.settings.value("extra_profiles", []) or []
        if isinstance(extra, str):
            extra = [extra]
        if p not in extra:
            extra.append(p)
        self.settings.setValue("extra_profiles", extra)
        self.settings.setValue("cmyk_profile", p)
        self._fill_profiles()
        self._colour_changed()

    # -- open ---------------------------------------------------------
    def open_dialog(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Photoshop file", self.settings.value("last_dir", ""),
            "Photoshop (*.psd *.psb);;All files (*)")
        if p:
            self.open(p)

    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        for u in ev.mimeData().urls():
            p = u.toLocalFile()
            if p.lower().endswith((".psd", ".psb")):
                self.open(p)
                break

    def open(self, path):
        self.settings.setValue("last_dir", os.path.dirname(path))
        self.log(f"Opening {path}")
        self._run(_load, path, on_done=lambda doc: self._loaded(path, doc))

    def _loaded(self, path, doc):
        self.doc, self.path = doc, path
        self.rgb_cache = {}
        longest = max(doc.width, doc.height)
        self.scale = max(1, math.ceil(longest / PREVIEW_MAX))
        self.setWindowTitle(f"PSD Prep {__version__} — {os.path.basename(path)}")
        for sp in self.spins:
            sp.blockSignals(True)
            sp.setMaximum(max(doc.width, doc.height))
            sp.blockSignals(False)
        self.view.crop = None
        self._fill_tree()
        self._update_info()
        self._set_enabled(True)
        self.chk_convert.setChecked(doc.mode != MODE_RGB or bool(doc.icc_profile))
        self._colour_changed()
        self.refresh_preview(fit=True)
        self._sync_spins()

    def _update_info(self):
        d = self.doc
        prof = d.icc_profile
        pdesc = ops.profile_info(prof)[1] if prof else "none"
        res = d.resolution
        kinds = {}
        for L in d.layers:
            if not L.is_divider:
                k = L.kind()
                kinds[k] = kinds.get(k, 0) + 1
        fx = sum(1 for L in d.layers if L.has_effects())
        lines = [f"<b>{os.path.basename(self.path)}</b>",
                 f"{d.width} × {d.height}px · {MODE_NAMES.get(d.mode)} · {d.depth}-bit"
                 + (f" · {res:g} dpi" if res else ""),
                 f"Profile: {pdesc}",
                 "Layers: " + ", ".join(f"{v} {k}" for k, v in kinds.items()) if kinds else "No layers"]
        if fx:
            lines.append(f"<span style='color:#e0a030'>{fx} layer(s) with styles/effects</span>")
        if not ops.merged_is_usable(d):
            lines.append("<span style='color:#999'>No saved composite (Maximize compatibility off)</span>")
        self.info.setText("<br>".join(lines))

    # -- layer tree -----------------------------------------------------
    def _fill_tree(self):
        self._building = True
        self.tree.clear()
        self.items = {}
        index_of = {id(L): i for i, L in enumerate(self.doc.layers)}

        def add(nodes, parent):
            for n in reversed(nodes):       # top first
                L = n.layer
                it = QtWidgets.QTreeWidgetItem([L.name, "", ops.BLEND_NAMES.get(L.blend(), L.blend()),
                                                self._note(L)])
                flags = it.flags() | E(Qt, "ItemFlag", "ItemIsUserCheckable")
                it.setFlags(flags)
                it.setCheckState(0, CHECKED)
                it.setCheckState(1, CHECKED if L.visible else UNCHECKED)
                it.setData(0, E(Qt, "ItemDataRole", "UserRole"), index_of[id(L)])
                if L.opacity < 255:
                    it.setText(2, it.text(2) + f" {round(L.opacity / 2.55)}%")
                if L.clipping:
                    it.setText(0, "↳ " + L.name)
                (parent.addChild(it) if isinstance(parent, QtWidgets.QTreeWidgetItem)
                 else self.tree.addTopLevelItem(it))
                self.items[index_of[id(L)]] = it
                if n.is_group:
                    add(n.children, it)
                    it.setExpanded(True)
        add(ops.build_tree(self.doc.layers), self.tree)
        self._building = False

    @staticmethod
    def _note(L):
        parts = []
        k = KIND_LABEL.get(L.kind(), "")
        if k and L.kind() != "group":
            parts.append(k)
        if L.has_effects():
            parts.append("FX not rendered")
        if L.mask is not None and L.mask.data is not None:
            parts.append("mask" + (" (off)" if L.mask.disabled else ""))
        return ", ".join(parts)

    def _item_changed(self, item, col):
        if self._building:
            return
        if col == 0:
            keep = item.checkState(0) == CHECKED
            self._building = True
            self._set_children_enabled(item, keep)
            self._building = False
        self.schedule_preview()

    def _set_children_enabled(self, item, on):
        for i in range(item.childCount()):
            ch = item.child(i)
            ch.setDisabled(not on)
            self._set_children_enabled(ch, on and ch.checkState(0) == CHECKED)

    def remove_hidden(self):
        self._building = True
        for idx, it in self.items.items():
            if it.checkState(1) != CHECKED:
                it.setCheckState(0, UNCHECKED)
        self._building = False
        for it in self.items.values():
            self._set_children_enabled(it, it.checkState(0) == CHECKED)
        self.schedule_preview()

    def keep_all(self):
        self._building = True
        for it in self.items.values():
            it.setCheckState(0, CHECKED)
            it.setDisabled(False)
        self._building = False
        self.schedule_preview()

    def _removed(self):
        return {i for i, it in self.items.items() if it.checkState(0) != CHECKED}

    def _visibility(self):
        return {i: it.checkState(1) == CHECKED for i, it in self.items.items()}

    # -- crop sync --------------------------------------------------------
    def _aspect_changed(self, idx):
        a = ASPECTS[idx][1]
        if a == "orig" and self.doc:
            a = self.doc.width / self.doc.height
        self.view.aspect = a
        if a and self.doc:
            x, y, w, h = self.view.crop or (0, 0, self.doc.width, self.doc.height)
            # largest box of that aspect centred in the current one
            if w / h > a:
                nw, nh = h * a, h
            else:
                nw, nh = w, w / a
            self.view.set_crop((x + (w - nw) / 2, y + (h - nh) / 2, nw, nh), emit=True)

    def _crop_from_view(self, rect):
        self._sync_spins()

    def _sync_spins(self):
        if not self.doc:
            return
        vals = self.view.crop or (0, 0, self.doc.width, self.doc.height)
        for sp, v in zip(self.spins, vals):
            sp.blockSignals(True)
            sp.setValue(int(v))
            sp.blockSignals(False)
        c = self.view.crop
        self.status.setText(f"Crop: {c[2]} × {c[3]} at {c[0]},{c[1]}" if c else
                            f"Full canvas {self.doc.width} × {self.doc.height}")

    def _crop_from_spins(self):
        if not self.doc:
            return
        x, y, w, h = [s.value() for s in self.spins]
        self.view.set_crop((x, y, w, h))
        self._sync_spins()

    # -- colour / preview ---------------------------------------------------
    def _settings(self) -> Settings:
        depth = {0: None, 1: 8, 2: 16}[self.cmb_depth.currentIndex()]
        return Settings(
            remove=self._removed(), visibility=self._visibility(), crop=self.view.crop,
            delete_cropped=self.chk_delete.isChecked(), convert=self.chk_convert.isChecked(),
            use_embedded=self.chk_embedded.isChecked(), cmyk_profile=self.cmb_profile.currentData(),
            intent=self.cmb_intent.currentText(), bpc=self.chk_bpc.isChecked(),
            flatten_source=["auto", "layers", "composite"][self.cmb_source.currentIndex()],
            flat_depth=depth, flat_alpha=self.chk_alpha.isChecked())

    def _colour_changed(self, *a):
        self.settings.setValue("intent", self.cmb_intent.currentText())
        if self.cmb_profile.currentData():
            self.settings.setValue("cmyk_profile", self.cmb_profile.currentData())
        self.rgb_cache = {}
        if self.doc:
            conv = make_converter(self.doc, self._settings())
            txt = f"Source: {conv.source_desc}"
            if self.doc.mode == MODE_CMYK and conv.method == "naive":
                txt += " — pick a CMYK profile for accurate colour"
            self.lbl_src.setText(txt)
        self.schedule_preview()

    def schedule_preview(self, *a):
        if self.doc:
            self._preview_timer.start()

    def refresh_preview(self, fit=False):
        if not self.doc:
            return
        doc, s = self.doc, self._settings()
        conv = make_converter(doc, s)
        if doc.mode == MODE_RGB and not (s.convert and conv.method == "icc"):
            conv = None
        try:
            if use_saved_composite(doc, s):
                rgb, alpha = ops.merged_rgba(doc, conv, self.scale)
                src = "Photoshop composite"
            else:
                hidden = {id(doc.layers[i]) for i in s.remove}
                hidden |= {id(doc.layers[i]) for i, v in s.visibility.items() if not v}
                # visibility override: layers hidden in file but ticked visible
                force_vis = {i for i, v in s.visibility.items() if v and not doc.layers[i].visible}
                saved = {i: doc.layers[i].flags for i in force_vis}
                for i in force_vis:
                    doc.layers[i].visible = True
                try:
                    comp = ops.Compositor(doc, self._cached_rgb(conv), self.scale, hidden)
                    rgb, alpha = comp.run()
                finally:
                    for i, fl in saved.items():
                        doc.layers[i].flags = fl
                src = "rebuilt from layers"
                if comp.unsupported:
                    src += " (drawn as Normal: " + ", ".join(sorted(comp.unsupported)) + ")"
        except Exception as e:
            self.log(f"Preview failed: {e}")
            return
        img = self._to_qimage(rgb, alpha)
        self.view.set_image(img, doc.width, doc.height, keep_view=not fit)
        self.setStatusTip(src)
        self.statusBar().showMessage(f"Preview: {src}" + (f" · 1/{self.scale} scale" if self.scale > 1 else ""))

    def _cached_rgb(self, conv):
        nc = self.doc.color_channels
        maxv = self.doc.max_value

        def get(L, src):
            key = id(L)
            if key not in self.rgb_cache:
                if not all(c in L.channels for c in range(nc)):
                    self.rgb_cache[key] = None
                else:
                    chans = [np.ascontiguousarray(L.channels[c][src]) for c in range(nc)]
                    if conv is not None:
                        self.rgb_cache[key] = conv.to_rgb_float(chans)
                    else:
                        self.rgb_cache[key] = np.stack(chans[:3], -1).astype(np.float32) / maxv
            return self.rgb_cache[key]
        return get

    @staticmethod
    def _to_qimage(rgb, alpha):
        h, w = alpha.shape
        yy, xx = np.mgrid[0:h, 0:w]
        checker = np.where(((yy // 12) + (xx // 12)) % 2 == 0, 0.42, 0.32).astype(np.float32)
        out = rgb * alpha[..., None] + checker[..., None] * (1 - alpha[..., None])
        buf = np.ascontiguousarray(np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8))
        img = QtGui.QImage(buf.tobytes(), w, h, 3 * w, E(QtGui.QImage, "Format", "Format_RGB888"))
        return img.copy()

    # -- export ------------------------------------------------------------
    def export(self, psd=True, flat=True):
        if not self.doc:
            return
        s = self._settings()
        base = os.path.splitext(self.path)[0]
        outdir = self.settings.value("out_dir", os.path.dirname(self.path))
        stem = os.path.join(outdir, os.path.basename(base))
        if psd:
            p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save PSD", stem + "_edit.psd",
                                                         "Photoshop (*.psd)")
            if not p:
                return
            if not p.lower().endswith((".psd", ".psb")):
                p += ".psd"
            if os.path.abspath(p) == os.path.abspath(self.path):
                QtWidgets.QMessageBox.warning(self, "PSD Prep", "Choose a new name — the original "
                                              "file is never overwritten.")
                return
            s.out_psd = p
            stem = os.path.splitext(p)[0]
        if flat:
            fmt = self.cmb_fmt.currentText()
            ext = ".png" if fmt == "PNG" else ".tif"
            p, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Export flattened image", stem + ext,
                "PNG (*.png);;TIFF (*.tif *.tiff)" if fmt == "PNG" else "TIFF (*.tif *.tiff);;PNG (*.png)")
            if not p:
                return
            if not p.lower().endswith((".png", ".tif", ".tiff")):
                p += ext
            s.out_flat = p
            self.settings.setValue("fmt", fmt)
        self.settings.setValue("out_dir", os.path.dirname(s.out_psd or s.out_flat))
        self._run(process, self.doc, s, on_done=self._exported)

    def _exported(self, res):
        msg = "Done"
        if res.get("psd"):
            msg += f" — {os.path.basename(res['psd'])}"
        if res.get("flat"):
            msg += f" — {os.path.basename(res['flat'])}"
        self.statusBar().showMessage(msg + f" (image {res['composite_source']})", 10000)

    # -- threading ------------------------------------------------------------
    def _run(self, fn, *args, on_done=None):
        if self.thread is not None:
            return
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.setEnabled(False)
        self.thread = QtCore.QThread(self)
        self.worker = Worker(fn, *args)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self.log)
        self.worker.progress.connect(self._progress)
        self.worker.done.connect(lambda r: self._finish(r, on_done))
        self.worker.failed.connect(self._failed)
        self.thread.start()

    def _progress(self, a, b):
        self.progress.setRange(0, b)
        self.progress.setValue(a)

    def _cleanup(self):
        self.thread.quit()
        self.thread.wait()
        self.thread = None
        self.progress.setVisible(False)
        self.setEnabled(True)

    def _finish(self, res, on_done):
        self._cleanup()
        if on_done:
            try:
                on_done(res)
            except Exception as e:
                self._failed(f"{e}\n\n{traceback.format_exc()}")

    def _failed(self, msg):
        if self.thread is not None:
            self._cleanup()
        self.log("ERROR: " + msg.split("\n")[0])
        box = QtWidgets.QMessageBox(self)
        box.setIcon(E(QtWidgets.QMessageBox, "Icon", "Critical"))
        box.setWindowTitle("PSD Prep")
        box.setText(msg.split("\n")[0])
        box.setDetailedText(msg)
        box.exec() if hasattr(box, "exec") else box.exec_()


def main(argv=None):
    argv = sys.argv if argv is None else argv
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    app.setApplicationName("PSD Prep")
    app.setStyle("Fusion")
    pal = QtGui.QPalette()
    for role, col in (("Window", (45, 45, 48)), ("WindowText", (220, 220, 220)),
                      ("Base", (30, 30, 32)), ("AlternateBase", (45, 45, 48)),
                      ("Text", (220, 220, 220)), ("Button", (55, 55, 58)),
                      ("ButtonText", (220, 220, 220)), ("Highlight", (230, 150, 30)),
                      ("HighlightedText", (0, 0, 0)), ("ToolTipBase", (30, 30, 32)),
                      ("ToolTipText", (220, 220, 220))):
        pal.setColor(E(QtGui.QPalette, "ColorRole", role), QtGui.QColor(*col))
    app.setPalette(pal)
    w = MainWindow()
    w.show()
    for a in argv[1:]:
        if a.lower().endswith((".psd", ".psb")) and os.path.isfile(a):
            w.open(a)
            break
    return app.exec() if hasattr(app, "exec") else app.exec_()

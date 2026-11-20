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

from . import artboards as ab_mod, check as check_mod, colorspaces, naming, ops, __version__
from .version import BUILD
from .pipeline import (Settings, colour_tag, export_artboards, make_converter, process,
                       use_saved_composite)
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


def _run_check(doc, conv, scale, hidden, log=None, progress=None):
    return check_mod.conversion_check(doc, conv, scale, hidden)


def _load_vector(path, dpi, long_side, log=None, progress=None):
    from .vector import load_vector
    return load_vector(path, dpi, long_side, progress)


class ResolutionDialog(QtWidgets.QDialog):
    """Asked when opening an Illustrator/PDF file: vector art has no pixel
    size, so pick one."""

    def __init__(self, parent, path, pages, settings):
        super().__init__(parent)
        self.setWindowTitle("Render Illustrator / PDF")
        self.pages = pages
        v = QtWidgets.QVBoxLayout(self)
        v.addWidget(QtWidgets.QLabel(f"<b>{os.path.basename(path)}</b> — {len(pages)} artboard(s), "
                                     f"first one {pages[0].size_text}"))
        f = QtWidgets.QGridLayout()
        self.rad_dpi = QtWidgets.QRadioButton("Resolution (DPI)")
        self.rad_long = QtWidgets.QRadioButton("Longest side (px)")
        self.spn_dpi = QtWidgets.QSpinBox()
        self.spn_dpi.setRange(18, 2400)
        self.spn_dpi.setValue(int(settings.value("vec_dpi", 300)))
        self.spn_long = QtWidgets.QSpinBox()
        self.spn_long.setRange(16, 20000)
        self.spn_long.setSingleStep(100)
        self.spn_long.setValue(int(settings.value("vec_long", 4000)))
        (self.rad_long if settings.value("vec_mode", "dpi") == "long" else self.rad_dpi).setChecked(True)
        f.addWidget(self.rad_dpi, 0, 0)
        f.addWidget(self.spn_dpi, 0, 1)
        f.addWidget(self.rad_long, 1, 0)
        f.addWidget(self.spn_long, 1, 1)
        v.addLayout(f)
        self.lbl = QtWidgets.QLabel("")
        self.lbl.setStyleSheet("color: #999;")
        self.lbl.setWordWrap(True)
        v.addWidget(self.lbl)
        bb = QtWidgets.QDialogButtonBox(E(QtWidgets.QDialogButtonBox, "StandardButton", "Ok")
                                        | E(QtWidgets.QDialogButtonBox, "StandardButton", "Cancel"))
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)
        for w in (self.rad_dpi, self.rad_long):
            w.toggled.connect(self._update)
        self.spn_dpi.valueChanged.connect(self._update)
        self.spn_long.valueChanged.connect(self._update)
        self._update()

    def values(self):
        if self.rad_long.isChecked():
            return None, self.spn_long.value()
        return self.spn_dpi.value(), None

    def _update(self, *a):
        dpi, long_side = self.values()
        sizes = [p.pixels(dpi, long_side) for p in self.pages[:4]]
        txt = ", ".join(f"{w} × {h}" for w, h in sizes)
        if len(self.pages) > 4:
            txt += " …"
        self.lbl.setText(f"Pixel size: {txt}\n300 DPI = print quality at real size. For screen/"
                         f"comp use, 'Longest side' gives an exact pixel size.")

    def remember(self, settings):
        settings.setValue("vec_mode", "long" if self.rad_long.isChecked() else "dpi")
        settings.setValue("vec_dpi", self.spn_dpi.value())
        settings.setValue("vec_long", self.spn_long.value())


# ---------------------------------------------------------------------------
# Preview with crop box
# ---------------------------------------------------------------------------
class PreviewView(QtWidgets.QGraphicsView):
    cropChanged = Signal(object)      # (x, y, w, h) in document pixels, or None
    hovered = Signal(int, int)        # document pixel under the mouse

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
        self.hl = QtWidgets.QGraphicsRectItem()
        hpen = QtGui.QPen(QtGui.QColor(60, 200, 255))
        hpen.setCosmetic(True)
        hpen.setWidth(2)
        hpen.setStyle(E(Qt, "PenStyle", "DashLine"))
        self.hl.setPen(hpen)
        self.hl.setZValue(7)
        self.hl.setVisible(False)
        self.scene().addItem(self.hl)
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

    def set_highlight(self, rect):
        """Dashed outline (artboard) without touching the crop."""
        if rect:
            self.hl.setRect(QtCore.QRectF(*rect))
            self.hl.setVisible(True)
        else:
            self.hl.setVisible(False)

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
            self.hovered.emit(int(sp.x()), int(sp.y()))
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
        self.setWindowTitle(f"PSD Prep {__version__} ({BUILD})")
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
        self.act_psd = tb.addAction("Save PSD", lambda: self.export(psd=True, flat=False))
        self.act_flat = tb.addAction("Export flattened", lambda: self.export(psd=False, flat=True))
        self.act_both = tb.addAction("Save both", lambda: self.export(psd=True, flat=True))
        tb.addSeparator()
        self.act_rerender = tb.addAction("Change resolution…", self._rerender)
        self.act_rerender.setVisible(False)
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
        grid = QtWidgets.QGridLayout()
        for label, fn, tip, (row, col, span) in (
                ("Show all", self.show_all, "Turn every layer's eye on", (0, 0, 1)),
                ("Hide all", self.hide_all, "Turn every layer's eye off", (0, 1, 1)),
                ("Reset to original", self.reset_layers,
                 "Back to how the file was saved: same visible/hidden layers, nothing removed",
                 (1, 0, 2)),
                ("Remove hidden", self.remove_hidden, "Untick every layer whose eye is off", (2, 0, 1)),
                ("Keep all", self.keep_all, "Tick every layer again (nothing removed)", (2, 1, 1))):
            btn = QtWidgets.QPushButton(label)
            btn.setToolTip(tip)
            btn.clicked.connect(fn)
            grid.addWidget(btn, row, col, 1, span)
        lv.addLayout(grid)
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
        right_outer = QtWidgets.QWidget()
        ro = QtWidgets.QVBoxLayout(right_outer)
        ro.setContentsMargins(0, 0, 0, 0)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(E(QtWidgets.QFrame, "Shape", "NoFrame"))
        right = QtWidgets.QWidget()
        scroll.setWidget(right)
        ro.addWidget(scroll, 3)
        rv = QtWidgets.QVBoxLayout(right)
        rv.setContentsMargins(6, 6, 6, 6)

        g = QtWidgets.QGroupBox("Original (never modified)")
        f = QtWidgets.QFormLayout(g)
        self.orig = {}
        for key, label in (("file", "File"), ("size", "Size"), ("mode", "Colour mode"),
                           ("depth", "Bit depth"), ("profile", "Colour profile"),
                           ("space", "Profile space"), ("res", "Resolution"),
                           ("layers", "Layers"), ("boards", "Artboards")):
            lab = QtWidgets.QLabel("—")
            lab.setWordWrap(True)
            lab.setTextInteractionFlags(E(Qt, "TextInteractionFlag", "TextSelectableByMouse"))
            f.addRow(label, lab)
            self.orig[key] = lab
        self.info = QtWidgets.QLabel("")
        self.info.setWordWrap(True)
        f.addRow(self.info)
        rv.addWidget(g)

        # colour
        g = QtWidgets.QGroupBox("Colour")
        f = QtWidgets.QFormLayout(g)
        self.cmb_target = QtWidgets.QComboBox()
        self.cmb_target.addItem("No conversion", None)
        for key, t in colorspaces.TARGETS.items():
            self.cmb_target.addItem(t.label, key)
            self.cmb_target.setItemData(self.cmb_target.count() - 1, t.tooltip,
                                        E(Qt, "ItemDataRole", "ToolTipRole"))
        last = self.settings.value("target", colorspaces.DEFAULT_TARGET)
        self.cmb_target.setCurrentIndex(max(1, self.cmb_target.findData(last)))
        f.addRow("Convert to", self.cmb_target)
        self.btn_check = QtWidgets.QPushButton("Check conversion")
        self.btn_check.setToolTip("Measure every converted pixel against the source colour profile "
                                  "(ΔE2000) and highlight anything that changed visibly")
        self.btn_check.clicked.connect(self.check_conversion)
        f.addRow(self.btn_check)
        self.chk_highlight = QtWidgets.QCheckBox("Highlight pixels with ΔE > 2")
        self.chk_highlight.setEnabled(False)
        self.chk_highlight.toggled.connect(lambda *_: self.refresh_preview())
        f.addRow(self.chk_highlight)
        self.lbl_check = QtWidgets.QLabel("Hover the image to compare source and converted values.")
        self.lbl_check.setWordWrap(True)
        self.lbl_check.setStyleSheet("color: #999;")
        self.lbl_check.setTextInteractionFlags(E(Qt, "TextInteractionFlag", "TextSelectableByMouse"))
        f.addRow(self.lbl_check)
        rv.addWidget(g)
        self.de_map = None
        # Fixed conversion settings (Photoshop's defaults): the file's embedded
        # profile, else the bundled generic CMYK profile; relative colorimetric + BPC.
        found = ops.find_cmyk_profiles()
        self.cmyk_profile = found[0][0] if found else None

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

        # where to save
        g = QtWidgets.QGroupBox("Save to")
        f = QtWidgets.QFormLayout(g)
        self.rad_auto = QtWidgets.QRadioButton("Next to the original, auto-named")
        self.rad_custom = QtWidgets.QRadioButton("Other folder and name")
        self.rad_ask = QtWidgets.QRadioButton("Ask where to save each time")
        mode = self.settings.value("save_mode", "auto")
        {"ask": self.rad_ask, "custom": self.rad_custom}.get(mode, self.rad_auto).setChecked(True)
        f.addRow(self.rad_auto)
        self.txt_suffix = QtWidgets.QLineEdit(self.settings.value("suffix", naming.DEFAULT_SUFFIX))
        self.txt_suffix.setPlaceholderText("e.g. cleanup")
        f.addRow("Name suffix", self.txt_suffix)
        f.addRow(self.rad_custom)
        frow = QtWidgets.QHBoxLayout()
        self.txt_folder = QtWidgets.QLineEdit(self.settings.value("custom_folder", ""))
        self.txt_folder.setPlaceholderText("output folder")
        bf = QtWidgets.QToolButton()
        bf.setText("…")
        bf.clicked.connect(self._browse_folder)
        frow.addWidget(self.txt_folder, 1)
        frow.addWidget(bf)
        f.addRow("Folder", frow)
        self.txt_name = QtWidgets.QLineEdit("")
        self.txt_name.setPlaceholderText("file name (version and colour space are added)")
        f.addRow("Name", self.txt_name)
        f.addRow(self.rad_ask)
        self.lbl_next = QtWidgets.QLabel("")
        self.lbl_next.setWordWrap(True)
        self.lbl_next.setStyleSheet("color: #999;")
        f.addRow(self.lbl_next)
        for w in (self.rad_auto, self.rad_custom, self.rad_ask):
            w.toggled.connect(self._update_next_name)
        for w in (self.txt_suffix, self.txt_folder, self.txt_name):
            w.textChanged.connect(self._update_next_name)
        self.cmb_fmt.currentIndexChanged.connect(self._update_next_name)
        rv.addWidget(g)

        # artboards
        self.grp_boards = QtWidgets.QGroupBox("Artboards")
        bv = QtWidgets.QVBoxLayout(self.grp_boards)
        bv.addWidget(QtWidgets.QLabel("Tick to export · drag to reorder (sets the numbering)"))
        self.lst_boards = QtWidgets.QListWidget()
        self.lst_boards.setDragDropMode(E(QtWidgets.QAbstractItemView, "DragDropMode", "InternalMove"))
        self.lst_boards.setMinimumHeight(110)
        self.lst_boards.currentRowChanged.connect(self._board_selected)
        bv.addWidget(self.lst_boards)
        row = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("Reading order")
        b.setToolTip("Rows top to bottom, then left to right")
        b.clicked.connect(lambda: self._order_boards("position"))
        row.addWidget(b)
        b = QtWidgets.QPushButton("Layers order")
        b.clicked.connect(lambda: self._order_boards("layers"))
        row.addWidget(b)
        bv.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        self.chk_board_psd = QtWidgets.QCheckBox("PSD")
        self.chk_board_psd.setChecked(True)
        self.chk_board_flat = QtWidgets.QCheckBox("Flattened")
        self.chk_board_flat.setChecked(True)
        row.addWidget(self.chk_board_psd)
        row.addWidget(self.chk_board_flat)
        bv.addLayout(row)
        self.chk_board_transp = QtWidgets.QCheckBox("Transparent background (ignore artboard colour)")
        bv.addWidget(self.chk_board_transp)
        b = QtWidgets.QPushButton("Export artboards")
        b.clicked.connect(self.export_boards)
        bv.addWidget(b)
        self.grp_boards.setVisible(False)
        rv.addWidget(self.grp_boards)
        rv.addStretch(1)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setVisible(False)
        ro.addWidget(self.progress)
        self.logbox = QtWidgets.QPlainTextEdit()
        self.logbox.setReadOnly(True)
        self.logbox.setMaximumBlockCount(500)
        ro.addWidget(self.logbox, 1)
        split.addWidget(right_outer)
        split.setSizes([330, 820, 350])
        self.boards = []

        self.cmb_target.currentIndexChanged.connect(self._colour_changed)
        self.view.hovered.connect(self._probe)
        self.cmb_source.currentIndexChanged.connect(self.schedule_preview)

    def _set_enabled(self, on):
        vector = self.doc is not None and self.doc.source_kind == "vector"
        self.act_flat.setEnabled(on)
        # Illustrator/PDF sources export flattened PNG/TIFF only
        self.act_psd.setEnabled(on and not vector)
        self.act_both.setEnabled(on and not vector)
        self.act_rerender.setVisible(on and vector)
        self.chk_board_psd.setEnabled(not vector)
        if vector:
            self.chk_board_psd.setChecked(False)
            self.chk_board_flat.setChecked(True)

    def log(self, msg):
        self.logbox.appendPlainText(msg)

    # -- open ---------------------------------------------------------
    def open_dialog(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open file", self.settings.value("last_dir", ""),
            "Photoshop / Illustrator / PDF (*.psd *.psb *.ai *.pdf);;All files (*)")
        if p:
            self.open(p)

    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        for u in ev.mimeData().urls():
            p = u.toLocalFile()
            if p.lower().endswith((".psd", ".psb", ".ai", ".pdf")):
                self.open(p)
                break

    def open(self, path):
        self.settings.setValue("last_dir", os.path.dirname(path))
        from .vector import is_vector, page_infos, VectorError
        if is_vector(path):
            try:
                pages = page_infos(path)
            except VectorError as e:
                QtWidgets.QMessageBox.warning(self, "PSD Prep", str(e))
                return
            dlg = ResolutionDialog(self, path, pages, self.settings)
            ok = dlg.exec() if hasattr(dlg, "exec") else dlg.exec_()
            if not ok:
                return
            dlg.remember(self.settings)
            self._open_vector(path, *dlg.values())
            return
        self.log(f"Opening {path}")
        self._run(_load, path, on_done=lambda doc: self._loaded(path, doc))

    def _open_vector(self, path, dpi=None, long_side=None):
        how = f"{dpi} DPI" if dpi else f"longest side {long_side} px"
        self.log(f"Rendering {path} at {how}")
        self._run(_load_vector, path, dpi, long_side, on_done=lambda doc: self._loaded(path, doc))

    def _rerender(self):
        if self.path and self.doc is not None and self.doc.source_kind == "vector":
            self.open(self.path)

    def _loaded(self, path, doc):
        self.doc, self.path = doc, path
        self.rgb_cache = {}
        longest = max(doc.width, doc.height)
        self.scale = max(1, math.ceil(longest / PREVIEW_MAX))
        self.setWindowTitle(f"PSD Prep {__version__} ({BUILD}) — {os.path.basename(path)}")
        for sp in self.spins:
            sp.blockSignals(True)
            sp.setMaximum(max(doc.width, doc.height))
            sp.blockSignals(False)
        self.view.crop = None
        self._fill_tree()
        self._update_info()
        self._set_enabled(True)
        self.de_map = None
        self.chk_highlight.blockSignals(True)
        self.chk_highlight.setChecked(False)
        self.chk_highlight.setEnabled(False)
        self.chk_highlight.blockSignals(False)
        self.lbl_check.setText("Hover the image to compare source and converted values.")
        self._colour_changed()
        self.txt_name.blockSignals(True)
        self.txt_name.setText(naming.base_name(path, self.txt_suffix.text()))
        self.txt_name.blockSignals(False)
        self.boards = ab_mod.find_artboards(doc)
        self._fill_boards(self.boards)
        self.refresh_preview(fit=True)
        self._sync_spins()
        self._update_next_name()

    # -- artboards ------------------------------------------------------------
    def _fill_boards(self, boards):
        self.lst_boards.clear()
        self.view.set_highlight(None)
        for b in boards:
            it = QtWidgets.QListWidgetItem(b.label)
            it.setFlags(it.flags() | E(Qt, "ItemFlag", "ItemIsUserCheckable"))
            it.setCheckState(CHECKED)
            it.setData(E(Qt, "ItemDataRole", "UserRole"), self.boards.index(b))
            self.lst_boards.addItem(it)
        self.grp_boards.setVisible(bool(self.boards))

    def _order_boards(self, how):
        checked = {self.lst_boards.item(i).data(E(Qt, "ItemDataRole", "UserRole")):
                   self.lst_boards.item(i).checkState() for i in range(self.lst_boards.count())}
        order = ab_mod.sort_by_position(self.boards) if how == "position" else list(self.boards)
        self._fill_boards(order)
        for i in range(self.lst_boards.count()):
            it = self.lst_boards.item(i)
            it.setCheckState(checked.get(it.data(E(Qt, "ItemDataRole", "UserRole")), CHECKED))

    def _board_selected(self, row):
        if 0 <= row < self.lst_boards.count():
            b = self.boards[self.lst_boards.item(row).data(E(Qt, "ItemDataRole", "UserRole"))]
            self.view.set_highlight(b.rect)
        else:
            self.view.set_highlight(None)

    def _selected_boards(self):
        out = []
        for i in range(self.lst_boards.count()):
            it = self.lst_boards.item(i)
            if it.checkState() == CHECKED:
                out.append(self.boards[it.data(E(Qt, "ItemDataRole", "UserRole"))])
        return out

    def _update_vector_info(self):
        d = self.doc
        o = self.orig
        pages = d.vector_pages
        o["file"].setText(os.path.basename(self.path))
        o["file"].setToolTip(self.path)
        kind = "Adobe Illustrator" if self.path.lower().endswith(".ai") else "PDF"
        o["size"].setText(", ".join(p.size_text for p in pages[:3]) + (" …" if len(pages) > 3 else ""))
        o["mode"].setText(f"<b>{kind} (vector)</b> — rendered to RGB")
        o["depth"].setText("8-bit")
        o["profile"].setText("— (colour handled by the PDF renderer)")
        o["space"].setText("—")
        vs = d.vector_settings
        o["res"].setText(f"{vs['dpi']} dpi" if vs.get("dpi") else f"longest side {vs.get('long_side')} px")
        o["layers"].setText("flattened per artboard (Illustrator layers can't be read)")
        o["boards"].setText(str(len(pages)))
        self.info.setVisible(False)

    def _update_info(self):
        d = self.doc
        if d.source_kind == "vector":
            self._update_vector_info()
            return
        prof = d.icc_profile
        pdesc = ops.profile_info(prof)[1] if prof else "none"
        res = d.resolution
        kinds = {}
        for L in d.layers:
            if not L.is_divider:
                k = L.kind()
                kinds[k] = kinds.get(k, 0) + 1
        fx = sum(1 for L in d.layers if L.has_effects())
        space = ops.profile_info(prof)[0] if prof else ""
        space_name = {"RGB": "RGB", "CMYK": "CMYK", "GRAY": "Gray", "Lab": "Lab"}.get(space, space)
        o = self.orig
        o["file"].setText(os.path.basename(self.path))
        o["file"].setToolTip(self.path)
        o["size"].setText(f"{d.width} × {d.height} px")
        o["mode"].setText(f"<b>{MODE_NAMES.get(d.mode, d.mode)}</b>")
        o["depth"].setText(f"{d.depth}-bit")
        o["profile"].setText(f"<b>{pdesc}</b>" if prof else
                             "<span style='color:#e0a030'>none embedded</span>"
                             + (" (assumed sRGB)" if d.mode == MODE_RGB else ""))
        o["space"].setText(space_name or "—")
        o["res"].setText(f"{res:g} dpi" if res else "—")
        o["layers"].setText(", ".join(f"{v} {k}" for k, v in kinds.items()) if kinds else "none")
        nb = len(ab_mod.find_artboards(d))
        o["boards"].setText(str(nb) if nb else "none")
        lines = []
        if fx:
            lines.append(f"<span style='color:#e0a030'>{fx} layer(s) with styles/effects</span>")
        if not ops.merged_is_usable(d):
            lines.append("<span style='color:#999'>No saved composite (Maximize compatibility off)</span>")
        self.info.setText("<br>".join(lines))
        self.info.setVisible(bool(lines))

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
        if L.unrendered:
            parts.append(L.unrendered)
        if L.fill_opacity < 255:
            parts.append(f"fill {round(L.fill_opacity / 2.55)}%")
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

    def _set_all_eyes(self, state_for):
        self._building = True
        for idx, it in self.items.items():
            it.setCheckState(1, state_for(idx))
        self._building = False
        self.schedule_preview()

    def show_all(self):
        self._set_all_eyes(lambda i: CHECKED)

    def hide_all(self):
        self._set_all_eyes(lambda i: UNCHECKED)

    def reset_layers(self):
        """Back to the file as saved: original eyes, nothing removed."""
        self._building = True
        for idx, it in self.items.items():
            it.setCheckState(0, CHECKED)
            it.setDisabled(False)
            it.setCheckState(1, CHECKED if self.doc.layers[idx].visible else UNCHECKED)
        self._building = False
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
            delete_cropped=self.chk_delete.isChecked(),
            convert=self.cmb_target.currentData() is not None,
            target=self.cmb_target.currentData() or "srgb",
            use_embedded=True, cmyk_profile=self.cmyk_profile,
            intent="Relative Colorimetric", bpc=True,
            flatten_source=["auto", "layers", "composite"][self.cmb_source.currentIndex()],
            flat_depth=depth, flat_alpha=self.chk_alpha.isChecked())

    def _colour_changed(self, *a):
        self.rgb_cache = {}
        if self.cmb_target.currentData():
            self.settings.setValue("target", self.cmb_target.currentData())
        self.de_map = None
        self.chk_highlight.setChecked(False)
        self.chk_highlight.setEnabled(False)
        if self.doc:
            conv = make_converter(self.doc, self._settings())
            self.cmb_target.setToolTip(f"Source: {conv.source_desc}")
        self._update_next_name()
        self.schedule_preview()

    def _display_converter(self):
        """Converter for what the preview shows: the chosen target, or sRGB
        display of an unconverted file."""
        s = self._settings()
        conv = make_converter(self.doc, s)
        if self.doc.mode == MODE_RGB and not s.convert:
            return None
        return conv

    def check_conversion(self):
        if not self.doc:
            return
        s = self._settings()
        if not s.convert:
            self.lbl_check.setText("Choose a colour space in 'Convert to' first.")
            return
        hidden = {id(self.doc.layers[i]) for i in s.remove}
        hidden |= {id(self.doc.layers[i]) for i, v in s.visibility.items() if not v}
        conv = make_converter(self.doc, s)
        self._run(_run_check, self.doc, conv, self.scale, hidden, on_done=self._checked)

    def _checked(self, res):
        self.lbl_check.setText(res.message)
        self.log(res.message)
        self.de_map = res.de_map
        self.chk_highlight.setEnabled(res.de_map is not None)
        if res.de_map is not None and not res.ok:
            self.chk_highlight.setChecked(True)
        self.refresh_preview()

    def _probe(self, x, y):
        if not self.doc or self.thread is not None:
            return
        s = self._settings()
        conv = make_converter(self.doc, s)
        hidden = {id(self.doc.layers[i]) for i in s.remove}
        hidden |= {id(self.doc.layers[i]) for i, v in s.visibility.items() if not v}
        try:
            r = check_mod.probe(self.doc, conv, x, y, hidden)
        except Exception:
            r = None
        if r is None:
            return
        txt = f"{x}, {y} · {r['layer']}: {r['source']}"
        if s.convert:
            txt += f"  →  {r['target']} {r['out']}"
            if r["de"] is not None:
                txt += f"  (ΔE {r['de']:.1f})"
        self.statusBar().showMessage(txt)

    def schedule_preview(self, *a):
        if self.doc:
            self._preview_timer.start()

    def refresh_preview(self, fit=False):
        if not self.doc:
            return
        doc, s = self.doc, self._settings()
        conv = self._display_converter()
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
        overlay = self.de_map if (self.chk_highlight.isChecked() and self.de_map is not None) else None
        img = self._to_qimage(rgb, alpha, overlay)
        self.view.set_image(img, doc.width, doc.height, keep_view=not fit)
        self.setStatusTip(src)
        self.statusBar().showMessage(f"Preview: {src}" + (f" · 1/{self.scale} scale" if self.scale > 1 else ""))

    PREVIEW_CACHE_BYTES = 1 << 30      # converted preview layers kept in memory (8-bit)

    def _cached_rgb(self, conv):
        """Colour-converted preview pixels per layer, stored 8-bit (display
        precision) in an LRU with a memory budget."""
        from collections import OrderedDict
        if not isinstance(self.rgb_cache, OrderedDict):
            self.rgb_cache = OrderedDict()
        cache = self.rgb_cache
        nc = self.doc.color_channels
        maxv = self.doc.max_value

        def get(L, src):
            key = id(L)
            if key in cache:
                cache.move_to_end(key)
                v = cache[key]
                return None if v is None else v.astype(np.float32) * (1 / 255.0)
            if not all(c in L.channels for c in range(nc)):
                cache[key] = None
                return None
            chans = [np.ascontiguousarray(L.channels[c][src]) for c in range(nc)]
            if conv is not None:
                rgb = conv.to_rgb_float(chans)
            else:
                rgb = np.stack(chans[:3], -1).astype(np.float32) / maxv
            cache[key] = np.clip(rgb * 255 + 0.5, 0, 255).astype(np.uint8)
            used = sum(v.nbytes for v in cache.values() if v is not None)
            while used > self.PREVIEW_CACHE_BYTES and len(cache) > 1:
                _, old = cache.popitem(last=False)
                used -= old.nbytes if old is not None else 0
            return rgb
        return get

    @staticmethod
    def _to_qimage(rgb, alpha, de_map=None):
        h, w = alpha.shape
        yy, xx = np.mgrid[0:h, 0:w]
        checker = np.where(((yy // 12) + (xx // 12)) % 2 == 0, 0.42, 0.32).astype(np.float32)
        out = rgb * alpha[..., None] + checker[..., None] * (1 - alpha[..., None])
        if de_map is not None and de_map.shape == alpha.shape:
            bad = np.nan_to_num(de_map, nan=0.0) > check_mod.THRESHOLD
            stripes = ((yy + xx) // 4) % 2 == 0
            out = out.copy()
            out[bad & stripes] = (1.0, 0.0, 1.0)            # magenta hatch where ΔE > 2
        buf = np.ascontiguousarray(np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8))
        img = QtGui.QImage(buf.tobytes(), w, h, 3 * w, E(QtGui.QImage, "Format", "Format_RGB888"))
        return img.copy()

    # -- export ------------------------------------------------------------
    def export(self, psd=True, flat=True):
        if not self.doc:
            return
        s = self._settings()
        s.source_path = self.path
        fmt = self.cmb_fmt.currentText()
        ext = ".png" if fmt == "PNG" else ".tif"
        self._remember_save_options()
        tag = colour_tag(self.doc, s)
        if not self.rad_ask.isChecked():
            stem = self._versioned_stem()
            if stem is None:
                return
            stem = naming.with_tag(stem, tag)
            s.out_psd = stem + ".psd" if psd else None
            s.out_flat = stem + ext if flat else None
            for p in (s.out_psd, s.out_flat):
                if p and os.path.exists(p):          # can't happen, but never overwrite
                    QtWidgets.QMessageBox.warning(self, "PSD Prep", f"{p} already exists.")
                    return
            self._run(process, self.doc, s, on_done=self._exported)
            return
        outdir = self.settings.value("out_dir", os.path.dirname(self.path))
        stem = naming.with_tag(naming.custom_stem(outdir, naming.base_name(self.path, self.txt_suffix.text())), tag)
        if psd:
            p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save PSD", stem + ".psd",
                                                         "Photoshop (*.psd)")
            if not p:
                return
            chosen = p
            if not p.lower().endswith((".psd", ".psb")):
                p += ".psd"
            if naming.is_same_file(p, self.path):
                QtWidgets.QMessageBox.warning(self, "PSD Prep", "Choose a new name — the original "
                                              "file is never overwritten.")
                return
            if not self._ok_to_replace(p, chosen):
                return
            s.out_psd = p
            stem = os.path.splitext(p)[0]
        if flat:
            p, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Export flattened image", stem + ext,
                "PNG (*.png);;TIFF (*.tif *.tiff)" if fmt == "PNG" else "TIFF (*.tif *.tiff);;PNG (*.png)")
            if not p:
                return
            chosen = p
            if not p.lower().endswith((".png", ".tif", ".tiff")):
                p += ext
            if naming.is_same_file(p, self.path):
                QtWidgets.QMessageBox.warning(self, "PSD Prep", "The original file is never overwritten.")
                return
            if not self._ok_to_replace(p, chosen):
                return
            s.out_flat = p
            self.settings.setValue("fmt", fmt)
        self.settings.setValue("out_dir", os.path.dirname(s.out_psd or s.out_flat))
        s.overwrite = True          # every existing target was confirmed above
        self._run(process, self.doc, s, on_done=self._exported)

    def _ok_to_replace(self, path, chosen):
        """The file dialog only confirms the exact name typed; if we added the
        extension, ask ourselves before replacing an existing file."""
        if not os.path.exists(path) or path == chosen:
            return True
        btn = QtWidgets.QMessageBox.question(
            self, "PSD Prep", f"{os.path.basename(path)} already exists.\nReplace it?")
        return btn == E(QtWidgets.QMessageBox, "StandardButton", "Yes")

    def _exported(self, res):
        files = res.get("written") or [res[k] for k in ("psd", "flat") if res.get(k)]
        msg = "Saved " + ", ".join(os.path.basename(f) for f in files[:4])
        if len(files) > 4:
            msg += f" … ({len(files)} files)"
        self.statusBar().showMessage(msg, 15000)
        self.log(msg)
        self._update_next_name()

    def _remember_save_options(self):
        mode = "auto" if self.rad_auto.isChecked() else "custom" if self.rad_custom.isChecked() else "ask"
        self.settings.setValue("save_mode", mode)
        self.settings.setValue("suffix", self.txt_suffix.text())
        self.settings.setValue("custom_folder", self.txt_folder.text())
        self.settings.setValue("fmt", self.cmb_fmt.currentText())

    def _browse_folder(self):
        start = self.txt_folder.text() or (os.path.dirname(self.path) if self.path else "")
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Output folder", start)
        if d:
            self.txt_folder.setText(d)
            self.rad_custom.setChecked(True)

    def _versioned_stem(self, quiet=False):
        """<folder>/<name>_v<N> for the auto / custom modes (None if unusable)."""
        if self.rad_auto.isChecked():
            return naming.auto_stem(self.path, self.txt_suffix.text())
        folder = self.txt_folder.text().strip()
        if not folder or not os.path.isdir(folder):
            if not quiet:
                QtWidgets.QMessageBox.information(self, "PSD Prep", "Pick an existing output folder first.")
            return None
        if not os.access(folder, os.W_OK):
            if not quiet:
                QtWidgets.QMessageBox.warning(self, "PSD Prep", f"Can't write to {folder}.")
            return None
        name = self.txt_name.text().strip() or naming.base_name(self.path, self.txt_suffix.text())
        return naming.custom_stem(folder, name)

    def _update_next_name(self, *a):
        if not self.path or not self.doc:
            self.lbl_next.setText("")
            return
        if self.rad_ask.isChecked():
            self.lbl_next.setText("You pick the file name each time.")
            return
        stem = self._versioned_stem(quiet=True)
        if stem is None:
            self.lbl_next.setText("Pick an output folder.")
            return
        stem = naming.with_tag(stem, colour_tag(self.doc, self._settings()))
        ext = ".png" if self.cmb_fmt.currentText() == "PNG" else ".tif"
        self.lbl_next.setText(f"Next: {os.path.basename(stem)}.psd / {ext}\nin {os.path.dirname(stem)}")

    def export_boards(self):
        if not self.doc:
            return
        boards = self._selected_boards()
        if not boards:
            QtWidgets.QMessageBox.information(self, "PSD Prep", "Tick at least one artboard.")
            return
        want_psd = self.chk_board_psd.isChecked()
        ext = (".png" if self.cmb_fmt.currentText() == "PNG" else ".tif") \
            if self.chk_board_flat.isChecked() else None
        if not want_psd and not ext:
            QtWidgets.QMessageBox.information(self, "PSD Prep", "Tick PSD and/or Flattened.")
            return
        self._remember_save_options()
        if not self.rad_ask.isChecked():
            stem = self._versioned_stem()
            if stem is None:
                return
        else:
            folder = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Folder for the artboards", self.settings.value("out_dir", os.path.dirname(self.path)))
            if not folder:
                return
            self.settings.setValue("out_dir", folder)
            stem = naming.custom_stem(folder, naming.base_name(self.path, self.txt_suffix.text()))
        s = self._settings()
        s.source_path = self.path
        s.artboard_transparent = self.chk_board_transp.isChecked()
        self._run(export_artboards, self.doc, s, boards, stem, want_psd, ext, on_done=self._exported,
                  tag=colour_tag(self.doc, s))

    # -- threading ------------------------------------------------------------
    def _run(self, fn, *args, on_done=None, **kwargs):
        if self.thread is not None:
            return
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.setEnabled(False)
        self.thread = QtCore.QThread(self)
        self.worker = Worker(fn, *args, **kwargs)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self.log)
        self.worker.progress.connect(self._progress)
        self._on_done = on_done
        # bound methods of this window -> queued to the GUI thread (a lambda
        # would run in the worker thread and crash Qt)
        self.worker.done.connect(self._finish)
        self.worker.failed.connect(self._failed)
        self.thread.start()

    def _progress(self, a, b):
        self.progress.setRange(0, b)
        self.progress.setValue(a)

    def _cleanup(self):
        self.thread.quit()
        self.thread.wait()
        self.worker.deleteLater()
        self.thread.deleteLater()
        self.thread = None
        self.progress.setVisible(False)
        self.setEnabled(True)

    def _finish(self, res):
        on_done, self._on_done = self._on_done, None
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


LOG_PATH = os.path.join(os.path.expanduser("~"), ".psdprep.log")


def _install_crash_handlers():
    """Python errors -> dialog + log file instead of a silent exit;
    hard crashes -> native traceback in ~/.psdprep.log."""
    try:
        import faulthandler
        _install_crash_handlers.f = open(LOG_PATH, "a", buffering=1)
        faulthandler.enable(_install_crash_handlers.f)
    except Exception:
        pass

    def hook(et, ev, tb):
        msg = "".join(traceback.format_exception(et, ev, tb))
        try:
            with open(LOG_PATH, "a") as f:
                f.write(msg + "\n")
        except Exception:
            pass
        sys.__stderr__.write(msg)
        try:
            box = QtWidgets.QMessageBox()
            box.setWindowTitle("PSD Prep error")
            box.setText(f"{et.__name__}: {ev}")
            box.setInformativeText(f"Details saved to {LOG_PATH}")
            box.setDetailedText(msg)
            box.exec() if hasattr(box, "exec") else box.exec_()
        except Exception:
            pass
    sys.excepthook = hook


def main(argv=None):
    argv = sys.argv if argv is None else argv
    _install_crash_handlers()
    from .desktop import auto_register
    auto_register()
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
        if a.lower().endswith((".psd", ".psb", ".ai", ".pdf")) and os.path.isfile(a):
            w.open(a)
            break
    return app.exec() if hasattr(app, "exec") else app.exec_()

"""Headless self-test of the packaged app:  PSDPrep selftest [--log FILE]

Checks the PSD reader/writer, colour conversion, PNG/TIFF export, and drives
the window offscreen (load, layer tree, remove hidden, crop, aspect, preview,
export). Exit code 0 = everything passed."""
from __future__ import annotations

import os
import sys
import tempfile
import traceback

import numpy as np


class Report:
    def __init__(self):
        self.lines, self.failed = [], 0

    def step(self, name, fn):
        try:
            detail = fn()
            self.lines.append(f"PASS  {name}" + (f"  ({detail})" if detail else ""))
        except Exception:
            self.failed += 1
            self.lines.append(f"FAIL  {name}\n" + traceback.format_exc())

    def skip(self, name, why):
        self.lines.append(f"SKIP  {name}  ({why})")


def _profile():
    from . import ops
    found = ops.find_cmyk_profiles()
    return found[0] if found else (None, None)


def _roundtrip(tmp):
    from .fixtures import make_doc
    from .psdfile import read_psd, write_psd
    for depth in (8, 16):
        a = make_doc(depth)
        p = os.path.join(tmp, f"rt{depth}.psd")
        write_psd(a, p)
        b = read_psd(p)
        assert [L.name for L in a.layers] == [L.name for L in b.layers]
        for la, lb in zip(a.layers, b.layers):
            for k, v in la.channels.items():
                assert np.array_equal(v, lb.channels[k]), (depth, la.name, k)
    return "8 and 16-bit layers identical after save + reopen"


def _pipeline(tmp):
    from .fixtures import make_doc
    from .pipeline import Settings, process
    from .psdfile import read_psd
    path, desc = _profile()
    out = []
    for depth in (8, 16):
        d = make_doc(depth)
        s = Settings(remove={0}, crop=(10, 10, 300, 180), cmyk_profile=path,
                     out_psd=os.path.join(tmp, f"o{depth}.psd"),
                     out_flat=os.path.join(tmp, f"o{depth}.tif" if depth == 16 else f"o{depth}.png"))
        process(d, s, log=lambda *a: None)
        o = read_psd(s.out_psd)
        assert (o.width, o.height, o.mode) == (300, 180, 3), (o.width, o.height, o.mode)
        assert o.merged_has_alpha and "Background" not in [L.name for L in o.layers]
        assert os.path.getsize(s.out_flat) > 1000
        out.append(f"{depth}-bit ok")
    return ", ".join(out) + f"; CMYK profile: {desc or 'none found (simple formula)'}"


def _gui(tmp):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from .fixtures import make_doc
    from .psdfile import write_psd, read_psd
    from . import gui
    from .pipeline import process  # noqa: F401

    src = os.path.join(tmp, "gui_in.psd")
    write_psd(make_doc(8), src)
    app = gui.QtWidgets.QApplication.instance() or gui.QtWidgets.QApplication([])
    w = gui.MainWindow()
    w.show()
    import time
    w.open(src)                                    # same path as drag & drop (worker thread)
    t0 = time.time()
    while (w.doc is None or w.thread is not None) and time.time() - t0 < 30:
        app.processEvents()
        time.sleep(0.01)
    assert w.doc is not None, "file did not load"
    app.processEvents()
    assert len(w.items) == 6, f"tree has {len(w.items)} items"
    assert w.view.pix.pixmap().width() > 0, "empty preview"
    w.remove_hidden()
    removed = {w.doc.layers[i].name for i in w._removed()}
    assert removed == {"Hidden"}, removed
    w.cmb_aspect.setCurrentIndex([n for n, _ in gui.ASPECTS].index("16:9"))
    app.processEvents()
    x, y, cw, ch = w.view.crop
    assert abs(cw / ch - 16 / 9) < 0.02, w.view.crop
    w.spins[0].setValue(5)
    app.processEvents()
    assert w.view.crop[0] == 5
    w.chk_convert.setChecked(True)
    w.refresh_preview()
    app.processEvents()
    def wait(limit=60):
        t0 = time.time()
        while w.thread is not None and time.time() - t0 < limit:
            app.processEvents()
            time.sleep(0.01)
        app.processEvents()

    before = open(src, "rb").read()
    # auto-named save next to the original: v1, then v2
    w.rad_auto.setChecked(True)
    w.txt_suffix.setText("cleanup")
    w.cmb_fmt.setCurrentText("PNG")
    w.export(psd=True, flat=True)
    wait()
    w.export(psd=True, flat=False)
    wait()
    made = sorted(f for f in os.listdir(tmp) if f.startswith("gui_in_cleanup"))
    assert made == ["gui_in_cleanup_v1.png", "gui_in_cleanup_v1.psd", "gui_in_cleanup_v2.psd"], made
    o = read_psd(os.path.join(tmp, "gui_in_cleanup_v1.psd"))
    assert (o.width, o.height) == (w.view.crop[2], w.view.crop[3])
    assert "v3" in w.lbl_next.text(), w.lbl_next.text()
    assert open(src, "rb").read() == before, "original changed"
    assert w.orig["mode"].text().find("CMYK") >= 0

    # artboards: list, reorder, export each one
    from .fixtures import make_artboard_doc
    bsrc = os.path.join(tmp, "boards.psd")
    write_psd(make_artboard_doc(), bsrc)
    w.open(bsrc)
    wait()
    assert w.grp_boards.isVisibleTo(w) and w.lst_boards.count() == 3
    w._order_boards("position")
    names = [w.lst_boards.item(i).text().split("  ")[0] for i in range(3)]
    assert names == ["Square", "Banner", "Small"], names
    w.lst_boards.setCurrentRow(1)
    assert w.view.hl.isVisible()
    w.lst_boards.item(2).setCheckState(gui.UNCHECKED)       # skip "Small"
    w.export_boards()
    wait()
    made = sorted(f for f in os.listdir(tmp) if f.startswith("boards_cleanup"))
    assert made == ["boards_cleanup_v1_01_Square.png", "boards_cleanup_v1_01_Square.psd",
                    "boards_cleanup_v1_02_Banner.png", "boards_cleanup_v1_02_Banner.psd"], made
    w.close()
    app.processEvents()
    return f"window ok; auto-named v1/v2; {len(made)} artboard files"


def run(log_path=None) -> int:
    from . import __version__
    from .version import BUILD
    r = Report()
    r.lines.append(f"PSD Prep {__version__} ({BUILD}) self-test, Python {sys.version.split()[0]}")
    with tempfile.TemporaryDirectory() as tmp:
        r.step("PSD read/write", lambda: _roundtrip(tmp))
        r.step("Edit + convert + export", lambda: _pipeline(tmp))
        try:
            import PySide6  # noqa: F401
            r.step("Window (offscreen)", lambda: _gui(tmp))
        except ImportError:
            r.skip("Window (offscreen)", "PySide6 not installed")
    r.lines.append("RESULT: " + ("all passed" if not r.failed else f"{r.failed} failed"))
    text = "\n".join(r.lines)
    print(text)
    if log_path:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    return 1 if r.failed else 0

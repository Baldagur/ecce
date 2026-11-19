"""Run: python3 -m pytest tests  (or python3 tests/test_roundtrip.py)"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from psdprep import ops  # noqa: E402
from psdprep.pipeline import Settings, process  # noqa: E402
from psdprep.psdfile import (MODE_CMYK, PSD, SECTION_DIVIDER, SECTION_OPEN, Layer, Mask,  # noqa: E402
                             read_psd, write_psd)


def make_doc(depth=8):
    mx = 255 if depth == 8 else 65535
    dt = np.uint8 if depth == 8 else np.uint16
    W, H = 320, 200
    d = PSD(width=W, height=H, depth=depth, mode=MODE_CMYK, num_channels=4)

    def L(name, t, l, b, r, cmyk, alpha=None, **kw):
        ch = {i: np.full((b - t, r - l), int(mx - cmyk[i] * mx), dt) for i in range(4)}
        if alpha is not None:
            ch[-1] = alpha
        return Layer(name=name, top=t, left=l, bottom=b, right=r, channels=ch, **kw)

    yy, xx = np.mgrid[0:100, 0:100]
    circ = ((((yy - 50) ** 2 + (xx - 50) ** 2) < 45 ** 2) * mx).astype(dt)
    grad = np.tile(np.linspace(0, mx, 150).astype(dt), (75, 1))
    bg = L("Background", 0, 0, H, W, (0.05, 0, 0.1, 0))
    c1 = L("Cyan circle é", 25, 20, 125, 120, (1, 0, 0, 0), circ)
    m = L("Magenta", 50, 75, 125, 225, (0, 1, 0, 0), blend_mode="mul ")
    m.mask = Mask(50, 75, 125, 225, 0, 0, grad)
    grp = Layer(name="Group", section=SECTION_OPEN, section_blend="pass")
    hid = L("Hidden", 0, 0, 50, 50, (0, 0, 1, 0))
    hid.visible = False
    off = L("Off canvas", 150, 275, 250, 375, (0, 0, 0, 1), np.full((100, 100), mx // 2, dt))
    d.layers = [bg, Layer(name="</Layer group>", section=SECTION_DIVIDER), c1, m, grp, hid, off]
    return d


def test_roundtrip():
    for depth in (8, 16):
        a = make_doc(depth)
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "a.psd")
            write_psd(a, p)
            b = read_psd(p)
        assert [L.name for L in a.layers] == [L.name for L in b.layers]
        for la, lb in zip(a.layers, b.layers):
            for k, v in la.channels.items():
                assert np.array_equal(v, lb.channels[k])
            assert la.visible == lb.visible and la.section == lb.section


def test_process_outputs():
    d = make_doc(8)
    with tempfile.TemporaryDirectory() as tmp:
        s = Settings(remove={0}, crop=(10, 10, 300, 180),
                     out_psd=os.path.join(tmp, "o.psd"), out_flat=os.path.join(tmp, "o.png"))
        process(d, s, log=lambda *a: None)
        o = read_psd(s.out_psd)
        assert (o.width, o.height, o.mode) == (300, 180, 3)
        assert "Background" not in [L.name for L in o.layers]
        assert o.merged_has_alpha
        assert os.path.getsize(s.out_flat) > 0


def test_tree_remove_group():
    d = make_doc(8)
    grp = [L for L in d.layers if L.is_group][0]
    ops.remove_layers(d, {id(grp)})
    assert [L.name for L in d.layers] == ["Background", "Hidden", "Off canvas"]


if __name__ == "__main__":
    test_roundtrip(); test_process_outputs(); test_tree_remove_group()
    print("all tests passed")

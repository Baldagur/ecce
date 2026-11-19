"""Synthetic PSD used by the tests and the self-test."""
from __future__ import annotations

import numpy as np

from .psdfile import MODE_CMYK, PSD, SECTION_DIVIDER, SECTION_OPEN, Layer, Mask


def make_doc(depth: int = 8, icc: bytes = None) -> PSD:
    """8/16-bit CMYK doc: background, a pass-through group with a transparent
    circle and a masked multiply layer, a hidden layer and one off canvas."""
    mx = 255 if depth == 8 else 65535
    dt = np.uint8 if depth == 8 else np.uint16
    W, H = 320, 200
    d = PSD(width=W, height=H, depth=depth, mode=MODE_CMYK, num_channels=4)
    if icc:
        d.set_resource(1039, icc)

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

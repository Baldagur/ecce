"""Prove a colour conversion: compare every converted pixel with what the
source ICC profile says the colour is (CIE Lab), as Delta E 2000.

Delta E < 1: not visible; < 2: only visible side by side; > 3-5: a real
difference (almost always a colour outside the target gamut, which has to
be clipped by any converter, Photoshop included)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import ops
from .colorspaces import TARGETS, delta_e_2000, rgb_to_lab
from .psdfile import MODE_CMYK, MODE_GRAY, MODE_LAB, PSD

THRESHOLD = 2.0


@dataclass
class CheckResult:
    ok: bool
    message: str
    stats: dict = field(default_factory=dict)
    de_map: Optional[np.ndarray] = None          # preview-scale Delta E, NaN where no pixel

    def summary(self) -> str:
        return self.message


GAMUT_TOL = 0.02        # linear-light tolerance (8-bit Lab reference quantisation)


def _de_for(conv: ops.ColorConverter, chans: list, with_gamut: bool = False):
    """Delta E per pixel; with_gamut also returns which SOURCE colours lie
    outside the target gamut (decided from the source colour itself, so a
    broken conversion can't hide its errors as 'out of gamut')."""
    from .colorspaces import lab_to_linear
    ref = conv.source_lab(chans)
    out = conv.to_rgb_float(chans)
    de = delta_e_2000(ref, rgb_to_lab(out.astype(np.float64), conv.target)).astype(np.float32)
    if not with_gamut:
        return de
    lin = lab_to_linear(ref)
    outside = ((lin < -GAMUT_TOL) | (lin > 1 + GAMUT_TOL)).any(axis=-1)
    return de, outside


def conversion_check(doc: PSD, conv: ops.ColorConverter, scale: int = 1,
                     hidden: set = None) -> CheckResult:
    target = TARGETS[conv.target].label
    if doc.mode == MODE_LAB:
        return CheckResult(False, "Lab files can't be checked yet.")
    if conv.src_profile is None:
        if doc.mode == MODE_CMYK:
            return CheckResult(False, "This CMYK file has no colour profile and none is bundled, "
                                      "so it's converted with a simple formula: there is no "
                                      "reference to measure against.")
        return CheckResult(True, f"Untagged {'gray' if doc.mode == MODE_GRAY else 'RGB'} is "
                                 f"treated as sRGB: the conversion to {target} is exact maths "
                                 f"(same primaries, only the curve changes).")
    comp = ops.Compositor(doc, lambda L, s: None, scale, set())
    de_map = np.full((comp.H, comp.W), np.nan, np.float32)
    values, clipped = [], []
    nc = doc.color_channels
    # hidden: ids that don't show (ops.effective_hidden); default: as saved
    if hidden is None:
        hidden = ops.effective_hidden(doc)
    layers = [L for L in doc.layers if not (L.is_group or L.is_divider)
              and id(L) not in hidden and all(c in L.channels for c in range(nc))]
    for L in layers:
        p = comp._place(L.top, L.left, L.bottom, L.right)
        if p is None:
            continue
        src, dst = p
        chans = [np.ascontiguousarray(L.channels[c][src]) for c in range(nc)]
        if chans[0].size == 0:
            continue
        de, gam = _de_for(conv, chans, True)
        a = (L.channels[-1][src].astype(np.float32) / doc.max_value
             if -1 in L.channels else np.ones(de.shape, np.float32))
        solid = a > 0.5
        values.append(de[solid])
        clipped.append(gam[solid])
        region = de_map[dst]
        region[solid] = de[solid]                  # upper layers overwrite lower ones
        de_map[dst] = region
    if not values and doc.merged is not None:
        m = doc.merged[:nc, ::comp.s, ::comp.s]
        de, gam = _de_for(conv, [m[c] for c in range(nc)], True)
        de_map[:de.shape[0], :de.shape[1]] = de
        values.append(de.ravel())
        clipped.append(gam.ravel())
    v = np.concatenate(values) if values else np.zeros(0, np.float32)
    if v.size == 0:
        return CheckResult(False, "Nothing visible to check.")
    out_g = np.concatenate(clipped)
    inside = v[~out_g]
    if inside.size == 0:
        return CheckResult(False, f"Every measured colour is outside the {target} gamut, so there "
                                  f"is nothing exact to compare (max ΔE {float(v.max()):.1f}).",
                           {"pixels": int(v.size), "out_of_gamut": 1.0, "ok": False}, de_map)
    stats = {"pixels": int(v.size), "out_of_gamut": float(out_g.mean()),
             "median": float(np.median(inside)) if inside.size else 0.0,
             "p95": float(np.percentile(inside, 95)) if inside.size else 0.0,
             "max": float(inside.max()) if inside.size else 0.0,
             "max_all": float(v.max()), "under2": float((inside < THRESHOLD).mean()) if inside.size else 1.0}
    if stats["p95"] < 1:
        verdict = "Excellent: differences are invisible"
    elif stats["p95"] < 2:
        verdict = "Very good: differences are only visible side by side"
    else:
        verdict = "Check the highlighted areas"
    msg = (f"{stats['pixels']:,} pixels measured against the source profile "
           f"({conv.source_desc}) → {target}.\n"
           f"Colours the target can show: ΔE2000 median {stats['median']:.2f}, 95% under "
           f"{stats['p95']:.2f}, max {stats['max']:.2f}. {verdict}.")
    if stats["out_of_gamut"] > 0:
        msg += (f"\n{stats['out_of_gamut'] * 100:.1f}% of pixels are outside the {target} gamut "
                f"(e.g. pure cyan): no {target} value can show them, so they're clipped to the "
                f"nearest colour (up to ΔE {stats['max_all']:.1f}), as Photoshop does.")
    stats["ok"] = stats["p95"] < 2
    return CheckResult(stats["ok"], msg, stats, de_map)


def probe(doc: PSD, conv: ops.ColorConverter, x: int, y: int, hidden: set = None) -> Optional[dict]:
    """Source and converted values of the top visible pixel at (x, y)."""
    if not (0 <= x < doc.width and 0 <= y < doc.height):
        return None
    nc = doc.color_channels
    if hidden is None:
        hidden = ops.effective_hidden(doc)
    for L in reversed(doc.layers):
        if L.is_group or L.is_divider or id(L) in hidden:
            continue
        if not (L.left <= x < L.right and L.top <= y < L.bottom):
            continue
        if not all(c in L.channels for c in range(nc)):
            continue
        yy, xx = y - L.top, x - L.left
        if -1 in L.channels and L.channels[-1][yy, xx] / doc.max_value < 0.5:
            continue
        chans = [L.channels[c][yy:yy + 1, xx:xx + 1] for c in range(nc)]
        raw = [float(c[0, 0]) / doc.max_value for c in chans]
        if doc.mode == MODE_CMYK:
            src = "CMYK " + " ".join(f"{(1 - v) * 100:.0f}" for v in raw) + " %"
        elif doc.mode == MODE_GRAY:
            src = f"Gray {(1 - raw[0]) * 100:.0f} % K"
        else:
            src = "RGB " + " ".join(f"{v * 255:.0f}" for v in raw[:3])
        out = conv.to_rgb_float(chans)[0, 0]
        res = {"layer": L.name, "source": src,
               "out": " ".join(f"{v * 255:.0f}" for v in out),
               "target": TARGETS[conv.target].tag, "de": None}
        if conv.src_profile is not None and doc.mode != MODE_LAB:
            res["de"] = float(_de_for(conv, chans)[0, 0])
        return res
    return None

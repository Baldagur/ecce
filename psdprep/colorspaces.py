"""Output colour spaces (sRGB, Rec.709), their ICC profiles, and the maths
used to check a conversion (Lab, Delta E 2000)."""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


# --- transfer functions (encoded <-> linear, 0..1) -------------------------
def srgb_decode(v):
    v = np.asarray(v, np.float64)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def srgb_encode(l):
    l = np.clip(np.asarray(l, np.float64), 0, 1)
    return np.where(l <= 0.0031308, l * 12.92, 1.055 * l ** (1 / 2.4) - 0.055)


def g24_decode(v):
    return np.clip(np.asarray(v, np.float64), 0, 1) ** 2.4


def g24_encode(l):
    return np.clip(np.asarray(l, np.float64), 0, 1) ** (1 / 2.4)


def oetf709_encode(l):
    l = np.clip(np.asarray(l, np.float64), 0, 1)
    return np.where(l < 0.018, 4.5 * l, 1.099 * l ** 0.45 - 0.099)


def oetf709_decode(v):
    v = np.clip(np.asarray(v, np.float64), 0, 1)
    return np.where(v < 0.081, v / 4.5, ((v + 0.099) / 1.099) ** (1 / 0.45))


@dataclass
class Target:
    key: str
    label: str
    tag: str                      # added at the end of file names
    decode: Callable
    encode: Callable
    tooltip: str = ""


TARGETS = {
    "srgb": Target("srgb", "sRGB", "sRGB", srgb_decode, srgb_encode,
                   "Standard sRGB (IEC 61966-2-1): web, screens, most graphics."),
    "rec709": Target("rec709", "Rec.709 (gamma 2.4)", "Rec709", g24_decode, g24_encode,
                     "Rec.709 primaries, BT.1886 gamma 2.4 display encoding: what a Rec.709 "
                     "video monitor shows, and Flame's 'Rec.709 video'. Use for graphics going into a "
                     "Rec.709 deliverable."),
    "rec709_oetf": Target("rec709_oetf", "Rec.709 (camera OETF)", "Rec709-OETF",
                          oetf709_decode, oetf709_encode,
                          "Rec.709 primaries with the ITU-R BT.709 camera curve (Nuke's 'rec709' "
                          "colorspace). Brighter shadows than gamma 2.4."),
}
DEFAULT_TARGET = "srgb"

# sRGB / Rec.709 primaries, D65 -> XYZ, then Bradford-adapted to D50 (ICC PCS)
_RGB2XYZ_D65 = np.array([[0.4124564, 0.3575761, 0.1804375],
                         [0.2126729, 0.7151522, 0.0721750],
                         [0.0193339, 0.1191920, 0.9503041]])
_BRADFORD_D65_D50 = np.array([[1.0478112, 0.0228866, -0.0501270],
                              [0.0295424, 0.9904844, -0.0170491],
                              [-0.0092345, 0.0150436, 0.7521316]])
RGB2XYZ_D50 = _BRADFORD_D65_D50 @ _RGB2XYZ_D65
D50 = np.array([0.9642, 1.0, 0.8249])


def rgb_to_lab(rgb: np.ndarray, target: str = "srgb") -> np.ndarray:
    """Encoded RGB (..., 3) 0..1 in the target space -> CIE Lab (D50)."""
    lin = TARGETS[target].decode(rgb)
    xyz = lin @ RGB2XYZ_D50.T / D50
    e = 216 / 24389
    k = 24389 / 27
    f = np.where(xyz > e, np.cbrt(xyz), (k * xyz + 16) / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], -1)


def lab_to_linear(lab: np.ndarray) -> np.ndarray:
    """CIE Lab (D50) -> linear RGB with sRGB/Rec.709 primaries, NOT clipped
    (values outside 0..1 mean the colour is outside the gamut)."""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200
    e, k = 216 / 24389, 24389 / 27

    def finv(f):
        return np.where(f ** 3 > e, f ** 3, (116 * f - 16) / k)
    xyz = np.stack([finv(fx), np.where(L > k * e, fy ** 3, L / k), finv(fz)], -1) * D50
    return xyz @ np.linalg.inv(RGB2XYZ_D50).T


def delta_e_2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    Cb = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cb ** 7 / (Cb ** 7 + 25 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, np.where(dhp < -180, dhp + 360, dhp))
    dhp = np.where(C1p * C2p == 0, 0, dhp)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2))
    Lbp = (L1 + L2) / 2
    Cbp = (C1p + C2p) / 2
    hsum = h1p + h2p
    hbp = np.where(np.abs(h1p - h2p) > 180,
                   np.where(hsum < 360, (hsum + 360) / 2, (hsum - 360) / 2), hsum / 2)
    hbp = np.where(C1p * C2p == 0, hsum, hbp)
    T = (1 - 0.17 * np.cos(np.radians(hbp - 30)) + 0.24 * np.cos(np.radians(2 * hbp))
         + 0.32 * np.cos(np.radians(3 * hbp + 6)) - 0.20 * np.cos(np.radians(4 * hbp - 63)))
    dtheta = 30 * np.exp(-(((hbp - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cbp ** 7 / (Cbp ** 7 + 25 ** 7))
    Sl = 1 + 0.015 * (Lbp - 50) ** 2 / np.sqrt(20 + (Lbp - 50) ** 2)
    Sc = 1 + 0.045 * Cbp
    Sh = 1 + 0.015 * Cbp * T
    Rt = -np.sin(np.radians(2 * dtheta)) * Rc
    return np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2
                   + Rt * (dCp / Sc) * (dHp / Sh))


# --- ICC profiles -------------------------------------------------------------
def _s15(v):
    return struct.pack(">i", int(round(v * 65536)))


def _xyz(x, y, z):
    return b"XYZ \0\0\0\0" + _s15(x) + _s15(y) + _s15(z)


def _desc(text: str) -> bytes:
    t = text.encode("ascii") + b"\0"
    return (b"desc\0\0\0\0" + struct.pack(">I", len(t)) + t + struct.pack(">II", 0, 0)
            + b"\0\0\0" + b"\0" * 67)


def _text(text: str) -> bytes:
    return b"text\0\0\0\0" + text.encode("ascii") + b"\0"


def _curve(decode: Callable, gamma: Optional[float]) -> bytes:
    if gamma:
        return b"curv\0\0\0\0" + struct.pack(">IH", 1, int(round(gamma * 256)))
    n = 4096
    vals = np.clip(np.rint(decode(np.linspace(0, 1, n)) * 65535), 0, 65535).astype(">u2")
    return b"curv\0\0\0\0" + struct.pack(">I", n) + vals.tobytes()


def _srgb_colorants() -> np.ndarray:
    """rXYZ/gXYZ/bXYZ columns exactly as lcms's built-in sRGB profile has them,
    so sRGB -> Rec.709 never shifts primaries (pure green stays 0,255,0)."""
    try:
        from PIL import ImageCms
        icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        n = struct.unpack(">I", icc[128:132])[0]
        tags = {}
        for i in range(n):
            sig, off, size = struct.unpack(">4sII", icc[132 + 12 * i:144 + 12 * i])
            tags[sig] = icc[off:off + size]
        cols = []
        for sig in (b"rXYZ", b"gXYZ", b"bXYZ"):
            v = struct.unpack(">3i", tags[sig][8:20])
            cols.append([x / 65536 for x in v])
        return np.array(cols).T
    except Exception:
        return RGB2XYZ_D50


def build_icc(target: str) -> bytes:
    """ICC v2 matrix/TRC display profile for a target (lcms and Photoshop read it)."""
    t = TARGETS[target]
    gamma = 2.4 if target == "rec709" else None   # pure gamma: lcms inverts it analytically
    names = {"srgb": "sRGB (PSD Prep)", "rec709": "Rec.709 gamma 2.4 (BT.1886)",
             "rec709_oetf": "Rec.709 (ITU-R BT.709 OETF)"}
    trc = _curve(t.decode, gamma)
    cols = _srgb_colorants()
    tags = [(b"desc", _desc(names[target])), (b"cprt", _text("No copyright, use freely")),
            (b"wtpt", _xyz(*D50)),
            (b"rXYZ", _xyz(*cols[:, 0])), (b"gXYZ", _xyz(*cols[:, 1])),
            (b"bXYZ", _xyz(*cols[:, 2])),
            (b"rTRC", trc), (b"gTRC", trc), (b"bTRC", trc)]
    off = 128 + 4 + 12 * len(tags)
    table, body, seen = b"", b"", {}
    for sig, data in tags:
        data += b"\0" * ((-len(data)) % 4)
        if data in seen:                               # share identical curves
            table += sig + struct.pack(">II", *seen[data])
            continue
        seen[data] = (off + len(body), len(data))
        table += sig + struct.pack(">II", off + len(body), len(data))
        body += data
    size = off + len(body)
    hdr = (struct.pack(">I", size) + b"lcms" + struct.pack(">I", 0x02100000) + b"mntrRGB XYZ "
           + struct.pack(">6H", 2026, 1, 1, 0, 0, 0) + b"acsp" + b"\0" * 24
           + struct.pack(">I", 0) + _s15(D50[0]) + _s15(D50[1]) + _s15(D50[2]) + b"\0" * 48)
    return hdr + struct.pack(">I", len(tags)) + table + body


_ICC_CACHE: dict = {}


def icc_bytes(target: str) -> Optional[bytes]:
    if target == "srgb":
        try:
            from PIL import ImageCms
            return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        except Exception:
            pass
    if target not in _ICC_CACHE:
        _ICC_CACHE[target] = build_icc(target)
    return _ICC_CACHE[target]


def pil_profile(target: str):
    """Pillow/lcms profile object for the target."""
    from PIL import ImageCms
    import io
    if target == "srgb":
        return ImageCms.createProfile("sRGB")
    return ImageCms.ImageCmsProfile(io.BytesIO(icc_bytes(target)))


def reencode_from_srgb(rgb: np.ndarray, target: str) -> np.ndarray:
    """sRGB-encoded floats -> target encoding (same primaries)."""
    if target == "srgb":
        return rgb
    return TARGETS[target].encode(srgb_decode(rgb)).astype(np.float32)

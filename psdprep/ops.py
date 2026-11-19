"""Edits on a PSD: layer tree, remove layers, crop, colour conversion,
compositing (with alpha)."""
from __future__ import annotations

import copy
import glob
import os
import struct
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .psdfile import (PSD, Layer, Mask, MODE_CMYK, MODE_GRAY, MODE_LAB, MODE_RGB,
                      SECTION_CLOSED, SECTION_DIVIDER, SECTION_OPEN)


# ---------------------------------------------------------------------------
# Layer tree
# ---------------------------------------------------------------------------
@dataclass
class Node:
    layer: Layer
    children: list = field(default_factory=list)   # bottom -> top
    divider: Optional[Layer] = None

    @property
    def is_group(self):
        return self.layer.is_group


def build_tree(layers: list) -> list:
    """File order (bottom->top) -> list of root Nodes (bottom->top)."""
    root: list = []
    stack = [(root, None)]
    for L in layers:
        if L.is_divider:
            stack.append(([], L))
        elif L.is_group:
            if len(stack) > 1:
                children, div = stack.pop()
            else:
                children, div = [], None
            stack[-1][0].append(Node(L, children, div))
        else:
            stack[-1][0].append(Node(L))
    # unterminated groups: lift children up
    while len(stack) > 1:
        children, _ = stack.pop()
        stack[-1][0].extend(children)
    return root


def flatten_tree(nodes: list) -> list:
    out = []
    for n in nodes:
        if n.is_group:
            div = n.divider or Layer(name="</Layer group>", section=SECTION_DIVIDER)
            out.append(div)
            out.extend(flatten_tree(n.children))
            out.append(n.layer)
        else:
            out.append(n.layer)
    return out


def walk(nodes: list, depth: int = 0):
    """Yield (node, depth) top -> bottom (UI order)."""
    for n in reversed(nodes):
        yield n, depth
        if n.is_group:
            yield from walk(n.children, depth + 1)


# ---------------------------------------------------------------------------
# Remove / crop
# ---------------------------------------------------------------------------
def remove_layers(doc: PSD, remove: set) -> PSD:
    """remove: set of id(layer). Removing a group removes its contents."""
    def prune(nodes):
        out = []
        for n in nodes:
            if id(n.layer) in remove:
                continue
            if n.is_group:
                n = Node(n.layer, prune(n.children), n.divider)
            out.append(n)
        return out
    doc.layers = flatten_tree(prune(build_tree(doc.layers)))
    return doc


def _crop_array(arr, top, left, new_top, new_left, new_bottom, new_right):
    return arr[new_top - top:new_bottom - top, new_left - left:new_right - left]


def crop(doc: PSD, x: int, y: int, w: int, h: int, delete_pixels: bool = True) -> PSD:
    x = max(0, int(x)); y = max(0, int(y))
    w = max(1, min(int(w), doc.width - x)); h = max(1, min(int(h), doc.height - y))
    for L in doc.layers:
        if L.is_group or L.is_divider:
            if L.mask is not None and L.mask.data is not None:
                _crop_mask(L.mask, x, y, w, h, delete_pixels)
            continue
        t, l, b, r = L.top - y, L.left - x, L.bottom - y, L.right - x
        if delete_pixels and L.width and L.height:
            nt, nl, nb, nr = max(t, 0), max(l, 0), min(b, h), min(r, w)
            if nb <= nt or nr <= nl:
                nt = nl = nb = nr = 0
                L.channels = {k: np.zeros((0, 0), v.dtype) for k, v in L.channels.items()}
            else:
                L.channels = {k: np.ascontiguousarray(_crop_array(v, t, l, nt, nl, nb, nr))
                              for k, v in L.channels.items()}
            t, l, b, r = nt, nl, nb, nr
        L.top, L.left, L.bottom, L.right = t, l, b, r
        if L.mask is not None and L.mask.data is not None:
            _crop_mask(L.mask, x, y, w, h, delete_pixels)
    if doc.merged is not None:
        doc.merged = np.ascontiguousarray(doc.merged[:, y:y + h, x:x + w])
    doc.width, doc.height = w, h
    return doc


def _crop_mask(m: Mask, x, y, w, h, delete_pixels):
    t, l, b, r = m.top - y, m.left - x, m.bottom - y, m.right - x
    if delete_pixels and m.width and m.height:
        nt, nl, nb, nr = max(t, 0), max(l, 0), min(b, h), min(r, w)
        if nb <= nt or nr <= nl:
            m.data = np.zeros((0, 0), m.data.dtype)
            nt = nl = nb = nr = 0
        else:
            m.data = np.ascontiguousarray(_crop_array(m.data, t, l, nt, nl, nb, nr))
        t, l, b, r = nt, nl, nb, nr
    m.top, m.left, m.bottom, m.right = t, l, b, r


# ---------------------------------------------------------------------------
# Colour conversion
# ---------------------------------------------------------------------------
INTENTS = {"Perceptual": 0, "Relative Colorimetric": 1, "Saturation": 2,
           "Absolute Colorimetric": 3}

CMYK_PROFILE_DIRS = [
    "/usr/share/color/icc", "/usr/share/color/icc/*", "/usr/local/share/color/icc",
    "/usr/share/ghostscript/iccprofiles", "/usr/share/ghostscript/*/iccprofiles",
    os.path.expanduser("~/.color/icc"), os.path.expanduser("~/.local/share/icc"),
    "/Library/ColorSync/Profiles", "/System/Library/ColorSync/Profiles",
    "C:/Windows/System32/spool/drivers/color",
]


def profile_info(data: bytes) -> tuple:
    """(colour space, description) of an ICC profile, best effort."""
    space = data[16:20].decode("latin-1", "replace").strip() if len(data) > 20 else "?"
    desc = ""
    try:
        from PIL import ImageCms
        import io
        p = ImageCms.ImageCmsProfile(io.BytesIO(data))
        desc = ImageCms.getProfileDescription(p).strip()
    except Exception:
        pass
    return space, desc


def find_cmyk_profiles() -> list:
    """List of (path, description) of CMYK ICC profiles found on this machine.
    Also looks in a 'profiles' folder next to the app."""
    here = os.path.dirname(os.path.abspath(__file__))
    dirs = list(CMYK_PROFILE_DIRS) + [os.path.join(here, "profiles"),
                                       os.path.join(os.path.dirname(here), "profiles")]
    import sys
    if getattr(sys, "_MEIPASS", None):
        dirs.append(os.path.join(sys._MEIPASS, "profiles"))
    if getattr(sys, "executable", None):
        dirs.append(os.path.join(os.path.dirname(sys.executable), "profiles"))
    seen, out = set(), []
    for d in dirs:
        for folder in glob.glob(d):
            for p in sorted(glob.glob(os.path.join(folder, "*"))):
                if not p.lower().endswith((".icc", ".icm")) or p in seen:
                    continue
                seen.add(p)
                try:
                    with open(p, "rb") as f:
                        head = f.read(128)
                    if head[16:20] == b"CMYK":
                        with open(p, "rb") as f:
                            out.append((p, profile_info(f.read())[1] or os.path.basename(p)))
                except OSError:
                    pass
    return out


def srgb_profile_bytes() -> Optional[bytes]:
    try:
        from PIL import ImageCms
        return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    except Exception:
        return None


class ColorConverter:
    """Converts raw PSD colour channels of one document to sRGB."""

    def __init__(self, mode: int, depth: int, embedded: Optional[bytes] = None,
                 fallback_profile: Optional[str] = None, intent: str = "Relative Colorimetric",
                 bpc: bool = True, use_embedded: bool = True):
        self.mode, self.depth = mode, depth
        self.max = 255.0 if depth == 8 else 65535.0
        self.intent = intent
        self.bpc = bpc
        self.source_desc = "none"
        self.transform = None
        self.method = "identity"
        self._lut = None
        if mode == MODE_GRAY:
            self.method = "gray"
            return
        prof_bytes = None
        if embedded and use_embedded:
            space = embedded[16:20]
            want = {MODE_CMYK: b"CMYK", MODE_RGB: b"RGB ", MODE_LAB: b"Lab "}.get(mode)
            if space == want:
                prof_bytes = embedded
                self.source_desc = "embedded: " + (profile_info(embedded)[1] or "unnamed")
        if prof_bytes is None and mode == MODE_CMYK and fallback_profile:
            try:
                with open(fallback_profile, "rb") as f:
                    prof_bytes = f.read()
                self.source_desc = "assigned: " + (profile_info(prof_bytes)[1]
                                                    or os.path.basename(fallback_profile))
            except OSError:
                prof_bytes = None
        try:
            from PIL import ImageCms
            import io
            dst = ImageCms.createProfile("sRGB")
            flags = 0
            if bpc:
                flags |= int(getattr(getattr(ImageCms, "Flags", None), "BLACKPOINTCOMPENSATION", 0x2000))
            ri = INTENTS.get(intent, 1)
            if mode == MODE_LAB:
                src = (ImageCms.ImageCmsProfile(io.BytesIO(prof_bytes)) if prof_bytes
                       else ImageCms.createProfile("LAB"))
                self.transform = ImageCms.buildTransform(src, dst, "LAB", "RGB", ri, flags)
                self.method = "icc"
                if not prof_bytes:
                    self.source_desc = "Lab D50"
            elif prof_bytes:
                src = ImageCms.ImageCmsProfile(io.BytesIO(prof_bytes))
                in_mode = "CMYK" if mode == MODE_CMYK else "RGB"
                self.transform = ImageCms.buildTransform(src, dst, in_mode, "RGB", ri, flags)
                self.method = "icc"
        except Exception as e:  # pragma: no cover - depends on profiles
            self.transform = None
            self.source_desc = f"profile error ({e})"
        if self.transform is None:
            self.method = "naive" if mode == MODE_CMYK else "identity"
            if mode == MODE_CMYK:
                self.source_desc = "no profile - simple formula"
            elif mode == MODE_RGB and self.source_desc == "none":
                self.source_desc = "untagged RGB (assumed sRGB)"

    # -- helpers ---------------------------------------------------------
    def _to_pil_input(self, chans: list) -> np.ndarray:
        """raw PSD channels (8-bit) -> HxWxN uint8 in Pillow's convention."""
        a = np.stack(chans, axis=-1)
        if self.mode == MODE_CMYK:
            a = 255 - a  # PSD stores CMYK inverted
        return np.ascontiguousarray(a, dtype=np.uint8)

    def _apply_pil(self, arr: np.ndarray) -> np.ndarray:
        from PIL import Image, ImageCms
        mode = {MODE_CMYK: "CMYK", MODE_LAB: "LAB", MODE_RGB: "RGB"}[self.mode]
        im = Image.frombytes(mode, (arr.shape[1], arr.shape[0]), arr.tobytes())
        out = ImageCms.applyTransform(im, self.transform)
        return np.asarray(out, dtype=np.uint8).reshape(arr.shape[0], arr.shape[1], 3)

    def _build_lut(self):
        n_in = 4 if self.mode == MODE_CMYK else 3
        N = 18 if n_in == 4 else 52          # 255 / (N-1) is an integer
        nodes = np.arange(N, dtype=np.int32) * (255 // (N - 1))
        grids = np.meshgrid(*([nodes] * n_in), indexing="ij")
        pts = np.stack([g.reshape(-1) for g in grids], axis=-1).astype(np.uint8)
        side = int(np.ceil(np.sqrt(pts.shape[0])))
        padded = np.zeros((side * side, n_in), np.uint8)
        padded[:pts.shape[0]] = pts
        rgb = self._apply_pil(padded.reshape(side, side, n_in)).reshape(-1, 3)[:pts.shape[0]]
        self._lut = (N, n_in, rgb.astype(np.float32) / 255.0)

    def _interp(self, x: np.ndarray) -> np.ndarray:
        """x: (P, n_in) float in 0..1 (Pillow convention). Returns (P,3) 0..1."""
        if self._lut is None:
            self._build_lut()
        N, n_in, lut = self._lut
        strides = [N ** (n_in - 1 - d) for d in range(n_in)]
        out = np.empty((x.shape[0], 3), np.float32)
        step = 400_000
        for s in range(0, x.shape[0], step):
            t = np.clip(x[s:s + step], 0, 1) * (N - 1)
            i0 = np.minimum(t.astype(np.int32), N - 2)
            f = (t - i0).astype(np.float32)
            acc = np.zeros((t.shape[0], 3), np.float32)
            for corner in range(1 << n_in):
                idx = np.zeros(t.shape[0], np.int64)
                w = np.ones(t.shape[0], np.float32)
                for d in range(n_in):
                    bit = (corner >> d) & 1
                    idx += (i0[:, d] + bit) * strides[d]
                    w *= f[:, d] if bit else (1 - f[:, d])
                acc += lut[idx] * w[:, None]
            out[s:s + step] = acc
        return out

    # -- public ----------------------------------------------------------
    def to_rgb_float(self, chans: list) -> np.ndarray:
        """raw colour channels -> HxWx3 float32 sRGB 0..1."""
        h, w = chans[0].shape
        if h == 0 or w == 0:
            return np.zeros((h, w, 3), np.float32)
        if self.method == "gray":
            g = chans[0].astype(np.float32) / self.max
            return np.repeat(g[:, :, None], 3, axis=2)
        if self.method == "identity":
            return np.stack(chans[:3], axis=-1).astype(np.float32) / self.max
        if self.method == "naive":
            ink = 1.0 - np.stack(chans, axis=-1).astype(np.float32) / self.max
            c, m, y, k = ink[..., 0], ink[..., 1], ink[..., 2], ink[..., 3]
            return np.stack([(1 - c) * (1 - k), (1 - m) * (1 - k), (1 - y) * (1 - k)], axis=-1)
        if self.depth == 8:
            return self._apply_pil(self._to_pil_input(chans)).astype(np.float32) / 255.0
        # 16-bit: interpolate through a LUT sampled with littleCMS
        x = np.stack(chans, axis=-1).reshape(-1, len(chans)).astype(np.float32) / self.max
        if self.mode == MODE_CMYK:
            x = 1.0 - x
        return self._interp(x).reshape(h, w, 3)

    def to_rgb_raw(self, chans: list, dtype) -> list:
        rgb = self.to_rgb_float(chans)
        m = 255.0 if dtype == np.uint8 else 65535.0
        out = np.clip(np.rint(rgb * m), 0, m).astype(dtype)
        return [np.ascontiguousarray(out[..., i]) for i in range(3)]


def convert_to_srgb(doc: PSD, conv: ColorConverter, progress: Callable = None) -> PSD:
    nc = doc.color_channels
    pixel_layers = [L for L in doc.layers if not (L.is_group or L.is_divider)]
    for i, L in enumerate(pixel_layers):
        if L.width and L.height and all(c in L.channels for c in range(nc)):
            rgb = conv.to_rgb_raw([L.channels[c] for c in range(nc)], doc.dtype)
            for c in range(nc):
                L.channels.pop(c, None)
            for c in range(3):
                L.channels[c] = rgb[c]
        else:
            for c in range(nc):
                L.channels.pop(c, None)
            for c in range(3):
                L.channels[c] = np.zeros((L.height, L.width), doc.dtype)
        L.blending_ranges = b""
        if progress:
            progress(i + 1, len(pixel_layers))
    if doc.merged is not None:
        color = [doc.merged[c] for c in range(nc)]
        rgb = conv.to_rgb_raw(color, doc.dtype)
        extra = [doc.merged[c] for c in range(nc, doc.merged.shape[0])]
        doc.merged = np.stack(rgb + extra)
    doc.mode = MODE_RGB
    doc.num_channels = 3
    doc.color_mode_data = b""
    doc.set_resource(1039, srgb_profile_bytes())
    return doc


def prepare_for_save(doc: PSD) -> PSD:
    """Drop image resources that may no longer match the edited document."""
    keep = {1005, 1039, 1060, 1062}
    doc.resources = [r for r in doc.resources if r[0] in keep]
    return doc


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------
def _soft_light(cb, cs):
    d = np.where(cb <= 0.25, ((16 * cb - 12) * cb + 4) * cb, np.sqrt(cb))
    return np.where(cs <= 0.5, cb - (1 - 2 * cs) * cb * (1 - cb), cb + (2 * cs - 1) * (d - cb))


def _hard_light(cb, cs):
    return np.where(cs <= 0.5, cb * 2 * cs, 1 - (1 - cb) * (1 - (2 * cs - 1)))


def _safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / np.maximum(b, 1e-12), 1.0)


BLENDS = {
    "norm": lambda cb, cs: cs,
    "pass": lambda cb, cs: cs,
    "mul ": lambda cb, cs: cb * cs,
    "scrn": lambda cb, cs: cb + cs - cb * cs,
    "over": lambda cb, cs: _hard_light(cs, cb),
    "hLit": _hard_light,
    "sLit": _soft_light,
    "dark": np.minimum,
    "lite": np.maximum,
    "diff": lambda cb, cs: np.abs(cb - cs),
    "smud": lambda cb, cs: cb + cs - 2 * cb * cs,
    "lddg": lambda cb, cs: np.minimum(1, cb + cs),
    "lbrn": lambda cb, cs: np.maximum(0, cb + cs - 1),
    "fsub": lambda cb, cs: np.maximum(0, cb - cs),
    "fdiv": lambda cb, cs: np.clip(_safe_div(cb, cs), 0, 1),
    "div ": lambda cb, cs: np.where(cb <= 0, 0, np.where(cs >= 1, 1, np.minimum(1, _safe_div(cb, 1 - cs)))),
    "idiv": lambda cb, cs: np.where(cb >= 1, 1, np.where(cs <= 0, 0, 1 - np.minimum(1, _safe_div(1 - cb, cs)))),
}
BLEND_NAMES = {
    "norm": "Normal", "pass": "Pass Through", "diss": "Dissolve", "dark": "Darken",
    "mul ": "Multiply", "idiv": "Color Burn", "lbrn": "Linear Burn", "dkCl": "Darker Color",
    "lite": "Lighten", "scrn": "Screen", "div ": "Color Dodge", "lddg": "Linear Dodge (Add)",
    "lgCl": "Lighter Color", "over": "Overlay", "sLit": "Soft Light", "hLit": "Hard Light",
    "vLit": "Vivid Light", "lLit": "Linear Light", "pLit": "Pin Light", "hMix": "Hard Mix",
    "diff": "Difference", "smud": "Exclusion", "fsub": "Subtract", "fdiv": "Divide",
    "hue ": "Hue", "sat ": "Saturation", "colr": "Color", "lum ": "Luminosity",
}


class Compositor:
    """Composites a document into straight RGB + alpha float buffers.

    get_rgb(layer, src) must return rgb (h, w, 3) float 0..1 for the layer's
    colour channels sliced with src (a pair of slices); it lets the caller
    cache colour-converted pixels and convert only what is drawn.
    hidden: optional set of id(layer) treated as invisible (UI toggles).
    """

    def __init__(self, doc: PSD, get_rgb: Callable, scale: int = 1, hidden: set = None):
        self.doc = doc
        self.get_rgb = get_rgb
        self.s = max(1, int(scale))
        self.hidden = hidden or set()
        self.H = (doc.height + self.s - 1) // self.s
        self.W = (doc.width + self.s - 1) // self.s
        self.unsupported = set()

    # place a rect (top,left,bottom,right) onto the scaled canvas
    def _place(self, top, left, bottom, right):
        s = self.s
        y0 = max(top, 0); x0 = max(left, 0)
        y0 = -(-y0 // s) * s; x0 = -(-x0 // s) * s
        y1 = min(bottom, self.doc.height); x1 = min(right, self.doc.width)
        if y1 <= y0 or x1 <= x0:
            return None
        src = (slice(y0 - top, y1 - top, s), slice(x0 - left, x1 - left, s))
        cy0, cx0 = y0 // s, x0 // s
        hh = len(range(y0, y1, s)); ww = len(range(x0, x1, s))
        dst = (slice(cy0, cy0 + hh), slice(cx0, cx0 + ww))
        return src, dst

    def _mask_canvas(self, L: Layer) -> Optional[np.ndarray]:
        m = L.mask
        if m is None or m.data is None or m.disabled:
            return None
        out = np.full((self.H, self.W), m.default_color / 255.0, np.float32)
        p = self._place(m.top, m.left, m.bottom, m.right)
        if p is not None and m.data.size:
            src, dst = p
            out[dst] = m.data[src].astype(np.float32) / self.doc.max_value
        return out

    def _layer_rgba(self, L: Layer):
        """-> (dst slices, rgb, alpha) on the canvas, or None."""
        p = self._place(L.top, L.left, L.bottom, L.right)
        if p is None:
            return None
        src, dst = p
        rgb = self.get_rgb(L, src)
        if rgb is None:
            return None
        if -1 in L.channels:
            a = L.channels[-1][src].astype(np.float32) / self.doc.max_value
        else:
            a = np.ones(rgb.shape[:2], np.float32)
        mc = self._mask_canvas(L)
        if mc is not None:
            a = a * mc[dst]
        a = a * (L.opacity / 255.0)
        return dst, rgb, a

    @staticmethod
    def _blend_into(cb, ab, cs, as_, mode):
        fn = BLENDS.get(mode, BLENDS["norm"])
        as3 = as_[..., None]
        ab3 = ab[..., None]
        if mode in ("norm", "pass"):
            mixed = cs
        else:
            mixed = (1 - ab3) * cs + ab3 * fn(cb, cs)
        ao = as_ + ab * (1 - as_)
        ao3 = ao[..., None]
        with np.errstate(divide="ignore", invalid="ignore"):
            co = np.where(ao3 > 0, (as3 * mixed + ab3 * (1 - as3) * cb) / np.maximum(ao3, 1e-12), 0)
        return co.astype(np.float32), ao.astype(np.float32)

    def _visible(self, L: Layer) -> bool:
        return L.visible and id(L) not in self.hidden

    def _composite_nodes(self, nodes, rgb, alpha):
        base_alpha = None   # canvas alpha of the current clipping base
        base_visible = True
        for n in nodes:
            L = n.layer
            if not L.clipping:
                base_alpha, base_visible = None, self._visible(L)
            elif not base_visible:
                continue
            if not self._visible(L):
                continue
            mode = L.blend()
            if mode not in BLENDS:
                self.unsupported.add(BLEND_NAMES.get(mode, mode))
            if n.is_group:
                op = L.opacity / 255.0
                mc = self._mask_canvas(L)
                if mode == "pass":
                    work_rgb, work_a = rgb.copy(), alpha.copy()
                    self._composite_nodes(n.children, work_rgb, work_a)
                    f = np.float32(op) if mc is None else (op * mc)
                    if np.ndim(f) == 0 and f == 1:
                        rgb[:], alpha[:] = work_rgb, work_a
                    else:
                        f2 = f if np.ndim(f) else np.full(alpha.shape, f, np.float32)
                        rgb[:] = rgb + (work_rgb - rgb) * f2[..., None]
                        alpha[:] = alpha + (work_a - alpha) * f2
                else:
                    g_rgb = np.zeros_like(rgb)
                    g_a = np.zeros_like(alpha)
                    self._composite_nodes(n.children, g_rgb, g_a)
                    g_a *= op
                    if mc is not None:
                        g_a *= mc
                    rgb[:], alpha[:] = self._blend_into(rgb, alpha, g_rgb, g_a, mode)
                if not L.clipping:
                    base_alpha = None  # clipping to groups: not supported
                continue
            r = self._layer_rgba(L)
            if r is None:
                if not L.clipping:
                    base_alpha = np.zeros_like(alpha)
                continue
            dst, cs, as_ = r
            if L.clipping:
                if base_alpha is not None:
                    as_ = as_ * base_alpha[dst]
            else:
                base_alpha = np.zeros_like(alpha)
                base_alpha[dst] = as_ / max(L.opacity / 255.0, 1e-6) if L.opacity else 0
                base_alpha = np.clip(base_alpha, 0, 1)
            rgb[dst], alpha[dst] = self._blend_into(rgb[dst], alpha[dst], cs, as_, mode)

    def run(self):
        rgb = np.zeros((self.H, self.W, 3), np.float32)
        alpha = np.zeros((self.H, self.W), np.float32)
        self._composite_nodes(build_tree(self.doc.layers), rgb, alpha)
        return rgb, alpha


def default_get_rgb(doc: PSD, conv: Optional[ColorConverter] = None):
    nc = doc.color_channels

    def get(L: Layer, src=(slice(None), slice(None))):
        if not all(c in L.channels for c in range(nc)):
            return None
        chans = [np.ascontiguousarray(L.channels[c][src]) for c in range(nc)]
        if conv is not None:
            return conv.to_rgb_float(chans)
        return np.stack(chans[:3], axis=-1).astype(np.float32) / doc.max_value
    return get


def merged_rgba(doc: PSD, conv: Optional[ColorConverter] = None, scale: int = 1):
    """Photoshop's saved composite as straight (rgb, alpha) floats, or None."""
    if doc.merged is None:
        return None
    s = max(1, scale)
    nc = doc.color_channels
    m = doc.merged[:, ::s, ::s]
    chans = [m[c] for c in range(nc)]
    if conv is not None:
        rgb = conv.to_rgb_float(chans)
    else:
        rgb = np.stack(chans[:3], axis=-1).astype(np.float32) / doc.max_value
    if doc.merged_has_alpha and m.shape[0] > nc:
        a = m[nc].astype(np.float32) / doc.max_value
        # Photoshop mattes the composite against white: undo that
        with np.errstate(divide="ignore", invalid="ignore"):
            rgb = np.where(a[..., None] > 0,
                           (rgb - (1 - a[..., None])) / np.maximum(a[..., None], 1e-6), 0)
        rgb = np.clip(rgb, 0, 1).astype(np.float32)
    else:
        a = np.ones(rgb.shape[:2], np.float32)
    return rgb, a


def merged_is_usable(doc: PSD) -> bool:
    """False when 'Maximize compatibility' was off (composite is blank)."""
    if doc.merged is None:
        return False
    if not doc.layers:
        return True
    m = doc.merged[:doc.color_channels, ::16, ::16]
    return not (m.min() == m.max())


def matte_white(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return rgb * alpha[..., None] + (1 - alpha[..., None])


def to_dtype(x: np.ndarray, dtype) -> np.ndarray:
    m = 255.0 if dtype == np.uint8 else 65535.0
    return np.clip(np.rint(x * m), 0, m).astype(dtype)


def clone(doc: PSD) -> PSD:
    """Deep-ish copy: arrays are shared until replaced (edits replace them)."""
    new = copy.copy(doc)
    new.resources = list(doc.resources)
    new.layers = []
    for L in doc.layers:
        nl = copy.copy(L)
        nl.channels = dict(L.channels)
        nl.tagged = list(L.tagged)
        nl.mask = copy.copy(L.mask) if L.mask else None
        new.layers.append(nl)
    return new

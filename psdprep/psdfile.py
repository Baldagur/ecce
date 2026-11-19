"""Minimal, dependency-light PSD/PSB reader and writer.

Only numpy is required. Pillow is used (if present) to decode RLE quickly.

Pixel data is kept exactly as stored in the file:
  * 8-bit  -> uint8
  * 16-bit -> uint16
  * CMYK values are kept inverted the way Photoshop stores them
    (255 = no ink); conversion code takes care of that.

Layers are kept in file order, i.e. bottom -> top. Groups follow the
Photoshop convention: a "divider" layer (section 3) sits below the group's
children and the group layer itself (section 1/2) sits above them.
"""
from __future__ import annotations

import io
import struct
import zlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

MODE_BITMAP, MODE_GRAY, MODE_INDEXED, MODE_RGB, MODE_CMYK = 0, 1, 2, 3, 4
MODE_MULTICHANNEL, MODE_DUOTONE, MODE_LAB = 7, 8, 9
MODE_NAMES = {0: "Bitmap", 1: "Grayscale", 2: "Indexed", 3: "RGB", 4: "CMYK",
              7: "Multichannel", 8: "Duotone", 9: "Lab"}
COLOR_CHANNELS = {MODE_GRAY: 1, MODE_RGB: 3, MODE_CMYK: 4, MODE_LAB: 3}

SECTION_NONE, SECTION_OPEN, SECTION_CLOSED, SECTION_DIVIDER = 0, 1, 2, 3

# Tagged blocks that use 8-byte lengths in PSB files.
_PSB_LONG_KEYS = {b"LMsk", b"Lr16", b"Lr32", b"Layr", b"Mt16", b"Mt32", b"Mtrn",
                  b"Alph", b"FMsk", b"lnk2", b"FEid", b"FXid", b"PxSD"}

# Keys that tell us a layer is more than plain pixels (for UI warnings).
TEXT_KEYS = {b"TySh", b"tySh"}
SMART_KEYS = {b"SoLd", b"PlLd", b"SoLE", b"plLd"}
FX_KEYS = {b"lfx2", b"lrFX", b"lmfx"}
ADJUST_KEYS = {b"levl", b"curv", b"brit", b"blnc", b"hue2", b"hue ", b"selc",
               b"thrs", b"nvrt", b"post", b"mixr", b"grdm", b"phfl", b"vibA",
               b"expA", b"clrL", b"blwh", b"CgEd"}
FILL_KEYS = {b"SoCo", b"GdFl", b"PtFl"}
# Tagged blocks we copy into the output as-is (safe after edits).
_KEEP_KEYS = {b"lyid", b"clbl", b"infx", b"knko", b"lspf", b"lclr", b"fxrp",
              b"artb", b"artd", b"abdd", b"iOpa"}  # artboards (dropped by the pipeline after a crop)


class PSDError(Exception):
    pass


@dataclass
class Mask:
    top: int = 0
    left: int = 0
    bottom: int = 0
    right: int = 0
    default_color: int = 0
    flags: int = 0
    data: Optional[np.ndarray] = None  # (h, w), same dtype as image

    @property
    def disabled(self) -> bool:
        return bool(self.flags & 2)

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)


@dataclass
class Layer:
    name: str = "Layer"
    top: int = 0
    left: int = 0
    bottom: int = 0
    right: int = 0
    channels: dict = field(default_factory=dict)   # id -> 2D array
    blend_mode: str = "norm"
    opacity: int = 255
    clipping: int = 0
    flags: int = 0
    mask: Optional[Mask] = None
    blending_ranges: bytes = b""
    section: int = SECTION_NONE
    section_blend: Optional[str] = None
    tagged: list = field(default_factory=list)     # [(sig, key, data)]
    unrendered: str = ""                           # why the content can't be drawn

    # -- convenience -------------------------------------------------------
    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

    @property
    def visible(self) -> bool:
        return not (self.flags & 2)

    @visible.setter
    def visible(self, value: bool):
        self.flags = (self.flags & ~2) if value else (self.flags | 2)

    @property
    def is_group(self) -> bool:
        return self.section in (SECTION_OPEN, SECTION_CLOSED)

    @property
    def is_divider(self) -> bool:
        return self.section == SECTION_DIVIDER

    def keys(self) -> set:
        return {k for _, k, _ in self.tagged}

    def kind(self) -> str:
        k = self.keys()
        if self.is_group:
            return "group"
        if k & TEXT_KEYS:
            return "text"
        if k & SMART_KEYS:
            return "smart"
        if k & ADJUST_KEYS:
            return "adjustment"
        if k & FILL_KEYS:
            return "fill"
        return "pixel"

    @property
    def fill_opacity(self) -> int:
        """Photoshop 'Fill' (0-255): scales the layer's own pixels, not its FX."""
        for _, k, d in self.tagged:
            if k == b"iOpa" and len(d) >= 1:
                return d[0]
        return 255

    def has_vector_mask(self) -> bool:
        return bool(self.keys() & {b"vmsk", b"vsms"})

    def has_effects(self) -> bool:
        return bool(self.keys() & FX_KEYS)

    def blend(self) -> str:
        """Effective blend key (groups store theirs in lsct)."""
        if self.is_group and self.section_blend:
            return self.section_blend
        return self.blend_mode


@dataclass
class PSD:
    version: int = 1
    width: int = 0
    height: int = 0
    depth: int = 8
    mode: int = MODE_RGB
    num_channels: int = 3
    color_mode_data: bytes = b""
    resources: list = field(default_factory=list)  # [(id, name_bytes, data)]
    layers: list = field(default_factory=list)     # bottom -> top
    merged: Optional[np.ndarray] = None            # (C, H, W) or None
    merged_has_alpha: bool = False

    @property
    def color_channels(self) -> int:
        return COLOR_CHANNELS.get(self.mode, 1)

    @property
    def dtype(self):
        return {8: np.uint8, 16: np.uint16, 32: np.float32}[self.depth]

    @property
    def max_value(self) -> float:
        return {8: 255.0, 16: 65535.0, 32: 1.0}[self.depth]

    def resource(self, rid: int) -> Optional[bytes]:
        for i, _, d in self.resources:
            if i == rid:
                return d
        return None

    def set_resource(self, rid: int, data: Optional[bytes]):
        self.resources = [r for r in self.resources if r[0] != rid]
        if data is not None:
            self.resources.append((rid, b"", data))
            self.resources.sort(key=lambda r: r[0])

    @property
    def icc_profile(self) -> Optional[bytes]:
        return self.resource(1039)

    @property
    def resolution(self) -> Optional[float]:
        d = self.resource(1005)
        if d and len(d) >= 4:
            return struct.unpack(">I", d[:4])[0] / 65536.0
        return None


# ---------------------------------------------------------------------------
# Reading helpers
# ---------------------------------------------------------------------------
class _Reader:
    def __init__(self, data: bytes, psb: bool = False):
        self.b = data
        self.p = 0
        self.psb = psb

    def read(self, n: int):
        """bytes for small reads; a zero-copy memoryview slice for big ones."""
        if self.p + n > len(self.b):
            raise PSDError("Unexpected end of file")
        r = self.b[self.p:self.p + n]
        self.p += n
        return bytes(r) if (n <= 256 and isinstance(r, memoryview)) else r

    def u8(self):
        return self.read(1)[0]

    def u16(self):
        return struct.unpack(">H", self.read(2))[0]

    def i16(self):
        return struct.unpack(">h", self.read(2))[0]

    def u32(self):
        return struct.unpack(">I", self.read(4))[0]

    def i32(self):
        return struct.unpack(">i", self.read(4))[0]

    def u64(self):
        return struct.unpack(">Q", self.read(8))[0]

    def length(self):
        """PSD: 4 byte length, PSB: 8 byte length."""
        return self.u64() if self.psb else self.u32()


def _fit(data, n: int) -> np.ndarray:
    """First n bytes of data as uint8 (zero padded if short), no copy if possible."""
    a = np.frombuffer(data, np.uint8, count=min(n, len(data)))
    if a.size < n:
        a = np.concatenate([a, np.zeros(n - a.size, np.uint8)])
    return a


def _unpackbits(data, rowbytes: int, rows: int) -> np.ndarray:
    """Decode a concatenation of PackBits rows into (rows, rowbytes) uint8."""
    if rows == 0 or rowbytes == 0:
        return np.zeros((rows, rowbytes), np.uint8)
    try:
        from PIL import Image
        im = Image.frombytes("L", (rowbytes, rows), data, "packbits", "L")
        return np.asarray(im, dtype=np.uint8).copy()
    except Exception:
        pass
    # Pure python fallback (slow, but always works).
    data = bytes(data)
    out = bytearray()
    i, n = 0, len(data)
    need = rowbytes * rows
    while i < n and len(out) < need:
        h = data[i]
        i += 1
        if h < 128:
            out += data[i:i + h + 1]
            i += h + 1
        elif h > 128:
            out += data[i:i + 1] * (257 - h)
            i += 1
    return _fit(bytes(out[:need]), need).reshape(rows, rowbytes).copy()


def _bytes_to_pixels(raw: np.ndarray, w: int, h: int, depth: int) -> np.ndarray:
    """raw: (h, w*bytes) uint8 big-endian -> (h, w) native array."""
    if depth == 8:
        return raw.reshape(h, w)
    if depth == 16:
        return raw.reshape(h, w * 2).view(">u2").astype(np.uint16)
    if depth == 32:
        return raw.reshape(h, w * 4).view(">f4").astype(np.float32)
    if depth == 1:
        bits = np.unpackbits(raw, axis=1)[:, :w]
        return (1 - bits).astype(np.uint8) * 255
    raise PSDError(f"Unsupported depth {depth}")


def _row_bytes(w: int, depth: int) -> int:
    return (w + 7) // 8 if depth == 1 else w * depth // 8


def _decode_channel(data: bytes, w: int, h: int, depth: int, psb: bool,
                    compression: Optional[int] = None) -> np.ndarray:
    """Decode one channel. If compression is None, it is read from data[0:2]."""
    if compression is None:
        if len(data) < 2:
            return np.zeros((h, w), {8: np.uint8, 16: np.uint16, 32: np.float32}.get(depth, np.uint8))
        compression = struct.unpack(">H", data[:2])[0]
        data = data[2:]
    rb = _row_bytes(w, depth)
    if w == 0 or h == 0:
        return np.zeros((h, w), {8: np.uint8, 16: np.uint16, 32: np.float32}.get(depth, np.uint8))
    if compression == 0:
        raw = _fit(data, rb * h).reshape(h, rb)
    elif compression == 1:
        csize = 4 if psb else 2
        raw = _unpackbits(data[h * csize:], rb, h)
    elif compression in (2, 3):
        dec = zlib.decompress(data)
        raw = _fit(dec, rb * h).reshape(h, rb)
        if compression == 3:
            if depth == 8:
                raw = np.cumsum(raw, axis=1, dtype=np.uint8)
            elif depth == 16:
                v = np.cumsum(raw.view(">u2").astype(np.uint16), axis=1, dtype=np.uint16)
                raw = v.astype(">u2").view(np.uint8).reshape(h, rb)
            elif depth == 32:
                d = np.cumsum(raw, axis=1, dtype=np.uint8)
                # bytes are stored planar per row: b0..., b1..., b2..., b3...
                raw = d.reshape(h, 4, w).transpose(0, 2, 1).reshape(h, rb).copy()
    else:
        raise PSDError(f"Unknown compression {compression}")
    return _bytes_to_pixels(raw, w, h, depth)


def _read_pascal(r: _Reader, pad: int) -> bytes:
    n = r.u8()
    s = r.read(n)
    total = n + 1
    if total % pad:
        r.read(pad - total % pad)
    return s


def _parse_tagged(data: bytes, psb: bool) -> list:
    out = []
    r = _Reader(data, psb)
    while r.p + 12 <= len(data):
        sig = r.read(4)
        if sig not in (b"8BIM", b"8B64"):
            break
        key = r.read(4)
        n = r.u64() if (psb and key in _PSB_LONG_KEYS) else r.u32()
        blk = r.read(min(n, len(data) - r.p))
        if key not in (b"Lr16", b"Lr32", b"Layr"):
            blk = bytes(blk)
        out.append((sig, key, blk))
    return out


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def read_psd(path_or_bytes, load_merged: bool = True) -> PSD:
    if isinstance(path_or_bytes, (bytes, bytearray)):
        data = bytes(path_or_bytes)
    else:
        with open(path_or_bytes, "rb") as f:
            data = f.read()
    data = memoryview(data)

    r = _Reader(data)
    if r.read(4) != b"8BPS":
        raise PSDError("Not a Photoshop file")
    doc = PSD()
    doc.version = r.u16()
    if doc.version not in (1, 2):
        raise PSDError(f"Unknown PSD version {doc.version}")
    r.psb = doc.version == 2
    r.read(6)
    doc.num_channels = r.u16()
    doc.height = r.u32()
    doc.width = r.u32()
    doc.depth = r.u16()
    doc.mode = r.u16()
    if doc.mode not in (MODE_GRAY, MODE_RGB, MODE_CMYK, MODE_LAB):
        raise PSDError(f"Colour mode '{MODE_NAMES.get(doc.mode, doc.mode)}' is not supported")
    if doc.depth not in (8, 16):
        raise PSDError(f"{doc.depth}-bit files are not supported yet (8 and 16-bit only)")

    n = r.u32()
    doc.color_mode_data = bytes(r.read(n))

    # image resources
    n = r.u32()
    end = r.p + n
    while r.p + 12 <= end:
        if r.read(4) not in (b"8BIM", b"MeSa", b"AgHg", b"PHUT", b"DCSR"):
            break
        rid = r.u16()
        name = _read_pascal(r, 2)
        size = r.u32()
        rdata = bytes(r.read(size))
        if size % 2:
            r.read(1)
        doc.resources.append((rid, name, rdata))
    r.p = end

    # layer and mask info
    lm_len = r.length()
    lm_end = r.p + lm_len
    layer_count = 0
    if lm_len:
        li_len = r.length()
        li_end = r.p + li_len
        if li_len:
            layer_count = _read_layer_info(r, doc)
        r.p = li_end
        if r.p + 4 <= lm_end:
            gm = r.u32()          # global layer mask info
            r.read(gm)
        # global tagged blocks: 16-bit files keep layers in Lr16
        if r.p < lm_end:
            for sig, key, blk in _parse_tagged(data[r.p:lm_end], r.psb):
                if key in (b"Lr16", b"Lr32", b"Layr") and not doc.layers:
                    sub = _Reader(blk, r.psb)
                    layer_count = _read_layer_info(sub, doc)
    r.p = lm_end

    # merged image data
    # A negative layer count means the first extra channel is transparency.
    doc.merged_has_alpha = layer_count < 0
    if load_merged and r.p + 2 <= len(data):
        comp = r.u16()
        w, h, c = doc.width, doc.height, doc.num_channels
        rb = _row_bytes(w, doc.depth)
        body = data[r.p:]
        try:
            if comp == 1:
                csize = 4 if r.psb else 2
                raw = _unpackbits(body[h * c * csize:], rb, h * c)
            elif comp == 0:
                raw = _fit(body, rb * h * c).reshape(h * c, rb)
            else:
                raw = None
                # ZIP: one stream for all channels; prediction restarts every row
                tall = _decode_channel(body, w, h * c, doc.depth, r.psb, comp)
                chans = [tall[i * h:(i + 1) * h] for i in range(c)]
            if raw is not None:
                chans = [_bytes_to_pixels(raw[i * h:(i + 1) * h], w, h, doc.depth) for i in range(c)]
            keep = doc.color_channels + (1 if doc.merged_has_alpha and c > doc.color_channels else 0)
            doc.merged = np.stack(chans[:keep])
            if doc.merged.shape[0] <= doc.color_channels:
                doc.merged_has_alpha = False
        except Exception:
            doc.merged = None
    return doc


def _read_layer_info(r: _Reader, doc: PSD) -> int:
    count = r.i16()
    n = abs(count)
    records = []
    for _ in range(n):
        L = Layer()
        L.top, L.left, L.bottom, L.right = r.i32(), r.i32(), r.i32(), r.i32()
        nch = r.u16()
        chinfo = [(r.i16(), r.length()) for _ in range(nch)]
        if r.read(4) != b"8BIM":
            raise PSDError("Bad layer record signature")
        L.blend_mode = r.read(4).decode("latin-1")
        L.opacity = r.u8()
        L.clipping = r.u8()
        L.flags = r.u8()
        r.u8()
        extra_len = r.u32()
        extra_end = r.p + extra_len
        # mask
        mlen = r.u32()
        mend = r.p + mlen
        real = None
        if mlen >= 18:
            m = Mask()
            m.top, m.left, m.bottom, m.right = r.i32(), r.i32(), r.i32(), r.i32()
            m.default_color = r.u8()
            m.flags = r.u8()
            L.mask = m
            if mlen >= 36:
                try:
                    if m.flags & 16:                     # mask parameters
                        params = r.u8()
                        for bit, size in ((1, 1), (2, 8), (4, 1), (8, 8)):
                            if params & bit:
                                r.read(size)
                    if mend - r.p >= 18:
                        real = Mask()
                        real.flags = r.u8()
                        real.default_color = r.u8()
                        real.top, real.left, real.bottom, real.right = r.i32(), r.i32(), r.i32(), r.i32()
                except PSDError:
                    real = None
        r.p = mend
        blen = r.u32()
        L.blending_ranges = r.read(blen)
        raw_name = _read_pascal(r, 4)
        L.name = raw_name.decode("mac_roman", "replace")
        if r.p < extra_end:
            L.tagged = _parse_tagged(r.b[r.p:extra_end], r.psb)
        r.p = extra_end
        for sig, key, d in L.tagged:
            if key == b"luni" and len(d) >= 4:
                cnt = struct.unpack(">I", d[:4])[0]
                L.name = d[4:4 + cnt * 2].decode("utf-16-be", "replace").rstrip("\0")
            elif key in (b"lsct", b"lsdk") and len(d) >= 4:
                L.section = struct.unpack(">I", d[:4])[0]
                if len(d) >= 12:
                    L.section_blend = d[8:12].decode("latin-1")
        L.tagged = [t for t in L.tagged if t[1] not in (b"luni", b"lsct", b"lsdk")]
        records.append((L, chinfo, real))

    # channel image data
    for L, chinfo, real in records:
        for cid, clen in chinfo:
            chunk = r.read(clen)
            if cid == -2 and L.mask is not None:
                w, h = L.mask.width, L.mask.height
            elif cid == -3 and real is not None:
                w, h = real.width, real.height
            elif cid < -1:
                continue
            else:
                w, h = L.width, L.height
            try:
                arr = _decode_channel(chunk, w, h, doc.depth, r.psb)
            except Exception:
                arr = np.zeros((h, w), doc.dtype)
            if cid == -2:
                L.mask.data = arr
            elif cid == -3:
                real.data = arr
            else:
                L.channels[cid] = arr
        if L.mask is not None and L.mask.data is None:
            L.mask.data = np.zeros((0, 0), doc.dtype)
        if real is not None and real.data is not None and L.mask is not None:
            L.mask = combine_masks(L.mask, real, doc.max_value, doc.dtype)
        _materialize_fill(L, doc)
        doc.layers.append(L)
    return count


def _fill_color(info: dict, doc: PSD) -> Optional[list]:
    """Descriptor colour -> raw channel values (0..1, PSD conventions) for doc.mode."""
    c = info.get("Clr ")
    if not isinstance(c, dict):
        return None
    try:
        if "Rd  " in c:
            rgb = [float(c[k]) / 255.0 for k in ("Rd  ", "Grn ", "Bl  ")]
        elif "Cyn " in c:
            cmyk = [float(c[k]) / 100.0 for k in ("Cyn ", "Mgnt", "Ylw ", "Blck")]
            if doc.mode == MODE_CMYK:
                return [1 - v for v in cmyk]                 # stored inverted
            rgb = [(1 - x) * (1 - cmyk[3]) for x in cmyk[:3]]
        elif "Gry " in c:
            g = 1 - float(c["Gry "]) / 100.0
            rgb = [g, g, g]
        else:
            return None
    except (TypeError, ValueError, KeyError):
        return None
    rgb = [min(1.0, max(0.0, v)) for v in rgb]
    if doc.mode == MODE_RGB:
        return rgb
    if doc.mode == MODE_CMYK:
        k = 1 - max(rgb)
        cmy = [(1 - v - k) / (1 - k) if k < 1 else 0 for v in rgb]
        return [1 - v for v in cmy + [k]]
    if doc.mode == MODE_GRAY:
        return [0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]]
    return None


def _materialize_fill(L: Layer, doc: PSD):
    """Solid-colour fill layers often carry no pixels: draw them as pixels.
    Shape layers (vector mask) and gradient/pattern fills are flagged instead."""
    keys = L.keys()
    has_pixels = L.width > 0 and L.height > 0 and all(c in L.channels for c in range(doc.color_channels))
    if b"SoCo" in keys and not has_pixels:
        if L.has_vector_mask() and not (L.mask is not None and L.mask.data is not None and L.mask.data.size):
            L.unrendered = "shape layer (vector) not rendered"
            return
        try:
            from . import descriptor
            info = descriptor.parse(next(bytes(d) for _, k, d in L.tagged if k == b"SoCo"))
            vals = _fill_color(info, doc)
        except Exception:
            vals = None
        if vals is None:
            L.unrendered = "fill colour not readable"
            return
        L.top, L.left, L.bottom, L.right = 0, 0, doc.height, doc.width
        L.channels = {i: np.full((doc.height, doc.width), round(v * doc.max_value), doc.dtype)
                      for i, v in enumerate(vals)}
    elif keys & {b"GdFl", b"PtFl"} and not has_pixels:
        L.unrendered = "gradient/pattern fill not rendered"
    elif L.has_vector_mask() and not has_pixels:
        L.unrendered = "shape layer (vector) not rendered"


def combine_masks(a: Mask, b: Mask, maxv: float, dtype) -> Mask:
    """Multiply two masks (vector mask raster x pixel mask) into one pixel mask."""
    rects = [(m.top, m.left, m.bottom, m.right) for m in (a, b) if m.width and m.height]
    if not rects:
        return Mask(0, 0, 0, 0, round(a.default_color * b.default_color / 255), 0,
                    np.zeros((0, 0), dtype))
    t = min(r[0] for r in rects); l = min(r[1] for r in rects)
    bt = max(r[2] for r in rects); rt = max(r[3] for r in rects)
    out = np.ones((bt - t, rt - l), np.float32)
    for m in (a, b):
        plane = np.full(out.shape, m.default_color / 255.0, np.float32)
        if m.width and m.height and m.data is not None and m.data.size:
            plane[m.top - t:m.bottom - t, m.left - l:m.right - l] = m.data.astype(np.float32) / maxv
        if not m.disabled:
            out *= plane
    data = np.clip(np.rint(out * maxv), 0, maxv).astype(dtype)
    default = round(255 * (a.default_color / 255 if not a.disabled else 1)
                    * (b.default_color / 255 if not b.disabled else 1))
    return Mask(t, l, bt, rt, default, 0, data)


# ---------------------------------------------------------------------------
# Writing helpers
# ---------------------------------------------------------------------------
def _packbits_block(rows: np.ndarray) -> tuple:
    """Vectorised PackBits of a block of byte rows (h, n) -> (counts[h], bytes).
    Each row is encoded independently, as PSD requires."""
    h, n = rows.shape
    if h == 0 or n == 0:
        return np.zeros(h, np.int64), b""
    flat = rows.reshape(-1)
    N = flat.size
    change = np.empty(N, bool)
    change[0] = True
    np.not_equal(flat[1:], flat[:-1], out=change[1:])
    change[::n] = True                                   # runs never cross rows
    starts = np.flatnonzero(change)
    lengths = np.diff(np.append(starts, N))
    run_row = starts // n
    rep = lengths >= 3
    # segments: every repeat run alone; consecutive literal runs in a row merge
    new = np.ones(starts.size, bool)
    new[1:] = rep[1:] | rep[:-1] | (run_row[1:] != run_row[:-1])
    seg_first = np.flatnonzero(new)
    seg_start = starts[seg_first]
    seg_len = np.add.reduceat(lengths, seg_first)
    seg_rep = rep[seg_first]
    seg_row = run_row[seg_first]
    # chunks of at most 128 bytes
    nch = (seg_len + 127) // 128
    ch_seg = np.repeat(np.arange(seg_first.size), nch)
    first_chunk = np.cumsum(nch) - nch
    ch_k = np.arange(ch_seg.size) - first_chunk[ch_seg]
    ch_start = seg_start[ch_seg] + 128 * ch_k
    ch_size = np.minimum(128, seg_len[ch_seg] - 128 * ch_k)
    ch_rep = seg_rep[ch_seg]
    ch_row = seg_row[ch_seg]
    ch_bytes = np.where(ch_rep, 2, 1 + ch_size)
    pos = np.cumsum(ch_bytes) - ch_bytes
    out = np.empty(int(ch_bytes.sum()), np.uint8)
    # headers: literal m bytes -> m-1 ; repeat m (2..128) -> 257-m ; a leftover
    # single byte of a repeat run is written as a 1-byte literal (header 0)
    hdr = np.where(ch_rep, np.where(ch_size >= 2, 257 - ch_size, 0), ch_size - 1)
    out[pos] = hdr.astype(np.uint8)
    r = ch_rep
    out[pos[r] + 1] = flat[ch_start[r]]
    lit = ~r
    if lit.any():
        sz = ch_size[lit]
        tot = int(sz.sum())
        within = np.arange(tot) - np.repeat(np.cumsum(sz) - sz, sz)
        out[np.repeat(pos[lit] + 1, sz) + within] = flat[np.repeat(ch_start[lit], sz) + within]
    counts = np.bincount(ch_row, weights=ch_bytes, minlength=h).astype(np.int64)
    return counts, out.tobytes()


def _to_bytes_rows(arr: np.ndarray, depth: int) -> np.ndarray:
    h, w = arr.shape
    if depth == 8:
        return np.ascontiguousarray(arr, dtype=np.uint8)
    if depth == 16:
        return np.ascontiguousarray(arr.astype(">u2")).view(np.uint8).reshape(h, w * 2)
    raise PSDError(f"Cannot write depth {depth}")


def encode_rle(arr: np.ndarray, depth: int, psb: bool) -> tuple:
    """Return (row_counts_bytes, data_bytes) for one channel."""
    rows = _to_bytes_rows(arr, depth)
    h, n = rows.shape
    step = max(1, (8 << 20) // max(1, n))      # ~8 MB of pixels per block
    counts, bodies = [], []
    for y in range(0, h, step):
        c, b = _packbits_block(rows[y:y + step])
        counts.append(c)
        bodies.append(b)
    counts = np.concatenate(counts) if counts else np.zeros(0, np.int64)
    dt = ">u4" if psb else ">u2"
    return counts.astype(dt).tobytes(), b"".join(bodies)


def _channel_blob(arr: Optional[np.ndarray], depth: int, psb: bool) -> bytes:
    if arr is None or arr.size == 0:
        return b"\0\0"
    counts, body = encode_rle(arr, depth, psb)
    return b"\0\1" + counts + body


def _pascal(s: str, pad: int) -> bytes:
    b = s.encode("mac_roman", "replace")[:255]
    out = bytes([len(b)]) + b
    if len(out) % pad:
        out += b"\0" * (pad - len(out) % pad)
    return out


def _tag(key: bytes, data: bytes, psb: bool, sig: bytes = b"8BIM") -> bytes:
    if len(data) % 4:
        data += b"\0" * (4 - len(data) % 4)
    if psb and key in _PSB_LONG_KEYS:
        return sig + key + struct.pack(">Q", len(data)) + data
    return sig + key + struct.pack(">I", len(data)) + data


def _default_blending_ranges(n_color: int) -> bytes:
    return (b"\0\0\xff\xff" * 2) * (1 + n_color)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
_PSD_LIMIT = (1 << 32) - (1 << 24)      # stay clear of 4-byte length fields


def _encode_layers(doc: PSD, psb: bool, progress=None) -> list:
    nc = doc.color_channels
    color_ids = list(range(nc))
    total = max(1, len(doc.layers))
    out = []
    for idx, L in enumerate(doc.layers):
        chans = []
        if L.is_group or L.is_divider:
            chans += [(cid, b"\0\0") for cid in [-1] + color_ids]
        else:
            for cid in ([-1] if -1 in L.channels else []) + color_ids:
                arr = L.channels.get(cid)
                if arr is None:
                    arr = np.zeros((L.height, L.width), doc.dtype)
                chans.append((cid, _channel_blob(arr, doc.depth, psb)))
        m = L.mask
        if m is not None and m.width and m.height and m.data is not None and m.data.size:
            chans.append((-2, _channel_blob(m.data, doc.depth, psb)))
        out.append(chans)
        if progress:
            progress(idx + 1, total)
    return out


def _layer_record(L: Layer, chans: list, nc: int, psb: bool) -> bytes:
    rec = io.BytesIO()
    top, left, bottom, right = ((0, 0, 0, 0) if (L.is_group or L.is_divider)
                                else (L.top, L.left, L.bottom, L.right))
    rec.write(struct.pack(">iiiiH", top, left, bottom, right, len(chans)))
    for cid, blob in chans:
        rec.write(struct.pack(">h", cid))
        rec.write(struct.pack(">Q" if psb else ">I", len(blob)))
    rec.write(b"8BIM" + L.blend_mode.encode("latin-1")[:4].ljust(4))
    rec.write(struct.pack(">BBBB", L.opacity, L.clipping, L.flags & 0x1F, 0))

    extra = io.BytesIO()
    m = L.mask
    if m is not None:
        # always written (even with no pixels: a "Hide All" mask is just a
        # default colour), so masked content stays hidden
        has_px = bool(m.width and m.height and m.data is not None and m.data.size)
        t, l, b, r = (m.top, m.left, m.bottom, m.right) if has_px else (0, 0, 0, 0)
        extra.write(struct.pack(">IiiiiBBxx", 20, t, l, b, r, m.default_color, m.flags & 0x0F))
    else:
        extra.write(struct.pack(">I", 0))
    br = L.blending_ranges
    if len(br) != 8 * (1 + nc):
        br = _default_blending_ranges(nc)
    extra.write(struct.pack(">I", len(br)) + br)
    extra.write(_pascal(L.name, 4))
    uni = L.name.encode("utf-16-be")
    extra.write(_tag(b"luni", struct.pack(">I", len(uni) // 2) + uni, psb))   # UTF-16 units
    if L.section:
        d = struct.pack(">I", L.section)
        if L.is_group:
            d += b"8BIM" + (L.section_blend or "pass").encode("latin-1")[:4].ljust(4)
        extra.write(_tag(b"lsct", d, psb))
    for sig, key, data in L.tagged:
        if key in _KEEP_KEYS:
            extra.write(_tag(key, bytes(data), psb, sig))
    e = extra.getvalue()
    rec.write(struct.pack(">I", len(e)) + e)
    return rec.getvalue()


def write_psd(doc: PSD, path: str, composite: Optional[np.ndarray] = None,
              composite_alpha: bool = False, progress=None):
    """Write doc to path (streamed to disk).

    composite: optional (C, H, W) merged image in the doc's dtype; C must be
    color_channels (+1 if composite_alpha). Colour must already be matted
    with white where alpha < 1 (Photoshop convention).
    Switches to PSB when the canvas is over 30000 px or the layer data would
    not fit PSD's 4 GB length fields.
    """
    nc = doc.color_channels
    has_alpha = composite_alpha and composite is not None
    nchan = nc + (1 if has_alpha else 0)
    psb = doc.width > 30000 or doc.height > 30000
    blobs = _encode_layers(doc, psb, progress)
    if not psb and sum(len(b) for ch in blobs for _, b in ch) > _PSD_LIMIT:
        psb = True
        blobs = _encode_layers(doc, psb, progress)
    records = [_layer_record(L, ch, nc, psb) for L, ch in zip(doc.layers, blobs)]
    L_LEN = ">Q" if psb else ">I"

    if composite is None:
        composite = np.full((nc, doc.height, doc.width), doc.max_value, doc.dtype)

    count = len(doc.layers)
    li_len = 0
    if doc.layers:
        li_len = 2 + sum(len(r) for r in records) + sum(len(b) for ch in blobs for _, b in ch)
    li_pad = (-li_len) % 4
    lm_len = (8 if psb else 4) + li_len + li_pad + 4

    with open(path, "wb") as f:
        f.write(b"8BPS" + struct.pack(">H", 2 if psb else 1) + b"\0" * 6)
        f.write(struct.pack(">HIIHH", nchan, doc.height, doc.width, doc.depth, doc.mode))
        f.write(struct.pack(">I", len(doc.color_mode_data)) + doc.color_mode_data)
        res = io.BytesIO()
        for rid, name, data in doc.resources:
            res.write(b"8BIM" + struct.pack(">H", rid))
            nm = bytes([len(name)]) + name
            if len(nm) % 2:
                nm += b"\0"
            res.write(nm)
            res.write(struct.pack(">I", len(data)) + data)
            if len(data) % 2:
                res.write(b"\0")
        rb = res.getvalue()
        f.write(struct.pack(">I", len(rb)) + rb)

        f.write(struct.pack(L_LEN, lm_len))
        f.write(struct.pack(L_LEN, li_len + li_pad))
        if doc.layers:
            f.write(struct.pack(">h", -count if has_alpha else count))
            for r in records:
                f.write(r)
            del records
            for ch in blobs:
                for _, blob in ch:
                    f.write(blob)
            f.write(b"\0" * li_pad)
        del blobs
        f.write(struct.pack(">I", 0))  # global layer mask info

        f.write(b"\0\1")
        enc = [encode_rle(composite[c], doc.depth, psb) for c in range(nchan)]
        for counts, _ in enc:
            f.write(counts)
        for _, body in enc:
            f.write(body)

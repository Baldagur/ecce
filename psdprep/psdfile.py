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
              b"artb", b"artd", b"abdd"}  # artboards (dropped by the pipeline after a crop)


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

    def read(self, n: int) -> bytes:
        if self.p + n > len(self.b):
            raise PSDError("Unexpected end of file")
        r = self.b[self.p:self.p + n]
        self.p += n
        return r

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


def _unpackbits(data: bytes, rowbytes: int, rows: int) -> np.ndarray:
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
    out = bytes(out[:need]).ljust(need, b"\0")
    return np.frombuffer(out, np.uint8).reshape(rows, rowbytes).copy()


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
        raw = np.frombuffer(data[:rb * h].ljust(rb * h, b"\0"), np.uint8).reshape(h, rb)
    elif compression == 1:
        csize = 4 if psb else 2
        raw = _unpackbits(data[h * csize:], rb, h)
    elif compression in (2, 3):
        dec = zlib.decompress(data)
        raw = np.frombuffer(dec[:rb * h].ljust(rb * h, b"\0"), np.uint8).reshape(h, rb)
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
        out.append((sig, key, r.read(min(n, len(data) - r.p))))
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
    doc.color_mode_data = r.read(n)

    # image resources
    n = r.u32()
    end = r.p + n
    while r.p + 12 <= end:
        if r.read(4) not in (b"8BIM", b"MeSa", b"AgHg", b"PHUT", b"DCSR"):
            break
        rid = r.u16()
        name = _read_pascal(r, 2)
        size = r.u32()
        rdata = r.read(size)
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
                need = rb * h * c
                raw = np.frombuffer(body[:need].ljust(need, b"\0"), np.uint8).reshape(h * c, rb)
            else:
                raw = None
                chans = [_decode_channel(body, w, h, doc.depth, r.psb, comp)]
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
        if mlen >= 18:
            m = Mask()
            m.top, m.left, m.bottom, m.right = r.i32(), r.i32(), r.i32(), r.i32()
            m.default_color = r.u8()
            m.flags = r.u8()
            L.mask = m
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
        records.append((L, chinfo))

    # channel image data
    for L, chinfo in records:
        for cid, clen in chinfo:
            chunk = r.read(clen)
            if cid == -2 and L.mask is not None:
                w, h = L.mask.width, L.mask.height
            elif cid < -2:
                continue  # real user mask / vector mask extras: not used
            else:
                w, h = L.width, L.height
            try:
                arr = _decode_channel(chunk, w, h, doc.depth, r.psb)
            except Exception:
                arr = np.zeros((h, w), doc.dtype)
            if cid == -2:
                if L.mask is not None:
                    L.mask.data = arr
            else:
                L.channels[cid] = arr
        doc.layers.append(L)
    return count


# ---------------------------------------------------------------------------
# Writing helpers
# ---------------------------------------------------------------------------
def _packbits_row(rb: bytes, row: np.ndarray, out: bytearray):
    n = row.size
    if n == 0:
        return
    change = np.empty(n, bool)
    change[0] = True
    np.not_equal(row[1:], row[:-1], out=change[1:])
    starts = np.flatnonzero(change)
    lengths = np.diff(np.append(starts, n))
    pos = 0

    def literal(a, b):
        while a < b:
            m = min(128, b - a)
            out.append(m - 1)
            out.extend(rb[a:a + m])
            a += m

    for i in np.flatnonzero(lengths >= 3):
        s = int(starts[i])
        L = int(lengths[i])
        literal(pos, s)
        v = rb[s]
        while L >= 2:
            m = min(128, L)
            out.append(257 - m)
            out.append(v)
            L -= m
        if L == 1:
            out.append(0)
            out.append(v)
        pos = s + int(lengths[i])
    literal(pos, n)


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
    counts = []
    body = bytearray()
    for row in rows:
        start = len(body)
        _packbits_row(row.tobytes(), row, body)
        counts.append(len(body) - start)
    fmt = ">%d%s" % (len(counts), "I" if psb else "H")
    return struct.pack(fmt, *counts), bytes(body)


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
def write_psd(doc: PSD, path: str, composite: Optional[np.ndarray] = None,
              composite_alpha: bool = False, progress=None):
    """Write doc to path.

    composite: optional (C, H, W) merged image in the doc's dtype; C must be
    color_channels (+1 if composite_alpha). Colour must already be matted
    with white where alpha < 1 (Photoshop convention).
    """
    psb = doc.width > 30000 or doc.height > 30000
    version = 2 if psb else 1
    nc = doc.color_channels
    color_ids = list(range(nc))
    f = io.BytesIO()

    # header
    has_alpha = composite_alpha and composite is not None
    nchan = nc + (1 if has_alpha else 0)
    f.write(b"8BPS" + struct.pack(">H", version) + b"\0" * 6)
    f.write(struct.pack(">HIIHH", nchan, doc.height, doc.width, doc.depth, doc.mode))
    # colour mode data
    f.write(struct.pack(">I", len(doc.color_mode_data)) + doc.color_mode_data)
    # resources
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

    # layer info
    total = max(1, len(doc.layers))
    li = io.BytesIO()
    count = len(doc.layers)
    li.write(struct.pack(">h", -count if (has_alpha and count) else count))
    blobs = []
    for idx, L in enumerate(doc.layers):
        chans = []
        if L.is_group or L.is_divider:
            ids = [-1] + color_ids
            for cid in ids:
                chans.append((cid, b"\0\0"))
        else:
            for cid in ([-1] if -1 in L.channels else []) + color_ids:
                arr = L.channels.get(cid)
                if arr is None:
                    arr = np.zeros((L.height, L.width), doc.dtype)
                chans.append((cid, _channel_blob(arr, doc.depth, psb)))
            if L.mask is not None and L.mask.data is not None and L.mask.data.size:
                chans.append((-2, _channel_blob(L.mask.data, doc.depth, psb)))
        blobs.append(chans)
        if progress:
            progress(idx + 1, total)

        top, left, bottom, right = ((0, 0, 0, 0) if (L.is_group or L.is_divider)
                                    else (L.top, L.left, L.bottom, L.right))
        li.write(struct.pack(">iiiiH", top, left, bottom, right, len(chans)))
        for cid, blob in chans:
            li.write(struct.pack(">h", cid))
            li.write(struct.pack(">Q" if psb else ">I", len(blob)))
        li.write(b"8BIM" + L.blend_mode.encode("latin-1")[:4].ljust(4))
        li.write(struct.pack(">BBBB", L.opacity, L.clipping, L.flags & 0x1F, 0))

        extra = io.BytesIO()
        m = L.mask
        if m is not None and m.data is not None and m.data.size:
            extra.write(struct.pack(">IiiiiBBxx", 20, m.top, m.left, m.bottom, m.right,
                                    m.default_color, m.flags & 0x0F))
        else:
            extra.write(struct.pack(">I", 0))
        br = L.blending_ranges
        if len(br) != 8 * (1 + nc):
            br = _default_blending_ranges(nc)
        extra.write(struct.pack(">I", len(br)) + br)
        extra.write(_pascal(L.name, 4))
        uni = L.name.encode("utf-16-be")
        extra.write(_tag(b"luni", struct.pack(">I", len(L.name)) + uni, psb))
        if L.section:
            d = struct.pack(">I", L.section)
            if L.is_group:
                d += b"8BIM" + (L.section_blend or "pass").encode("latin-1")[:4].ljust(4)
            extra.write(_tag(b"lsct", d, psb))
        for sig, key, data in L.tagged:
            if key in _KEEP_KEYS:
                extra.write(_tag(key, data, psb, sig))
        e = extra.getvalue()
        li.write(struct.pack(">I", len(e)) + e)

    for chans in blobs:
        for _, blob in chans:
            li.write(blob)
    lib = li.getvalue() if doc.layers else b""
    if len(lib) % 4:
        lib += b"\0" * (4 - len(lib) % 4)

    lm = io.BytesIO()
    lm.write(struct.pack(">Q" if psb else ">I", len(lib)) + lib)
    lm.write(struct.pack(">I", 0))  # global layer mask info
    lmb = lm.getvalue()
    f.write(struct.pack(">Q" if psb else ">I", len(lmb)) + lmb)

    # merged composite
    if composite is None:
        composite = np.full((nc, doc.height, doc.width), doc.max_value, doc.dtype)
        if doc.mode == MODE_CMYK:
            composite[:] = doc.max_value  # inverted: white = no ink
    f.write(b"\0\1")
    counts_all, bodies = [], []
    for c in range(nchan):
        counts, body = encode_rle(composite[c], doc.depth, psb)
        counts_all.append(counts)
        bodies.append(body)
    f.write(b"".join(counts_all))
    f.write(b"".join(bodies))

    with open(path, "wb") as out:
        out.write(f.getvalue())

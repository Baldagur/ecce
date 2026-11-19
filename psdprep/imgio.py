"""Small PNG and TIFF writers (8/16-bit, RGB or RGBA, embedded ICC).
Only numpy + zlib, so they work everywhere."""
from __future__ import annotations

import struct
import zlib
from typing import Optional

import numpy as np


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))


def write_png(path: str, pixels: np.ndarray, icc: Optional[bytes] = None,
              dpi: Optional[float] = None, level: int = 6):
    """pixels: HxWx3 or HxWx4, uint8 or uint16, straight (unpremultiplied) alpha."""
    h, w, c = pixels.shape
    bits = 16 if pixels.dtype == np.uint16 else 8
    ctype = 6 if c == 4 else 2
    if bits == 16:
        raw = np.ascontiguousarray(pixels.astype(">u2")).view(np.uint8).reshape(h, w * c * 2)
    else:
        raw = np.ascontiguousarray(pixels, dtype=np.uint8).reshape(h, w * c)
    bpp = c * bits // 8
    # Sub filter: byte minus the byte one pixel to the left
    filt = raw.copy()
    filt[:, bpp:] = raw[:, bpp:] - raw[:, :-bpp]
    data = np.empty((h, filt.shape[1] + 1), np.uint8)
    data[:, 0] = 1
    data[:, 1:] = filt
    out = [b"\x89PNG\r\n\x1a\n",
           _png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, bits, ctype, 0, 0, 0))]
    if icc:
        out.append(_png_chunk(b"iCCP", b"ICC Profile\0\0" + zlib.compress(icc)))
    if dpi:
        ppm = int(round(dpi / 0.0254))
        out.append(_png_chunk(b"pHYs", struct.pack(">IIB", ppm, ppm, 1)))
    out.append(_png_chunk(b"IDAT", zlib.compress(data.tobytes(), level)))
    out.append(_png_chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(b"".join(out))


def write_tiff(path: str, pixels: np.ndarray, icc: Optional[bytes] = None,
               dpi: Optional[float] = None, compress: bool = True, rows_per_strip: int = 32):
    """Little-endian baseline TIFF, Deflate + horizontal predictor.
    Alpha (if present) is written as unassociated (straight) alpha."""
    h, w, c = pixels.shape
    bits = 16 if pixels.dtype == np.uint16 else 8
    dt = np.dtype("<u2") if bits == 16 else np.dtype(np.uint8)
    px = np.ascontiguousarray(pixels.astype(dt))

    strips = []
    for y in range(0, h, rows_per_strip):
        block = px[y:y + rows_per_strip]
        if compress:
            d = block.copy()
            d[:, 1:] = block[:, 1:] - block[:, :-1]      # predictor 2 (wraps)
            strips.append(zlib.compress(d.tobytes(), 6))
        else:
            strips.append(block.tobytes())

    entries = []   # (tag, type, count, value_bytes)

    def add(tag, typ, values):
        fmt = {3: "H", 4: "I", 5: "II", 7: "B"}[typ]
        if typ == 5:
            b = b"".join(struct.pack("<II", *v) for v in values)
        elif typ == 7:
            b = bytes(values)
        else:
            b = struct.pack("<%d%s" % (len(values), fmt), *values)
        entries.append((tag, typ, len(values), b))

    n_strips = len(strips)
    add(256, 4, [w])
    add(257, 4, [h])
    add(258, 3, [bits] * c)
    add(259, 3, [8 if compress else 1])
    add(262, 3, [2])
    add(273, 4, [0] * n_strips)          # patched below
    add(277, 3, [c])
    add(278, 4, [rows_per_strip])
    add(279, 4, [len(s) for s in strips])
    res = int(round(dpi or 72))
    add(282, 5, [(res, 1)])
    add(283, 5, [(res, 1)])
    add(284, 3, [1])
    add(296, 3, [2])
    if compress:
        add(317, 3, [2])
    if c == 4:
        add(338, 3, [2])
    if icc:
        add(34675, 7, list(icc))
    entries.sort(key=lambda e: e[0])

    # layout: header | IFD | overflow values | strips
    ifd_off = 8
    ifd_size = 2 + 12 * len(entries) + 4
    extra_off = ifd_off + ifd_size
    extra = bytearray()
    placed = {}
    for tag, typ, cnt, b in entries:
        if len(b) > 4:
            if len(extra) % 2:
                extra += b"\0"
            placed[tag] = extra_off + len(extra)
            extra += b
    data_off = extra_off + len(extra)
    if data_off % 2:
        extra += b"\0"
        data_off += 1
    offsets, pos = [], data_off
    for s in strips:
        offsets.append(pos)
        pos += len(s)
    # patch strip offsets
    ob = struct.pack("<%dI" % n_strips, *offsets)
    if 273 in placed:
        o = placed[273] - extra_off
        extra[o:o + len(ob)] = ob
    ifd = bytearray(struct.pack("<H", len(entries)))
    for tag, typ, cnt, b in entries:
        if tag == 273:
            b = ob
        if len(b) > 4:
            ifd += struct.pack("<HHII", tag, typ, cnt, placed[tag])
        else:
            ifd += struct.pack("<HHI", tag, typ, cnt) + b.ljust(4, b"\0")
    ifd += struct.pack("<I", 0)
    with open(path, "wb") as f:
        f.write(b"II*\0" + struct.pack("<I", ifd_off))
        f.write(ifd)
        f.write(extra)
        for s in strips:
            f.write(s)

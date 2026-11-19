"""Reader (and a small writer, for tests) for Photoshop action descriptors,
the key/value format used by artboard data and many other tagged blocks."""
from __future__ import annotations

import struct


class _R:
    def __init__(self, b: bytes, p: int = 0):
        self.b, self.p = b, p

    def take(self, n):
        if self.p + n > len(self.b):
            raise ValueError("descriptor truncated")
        v = self.b[self.p:self.p + n]
        self.p += n
        return v

    def u32(self):
        return struct.unpack(">I", self.take(4))[0]

    def i32(self):
        return struct.unpack(">i", self.take(4))[0]

    def ustr(self):
        n = self.u32()
        return self.take(n * 2).decode("utf-16-be", "replace").rstrip("\0")

    def key(self):
        n = self.u32()
        return self.take(4 if n == 0 else n).decode("latin-1")


def _value(r: _R, t: bytes):
    if t in (b"Objc", b"GlbO"):
        return _descriptor(r)
    if t == b"VlLs":
        return [_value(r, r.take(4)) for _ in range(r.u32())]
    if t == b"doub":
        return struct.unpack(">d", r.take(8))[0]
    if t == b"UntF":
        r.take(4)
        return struct.unpack(">d", r.take(8))[0]
    if t == b"UnFl":
        r.take(4)
        n = r.u32()
        return list(struct.unpack(">%dd" % n, r.take(8 * n)))
    if t == b"TEXT":
        return r.ustr()
    if t == b"enum":
        r.key()
        return r.key()
    if t == b"long":
        return r.i32()
    if t == b"comp":
        return struct.unpack(">q", r.take(8))[0]
    if t == b"bool":
        return bool(r.take(1)[0])
    if t in (b"type", b"GlbC"):
        r.ustr()
        return r.key()
    if t in (b"alis", b"tdta", b"Pth "):
        return r.take(r.u32())
    if t == b"obj ":
        out = []
        for _ in range(r.u32()):
            rt = r.take(4)
            if rt == b"prop":
                r.ustr(); r.key(); out.append(r.key())
            elif rt == b"Clss":
                r.ustr(); out.append(r.key())
            elif rt == b"Enmr":
                r.ustr(); r.key(); r.key(); out.append(r.key())
            elif rt == b"rele":
                r.ustr(); r.key(); out.append(r.u32())
            elif rt in (b"Idnt", b"indx"):
                out.append(r.u32())
            elif rt == b"name":
                r.ustr(); r.key(); out.append(r.ustr())
            else:
                raise ValueError(f"unknown reference type {rt!r}")
        return out
    raise ValueError(f"unknown descriptor type {t!r}")


def _descriptor(r: _R) -> dict:
    r.ustr()            # name
    r.key()             # class id
    d = {}
    for _ in range(r.u32()):
        k = r.key()
        d[k] = _value(r, r.take(4))
    return d


def parse(data: bytes, with_version: bool = True) -> dict:
    """data: tagged-block payload starting with the u32 descriptor version (16)."""
    r = _R(data)
    if with_version:
        r.u32()
    return _descriptor(r)


# ---------------------------------------------------------------------------
# Writer (only what the tests/fixtures need)
# ---------------------------------------------------------------------------
def _wkey(k: str) -> bytes:
    b = k.encode("latin-1")
    return struct.pack(">I", 0) + b if len(b) == 4 else struct.pack(">I", len(b)) + b


def _wustr(s: str) -> bytes:
    return struct.pack(">I", len(s) + 1) + (s + "\0").encode("utf-16-be")


def _wvalue(v) -> bytes:
    if isinstance(v, bool):
        return b"bool" + bytes([int(v)])
    if isinstance(v, int):
        return b"long" + struct.pack(">i", v)
    if isinstance(v, float):
        return b"doub" + struct.pack(">d", v)
    if isinstance(v, str):
        return b"TEXT" + _wustr(v)
    if isinstance(v, dict):
        return b"Objc" + _wdesc(v)
    if isinstance(v, list):
        return b"VlLs" + struct.pack(">I", len(v)) + b"".join(_wvalue(x) for x in v)
    raise TypeError(type(v))


def _wdesc(d: dict, cls: str = "null") -> bytes:
    out = _wustr("") + _wkey(cls) + struct.pack(">I", len(d))
    for k, v in d.items():
        out += _wkey(k) + _wvalue(v)
    return out


def build(d: dict) -> bytes:
    return struct.pack(">I", 16) + _wdesc(d)

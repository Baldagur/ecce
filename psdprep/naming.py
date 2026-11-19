"""Output names next to the original: <name>_<suffix>_v<N>.<ext>, never
overwriting anything, and the guard that protects the original file."""
from __future__ import annotations

import os
import re

DEFAULT_SUFFIX = "cleanup"


class ProtectedFileError(Exception):
    pass


def clean_suffix(s: str) -> str:
    s = re.sub(r"[^\w\-]+", "_", (s or "").strip()).strip("_")
    return s


def safe_name(s: str) -> str:
    return re.sub(r"[^\w\-]+", "_", s.strip()).strip("_") or "artboard"


def base_name(source: str, suffix: str) -> str:
    stem = os.path.splitext(os.path.basename(source))[0]
    suf = clean_suffix(suffix)
    return f"{stem}_{suf}" if suf else stem


def next_version(folder: str, base: str) -> int:
    """1 + the highest v<N> already used by <base>_v<N>... in folder."""
    pat = re.compile(r"^" + re.escape(base) + r"_v(\d+)(?:[._]|$)", re.IGNORECASE)
    best = 0
    try:
        for f in os.listdir(folder):
            m = pat.match(f)
            if m:
                best = max(best, int(m.group(1)))
    except OSError:
        pass
    return best + 1


def auto_stem(source: str, suffix: str) -> str:
    """Full path without extension, e.g. /jobs/x/logo_cleanup_v3"""
    folder = os.path.dirname(os.path.abspath(source))
    base = base_name(source, suffix)
    return os.path.join(folder, f"{base}_v{next_version(folder, base)}")


def artboard_stem(stem: str, number: int, name: str, width: int = 2) -> str:
    return f"{stem}_{number:0{width}d}_{safe_name(name)}"


def is_same_file(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if os.path.realpath(a) == os.path.realpath(b):
        return True
    try:
        return os.path.exists(a) and os.path.exists(b) and os.path.samefile(a, b)
    except OSError:
        return False


def guard(source: str, outputs: list):
    """Raise if any output would overwrite the original PSD."""
    for p in outputs:
        if p and source and is_same_file(source, p):
            raise ProtectedFileError(f"Refusing to overwrite the original file: {p}")

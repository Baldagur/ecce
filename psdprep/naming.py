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


def custom_stem(folder: str, name: str) -> str:
    """<folder>/<name>_v<N>: a chosen folder and name, still versioned."""
    base = re.sub(r"[^\w\-.]+", "_", (name or "").strip()).strip("_.") or "output"
    base = re.sub(r"_v\d+$", "", base, flags=re.IGNORECASE)   # the version is added for you
    return os.path.join(folder, f"{base}_v{next_version(folder, base)}")


def with_tag(stem: str, tag: str) -> str:
    """Colour-space tag at the end of the name: logo_cleanup_v1 -> logo_cleanup_v1_sRGB"""
    return f"{stem}_{tag}" if tag else stem


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


OPEN_EXTS = (".psd", ".psb", ".ai", ".pdf")


def clean_pasted_path(text: str) -> str:
    """Turn whatever was pasted into a path: strips spaces/newlines and
    surrounding quotes, decodes file:// URLs (file managers copy those),
    expands ~ and $VARS, and takes the first line of a multi-line paste."""
    from urllib.parse import unquote, urlparse
    lines = [ln.strip() for ln in (text or "").replace("\r", "\n").split("\n") if ln.strip()]
    if not lines:
        return ""
    t = lines[0]
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'":
        t = t[1:-1].strip()
    if t.lower().startswith("file://"):
        t = unquote(urlparse(t).path)
    t = os.path.expandvars(os.path.expanduser(t))
    return os.path.normpath(t) if t else ""


def check_open_path(path: str) -> str:
    """'' if path can be opened, otherwise a short reason for the user."""
    if not path:
        return "Paste a file path first."
    if not os.path.exists(path):
        return f"Not found: {path}"
    if os.path.isdir(path):
        return f"That's a folder: {path}"
    if not path.lower().endswith(OPEN_EXTS):
        return "Only .psd, .psb, .ai and .pdf files can be opened."
    if not os.access(path, os.R_OK):
        return f"No permission to read {path}"
    return ""

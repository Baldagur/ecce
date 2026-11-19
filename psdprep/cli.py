"""Command line. With no arguments (or just a PSD path) the GUI opens."""
from __future__ import annotations

import argparse
import fnmatch
import os
import sys

from . import __version__, ops
from .pipeline import Settings, process
from .psdfile import MODE_NAMES, read_psd


def _wants_gui(argv):
    return not argv or (len(argv) == 1 and argv[0].lower().endswith((".psd", ".psb")))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if _wants_gui(argv):
        from .gui import main as gui_main
        return gui_main([sys.argv[0]] + argv)

    ap = argparse.ArgumentParser(prog="psdprep", description="Edit a PSD: remove layers, crop, "
                                 "convert to sRGB; write a layered PSD and/or a flattened PNG/TIFF.")
    ap.add_argument("psd")
    ap.add_argument("--list", action="store_true", help="list layers and exit")
    ap.add_argument("--out-psd", help="layered PSD to write")
    ap.add_argument("--out-flat", help="flattened .png / .tif to write")
    ap.add_argument("--remove", action="append", default=[],
                    help="layer name or glob to remove (repeatable)")
    ap.add_argument("--remove-hidden", action="store_true", help="remove layers hidden in the file")
    ap.add_argument("--crop", help="x,y,w,h in pixels")
    ap.add_argument("--keep-cropped", action="store_true", help="keep pixels outside the crop")
    ap.add_argument("--no-convert", action="store_true", help="keep the original colour mode")
    ap.add_argument("--profile", help="CMYK ICC profile when the file has none")
    ap.add_argument("--ignore-embedded", action="store_true", help="use --profile even if embedded")
    ap.add_argument("--intent", default="Relative Colorimetric", choices=list(ops.INTENTS))
    ap.add_argument("--no-bpc", action="store_true")
    ap.add_argument("--depth", type=int, choices=[8, 16], help="bit depth of the flattened file")
    ap.add_argument("--no-alpha", action="store_true", help="flattened file without alpha")
    ap.add_argument("--source", default="auto", choices=["auto", "layers", "composite"])
    ap.add_argument("--version", action="version", version=__version__)
    a = ap.parse_args(argv)

    doc = read_psd(a.psd)
    if a.list or not (a.out_psd or a.out_flat):
        print(f"{a.psd}: {doc.width}x{doc.height} {MODE_NAMES.get(doc.mode)} {doc.depth}-bit, "
              f"profile: {ops.profile_info(doc.icc_profile)[1] if doc.icc_profile else 'none'}")
        for node, depth in ops.walk(ops.build_tree(doc.layers)):
            L = node.layer
            flags = [L.kind()] + (["hidden"] if not L.visible else []) + \
                    (["fx"] if L.has_effects() else [])
            print(f"  {'  ' * depth}{L.name}  [{', '.join(flags)}]")
        return 0

    remove = set()
    for i, L in enumerate(doc.layers):
        if L.is_divider:
            continue
        if any(fnmatch.fnmatchcase(L.name, pat) for pat in a.remove) or (a.remove_hidden and not L.visible):
            remove.add(i)
    s = Settings(remove=remove, delete_cropped=not a.keep_cropped, convert=not a.no_convert,
                 use_embedded=not a.ignore_embedded, cmyk_profile=a.profile, intent=a.intent,
                 bpc=not a.no_bpc, flatten_source=a.source, out_psd=a.out_psd,
                 out_flat=a.out_flat, flat_depth=a.depth, flat_alpha=not a.no_alpha)
    if a.crop:
        s.crop = tuple(int(v) for v in a.crop.split(","))
    if s.cmyk_profile is None:
        found = ops.find_cmyk_profiles()
        if found:
            s.cmyk_profile = found[0][0]
    for p in (a.out_psd,):
        if p and os.path.abspath(p) == os.path.abspath(a.psd):
            ap.error("refusing to overwrite the input file")
    process(doc, s)
    return 0

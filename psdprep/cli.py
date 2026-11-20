"""Command line. With no arguments (or just a PSD path) the GUI opens."""
from __future__ import annotations

import argparse
import fnmatch
import os
import sys

from . import __version__, ops
from . import naming
from .artboards import find_artboards, is_artboard, sort_by_position
from .pipeline import Settings, export_artboards, process
from .psdfile import MODE_NAMES, read_psd


def _wants_gui(argv):
    return not argv or (len(argv) == 1 and argv[0].lower().endswith((".psd", ".psb", ".ai", ".pdf")))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "install-shortcut":
        from .desktop import install
        install()
        return 0
    if argv and argv[0] == "selftest":
        from .selftest import run
        log = argv[argv.index("--log") + 1] if "--log" in argv[:-1] else None
        return run(log)
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
    ap.add_argument("--to", default="srgb", choices=["none", "srgb", "rec709", "rec709_oetf"],
                    help="output colour space (default srgb; rec709 = gamma 2.4, rec709_oetf = camera curve)")
    ap.add_argument("--no-convert", action="store_true", help="same as --to none")
    ap.add_argument("--check", action="store_true",
                    help="measure the colour conversion against the source profile (ΔE2000) and print it")
    ap.add_argument("--folder", help="with --auto: write into this folder instead of next to the original")
    ap.add_argument("--name", help="with --auto: base file name (version and colour space are added)")
    ap.add_argument("--profile", help="CMYK ICC profile when the file has none")
    ap.add_argument("--ignore-embedded", action="store_true", help="use --profile even if embedded")
    ap.add_argument("--intent", default="Relative Colorimetric", choices=list(ops.INTENTS))
    ap.add_argument("--no-bpc", action="store_true")
    ap.add_argument("--depth", type=int, choices=[8, 16], help="bit depth of the flattened file")
    ap.add_argument("--no-alpha", action="store_true", help="flattened file without alpha")
    ap.add_argument("--source", default="auto", choices=["auto", "layers", "composite"])
    ap.add_argument("--auto", choices=["psd", "flat", "both"],
                    help="save next to the original as <name>_<suffix>_v<N> (never overwrites)")
    ap.add_argument("--suffix", default="cleanup", help="name suffix for --auto (default: cleanup)")
    ap.add_argument("--format", default="png", choices=["png", "tif"], help="flattened format for --auto")
    ap.add_argument("--artboards", action="store_true",
                    help="with --auto: export each artboard on its own, numbered in reading order")
    ap.add_argument("--layer-order", action="store_true", help="number artboards in layers-panel order")
    ap.add_argument("--dpi", type=float, help="Illustrator/PDF input: render resolution (default 300)")
    ap.add_argument("--long-side", type=int, help="Illustrator/PDF input: longest side in pixels")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow replacing existing --out-psd/--out-flat files (never the input)")
    ap.add_argument("--version", action="version", version=__version__)
    a = ap.parse_args(argv)

    from .vector import is_vector, load_vector, VectorError
    if is_vector(a.psd):
        try:
            doc = load_vector(a.psd, a.dpi if not a.long_side else None, a.long_side)
        except VectorError as e:
            ap.error(str(e))
    else:
        doc = read_psd(a.psd)
    if a.list or not (a.out_psd or a.out_flat or a.auto or a.check):
        print(f"{a.psd}: {doc.width}x{doc.height} {MODE_NAMES.get(doc.mode)} {doc.depth}-bit, "
              f"profile: {ops.profile_info(doc.icc_profile)[1] if doc.icc_profile else 'none'}")
        for node, depth in ops.walk(ops.build_tree(doc.layers)):
            L = node.layer
            flags = [L.kind() if not is_artboard(L) else "artboard"] + (["hidden"] if not L.visible else []) + \
                    (["fx"] if L.has_effects() else [])
            print(f"  {'  ' * depth}{L.name}  [{', '.join(flags)}]")
        return 0

    remove = set()
    for i, L in enumerate(doc.layers):
        if L.is_divider:
            continue
        if any(fnmatch.fnmatchcase(L.name, pat) for pat in a.remove) or (a.remove_hidden and not L.visible):
            remove.add(i)
    s = Settings(remove=remove, delete_cropped=not a.keep_cropped, convert=not (a.no_convert or a.to == "none"),
                 use_embedded=not a.ignore_embedded, cmyk_profile=a.profile, intent=a.intent,
                 bpc=not a.no_bpc, flatten_source=a.source, out_psd=a.out_psd,
                 out_flat=a.out_flat, flat_depth=a.depth, flat_alpha=not a.no_alpha)
    if a.crop:
        try:
            x, y, w, h = (int(v) for v in a.crop.split(","))
        except ValueError:
            ap.error("--crop needs four whole numbers: x,y,w,h")
        if w <= 0 or h <= 0:
            ap.error("--crop width and height must be positive")
        if x >= doc.width or y >= doc.height or x + w <= 0 or y + h <= 0:
            ap.error(f"--crop {a.crop} is outside the {doc.width}x{doc.height} image")
        s.crop = (x, y, w, h)
    s.overwrite = a.overwrite
    s.target = a.to if a.to != "none" else "srgb"
    if a.check:
        from .check import conversion_check
        from .pipeline import make_converter
        print(conversion_check(doc, make_converter(doc, s), 1).message)
        if not (a.out_psd or a.out_flat or a.auto):
            return 0
    if s.cmyk_profile is None:
        found = ops.find_cmyk_profiles()
        if found:
            s.cmyk_profile = found[0][0]
    s.source_path = a.psd
    for p in (a.out_psd, a.out_flat):
        if p and naming.is_same_file(p, a.psd):
            ap.error("refusing to overwrite the input file")
    ext = "." + a.format
    if a.auto:
        from .pipeline import colour_tag
        if a.folder:
            if not os.path.isdir(a.folder):
                ap.error(f"--folder {a.folder} doesn't exist")
            stem = naming.custom_stem(a.folder, a.name or naming.base_name(a.psd, a.suffix))
        else:
            stem = naming.auto_stem(a.psd, a.suffix)
        tag = colour_tag(doc, s)
        want_psd, want_flat = a.auto in ("psd", "both"), a.auto in ("flat", "both")
        if a.artboards:
            boards = find_artboards(doc)
            if not boards:
                ap.error("this file has no artboards")
            boards = boards if a.layer_order else sort_by_position(boards)
            export_artboards(doc, s, boards, stem, want_psd, ext if want_flat else None, tag=tag)
            return 0
        stem = naming.with_tag(stem, tag)
        s.out_psd = stem + ".psd" if want_psd else None
        s.out_flat = stem + ext if want_flat else None
    try:
        process(doc, s)
    except FileExistsError as e:
        ap.error(f"{e}. Use --overwrite to replace it, or --auto for versioned names.")
    return 0

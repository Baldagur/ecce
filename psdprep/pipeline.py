"""One call that applies all edits and writes the outputs."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from . import ops
from .artboards import Artboard, others_to_remove
from .naming import artboard_stem, guard
from .imgio import write_png, write_tiff
from .psdfile import MODE_RGB, PSD, write_psd


@dataclass
class Settings:
    remove: set = field(default_factory=set)          # indices into doc.layers
    visibility: dict = field(default_factory=dict)    # index -> bool
    crop: Optional[tuple] = None                      # (x, y, w, h)
    delete_cropped: bool = True
    convert: bool = True                              # to sRGB
    use_embedded: bool = True
    cmyk_profile: Optional[str] = None
    intent: str = "Relative Colorimetric"
    bpc: bool = True
    flatten_source: str = "auto"                      # auto | layers | composite
    out_psd: Optional[str] = None
    out_flat: Optional[str] = None                    # .png / .tif / .tiff
    flat_depth: Optional[int] = None                  # 8 / 16 / None = same as PSD
    flat_alpha: bool = True
    source_path: Optional[str] = None                 # original file: never written to
    artboard: Optional[Artboard] = None               # export just this artboard
    artboard_transparent: bool = False                # ignore the artboard's background colour
    overwrite: bool = False                           # allow replacing existing outputs (never the original)


def make_converter(doc: PSD, s: Settings) -> ops.ColorConverter:
    return ops.ColorConverter(doc.mode, doc.depth, doc.icc_profile, s.cmyk_profile,
                              s.intent, s.bpc, s.use_embedded)


def edited_doc(doc: PSD, s: Settings) -> PSD:
    """Removal, visibility and crop applied (no colour change)."""
    work = ops.clone(doc)
    for i, vis in s.visibility.items():
        if 0 <= i < len(work.layers):
            work.layers[i].visible = vis
    remove = set(s.remove)
    crop = s.crop
    if s.artboard is not None:
        remove |= others_to_remove(doc, s.artboard)
        crop = s.artboard.rect
    if remove:
        ids = {id(work.layers[i]) for i in remove if 0 <= i < len(work.layers)}
        ops.remove_layers(work, ids)
    if crop:
        ops.crop(work, *crop, delete_pixels=s.delete_cropped)
        # artboard rectangles no longer match the cropped canvas: plain groups
        from .artboards import ARTBOARD_KEYS
        for L in work.layers:
            L.tagged = [t for t in L.tagged if t[1] not in ARTBOARD_KEYS]
    return work


def use_saved_composite(doc: PSD, s: Settings) -> bool:
    if s.flatten_source == "layers" or (s.artboard is not None and s.flatten_source == "auto"):
        return False
    usable = ops.merged_is_usable(doc)
    if s.flatten_source == "composite":
        return usable
    changed = bool(s.remove) or any(doc.layers[i].visible != v for i, v in s.visibility.items()
                                    if 0 <= i < len(doc.layers))
    return usable and not changed


def process(doc: PSD, s: Settings, log: Callable = print, progress: Callable = None) -> dict:
    result = {"warnings": []}
    guard(s.source_path, [s.out_psd, s.out_flat])
    if not s.overwrite:
        for p in (s.out_psd, s.out_flat):
            if p and os.path.exists(p):
                raise FileExistsError(f"{p} already exists (not overwritten)")
    work = edited_doc(doc, s)
    conv = make_converter(work, s)
    saved = use_saved_composite(doc, s)
    result["composite_source"] = "Photoshop composite" if saved else "rebuilt from layers"

    needs_convert = s.convert and (work.mode != MODE_RGB or conv.method == "icc")
    if needs_convert:
        how = f", {s.intent}{', BPC' if s.bpc else ''}" if conv.method == "icc" else ""
        log(f"Converting to sRGB ({conv.source_desc}{how})")
        ops.convert_to_srgb(work, conv, progress)
        conv_rgb = None
    else:
        conv_rgb = conv if work.mode != MODE_RGB else None

    # composite (straight rgb + alpha)
    if saved:
        rgb, alpha = ops.merged_rgba(work, conv_rgb)
    else:
        comp = ops.Compositor(work, ops.default_get_rgb(work, conv_rgb))
        rgb, alpha = comp.run()
        if comp.unsupported:
            result["warnings"].append("Blend modes drawn as Normal: " + ", ".join(sorted(comp.unsupported)))
        fx = [L.name for L in work.layers if L.has_effects() and L.visible]
        if fx:
            result["warnings"].append("Layer styles not rendered: " + ", ".join(fx[:6]))
        unr = [f"{L.name} ({L.unrendered})" for L in work.layers if L.unrendered and L.visible]
        if unr:
            result["warnings"].append("Not drawn: " + ", ".join(unr[:6]))
        adj = [L.name for L in work.layers if L.kind() == "adjustment" and L.visible]
        if adj:
            result["warnings"].append("Adjustment layers ignored: " + ", ".join(adj[:6]))

    bg = None if (s.artboard is None or s.artboard_transparent) else s.artboard.background
    if bg is not None:
        rgb = rgb * alpha[..., None] + np.asarray(bg, np.float32) * (1 - alpha[..., None])
        alpha = np.ones_like(alpha)
        _add_background_layer(work, bg)

    if s.out_psd:
        ops.prepare_for_save(work)
        if work.mode == MODE_RGB:
            merged = np.moveaxis(ops.to_dtype(ops.matte_white(rgb, alpha), work.dtype), -1, 0)
            has_a = bool(alpha.min() < 1.0)
            if has_a:
                merged = np.concatenate([merged, ops.to_dtype(alpha, work.dtype)[None]], 0)
        elif saved and work.merged is not None:
            merged, has_a = work.merged, work.merged_has_alpha
        else:
            merged, has_a = None, False
            result["warnings"].append("PSD kept in its original colour mode: the stored preview "
                                      "composite is blank (Photoshop rebuilds it on open).")
        write_psd(work, s.out_psd, merged, has_a)
        log(f"Wrote {s.out_psd}")
        result["psd"] = s.out_psd

    if s.out_flat:
        depth = s.flat_depth or work.depth
        dtype = np.uint16 if depth == 16 else np.uint8
        px = ops.to_dtype(rgb, dtype)
        if s.flat_alpha:
            px = np.concatenate([px, ops.to_dtype(alpha, dtype)[..., None]], -1)
        # sRGB when converted; an unconverted RGB file keeps its own profile
        icc = (work.icc_profile if (not needs_convert and work.mode == MODE_RGB and work.icc_profile)
               else ops.srgb_profile_bytes())
        ext = os.path.splitext(s.out_flat)[1].lower()
        if ext in (".tif", ".tiff"):
            write_tiff(s.out_flat, px, icc, doc.resolution)
        else:
            write_png(s.out_flat, px, icc, doc.resolution)
        log(f"Wrote {s.out_flat}")
        result["flat"] = s.out_flat

    for w in result["warnings"]:
        log("Note: " + w)
    return result


def _add_background_layer(doc: PSD, rgb: tuple):
    """Artboard background colour as a solid bottom layer (artboards keep it
    outside the layers, so a stand-alone PSD would otherwise lose it)."""
    from .psdfile import Layer, MODE_CMYK, MODE_GRAY
    r, g, b = rgb
    if doc.mode == MODE_RGB:
        vals = [r, g, b]
    elif doc.mode == MODE_CMYK:
        k = 1 - max(r, g, b)
        c, m, y = ((1 - v - k) / (1 - k) if k < 1 else 0 for v in (r, g, b))
        vals = [1 - c, 1 - m, 1 - y, 1 - k]       # PSD stores CMYK inverted
    elif doc.mode == MODE_GRAY:
        vals = [0.2126 * r + 0.7152 * g + 0.0722 * b]
    else:
        return
    ch = {i: np.full((doc.height, doc.width), round(v * doc.max_value), doc.dtype)
          for i, v in enumerate(vals)}
    doc.layers.insert(0, Layer(name="Artboard background", top=0, left=0, bottom=doc.height,
                               right=doc.width, channels=ch))


def export_artboards(doc: PSD, base: Settings, boards: list, stem: str, want_psd: bool,
                     flat_ext: Optional[str], log: Callable = print, progress: Callable = None) -> dict:
    """Export each artboard on its own, numbered in the given order:
    <stem>_01_<name>.psd / .png ..."""
    import copy
    width = max(2, len(str(len(boards))))
    written = []
    for n, board in enumerate(boards, 1):
        s = copy.copy(base)
        s.artboard = board
        s.crop = None
        st = artboard_stem(stem, n, board.name, width)
        s.out_psd = st + ".psd" if want_psd else None
        s.out_flat = st + flat_ext if flat_ext else None
        for p in (s.out_psd, s.out_flat):
            if p and os.path.exists(p):
                raise FileExistsError(f"{p} already exists")
        log(f"[{n}/{len(boards)}] {board.name}")
        process(doc, s, log=log)
        written += [p for p in (s.out_psd, s.out_flat) if p]
        if progress:
            progress(n, len(boards))
    return {"written": written, "warnings": [], "composite_source": "artboards"}

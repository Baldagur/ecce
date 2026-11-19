"""One call that applies all edits and writes the outputs."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from . import ops
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


def make_converter(doc: PSD, s: Settings) -> ops.ColorConverter:
    return ops.ColorConverter(doc.mode, doc.depth, doc.icc_profile, s.cmyk_profile,
                              s.intent, s.bpc, s.use_embedded)


def edited_doc(doc: PSD, s: Settings) -> PSD:
    """Removal, visibility and crop applied (no colour change)."""
    work = ops.clone(doc)
    for i, vis in s.visibility.items():
        if 0 <= i < len(work.layers):
            work.layers[i].visible = vis
    if s.remove:
        ids = {id(work.layers[i]) for i in s.remove if 0 <= i < len(work.layers)}
        ops.remove_layers(work, ids)
    if s.crop:
        ops.crop(work, *s.crop, delete_pixels=s.delete_cropped)
    return work


def use_saved_composite(doc: PSD, s: Settings) -> bool:
    if s.flatten_source == "layers":
        return False
    usable = ops.merged_is_usable(doc)
    if s.flatten_source == "composite":
        return usable
    changed = bool(s.remove) or any(doc.layers[i].visible != v for i, v in s.visibility.items()
                                    if 0 <= i < len(doc.layers))
    return usable and not changed


def process(doc: PSD, s: Settings, log: Callable = print, progress: Callable = None) -> dict:
    result = {"warnings": []}
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
        adj = [L.name for L in work.layers if L.kind() == "adjustment" and L.visible]
        if adj:
            result["warnings"].append("Adjustment layers ignored: " + ", ".join(adj[:6]))

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
        icc = ops.srgb_profile_bytes()
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

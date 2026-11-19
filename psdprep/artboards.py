"""Photoshop artboards ("pages"): find them, order them, and the settings to
export one on its own."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import descriptor, ops
from .psdfile import PSD, Layer

ARTBOARD_KEYS = (b"artb", b"artd", b"abdd")


@dataclass
class Artboard:
    index: int                      # index of the artboard's group layer in doc.layers
    name: str
    rect: tuple                     # (x, y, w, h) in document pixels
    background: Optional[tuple]     # (r, g, b) 0..1, or None for transparent
    layer_order: int = 0            # 0 = top of the layers panel

    @property
    def label(self):
        x, y, w, h = self.rect
        return f"{self.name}  ({w}×{h})"


def _children_bbox(node) -> Optional[tuple]:
    boxes = []
    for L in ops.flatten_tree(node.children):
        if not (L.is_group or L.is_divider) and L.width and L.height:
            boxes.append((L.left, L.top, L.right, L.bottom))
    if not boxes:
        return None
    a = np.array(boxes)
    l, t, r, b = a[:, 0].min(), a[:, 1].min(), a[:, 2].max(), a[:, 3].max()
    return int(l), int(t), int(r - l), int(b - t)


def _clamp(rect, doc: PSD) -> Optional[tuple]:
    x, y, w, h = rect
    l, t = max(0, x), max(0, y)
    r, b = min(doc.width, x + w), min(doc.height, y + h)
    if r - l < 1 or b - t < 1:
        return None
    return int(l), int(t), int(r - l), int(b - t)


def _info(layer: Layer) -> dict:
    for _, key, data in layer.tagged:
        if key in ARTBOARD_KEYS:
            try:
                return descriptor.parse(data)
            except Exception:
                return {}
    return {}


def is_artboard(layer: Layer) -> bool:
    return layer.is_group and bool(layer.keys() & set(ARTBOARD_KEYS))


def find_artboards(doc: PSD) -> list:
    """Artboards in layers-panel order (top first)."""
    index_of = {id(L): i for i, L in enumerate(doc.layers)}
    out = []
    for order, (node, depth) in enumerate(
            (n, d) for n, d in ops.walk(ops.build_tree(doc.layers)) if d == 0):
        L = node.layer
        if not is_artboard(L):
            continue
        info = _info(L)
        rect = None
        r = info.get("artboardRect")
        if isinstance(r, dict):
            try:
                t, l, b, rr = (float(r[k]) for k in ("Top ", "Left", "Btom", "Rght"))
                rect = _clamp((round(l), round(t), round(rr - l), round(b - t)), doc)
            except (KeyError, TypeError, ValueError):
                rect = None
        if rect is None:
            bb = _children_bbox(node)
            rect = _clamp(bb, doc) if bb else None
        if rect is None:
            continue
        bg_type = info.get("artboardBackgroundType", 1)
        if bg_type == 3:
            bg = None
        elif bg_type == 2:
            bg = (0.0, 0.0, 0.0)
        elif bg_type == 4 and isinstance(info.get("Clr "), dict):
            c = info["Clr "]
            try:
                bg = tuple(min(1.0, max(0.0, float(c[k]) / 255.0)) for k in ("Rd  ", "Grn ", "Bl  "))
            except (KeyError, TypeError, ValueError):
                bg = (1.0, 1.0, 1.0)
        else:
            bg = (1.0, 1.0, 1.0)
        out.append(Artboard(index_of[id(L)], L.name, rect, bg, len(out)))
    return out


def sort_by_position(boards: list) -> list:
    """Reading order: rows top to bottom, then left to right."""
    if not boards:
        return []
    tol = min(b.rect[3] for b in boards) / 2
    rows = []
    for b in sorted(boards, key=lambda b: b.rect[1]):
        if rows and abs(b.rect[1] - rows[-1][0].rect[1]) <= tol:
            rows[-1].append(b)
        else:
            rows.append([b])
    return [b for row in rows for b in sorted(row, key=lambda b: b.rect[0])]


def others_to_remove(doc: PSD, board: Artboard) -> set:
    """Indices of every top-level layer/group except this artboard."""
    index_of = {id(L): i for i, L in enumerate(doc.layers)}
    return {index_of[id(n.layer)] for n in ops.build_tree(doc.layers)
            if index_of[id(n.layer)] != board.index}

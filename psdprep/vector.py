"""Illustrator (.ai) and PDF files: rendered to pixels with PDFium (pypdfium2),
then laid out as a normal RGB document, one transparent artboard per page,
so crop, preview and the numbered artboard export work unchanged.

Illustrator files open when they were saved with "Create PDF Compatible
File" (Illustrator's default): such files carry a full PDF copy of the
artwork, which is what gets rendered."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from . import descriptor
from .psdfile import MODE_RGB, PSD, SECTION_DIVIDER, SECTION_OPEN, Layer

VECTOR_EXTS = (".ai", ".pdf")
MAX_PIXELS = 250_000_000          # per page; ~15800 x 15800
MAX_CANVAS = 300_000_000          # all artboards together (~5 GB of float buffers)


class VectorError(Exception):
    pass


def is_vector(path: str) -> bool:
    return str(path).lower().endswith(VECTOR_EXTS)


@dataclass
class PageInfo:
    index: int
    label: str
    width_pt: float
    height_pt: float

    def pixels(self, dpi: Optional[float] = None, long_side: Optional[int] = None) -> tuple:
        s = scale_for(self, dpi, long_side)
        return max(1, round(self.width_pt * s)), max(1, round(self.height_pt * s))

    @property
    def size_text(self) -> str:
        mm = 25.4 / 72
        return f"{self.width_pt * mm:.0f} × {self.height_pt * mm:.0f} mm"


def scale_for(page: PageInfo, dpi: Optional[float], long_side: Optional[int]) -> float:
    """PDF points (1/72 in) -> pixels."""
    if long_side:
        return long_side / max(page.width_pt, page.height_pt, 1e-6)
    return (dpi or 300) / 72.0


def _check_header(path: str):
    with open(path, "rb") as f:
        head = f.read(1024)
    if b"%PDF" in head:
        return
    if head.startswith(b"%!PS"):
        raise VectorError("This Illustrator file was saved without PDF compatibility "
                          "(or is a very old version), so it can't be opened.\n\n"
                          "In Illustrator: File > Save As > tick \"Create PDF Compatible File\".")
    raise VectorError("Not an Illustrator/PDF file this tool can read.")


def _pdfium():
    try:
        import pypdfium2 as pdfium
        return pdfium
    except ImportError as e:
        raise VectorError("Illustrator/PDF support needs pypdfium2 (bundled in the app).") from e


def _page_label(pdf, i: int) -> Optional[str]:
    try:
        import ctypes
        import pypdfium2.raw as raw
        n = raw.FPDF_GetPageLabel(pdf.raw, i, None, 0)
        if n <= 2:
            return None
        buf = ctypes.create_string_buffer(n)
        raw.FPDF_GetPageLabel(pdf.raw, i, buf, n)
        return buf.raw[:n - 2].decode("utf-16-le").strip() or None
    except Exception:
        return None


def page_infos(path: str) -> list:
    _check_header(path)
    pdfium = _pdfium()
    try:
        pdf = pdfium.PdfDocument(path)
    except Exception as e:
        raise VectorError(f"Could not open {os.path.basename(path)}: {e}") from e
    try:
        out = []
        for i in range(len(pdf)):
            w, h = pdf[i].get_size()
            label = _page_label(pdf, i)
            out.append(PageInfo(i, label or f"Artboard {i + 1}", float(w), float(h)))
        return out
    finally:
        pdf.close()


def render_page(path: str, index: int, dpi: Optional[float] = None,
                long_side: Optional[int] = None) -> np.ndarray:
    """-> (h, w, 4) uint8 straight RGBA with transparent background."""
    pdfium = _pdfium()
    pdf = pdfium.PdfDocument(path)
    try:
        page = pdf[index]
        w, h = page.get_size()
        info = PageInfo(index, "", float(w), float(h))
        pw, ph = info.pixels(dpi, long_side)
        if pw * ph > MAX_PIXELS:
            raise VectorError(f"Page {index + 1} would be {pw} × {ph} px — too large. "
                              "Lower the resolution.")
        bmp = page.render(scale=scale_for(info, dpi, long_side), fill_color=(0, 0, 0, 0),
                          rev_byteorder=True, draw_annots=True, may_draw_forms=True)
        arr = bmp.to_numpy()
        if arr.shape[2] == 3:
            arr = np.concatenate([arr, np.full(arr.shape[:2] + (1,), 255, np.uint8)], 2)
        # PDFium's BGRA/RGBA bitmaps are straight (not premultiplied) alpha
        return np.ascontiguousarray(arr, dtype=np.uint8)
    finally:
        pdf.close()


def load_vector(path: str, dpi: Optional[float] = None, long_side: Optional[int] = None,
                progress: Callable = None) -> PSD:
    """Render every page and lay them out in a grid as transparent artboards."""
    pages = page_infos(path)
    if not pages:
        raise VectorError("The file has no pages/artboards.")
    rendered = []
    for n, p in enumerate(pages):
        rendered.append((p, render_page(path, p.index, dpi, long_side)))
        if progress:
            progress(n + 1, len(pages))
    cols = max(1, math.ceil(math.sqrt(len(rendered))))
    gap = max(20, int(0.04 * max(r.shape[1] for _, r in rendered)))
    rows = [rendered[i:i + cols] for i in range(0, len(rendered), cols)]
    widths = [sum(r.shape[1] for _, r in row) + gap * (len(row) - 1) for row in rows]
    heights = [max(r.shape[0] for _, r in row) for row in rows]
    W, H = max(widths), sum(heights) + gap * (len(rows) - 1)
    if W * H > MAX_CANVAS:
        raise VectorError(f"All artboards together would be {W} × {H} px (about "
                          f"{W * H * 16 / 1e9:.0f} GB of working memory). Lower the resolution.")

    doc = PSD(width=W, height=H, depth=8, mode=MODE_RGB, num_channels=3)
    doc.source_kind = "vector"
    doc.vector_pages = pages
    doc.vector_settings = {"dpi": dpi, "long_side": long_side}
    layers = []
    y = 0
    for row, rh in zip(rows, heights):
        x = 0
        for p, img in row:
            h, w = img.shape[:2]
            info = {"artboardRect": {"Top ": float(y), "Left": float(x), "Btom": float(y + h),
                                     "Rght": float(x + w)},
                    "artboardBackgroundType": 3}            # transparent
            content = Layer(name=p.label, top=y, left=x, bottom=y + h, right=x + w,
                            channels={0: img[..., 0].copy(), 1: img[..., 1].copy(),
                                      2: img[..., 2].copy(), -1: img[..., 3].copy()})
            group = Layer(name=p.label, section=SECTION_OPEN, section_blend="pass",
                          tagged=[(b"8BIM", b"artb", descriptor.build(info))])
            layers += [Layer(name="</Layer group>", section=SECTION_DIVIDER), content, group]
            x += w + gap
        y += rh + gap
    # first page at the top of the layers panel
    blocks = [layers[i:i + 3] for i in range(0, len(layers), 3)]
    doc.layers = [L for b in reversed(blocks) for L in b]
    return doc


def make_test_pdf(path: str):
    """Two-page PDF (no dependencies) used by tests and the self-test:
    page 1: 200x100 pt, red box at (20,20)-(120,70) on a transparent page;
    page 2: 100x100 pt, half-transparent blue box filling the page."""
    objs = []

    def add(s):
        objs.append(s)
        return len(objs)

    c1 = b"1 0 0 rg 20 30 100 50 re f"
    c2 = b"/GS1 gs 0 0 1 rg 0 0 100 100 re f"
    s1 = add(b"<< /Length %d >>\nstream\n" % len(c1) + c1 + b"\nendstream")
    s2 = add(b"<< /Length %d >>\nstream\n" % len(c2) + c2 + b"\nendstream")
    gs = add(b"<< /Type /ExtGState /ca 0.5 >>")
    pages_id = len(objs) + 4          # after p1, p2 and the catalog
    p1 = add(b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 200 100] /Contents %d 0 R >>"
             % (pages_id, s1))
    p2 = add(b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 100 100] /Contents %d 0 R "
             b"/Resources << /ExtGState << /GS1 %d 0 R >> >> >>" % (pages_id, s2, gs))
    add(b"<< /Type /Catalog /Pages %d 0 R >>" % pages_id)
    add(b"<< /Type /Pages /Kids [%d 0 R %d 0 R] /Count 2 >>" % (p1, p2))
    out = bytearray(b"%PDF-1.4\n")
    offs = []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for o in offs:
        out += b"%010d 00000 n \n" % o
    out += b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objs) + 1, len(objs) - 1, xref)
    with open(path, "wb") as f:
        f.write(out)

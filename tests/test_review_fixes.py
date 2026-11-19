"""Regression tests for the code-review findings (masks, fill opacity, clipping,
names, crop, gray profiles, overwrite protection, ZIP composites).
Run: python3 tests/test_review_fixes.py"""
import os
import struct
import sys
import tempfile
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from psdprep import descriptor, ops  # noqa: E402
from psdprep.psdfile import (MODE_GRAY, MODE_RGB, PSD, SECTION_DIVIDER, SECTION_OPEN, Layer,  # noqa: E402
                             Mask, combine_masks, read_psd, write_psd)

W, H = 8, 4


def rgb_doc():
    return PSD(width=W, height=H, depth=8, mode=MODE_RGB, num_channels=3)


def solid(name, color=(200, 50, 50), alpha=255, rect=(0, 0, H, W), **kw):
    t, l, b, r = rect
    ch = {i: np.full((b - t, r - l), color[i], np.uint8) for i in range(3)}
    ch[-1] = np.full((b - t, r - l), alpha, np.uint8)
    return Layer(name=name, top=t, left=l, bottom=b, right=r, channels=ch, **kw)


def composite(doc):
    return ops.Compositor(doc, ops.default_get_rgb(doc)).run()


def roundtrip(doc):
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "x.psd")
        write_psd(doc, p)
        return read_psd(p)


def test_hide_all_mask_survives():
    d = rgb_doc()
    L = solid("hidden by mask")
    L.mask = Mask(0, 0, 0, 0, 0, 0, np.zeros((0, 0), np.uint8))      # "Hide All"
    d.layers = [L]
    assert composite(d)[1].max() == 0
    back = roundtrip(d)
    assert back.layers[0].mask is not None
    assert composite(back)[1].max() == 0


def test_mask_cropped_away_keeps_hiding():
    d = rgb_doc()
    L = solid("half")
    L.mask = Mask(0, 0, H, 2, 0, 0, np.full((H, 2), 255, np.uint8))   # shows left 2 px only
    d.layers = [L]
    ops.crop(d, 4, 0, 4, H)                                          # mask fully cropped away
    assert composite(d)[1].max() == 0
    assert composite(roundtrip(d))[1].max() == 0


def test_group_mask_roundtrip():
    d = rgb_doc()
    g = Layer(name="G", section=SECTION_OPEN, section_blend="pass")
    g.mask = Mask(0, 0, H, W, 0, 0, np.tile(np.array([255] * 4 + [0] * 4, np.uint8), (H, 1)))
    d.layers = [Layer(name="</Layer group>", section=SECTION_DIVIDER), solid("in group"), g]
    a0 = composite(d)[1][0]
    back = roundtrip(d)
    assert back.layers[-1].mask is not None and back.layers[-1].mask.data.size
    assert np.allclose(composite(back)[1][0], a0) and a0[0] == 1 and a0[-1] == 0


def test_unicode_names():
    d = rgb_doc()
    name = "Logo 🎨 é ñ 日本"
    d.layers = [solid(name)]
    assert roundtrip(d).layers[0].name == name


def test_fill_opacity():
    d = rgb_doc()
    L = solid("fill 0")
    L.tagged.append((b"8BIM", b"iOpa", bytes([0, 0, 0, 0])))
    d.layers = [L]
    assert composite(d)[1].max() == 0
    back = roundtrip(d)
    assert back.layers[0].fill_opacity == 0 and composite(back)[1].max() == 0


def test_clipped_layer_takes_base_opacity():
    d = rgb_doc()
    base = solid("base", (0, 0, 255), opacity=128)
    clip = solid("clipped", (255, 0, 0), clipping=1)
    d.layers = [base, clip]
    rgb, a = composite(d)
    assert abs(a[0, 0] - 128 / 255) < 0.01
    assert rgb[0, 0, 0] > 0.99                    # clipped colour on top


def test_clipped_with_base_fill_zero_still_clips():
    d = rgb_doc()
    base = solid("base", rect=(0, 0, H, 2))
    base.tagged.append((b"8BIM", b"iOpa", bytes([0, 0, 0, 0])))
    clip = solid("clipped", (0, 255, 0), clipping=1)
    d.layers = [base, clip]
    rgb, a = composite(d)
    assert a[0, 0] == 1 and a[0, 5] == 0 and rgb[0, 0, 1] > 0.99


def test_clip_to_group_is_confined():
    d = rgb_doc()
    g = Layer(name="G", section=SECTION_OPEN, section_blend="pass")
    d.layers = [Layer(name="</Layer group>", section=SECTION_DIVIDER),
                solid("2px", rect=(0, 0, H, 2)), g, solid("clipped", (0, 255, 0), clipping=1)]
    a = composite(d)[1][0]
    assert list(a) == [1, 1, 0, 0, 0, 0, 0, 0], a


def test_hidden_base_hides_clipped():
    d = rgb_doc()
    base = solid("base")
    base.visible = False
    d.layers = [base, solid("clipped", clipping=1)]
    assert composite(d)[1].max() == 0


def test_solid_fill_layer_is_drawn():
    d = rgb_doc()
    info = {"Clr ": {"Rd  ": 10.0, "Grn ": 20.0, "Bl  ": 250.0}}
    L = Layer(name="Color Fill 1", tagged=[(b"8BIM", b"SoCo", descriptor.build(info))])
    # what Photoshop writes: a SoCo tagged block and no pixels
    from psdprep import psdfile
    psdfile._materialize_fill(L, d)
    assert L.width == W and tuple(int(L.channels[i][0, 0]) for i in range(3)) == (10, 20, 250)
    shape = Layer(name="Shape 1", tagged=[(b"8BIM", b"SoCo", descriptor.build(info)),
                                         (b"8BIM", b"vmsk", b"\0" * 8)])
    psdfile._materialize_fill(shape, d)
    assert shape.unrendered and shape.width == 0


def test_combined_vector_and_pixel_mask():
    a = Mask(0, 0, 2, 4, 0, 0, np.full((2, 4), 255, np.uint8))
    b = Mask(0, 2, 2, 6, 0, 0, np.full((2, 4), 255, np.uint8))
    m = combine_masks(a, b, 255.0, np.uint8)
    assert (m.left, m.right) == (0, 6)
    assert list(m.data[0]) == [0, 0, 255, 255, 0, 0]


def test_crop_outside_raises():
    d = rgb_doc()
    d.layers = [solid("x")]
    try:
        ops.crop(d, 100, 0, 20, 20)
        raise AssertionError("no error")
    except ValueError:
        pass
    ops.crop(d, -2, -1, 5, 3)                 # partly outside: intersected
    assert (d.width, d.height) == (3, 2)


def _gray_icc(gamma=1.0) -> bytes:
    def s15(v):
        return struct.pack(">i", round(v * 65536))
    desc_txt = b"Linear Gray\0"
    desc = (b"desc\0\0\0\0" + struct.pack(">I", len(desc_txt)) + desc_txt
            + struct.pack(">II", 0, 0) + b"\0\0\0" + b"\0" * 67)
    wtpt = b"XYZ \0\0\0\0" + s15(0.9642) + s15(1.0) + s15(0.8249)
    ktrc = b"curv\0\0\0\0" + struct.pack(">IH", 1, round(gamma * 256))
    tags = [(b"desc", desc), (b"wtpt", wtpt), (b"kTRC", ktrc)]
    off = 128 + 4 + 12 * len(tags)
    table, body = b"", b""
    for sig, data in tags:
        data += b"\0" * ((-len(data)) % 4)
        table += sig + struct.pack(">II", off + len(body), len(data))
        body += data
    size = off + len(body)
    hdr = (struct.pack(">I", size) + b"lcms" + struct.pack(">I", 0x02100000) + b"mntrGRAYXYZ "
           + b"\0" * 12 + b"acsp" + b"\0" * 24 + struct.pack(">I", 0)
           + s15(0.9642) + s15(1.0) + s15(0.8249) + b"\0" * 48)
    assert len(hdr) == 128
    return hdr + struct.pack(">I", len(tags)) + table + body


def test_gray_profile_is_used():
    icc = _gray_icc(1.0)                      # linear gray: 50% -> ~0.735 in sRGB
    conv = ops.ColorConverter(MODE_GRAY, 8, icc)
    v = conv.to_rgb_float([np.full((1, 1), 128, np.uint8)])[0, 0, 0]
    assert 0.70 < v < 0.77, v
    plain = ops.ColorConverter(MODE_GRAY, 8, None)
    assert abs(plain.to_rgb_float([np.full((1, 1), 128, np.uint8)])[0, 0, 0] - 128 / 255) < 1e-6


def test_zip_composite_all_channels():
    # minimal flat RGB PSD whose merged image uses ZIP-with-prediction (comp 3)
    img = np.random.default_rng(3).integers(0, 255, (3, H, W)).astype(np.uint8)
    pred = img.reshape(3 * H, W).astype(np.int16)
    pred[:, 1:] = (pred[:, 1:] - pred[:, :-1]) % 256
    data = (b"8BPS" + struct.pack(">H", 1) + b"\0" * 6 + struct.pack(">HIIHH", 3, H, W, 8, 3)
            + struct.pack(">I", 0) + struct.pack(">I", 0) + struct.pack(">I", 0)
            + struct.pack(">H", 3) + zlib.compress(pred.astype(np.uint8).tobytes()))
    d = read_psd(data)
    assert d.merged is not None and np.array_equal(d.merged, img)


def test_original_and_existing_files_protected():
    from psdprep import naming
    from psdprep.pipeline import Settings, process
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "a.psd")
        d = rgb_doc()
        d.layers = [solid("x")]
        write_psd(d, src)
        other = os.path.join(tmp, "existing.png")
        open(other, "wb").write(b"keep me")
        doc = read_psd(src)
        try:
            process(doc, Settings(source_path=src, out_flat=other, overwrite=False), log=lambda *a: None)
            raise AssertionError("existing file overwritten")
        except FileExistsError:
            pass
        assert open(other, "rb").read() == b"keep me"
        process(doc, Settings(source_path=src, out_flat=other, overwrite=True), log=lambda *a: None)
        try:
            process(doc, Settings(source_path=src, out_psd=src, overwrite=True), log=lambda *a: None)
            raise AssertionError("original overwritten")
        except naming.ProtectedFileError:
            pass


if __name__ == "__main__":
    n = 0
    for k, f in list(globals().items()):
        if k.startswith("test_") and callable(f):
            f()
            n += 1
    print(f"{n} review-fix tests passed")

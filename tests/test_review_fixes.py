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


def test_illustrator_pdf():
    try:
        import pypdfium2  # noqa: F401
    except ImportError:
        print("  (skipped Illustrator test: pypdfium2 not installed)")
        return
    from psdprep import vector
    with tempfile.TemporaryDirectory() as tmp:
        ai = os.path.join(tmp, "a.ai")
        vector.make_test_pdf(ai)
        img = vector.render_page(ai, 1, dpi=36)              # 50% blue
        assert tuple(img[10, 10][:3]) == (0, 0, 255) and 120 < img[10, 10, 3] < 135
        ps = os.path.join(tmp, "old.ai")
        open(ps, "wb").write(b"%!PS-Adobe-3.0\n%%Creator: Adobe Illustrator 8\n")
        try:
            vector.page_infos(ps)
            raise AssertionError("PostScript .ai accepted")
        except vector.VectorError as e:
            assert "PDF compatib" in str(e)


def test_delta_e_2000_reference():
    from psdprep.colorspaces import delta_e_2000
    # Sharma, Wu & Dalal (2005) test data
    for a, b, ref in (((50, 2.6772, -79.7751), (50, 0, -82.7485), 2.0425),
                      ((50, 2.5, 0), (73, 25, -18), 27.1492),
                      ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644)):
        assert abs(float(delta_e_2000(np.array(a, float), np.array(b, float))) - ref) < 1e-3


def test_rec709_profiles_match_maths():
    from PIL import Image, ImageCms
    from psdprep.colorspaces import pil_profile, reencode_from_srgb
    g = np.arange(256, dtype=np.uint8)
    ramp = np.stack([g, g, g], -1)[None]
    for t in ("rec709", "rec709_oetf"):
        tr = ImageCms.buildTransform(ImageCms.createProfile("sRGB"), pil_profile(t), "RGB", "RGB", 1)
        out = np.asarray(ImageCms.applyTransform(Image.fromarray(ramp), tr))[0, :, 0].astype(float)
        ref = np.rint(reencode_from_srgb(g / 255.0, t) * 255)
        assert np.abs(out - ref).max() <= 1, (t, np.abs(out - ref).max())
    # untagged RGB -> Rec.709 is exact maths: mid grey 128 -> gamma 2.4 code
    conv = ops.ColorConverter(MODE_RGB, 8, None, target="rec709")
    v = conv.to_rgb_float([np.full((1, 1), 128, np.uint8)] * 3)[0, 0, 0] * 255
    assert abs(v - 255 * (((128 / 255 + 0.055) / 1.055) ** 2.4) ** (1 / 2.4)) < 0.01


def test_check_catches_a_wrong_conversion():
    from psdprep import check, fixtures
    from psdprep.colorspaces import build_icc
    # a CMYK profile isn't available everywhere: measure an RGB file whose
    # embedded profile is our own Rec.709 one (RGB->sRGB through lcms)
    d = rgb_doc()
    d.set_resource(1039, build_icc("rec709"))
    d.layers = [solid("colours", (200, 120, 30))]
    conv = ops.ColorConverter(MODE_RGB, 8, d.icc_profile, target="srgb")
    good = check.conversion_check(d, conv, 1)
    assert good.ok and good.stats["p95"] < 1.0, good.message
    real = ops.ColorConverter.to_rgb_float
    conv.to_rgb_float = lambda chans: np.stack([1 - c.astype(np.float32) / 255 for c in chans[:3]], -1)
    bad = check.conversion_check(d, conv, 1)
    assert not bad.ok and bad.stats["p95"] > 10, bad.message
    # subtler: too bright by 1.5x and clipped (values pinned at 255 must not hide it)
    conv.to_rgb_float = lambda chans: np.clip(real(conv, chans) * 1.5, 0, 1)
    d.layers = [solid("colours", (200, 120, 30)), solid("white", (255, 255, 255), rect=(0, 0, 2, 2)),
                solid("pure", (230, 20, 20), rect=(2, 0, 4, 2))]
    bad2 = check.conversion_check(d, conv, 1)
    assert not bad2.ok, bad2.message


def test_colour_tags_and_custom_names():
    from psdprep import naming
    from psdprep.pipeline import Settings, colour_tag
    d = rgb_doc()
    assert colour_tag(d, Settings(convert=True, target="rec709")) == "Rec709"
    assert colour_tag(d, Settings(convert=True, target="rec709_oetf")) == "Rec709-OETF"
    assert colour_tag(d, Settings(convert=False)) == "RGB"
    from psdprep.fixtures import make_doc
    assert colour_tag(make_doc(8), Settings(convert=False)) == "CMYK"
    with tempfile.TemporaryDirectory() as tmp:
        open(os.path.join(tmp, "client_logo_v3_sRGB.png"), "w").close()
        stem = naming.custom_stem(tmp, "client logo")
        assert os.path.basename(stem) == "client_logo_v4"
        assert os.path.basename(naming.custom_stem(tmp, "client_logo_v1")) == "client_logo_v4"
        assert naming.with_tag(stem, "Rec709").endswith("client_logo_v4_Rec709")


def nuke_breakout(doc):
    """What Nuke's Read > Breakout Layers does (nukescripts/psd.py): every
    non-group layer in file order, merged with its own blend mode and opacity,
    masked by its mask channel; visibility, groups, clipping and fill ignored."""
    rgb = np.zeros((doc.height, doc.width, 3), np.float32)
    a = np.zeros((doc.height, doc.width), np.float32)
    for L in doc.layers:
        if L.is_group or L.is_divider or not L.width:
            continue
        t, l = max(0, L.top), max(0, L.left)
        b, r = min(doc.height, L.bottom), min(doc.width, L.right)
        if b <= t or r <= l:
            continue
        sl = (slice(t - L.top, b - L.top), slice(l - L.left, r - L.left))
        cs = np.stack([L.channels[c][sl] for c in range(3)], -1).astype(np.float32) / 255
        al = L.channels[-1][sl].astype(np.float32) / 255 if -1 in L.channels else np.ones(cs.shape[:2])
        if L.mask is not None and L.mask.data is not None and not L.mask.disabled:
            mk = np.zeros((doc.height, doc.width), np.float32)        # mask channel: 0 outside its box
            m = L.mask
            if m.width and m.height:
                mk[max(0, m.top):m.bottom, max(0, m.left):m.right] = \
                    m.data[max(0, -m.top):, max(0, -m.left):][:m.bottom - max(0, m.top), :m.right - max(0, m.left)] / 255
            al = al * mk[t:b, l:r]
        al = al * L.opacity / 255
        rgb[t:b, l:r], a[t:b, l:r] = ops.Compositor._blend_into(rgb[t:b, l:r], a[t:b, l:r], cs, al, L.blend_mode)
    return rgb, a


def tricky_doc():
    """Hidden layer, hidden group, 50% group, group mask, clipping, fill 0, mask."""
    d = PSD(width=16, height=8, depth=8, mode=MODE_RGB, num_channels=3)
    div = lambda: Layer(name="</Layer group>", section=SECTION_DIVIDER)  # noqa: E731
    bg = solid("bg", (40, 40, 40), rect=(0, 0, 8, 16))
    hid = solid("hidden", (255, 255, 0), rect=(0, 0, 8, 4))
    hid.visible = False
    hgrp = Layer(name="hidden group", section=SECTION_OPEN, section_blend="pass")
    hgrp.visible = False
    half = Layer(name="half group", section=SECTION_OPEN, section_blend="pass", opacity=128)
    half.mask = Mask(0, 0, 8, 12, 0, 0, np.full((8, 12), 255, np.uint8))     # hides x >= 12
    base = solid("clip base", (0, 0, 255), rect=(0, 4, 8, 10))
    clip = solid("clipped", (255, 0, 0), rect=(0, 0, 4, 16), clipping=1)
    fill0 = solid("fill 0", (0, 255, 0), rect=(4, 0, 8, 16))
    fill0.tagged.append((b"8BIM", b"iOpa", bytes([0, 0, 0, 0])))
    masked = solid("masked", (255, 255, 255), rect=(6, 0, 8, 16))
    masked.mask = Mask(0, 0, 0, 0, 255, 0, np.zeros((0, 0), np.uint8))          # reveal all (default 255)
    d.layers = [bg, div(), solid("in hidden group", (255, 0, 255)), hgrp, hid,
                div(), base, clip, fill0, half, masked]
    return d


def test_nuke_breakout_matches_after_baking():
    d = tricky_doc()
    want_rgb, want_a = composite(d)
    raw_rgb, raw_a = nuke_breakout(d)
    assert np.abs(raw_rgb * raw_a[..., None] - want_rgb * want_a[..., None]).max() > 0.3  # the bug
    baked = roundtrip(ops.nuke_friendly(d))
    names = [L.name for L in baked.layers]
    assert "hidden" not in names and "in hidden group" not in names and "fill 0" not in names
    assert not any(L.is_group or L.is_divider or L.clipping or L.mask for L in baked.layers)
    got_rgb, got_a = nuke_breakout(baked)
    err = np.abs(got_rgb * got_a[..., None] - want_rgb * want_a[..., None]).max()
    assert err < 0.02, err
    assert np.abs(got_a - want_a).max() < 0.02


def test_nuke_psd_through_pipeline():
    from psdprep.fixtures import make_doc
    from psdprep.pipeline import Settings, process
    with tempfile.TemporaryDirectory() as tmp:
        a, b = os.path.join(tmp, "plain.psd"), os.path.join(tmp, "nuke.psd")
        process(make_doc(8), Settings(out_psd=a), log=lambda *x: None)
        process(make_doc(8), Settings(out_psd=b, nuke_psd=True), log=lambda *x: None)
        want_rgb, want_a = composite(read_psd(a))
        got_rgb, got_a = nuke_breakout(read_psd(b))
        assert np.abs(got_a - want_a).max() < 0.02
        assert np.abs(got_rgb * got_a[..., None] - want_rgb * want_a[..., None]).max() < 0.02
        names = [L.name for L in read_psd(b).layers]
        assert all(n.isascii() for n in names) and "Cyan circle e" in names, names
        assert all(L.left >= 0 and L.right <= 320 for L in read_psd(b).layers)


def _nuke_err(d):
    want_rgb, want_a = composite(d)
    got_rgb, got_a = nuke_breakout(roundtrip(ops.nuke_friendly(d)))
    return max(np.abs(got_a - want_a).max(),
               np.abs(got_rgb * got_a[..., None] - want_rgb * want_a[..., None]).max())


def test_nuke_clipping_unit_with_base_opacity_and_blend():
    for kw in ({"opacity": 128}, {"blend_mode": "mul "}):
        d = rgb_doc()
        d.layers = [solid("bg", (200, 200, 200)), solid("base", (0, 0, 255), rect=(0, 2, 4, 6), **kw),
                    solid("clipped", (255, 0, 0), clipping=1)]
        assert _nuke_err(d) < 0.02, kw


def test_nuke_clip_to_group():
    d = rgb_doc()
    g = Layer(name="G", section=SECTION_OPEN, section_blend="pass")
    d.layers = [Layer(name="</Layer group>", section=SECTION_DIVIDER), solid("2px", rect=(0, 0, H, 2)), g,
                solid("texture", (0, 255, 0), clipping=1)]
    assert _nuke_err(d) < 0.02


def test_nuke_feathered_group_mask():
    d = rgb_doc()
    g = Layer(name="G", section=SECTION_OPEN, section_blend="pass")
    g.mask = Mask(0, 0, H, W, 0, 0, np.full((H, W), 128, np.uint8))
    d.layers = [Layer(name="</Layer group>", section=SECTION_DIVIDER), solid("a", (255, 0, 0)),
                solid("b", (0, 0, 255)), g]
    assert _nuke_err(d) < 0.02


def test_nuke_unique_names_and_off_canvas():
    d = rgb_doc()
    d.layers = [solid("foo"), solid("foo", (1, 2, 3)), solid("foo 2", (4, 5, 6)),
                solid("gone", rect=(0, 20, 4, 24))]
    names = [L.name for L in ops.nuke_friendly(d).layers]
    assert len(set(names)) == len(names) == 3, names


def test_rec709_icc_keeps_primaries():
    from PIL import Image, ImageCms
    from psdprep.colorspaces import pil_profile
    px = np.array([[[0, 255, 0], [0, 0, 255], [255, 0, 0]]], np.uint8)
    for t in ("rec709", "rec709_oetf"):
        tr = ImageCms.buildTransform(ImageCms.createProfile("sRGB"), pil_profile(t), "RGB", "RGB", 1)
        assert np.asarray(ImageCms.applyTransform(Image.fromarray(px), tr)).tolist() == px.tolist(), t


def test_flat_tag_and_oetf_tag():
    from psdprep.colorspaces import build_icc
    from psdprep.fixtures import make_doc
    from psdprep.pipeline import Settings, colour_tag
    cmyk = make_doc(8)
    assert colour_tag(cmyk, Settings(convert=False)) == "CMYK"
    assert colour_tag(cmyk, Settings(convert=False), flat=True) == "sRGB"
    d = rgb_doc()
    d.set_resource(1039, build_icc("rec709_oetf"))
    assert colour_tag(d, Settings(convert=False)) == "Rec709-OETF"
    d.set_resource(1039, build_icc("rec709"))
    assert colour_tag(d, Settings(convert=False), flat=True) == "Rec709"


def test_effective_hidden_follows_ui():
    d = rgb_doc()
    g = Layer(name="G", section=SECTION_OPEN, section_blend="pass")
    child = solid("child")
    hid = solid("hidden in file")
    hid.visible = False
    clip = solid("clipped", clipping=1)
    d.layers = [Layer(name="</Layer group>", section=SECTION_DIVIDER), child, g, hid, clip]
    gi = d.layers.index(g)
    h = ops.effective_hidden(d, remove={gi}, visibility={d.layers.index(hid): True})
    assert id(child) in h and id(g) in h          # removed group hides its child
    assert id(hid) not in h and id(clip) not in h  # eye turned on in the UI
    h2 = ops.effective_hidden(d)
    assert id(hid) in h2 and id(clip) in h2       # as saved: hidden base hides its clipped layer


def test_vector_canvas_cap():
    try:
        import pypdfium2  # noqa: F401
    except ImportError:
        return
    from psdprep import vector
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "big.pdf")
        vector.make_test_pdf(p)
        try:
            vector.load_vector(p, dpi=9000)
            raise AssertionError("huge render allowed")
        except vector.VectorError as e:
            assert "too large" in str(e) or "Lower the resolution" in str(e)


if __name__ == "__main__":
    n = 0
    for k, f in list(globals().items()):
        if k.startswith("test_") and callable(f):
            f()
            n += 1
    print(f"{n} review-fix tests passed")

"""Run: python3 -m pytest tests  (or python3 tests/test_roundtrip.py)"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from psdprep import ops  # noqa: E402
from psdprep.pipeline import Settings, process  # noqa: E402
from psdprep.fixtures import make_doc  # noqa: E402
from psdprep.psdfile import read_psd, write_psd  # noqa: E402


def test_roundtrip():
    for depth in (8, 16):
        a = make_doc(depth)
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "a.psd")
            write_psd(a, p)
            b = read_psd(p)
        assert [L.name for L in a.layers] == [L.name for L in b.layers]
        for la, lb in zip(a.layers, b.layers):
            for k, v in la.channels.items():
                assert np.array_equal(v, lb.channels[k])
            assert la.visible == lb.visible and la.section == lb.section


def test_process_outputs():
    d = make_doc(8)
    with tempfile.TemporaryDirectory() as tmp:
        s = Settings(remove={0}, crop=(10, 10, 300, 180),
                     out_psd=os.path.join(tmp, "o.psd"), out_flat=os.path.join(tmp, "o.png"))
        process(d, s, log=lambda *a: None)
        o = read_psd(s.out_psd)
        assert (o.width, o.height, o.mode) == (300, 180, 3)
        assert "Background" not in [L.name for L in o.layers]
        assert o.merged_has_alpha
        assert os.path.getsize(s.out_flat) > 0


def test_tree_remove_group():
    d = make_doc(8)
    grp = [L for L in d.layers if L.is_group][0]
    ops.remove_layers(d, {id(grp)})
    assert [L.name for L in d.layers] == ["Background", "Hidden", "Off canvas"]


def test_naming_and_protection():
    from psdprep import naming
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "logo.psd")
        write_psd(make_doc(8), src)
        before = open(src, "rb").read()
        stem = naming.auto_stem(src, "cleanup")
        assert os.path.basename(stem) == "logo_cleanup_v1"
        open(stem + ".png", "wb").close()
        assert os.path.basename(naming.auto_stem(src, "cleanup")) == "logo_cleanup_v2"
        open(os.path.join(tmp, "logo_cleanup_v7_01_Banner.psd"), "wb").close()
        assert os.path.basename(naming.auto_stem(src, "cleanup")) == "logo_cleanup_v8"
        doc = read_psd(src)
        for bad in (src, os.path.join(tmp, ".", "logo.psd")):
            try:
                process(doc, Settings(source_path=src, out_psd=bad), log=lambda *a: None)
                raise AssertionError("original was overwritten")
            except naming.ProtectedFileError:
                pass
        os.link(src, os.path.join(tmp, "hard.psd"))
        try:
            process(doc, Settings(source_path=src, out_flat=os.path.join(tmp, "hard.psd")),
                    log=lambda *a: None)
            raise AssertionError("hard link to original was overwritten")
        except naming.ProtectedFileError:
            pass
        assert open(src, "rb").read() == before


def test_artboards():
    from psdprep import artboards
    from psdprep.fixtures import make_artboard_doc
    from psdprep.pipeline import export_artboards
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "boards.psd")
        write_psd(make_artboard_doc(), src)
        doc = read_psd(src)
        boards = artboards.find_artboards(doc)
        assert [b.name for b in boards] == ["Banner", "Small", "Square"]
        assert [b.name for b in artboards.sort_by_position(boards)] == ["Square", "Banner", "Small"]
        sq = [b for b in boards if b.name == "Square"][0]
        assert sq.rect == (0, 0, 300, 300) and sq.background == (1.0, 1.0, 1.0)
        assert [b for b in boards if b.name == "Banner"][0].background is None
        assert [b for b in boards if b.name == "Small"][0].background == (1.0, 0.0, 0.0)
        ordered = artboards.sort_by_position(boards)
        res = export_artboards(doc, Settings(source_path=src), ordered,
                               os.path.join(tmp, "boards_cleanup_v1"), True, ".png",
                               log=lambda *a: None)
        names = sorted(os.path.basename(p) for p in res["written"])
        assert names == ["boards_cleanup_v1_01_Square.png", "boards_cleanup_v1_01_Square.psd",
                         "boards_cleanup_v1_02_Banner.png", "boards_cleanup_v1_02_Banner.psd",
                         "boards_cleanup_v1_03_Small.png", "boards_cleanup_v1_03_Small.psd"], names
        from PIL import Image
        import numpy as np
        sqp = np.asarray(Image.open(os.path.join(tmp, "boards_cleanup_v1_01_Square.png")))
        assert sqp.shape == (300, 300, 4)
        assert tuple(sqp[5, 5]) == (255, 255, 255, 255)        # white artboard background
        assert tuple(sqp[150, 150][:3]) == (200, 30, 30)       # content
        ban = np.asarray(Image.open(os.path.join(tmp, "boards_cleanup_v1_02_Banner.png")))
        assert ban.shape == (100, 350, 4) and ban[2, 2, 3] == 0  # transparent background
        sm = np.asarray(Image.open(os.path.join(tmp, "boards_cleanup_v1_03_Small.png")))
        assert tuple(sm[3, 3]) == (255, 0, 0, 255)             # custom colour background
        p = read_psd(os.path.join(tmp, "boards_cleanup_v1_03_Small.psd"))
        assert (p.width, p.height) == (200, 150)
        assert [L.name for L in p.layers if not L.is_divider] == ["Artboard background", "Small box", "Small"]


if __name__ == "__main__":
    test_roundtrip(); test_process_outputs(); test_tree_remove_group()
    test_naming_and_protection(); test_artboards()
    print("all tests passed")

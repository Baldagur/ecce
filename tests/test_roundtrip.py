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


if __name__ == "__main__":
    test_roundtrip(); test_process_outputs(); test_tree_remove_group()
    print("all tests passed")

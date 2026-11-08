import numpy as np
import pytest

pytest.importorskip("cv2")
from splat2sim.config import MaskCfg
from splat2sim.mask import postprocess as pp


def test_mask_cfg_default_prompts():
    assert MaskCfg().prompts == ["robot arm", "robotic gripper"] and MaskCfg().dilate_px == 15


def test_fill_holes_only_fills_enclosed():
    m = np.zeros((40, 40), bool)
    m[5:30, 5:30] = True
    m[15:20, 15:20] = False  # enclosed hole
    m[5:30, 0:5] = False
    f = pp.fill_holes(m)
    assert f[17, 17] and not f[2, 2]
    m2 = m.copy(); m2[15:20, 5:20] = False  # hole open to the left -> not filled
    assert not pp.fill_holes(m2)[17, 12]


def test_finalize_dilates_and_convention():
    m = np.zeros((60, 60), bool); m[30, 30] = True
    f = pp.finalize(m, 5, True)
    assert f[30, 35] and not f[30, 37]
    png = pp.to_png_mask(f)
    assert png.dtype == np.uint8 and png[30, 30] == 0 and png[0, 0] == 255
    assert not pp.finalize(np.zeros((9, 9), bool), 5).any()


def test_mask_stage_wiring(tmp_path, monkeypatch):
    from splat2sim import stages as st
    from splat2sim.runfolder import RunFolder

    monkeypatch.setenv("R2S_DATA", str(tmp_path))
    s = st.get("mask")
    cmd = st.compose_cmd(s, RunFolder("s"))
    assert s.implemented and "sam3" in cmd and cmd[-4:] == ["python", "-m", "splat2sim.mask", "mask"]
    assert "--user" in cmd and "R2S_SCENE=s" in cmd and s.config_sections == ("mask",)

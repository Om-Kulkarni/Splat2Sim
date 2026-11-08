from pathlib import Path

import pytest

from splat2sim import cli, stages as st
from splat2sim.config import load_lab
from splat2sim.runfolder import STAGE_DIRS, RunFolder

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("R2S_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("R2S_REPO", str(REPO))
    monkeypatch.setenv("R2S_GIT_REV", "testrev")
    cap = tmp_path / "cap.mcap"
    cap.write_bytes(b"x" * 10)
    return cap


def test_layout_and_marker(tmp_path):
    run = RunFolder("s", root=tmp_path)
    run.ensure()
    assert all((tmp_path / "s" / d).is_dir() for d in STAGE_DIRS)
    assert run.read_marker("ingest", "frames") is None
    m = run.write_marker("ingest", "frames", "abc", rev="r1")
    got = run.read_marker("ingest", "frames")
    assert got == m and got.inputs_hash == "abc" and got.git_rev == "r1" and got.timestamp
    run.clear_marker("ingest", "frames")
    assert run.read_marker("ingest", "frames") is None


def test_shared_dir_markers_do_not_collide(tmp_path):
    run = RunFolder("s", root=tmp_path)
    run.write_marker("poses", "poses", "a")
    assert run.read_marker("anchor", "poses") is None


def test_state_idempotency_and_staleness(tmp_path, env):
    lab = load_lab("robolab", labs_dir=REPO / "labs")
    run = RunFolder("s")
    ingest, mask = st.get("ingest"), st.get("mask")
    h = st.inputs_hash(ingest, run, lab, env)
    assert st.state(ingest, run, h) == "pending"
    run.write_marker("ingest", "frames", h)
    assert st.state(ingest, run, h) == "done"
    # same inputs -> same hash (idempotent)
    assert st.inputs_hash(ingest, run, lab, env) == h
    # capture changes -> stale
    env.write_bytes(b"y" * 11)
    assert st.state(ingest, run, st.inputs_hash(ingest, run, lab, env)) == "stale"
    # re-running upstream (new run_id) invalidates downstream
    env.write_bytes(b"x" * 10)
    hm = st.inputs_hash(mask, run, lab, env)
    run.write_marker("mask", "masks", hm)
    assert st.state(mask, run, st.inputs_hash(mask, run, lab, env)) == "done"
    run.write_marker("ingest", "frames", h)
    assert st.state(mask, run, st.inputs_hash(mask, run, lab, env)) == "stale"


def test_config_slice_invalidates_only_dependents(tmp_path, env):
    lab = load_lab("robolab", labs_dir=REPO / "labs")
    lab2 = lab.model_copy(update={"cubes": lab.cubes.model_copy(update={"count": 7})})
    run = RunFolder("s")
    for name in ("ingest", "anchor", "env"):
        s = st.get(name)
        same = st.inputs_hash(s, run, lab, env) == st.inputs_hash(s, run, lab2, env)
        assert same == (name != "env")


@pytest.fixture
def docker_calls(monkeypatch):
    """Replace the docker invocation so implemented stages 'succeed' without a daemon."""
    calls = []
    monkeypatch.setattr(st.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    return calls


def test_cli_status_empty_and_stub_failure(env, capsys, docker_calls):
    assert cli.main(["status", "empty"]) == 0
    out = capsys.readouterr().out
    assert out.count("pending") == len(st.STAGES)
    # every stage is implemented now (docker call patched): all run and get a marker
    rc = cli.main(["build", str(env), "--lab", "robolab", "--scene", "s1", "--to", "env"])
    assert rc == 0
    assert "not implemented yet" not in capsys.readouterr().out
    for stage in st.STAGES:
        assert RunFolder("s1").read_marker(stage.name, stage.out_dir) is not None, stage.name


def test_ingest_compose_command(env, docker_calls):
    assert cli.main(["build", str(env), "--lab", "robolab", "--scene", "s3", "--to", "ingest"]) == 0
    (cmd,) = docker_calls
    assert f"{env.resolve()}:/capture:ro" in cmd and "R2S_CAPTURE=/capture" in cmd and f"R2S_CAPTURE_SRC={env.resolve()}" in cmd and "R2S_SCENE=s3" in cmd
    assert "--user" in cmd and cmd[-4:] == ["python", "-m", "splat2sim.ingest", "ingest"]


def test_cli_skips_done_stage_and_force(env, capsys, docker_calls):
    args = ["build", str(env), "--lab", "robolab", "--scene", "s2", "--to", "ingest"]
    assert cli.main(args) == 0 and len(docker_calls) == 1
    capsys.readouterr()
    assert cli.main(args) == 0
    assert "[skip] ingest" in capsys.readouterr().out and len(docker_calls) == 1
    assert cli.main(args + ["--force", "ingest"]) == 0 and len(docker_calls) == 2


def test_stage_range_validation():
    assert [s.name for s in st.select("poses", "splat")] == ["poses", "anchor", "splat"]
    with pytest.raises(ValueError):
        st.select("splat", "mask")

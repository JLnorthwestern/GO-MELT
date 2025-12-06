import pytest
from pathlib import Path
from unittest.mock import MagicMock
import go_melt.core.go_melt as go_melt


# --- Helpers ---
def fake_state():
    s = MagicMock()
    s.ongoing_simulation = True
    s.t_add = 1
    s.Nonmesh = {"layer_num": 0}
    s.layer_check = 1
    return s


def fake_pre_execution(s, single=True):
    s.ongoing_simulation = False  # force termination
    return s, "laser", single


# --- Tests ---


def test_initialization(monkeypatch):
    state = fake_state()
    monkeypatch.setattr(go_melt, "pre_time_loop_initialization", lambda f: state)
    monkeypatch.setattr(go_melt, "time_loop_pre_execution", fake_pre_execution)
    monkeypatch.setattr(go_melt, "single_step_execution", lambda l, s: s)
    monkeypatch.setattr(go_melt, "time_loop_post_execution", lambda s, l, t: s)
    monkeypatch.setattr(go_melt, "post_time_loop_finalization", lambda s: None)

    go_melt.go_melt(Path("examples/example_unit.json"))


def test_immediate_termination(monkeypatch):
    state = fake_state()
    state.t_add = 0
    monkeypatch.setattr(go_melt, "pre_time_loop_initialization", lambda f: state)
    monkeypatch.setattr(go_melt, "time_loop_pre_execution", fake_pre_execution)
    monkeypatch.setattr(go_melt, "post_time_loop_finalization", lambda s: None)

    go_melt.go_melt(Path("examples/example_unit.json"))


def test_single_step_execution(monkeypatch):
    state = fake_state()
    state.Nonmesh["layer_num"] = 1
    state.layer_check = 1
    monkeypatch.setattr(go_melt, "pre_time_loop_initialization", lambda f: state)
    monkeypatch.setattr(go_melt, "time_loop_pre_execution", fake_pre_execution)
    monkeypatch.setattr(go_melt, "single_step_execution", lambda l, s: s)
    monkeypatch.setattr(go_melt, "time_loop_post_execution", lambda s, l, t: s)
    monkeypatch.setattr(go_melt, "post_time_loop_finalization", lambda s: None)

    go_melt.go_melt(Path("examples/example_unit.json"))


def test_multi_step_execution(monkeypatch):
    state = fake_state()
    monkeypatch.setattr(go_melt, "pre_time_loop_initialization", lambda f: state)

    # Wrap fake_pre_execution to force single=False
    def fake_multi(s):
        return fake_pre_execution(s, single=False)

    monkeypatch.setattr(go_melt, "time_loop_pre_execution", fake_multi)
    monkeypatch.setattr(go_melt, "multi_step_execution", lambda l, s: s)
    monkeypatch.setattr(go_melt, "time_loop_post_execution", lambda s, l, t: s)
    monkeypatch.setattr(go_melt, "post_time_loop_finalization", lambda s: None)

    go_melt.go_melt(Path("examples/example_unit.json"))


def test_finalization_called(monkeypatch):
    state = fake_state()
    monkeypatch.setattr(go_melt, "pre_time_loop_initialization", lambda f: state)
    monkeypatch.setattr(go_melt, "time_loop_pre_execution", fake_pre_execution)
    monkeypatch.setattr(go_melt, "single_step_execution", lambda l, s: s)
    monkeypatch.setattr(go_melt, "time_loop_post_execution", lambda s, l, t: s)

    called = []
    monkeypatch.setattr(
        go_melt, "post_time_loop_finalization", lambda s: called.append(True)
    )

    go_melt.go_melt(Path("examples/example_unit.json"))
    assert called, "Finalization should always be called"


if __name__ == "__main__":
    pytest.main([__file__])

import io
import numpy as np
import jax.numpy as jnp
import pytest
from pathlib import Path
import dill
import sys
import json

from go_melt.core.data_structures import SimulationState
from go_melt.core.go_melt_helper_functions import (
    pre_time_loop_initialization,
    time_loop_pre_execution,
    single_step_execution,
    multi_step_execution,
    time_loop_post_execution,
    post_time_loop_finalization,
    clear_jax_function_caches,
)
from go_melt.utils.testing_helper_functions import compare_values
import go_melt.core.go_melt_helper_functions as helper_functions


def test_single_step_execution_dwell():
    # Arrange: load input state
    with open(Path("tests/core/inputs/inputs_single_step_execution.pkl"), "rb") as f:
        laser_all, simulation_state = dill.load(f)

    # Act: run the function under test
    simulation_state.wait_inc = 201
    simulation_state.Levels[3]["Tprime0"] = jnp.ones_like(
        simulation_state.Levels[3]["Tprime0"]
    )
    simulation_state.laser_prev_z = 0.5
    laser_all = laser_all.at[:, 4].set(0)

    simulation_state_result = single_step_execution(laser_all, simulation_state)


def test_single_step_execution_edge_cases(monkeypatch):
    # Arrange: load input state
    with open(Path("tests/core/inputs/inputs_single_step_execution.pkl"), "rb") as f:
        laser_all, simulation_state = dill.load(f)

    # Act: run the function under test
    simulation_state.time_inc = 1
    simulation_state.Nonmesh["record_TAM"] = 1

    # Patch jnp.savez so it doesn't write files
    called = {}

    def fake_savez(filename, **kwargs):
        called["filename"] = filename
        called["kwargs"] = kwargs

    monkeypatch.setattr(jnp, "savez", fake_savez)

    called2 = {}
    monkeypatch.setattr(
        helper_functions,
        "save_object",
        lambda obj, filename: called2.update({"obj": obj, "filename": filename}),
    )
    called3 = {}
    monkeypatch.setattr("os.path.exists", lambda path: False)
    monkeypatch.setattr("os.makedirs", lambda path: called3.setdefault("path", path))

    simulation_state_result = single_step_execution(laser_all, simulation_state)

    # Assert: ensure fake_savez was called with expected args
    assert "filename" in called
    assert "accum_time" in called["kwargs"]
    # filename should include layer_num suffix
    assert (
        str(simulation_state_result.Nonmesh["layer_num"] - 1).zfill(4)
        in called["filename"]
    )
    assert called2["obj"][0] == simulation_state_result.Levels
    assert called3["path"] == simulation_state.checkpoint_path


def test_pre_time_loop_initialization_types_and_values():
    # Arrange
    input_file = Path("examples/example_unit.json")

    # Act
    state = pre_time_loop_initialization(input_file)

    # Assert: check dataclass type
    assert isinstance(state, SimulationState)

    # Core dicts
    assert isinstance(state.Levels, list)
    assert isinstance(state.Nonmesh, dict)
    assert isinstance(state.Properties, dict)

    # Static numbers
    assert isinstance(state.ne_nn, tuple)
    assert isinstance(state.substrate, tuple)
    assert isinstance(state.tmp_ne_nn, tuple)
    assert isinstance(state.subcycle, tuple)

    # Unchanging variables
    assert isinstance(state.laser_start, np.ndarray)
    assert isinstance(state.L1L2Eratio, list)
    assert all(isinstance(x, int) for x in state.L1L2Eratio)
    assert isinstance(state.L2L3Eratio, list)
    assert all(isinstance(x, int) for x in state.L2L3Eratio)
    assert isinstance(state.total_t_inc, int)
    assert isinstance(state.tool_path_file, io.TextIOWrapper)
    assert isinstance(state.layer_check, int)
    assert isinstance(state.level_names, list)
    assert all(isinstance(x, str) for x in state.level_names)

    # Changing variables
    assert isinstance(state.laser_prev_z, float)
    assert state.laser_prev_z == float("inf")
    assert isinstance(state.time_inc, int)
    assert isinstance(state.checkpoint_load, bool)
    assert isinstance(state.move_hist, list)
    assert all(isinstance(x, jnp.ndarray) for x in state.move_hist)
    assert isinstance(state.dwell_time_count, float)
    assert isinstance(state.accum_time, jnp.ndarray)
    assert isinstance(state.max_accum_time, jnp.ndarray)
    assert isinstance(state.record_inc, int)
    assert isinstance(state.wait_inc, int)
    assert isinstance(state.LInterp, list)
    assert isinstance(state.t_add, int)
    assert isinstance(state.tstart, float)
    assert isinstance(state.t_output, float)
    assert isinstance(state.ongoing_simulation, bool)
    assert state.ongoing_simulation is True

    # Paths
    assert isinstance(state.checkpoint_path, Path)
    assert state.checkpoint_path.name == "checkpoint"

    # Representative value checks
    assert state.dwell_time_count == 0.0
    assert state.move_hist[0].shape == ()  # scalar jnp array
    assert state.accum_time.shape == state.max_accum_time.shape

    # --- Value checks ---
    # Static metadata
    assert state.ne_nn == (600, 384, 726, 847, 567)
    assert state.substrate == (0, 0, 0, 0)
    assert state.tmp_ne_nn == (0, 0)
    assert state.subcycle == (4, 4, 16, 4.0, 4.0, 16.0, 1)

    # Laser start position
    np.testing.assert_array_equal(
        state.laser_start, np.array([2.5, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    )

    # Mesh ratios
    assert state.L1L2Eratio == [2, 2, 1]
    assert state.L2L3Eratio == [2, 2, 2]

    # Toolpath increments
    assert state.total_t_inc == 1480


class FakeToolPathFile(io.StringIO):
    def __init__(self, line="1.0,2.0,3.0\n"):
        # repeat the line many times so readline always has something
        super().__init__(line * 10)
        self._line = line

    def readline(self, *args, **kwargs):
        # always return the same line, regardless of position
        return self._line


def test_pre_time_loop_initialization_conditionals(monkeypatch):
    # --- Fake solver_input JSON ---
    solver_input = {
        "properties": {"layer_height": 1.0},
        "nonmesh": {
            "haste": 1,
            "use_txt": 1,
            "laser_center": [],  # empty list triggers branch
            "layer_num": 1,  # triggers checkpoint branch
            "restart_layer_num": 0,
            "toolpath": "fake_toolpath.txt",
            "save_path": "fake_save/",
            "record_step": 1,
        },
    }

    # Prepare fake file handles
    fake_json = io.StringIO(json.dumps(solver_input))
    fake_checkpoint = io.BytesIO()
    dill.dump((["levels"], "accum", "max_accum", 5, 10), fake_checkpoint)
    fake_checkpoint.seek(0)

    def fake_open(path, mode="r", *args, **kwargs):
        if "input.json" in str(path):
            return fake_json
        elif "toolpath" in str(path):
            return FakeToolPathFile()
        elif "Checkpoint" in str(path):
            return fake_checkpoint
        else:
            return io.StringIO("")

    monkeypatch.setattr("builtins.open", fake_open)

    # --- Monkeypatch heavy functions at helper level ---
    monkeypatch.setattr(
        helper_functions, "SetupProperties", lambda props: {"layer_height": 1.0}
    )
    monkeypatch.setattr(
        helper_functions,
        "SetupLevels",
        lambda solver_input, props: [
            {"nn": 1},
            {
                "h": [1, 1, 1],
                "node_coords": [jnp.arange(2), jnp.arange(2), jnp.arange(2)],
            },
            {
                "h": [1, 1, 1],
                "node_coords": [jnp.arange(2), jnp.arange(2), jnp.arange(2)],
            },
            {
                "h": [1, 1, 1],
                "node_coords": [jnp.arange(2), jnp.arange(2), jnp.arange(2)],
            },
        ],
    )
    monkeypatch.setattr(helper_functions, "SetupNonmesh", lambda nm: nm)
    monkeypatch.setattr(
        helper_functions, "SetupStaticNodesAndElements", lambda levels: (0, 0)
    )
    monkeypatch.setattr(helper_functions, "SetupStaticSubcycle", lambda nm: (1, 1, 1))
    monkeypatch.setattr(helper_functions, "count_lines", lambda path: 42)
    monkeypatch.setattr(
        helper_functions, "interpolatePointsMatrix", lambda a, b: np.array([[1]])
    )
    monkeypatch.setattr(
        helper_functions, "saveResults", lambda levels, nonmesh, savenum: None
    )

    # --- Act ---
    state = helper_functions.pre_time_loop_initialization("input.json")

    # --- Assert: check the conditional branches were exercised ---
    assert "HASTE" in state.level_names
    assert state.total_t_inc == 42
    assert np.allclose(state.laser_start, [1.0, 2.0, 3.0])
    assert state.checkpoint_load is True
    assert state.time_inc >= 5


def test_time_loop_pre_execution():
    # Arrange: load input state
    with open(Path("tests/core/inputs/inputs_time_loop_pre_execution.pkl"), "rb") as f:
        simulation_state = dill.load(f)

    # Arrange: load expected outputs
    with open(
        Path("tests/core/outputs/outputs_time_loop_pre_execution.pkl"), "rb"
    ) as f:
        simulation_state_output, laser_all_output, single_step_output = dill.load(f)

    # Assert: expect this to fail
    with pytest.raises(AssertionError):
        _assert_simulation_state_equal(simulation_state, simulation_state_output)

    # Act: run the function under test
    simulation_state_result, laser_all_result, single_step_result = (
        time_loop_pre_execution(simulation_state)
    )

    # Assert: exact equality with saved outputs
    assert simulation_state_result.ongoing_simulation == True
    _assert_simulation_state_equal(simulation_state_result, simulation_state_output)
    compare_values(laser_all_result, laser_all_output)
    compare_values(single_step_result, single_step_output)


def test_time_loop_pre_execution_edge_cases():
    # Arrange: load input state
    with open(Path("tests/core/inputs/inputs_time_loop_pre_execution.pkl"), "rb") as f:
        simulation_state = dill.load(f)

    num_lines_expected = (
        simulation_state.subcycle[0]
        * simulation_state.subcycle[1]
        * simulation_state.subcycle[-1]
    )
    raw_lines = [
        simulation_state.tool_path_file.readline().strip()
        for _ in range(num_lines_expected - 1)
    ]
    raw_lines.append("")
    simulation_state.tool_path_file = io.StringIO("\n".join(raw_lines) + "\n")
    # Act: run the function under test
    simulation_state_result, laser_all_result, single_step_result = (
        time_loop_pre_execution(simulation_state)
    )
    assert laser_all_result.shape[0] == 15
    assert single_step_result == True
    assert simulation_state_result.ongoing_simulation == False
    simulation_state_result, _, single_step_result = time_loop_pre_execution(
        simulation_state_result
    )
    assert single_step_result == False
    print("here")


def test_single_step_execution():
    # Arrange: load input state
    with open(Path("tests/core/inputs/inputs_single_step_execution.pkl"), "rb") as f:
        laser_all, simulation_state = dill.load(f)

    # Arrange: load expected outputs
    with open(Path("tests/core/outputs/outputs_single_step_execution.pkl"), "rb") as f:
        simulation_state_output = dill.load(f)

    # Assert: expect this to fail
    with pytest.raises(AssertionError):
        _assert_simulation_state_equal(simulation_state, simulation_state_output)

    # Act: run the function under test
    simulation_state_result = single_step_execution(laser_all, simulation_state)

    # Assert: exact equality with saved outputs
    _assert_simulation_state_equal(simulation_state_result, simulation_state_output)

    # Assert: now we expect this to fail
    with pytest.raises(AssertionError):
        # Mutate expected output to force mismatch
        simulation_state_result.Levels[1]["T0"] = jnp.ones_like(
            simulation_state_result.Levels[1]["T0"]
        )
        _assert_simulation_state_equal(simulation_state_result, simulation_state_output)


def test_multi_step_execution():
    # Arrange: load input state
    with open(Path("tests/core/inputs/inputs_multi_step_execution.pkl"), "rb") as f:
        laser_all, simulation_state = dill.load(f)

    # Arrange: load expected outputs
    with open(Path("tests/core/outputs/outputs_multi_step_execution.pkl"), "rb") as f:
        simulation_state_output = dill.load(f)

    # Assert: expect this to fail
    with pytest.raises(AssertionError):
        _assert_simulation_state_equal(simulation_state, simulation_state_output)

    # Act: run the function under test
    simulation_state_result = multi_step_execution(laser_all, simulation_state)

    # Assert: exact equality with saved outputs
    _assert_simulation_state_equal(simulation_state_result, simulation_state_output)

    # Assert: now we expect this to fail
    with pytest.raises(AssertionError):
        # Mutate expected output to force mismatch
        simulation_state_result.Levels[1]["T0"] = jnp.ones_like(
            simulation_state_result.Levels[1]["T0"]
        )
        _assert_simulation_state_equal(simulation_state_result, simulation_state_output)


def test_time_loop_post_execution():
    # Arrange: load input state
    with open(Path("tests/core/inputs/inputs_time_loop_post_execution.pkl"), "rb") as f:
        simulation_state, laser_all, t_loop = dill.load(f)

    # Arrange: load expected outputs
    with open(
        Path("tests/core/outputs/outputs_time_loop_post_execution.pkl"), "rb"
    ) as f:
        simulation_state_output = dill.load(f)

    # Assert: expect this to fail
    with pytest.raises(AssertionError):
        _assert_simulation_state_equal(simulation_state, simulation_state_output)

    # Act: run the function under test
    simulation_state_result = time_loop_post_execution(
        simulation_state, laser_all, t_loop
    )

    # Assert: exact equality with saved outputs
    _assert_simulation_state_equal(simulation_state_result, simulation_state_output)


def test_time_loop_post_execution_edge(monkeypatch):
    # Arrange: load input state from pickle
    with open(Path("tests/core/inputs/inputs_time_loop_post_execution.pkl"), "rb") as f:
        simulation_state, laser_all, t_loop = dill.load(f)

    # Patch saveResults so it doesn't execute
    called = {}
    monkeypatch.setattr(
        helper_functions,
        "saveResults",
        lambda levels, nonmesh, savenum: called.update(
            {"levels": levels, "nonmesh": nonmesh, "savenum": savenum}
        ),
    )

    # Force record_inc >= record_step so branch is taken
    simulation_state.record_inc = simulation_state.Nonmesh["record_step"]

    # Act
    simulation_state_result = time_loop_post_execution(
        simulation_state, laser_all, t_loop
    )

    # Assert: branch executed, saveResults was "called"
    assert "savenum" in called
    assert (
        called["savenum"]
        == int(simulation_state.time_inc / simulation_state.Nonmesh["record_step"]) + 1
    )
    assert simulation_state_result.record_inc == 0


def test_post_time_loop_finalization_runs(monkeypatch):
    # Arrange: load input state
    with open(
        Path("tests/core/inputs/inputs_post_time_loop_finalization.pkl"), "rb"
    ) as f:
        simulation_state = dill.load(f)

    # Force the branch: record_TAM == 1
    simulation_state.Nonmesh["record_TAM"] = 1

    # Patch side-effecting functions
    monkeypatch.setattr(
        "go_melt.io.save_results_functions.saveState", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "go_melt.io.save_results_functions.saveResultsFinal",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "go_melt.io.save_results_functions.saveCustom", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(jnp, "savez", lambda *args, **kwargs: None)

    # Act: just run the function
    post_time_loop_finalization(simulation_state)

    # Assert: no exception raised
    # (pytest will mark the test as passed if we reach here)
    pass


def test_clear_jax_function_caches(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    clear_jax_function_caches(funcs=None)
    output = buf.getvalue()
    assert "Cleared caches:" in output

    # Reset buffer for next case
    buf.truncate(0)
    buf.seek(0)

    def fake_function():
        return

    clear_jax_function_caches(funcs=[fake_function])
    output = buf.getvalue()
    assert "No caches cleared" in output


def _assert_simulation_state_equal(a, b):
    """Assert that two SimulationState objects are equal, field by field."""

    # --- Core dicts ---
    assert isinstance(a.Levels, list) and isinstance(b.Levels, list)
    assert len(a.Levels) == len(b.Levels)
    for d1, d2 in zip(a.Levels, b.Levels):
        assert d1.keys() == d2.keys()
        for k in d1:
            compare_values(d1[k], d2[k])

    compare_values(a.Nonmesh, b.Nonmesh)
    compare_values(a.Properties, b.Properties)

    # --- Static numbers ---
    assert a.ne_nn == b.ne_nn
    assert a.substrate == b.substrate
    assert a.tmp_ne_nn == b.tmp_ne_nn
    assert a.subcycle == b.subcycle

    # --- Unchanging variables ---
    np.testing.assert_array_equal(a.laser_start, b.laser_start)
    assert a.L1L2Eratio == b.L1L2Eratio
    assert a.L2L3Eratio == b.L2L3Eratio
    assert a.total_t_inc == b.total_t_inc
    assert isinstance(a.tool_path_file, io.TextIOWrapper)
    assert isinstance(b.tool_path_file, io.TextIOWrapper)
    assert a.layer_check == b.layer_check
    assert a.level_names == b.level_names

    # --- Changing variables ---
    assert a.laser_prev_z == b.laser_prev_z
    assert a.time_inc == b.time_inc
    assert a.checkpoint_load == b.checkpoint_load
    assert len(a.move_hist) == len(b.move_hist)
    for arr_a, arr_b in zip(a.move_hist, b.move_hist):
        assert jnp.array_equal(arr_a, arr_b)
    assert a.dwell_time_count == b.dwell_time_count
    np.testing.assert_array_equal(a.accum_time, b.accum_time)
    np.testing.assert_array_equal(a.max_accum_time, b.max_accum_time)
    assert a.record_inc == b.record_inc
    assert a.wait_inc == b.wait_inc
    assert a.t_add == b.t_add
    assert a.t_output == b.t_output
    assert a.ongoing_simulation == b.ongoing_simulation

    # --- Interpolation matrices ---
    assert len(a.LInterp) == len(b.LInterp)
    for m1, m2 in zip(a.LInterp, b.LInterp):
        np.testing.assert_array_equal(m1, m2)

    # --- Timing ---
    assert isinstance(a.tstart, float)
    assert isinstance(b.tstart, float)

    # --- Paths ---
    assert a.checkpoint_path == b.checkpoint_path


if __name__ == "__main__":
    pytest.main([__file__])

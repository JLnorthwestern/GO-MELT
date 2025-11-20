import os
import numpy as np
import pytest
import jax.numpy as jnp
import io
import dill

# Adjust this import to match where the functions actually live in your project.
# This test assumes the functions are in go_melt.utils.save_vtk
import pyevtk.hl as pyevtk_hl
from go_melt.io.save_results_functions import (
    saveResult,
    saveResults,
    saveResultsFinal,
    saveFinalResult,
    saveState,
    saveCustom,
    save_object,
)
import go_melt.io.save_results_functions as srf


def test_save_results_functions(monkeypatch, tmp_path):
    captured = {}

    def fake_gridToVTK(filename, cx, cy, cz, pointData=None):
        captured["filename"] = filename
        captured["cx"] = np.asarray(cx).copy()
        captured["cy"] = np.asarray(cy).copy()
        captured["cz"] = np.asarray(cz).copy()
        captured["pointData"] = {
            k: np.asarray(v).copy() for k, v in (pointData or {}).items()
        }

    # Monkeypatch both the module's gridToVTK reference and the pyevtk.hl function
    monkeypatch.setattr(pyevtk_hl, "gridToVTK", fake_gridToVTK)

    node_coords_x = jnp.linspace(0, 4, 5)
    node_coords_y = jnp.linspace(0, 4, 5)
    node_coords_z = jnp.linspace(0, 4, 5)

    nodes = [node_coords_x.size, node_coords_y.size, node_coords_z.size]
    total = nodes[0] * nodes[1] * nodes[2]
    T0 = np.arange(total).astype(float)
    S1 = (np.arange(total) % 2).astype(float)

    save_path = str(tmp_path) + os.sep

    Level = {
        "node_coords": [node_coords_x, node_coords_y, node_coords_z],
        "nodes": nodes,
        "T0": T0,
        "S1": S1,
    }
    Levels = [Level, Level, Level, Level]

    Nonmesh = {"output_files": 1, "Level1_record_step": 1, "save_path": save_path}

    saveResult(
        Level, save_str="Test_", record_lab=42, save_path=save_path, zoffset=0.123
    )
    saveResults(Levels, Nonmesh, savenum=42)
    saveResultsFinal(Levels, Nonmesh)
    saveFinalResult(Level, save_str="Test_", save_path=save_path, zoffset=0.123)
    saveState(
        Level, save_str="Test_", record_lab=42, save_path=save_path, zoffset=0.123
    )
    saveCustom(Level, Level["T0"], "Temperature (K)", save_path, "test_T", 0)


class NonClosingBytesIO(io.BytesIO):
    def close(self):
        # Don't actually close, just reset position
        self.seek(0)


def test_save_object_no_disk(monkeypatch):
    obj = {"a": 1, "b": [1, 2, 3]}
    fake_file = NonClosingBytesIO()

    monkeypatch.setattr("builtins.open", lambda f, mode: fake_file)
    monkeypatch.setattr(dill, "dump", lambda o, f, protocol: f.write(b"FAKE"))

    save_object(obj, "fake_filename.pkl")

    fake_file.seek(0)
    assert fake_file.read() == b"FAKE"


def test_record_first_call_invokes_save_object_once(monkeypatch):
    calls = []

    # Replace save_object with a dummy that just logs arguments
    monkeypatch.setattr(
        srf, "save_object", lambda obj, filename: calls.append((obj, filename))
    )

    # Force recording enabled
    monkeypatch.setattr(srf, "RECORD_INPUTS_OUTPUTS", True)
    srf._recorded_flags.clear()

    @srf.record_first_call("dummy")
    def dummy_func(x, y):
        return x + y

    # First call should trigger save_object twice (inputs + outputs)
    result1 = dummy_func(2, 3)
    assert result1 == 5
    assert len(calls) == 2
    assert "inputs_dummy.pkl" in calls[0][1]
    assert "outputs_dummy.pkl" in calls[1][1]

    # Second call should not trigger save_object again
    result2 = dummy_func(10, 20)
    assert result2 == 30
    assert len(calls) == 2  # still only the two calls from the first invocation


if __name__ == "__main__":
    pytest.main([__file__])

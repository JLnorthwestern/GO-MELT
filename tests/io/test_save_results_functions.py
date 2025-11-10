import os
import numpy as np
import pytest
import jax.numpy as jnp

# Adjust this import to match where the functions actually live in your project.
# This test assumes the functions are in go_melt.utils.save_vtk
import pyevtk.hl as pyevtk_hl
from go_melt.io.save_results_functions import (
    saveResult,
    saveResults,
    saveResultsFinal,
    saveFinalResult,
    saveState,
)


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


if __name__ == "__main__":
    pytest.main([__file__])

import pytest
import dill
import jax.numpy as jnp

from go_melt.core.move_mesh_functions import (
    moveEverything,
    move_fine_mesh,
    jit_constrain_v,
    update_overlap_nodes_coords,
    update_overlap_nodes_coords_L1L2,
    update_overlap_nodes_coords_L2,
)


def test_moveEverything():
    with open("tests/core/inputs/inputs_moveEverything.pkl", "rb") as f:
        (
            laser_pos,
            laser_start,
            Levels,
            move_hist,
            LInterp,
            L1L2Eratio,
            L2L3Eratio,
            layer_height,
        ) = dill.load(f)
    (Levels, Shapes, LInterp, move_hist) = moveEverything(
        laser_pos,
        laser_start,
        Levels,
        move_hist,
        LInterp,
        L1L2Eratio,
        L2L3Eratio,
        layer_height,
    )
    assert jnp.allclose(jnp.array(move_hist), jnp.array([0, 0, 1]))


def test_move_fine_mesh_positive_shift():
    # Original coordinates
    coords = [jnp.array([0.0, 1.0]), jnp.array([0.0, 1.0]), jnp.array([0.0, 1.0])]
    elem_sizes = [jnp.array(1.0), jnp.array(2.0), jnp.array(3.0)]
    displacement = [jnp.array(2.0), jnp.array(4.0), jnp.array(6.0)]

    new_coords, shifts = move_fine_mesh(coords, elem_sizes, displacement)

    # Expected integer shifts
    assert shifts == [2, 2, 2]
    # Coordinates should be shifted by elem_size * shift
    assert jnp.allclose(new_coords[0], coords[0] + 2.0)
    assert jnp.allclose(new_coords[1], coords[1] + 4.0)
    assert jnp.allclose(new_coords[2], coords[2] + 6.0)


def test_move_fine_mesh_zero_shift():
    coords = [jnp.array([0.0, 1.0]), jnp.array([0.0, 1.0]), jnp.array([0.0, 1.0])]
    elem_sizes = [jnp.array(1.0), jnp.array(1.0), jnp.array(1.0)]
    displacement = [jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)]

    new_coords, shifts = move_fine_mesh(coords, elem_sizes, displacement)

    # Expected integer shifts
    assert shifts == [0, 0, 0]
    # Coordinates should remain unchanged
    for i in range(3):
        assert jnp.allclose(new_coords[i], coords[i])


def test_move_fine_mesh_negative_shift():
    coords = [jnp.array([0.0, 1.0]), jnp.array([0.0, 1.0]), jnp.array([0.0, 1.0])]
    elem_sizes = [jnp.array(1.0), jnp.array(1.0), jnp.array(1.0)]
    displacement = [jnp.array(-1.0), jnp.array(-2.0), jnp.array(-3.0)]

    new_coords, shifts = move_fine_mesh(coords, elem_sizes, displacement)

    # Expected integer shifts (since truncates using int)
    assert shifts == [0, -1, -2]
    # Coordinates should be shifted negatively
    assert jnp.allclose(new_coords[0], coords[0] - 0.0)
    assert jnp.allclose(new_coords[1], coords[1] - 1.0)
    assert jnp.allclose(new_coords[2], coords[2] - 2.0)


def test_jit_constrain_v_inside_bounds():
    level = {
        "bounds": {
            "ix": (-5.0, 5.0),
            "iy": (-10.0, 10.0),
            "iz": (0.0, 20.0),
        }
    }
    laser_shift = [jnp.array(0.0), jnp.array(0.0), jnp.array(10.0)]
    result = jit_constrain_v(laser_shift, level)
    assert [float(r) for r in result] == [0.0, 0.0, 10.0]


def test_jit_constrain_v_below_bounds():
    level = {
        "bounds": {
            "ix": (-5.0, 5.0),
            "iy": (-10.0, 10.0),
            "iz": (0.0, 20.0),
        }
    }
    laser_shift = [jnp.array(-10.0), jnp.array(-20.0), jnp.array(-5.0)]
    result = jit_constrain_v(laser_shift, level)
    assert [float(r) for r in result] == [-5.0, -10.0, 0.0]


def test_jit_constrain_v_above_bounds():
    level = {
        "bounds": {
            "ix": (-5.0, 5.0),
            "iy": (-10.0, 10.0),
            "iz": (0.0, 20.0),
        }
    }
    laser_shift = [jnp.array(10.0), jnp.array(15.0), jnp.array(25.0)]
    result = jit_constrain_v(laser_shift, level)
    assert [float(r) for r in result] == [5.0, 10.0, 20.0]


def test_jit_constrain_v_mixed_case():
    level = {
        "bounds": {
            "ix": (-5.0, 5.0),
            "iy": (-10.0, 10.0),
            "iz": (0.0, 20.0),
        }
    }
    laser_shift = [jnp.array(-6.0), jnp.array(0.0), jnp.array(30.0)]
    result = jit_constrain_v(laser_shift, level)
    assert [float(r) for r in result] == [-5.0, 0.0, 20.0]


def test_update_overlap_nodes_coords_basic():
    Level = {
        "orig_overlap_nodes": [jnp.array([0]), jnp.array([0]), jnp.array([0])],
        "orig_overlap_coors": [jnp.array([0.0]), jnp.array([0.0]), jnp.array([0.0])],
    }
    displacement = [jnp.array(2.0), jnp.array(4.0), jnp.array(6.0)]
    elem_sizes = [jnp.array(1.0), jnp.array(2.0), jnp.array(3.0)]
    ratios = [1, 2, 3]

    updated = update_overlap_nodes_coords(Level, displacement, elem_sizes, ratios)

    # Shifts should be [2, 2, 2]
    assert jnp.all(updated["overlapNodes"][0] == 0 + 1 * 2)
    assert jnp.all(updated["overlapNodes"][1] == 0 + 2 * 2)
    assert jnp.all(updated["overlapNodes"][2] == 0 + 3 * 2)

    assert jnp.allclose(updated["overlapCoords"][0], 0.0 + 1.0 * 2)
    assert jnp.allclose(updated["overlapCoords"][1], 0.0 + 2.0 * 2)
    assert jnp.allclose(updated["overlapCoords"][2], 0.0 + 3.0 * 2)


def test_update_overlap_nodes_coords_L1L2_basic():
    Level = {
        "orig_overlap_nodes": [jnp.array([1]), jnp.array([2]), jnp.array([3])],
        "orig_overlap_coors": [jnp.array([1.0]), jnp.array([2.0]), jnp.array([3.0])],
    }
    displacement = [jnp.array(1.0), jnp.array(2.0), jnp.array(3.0)]
    elem_sizes = [jnp.array(1.0), jnp.array(1.0), jnp.array(1.0)]
    powder_layer_thickness = 2.0

    updated = update_overlap_nodes_coords_L1L2(
        Level, displacement, elem_sizes, powder_layer_thickness
    )

    # Shifts should be [1, 2, 3]
    assert jnp.all(updated["overlapNodes"][0] == 1 + 1)
    assert jnp.all(updated["overlapNodes"][1] == 2 + 2)
    assert jnp.all(updated["overlapNodes"][2] == 3 + 3)

    # Coordinates: x and y use elem_sizes * shift, z uses powder layer offset
    assert jnp.allclose(updated["overlapCoords"][0], 1.0 + 1.0 * 1)
    assert jnp.allclose(updated["overlapCoords"][1], 2.0 + 1.0 * 2)
    # z coordinate offset should be powder_layer_thickness * int(displacement[2]/powder_layer_thickness)
    expected_z_offset = powder_layer_thickness * int(
        displacement[2] / powder_layer_thickness
    )
    assert jnp.allclose(updated["overlapCoords"][2], 3.0 + expected_z_offset)


def test_update_overlap_nodes_coords_L2_basic():
    Level = {
        "orig_overlap_nodes_L2": [jnp.array([0]), jnp.array([0]), jnp.array([0])],
        "orig_overlap_coors_L2": [jnp.array([0.0]), jnp.array([0.0]), jnp.array([0.0])],
    }
    displacement = [jnp.array(2.0), jnp.array(4.0), jnp.array(6.0)]
    elem_sizes = [jnp.array(1.0), jnp.array(2.0), jnp.array(3.0)]
    ratios = [1, 2, 3]

    updated = update_overlap_nodes_coords_L2(Level, displacement, elem_sizes, ratios)

    # Shifts should be [2, 2, 2]
    assert jnp.all(updated["overlapNodes_L2"][0] == 0 + 1 * 2)
    assert jnp.all(updated["overlapNodes_L2"][1] == 0 + 2 * 2)
    assert jnp.all(updated["overlapNodes_L2"][2] == 0 + 3 * 2)

    assert jnp.allclose(updated["overlapCoords_L2"][0], 0.0 + 1.0 * 2)
    assert jnp.allclose(updated["overlapCoords_L2"][1], 0.0 + 2.0 * 2)
    assert jnp.allclose(updated["overlapCoords_L2"][2], 0.0 + 3.0 * 2)


if __name__ == "__main__":
    pytest.main([__file__])

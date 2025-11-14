import pytest
import dill
import jax.numpy as jnp

from go_melt.core.move_mesh_functions import moveEverything


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


if __name__ == "__main__":
    pytest.main([__file__])

import pytest
import jax.numpy as jnp
import dill

from go_melt.core.heat_source_functions import (
    computeLevelSource,
    computeSourcesL3,
    computeSources,
    computeSourceFunction_jax,
)


def test_computeSourceFunction_jax_runs():
    properties = {
        "laser_eta": 0.5,
        "laser_radius": 1.0,
        "laser_depth": 2.0,
    }
    x = jnp.array([0.0, 1.0])
    y = jnp.array([0.0, 1.0])
    z = jnp.array([0.0, 1.0])
    v = jnp.array([0.0, 0.0, 0.0])
    P = 10.0

    Q = computeSourceFunction_jax(x, y, z, v, properties, P)
    assert Q.shape == (2,)
    assert jnp.all(Q >= 0)


def test_computeLevelSource_runs():
    with open("tests/core/inputs/inputs_computeLevelSource.pkl", "rb") as f:
        Levels, ne_nn, laser_position, Shapes1, properties, laserP = dill.load(f)

    result = computeLevelSource(
        Levels, ne_nn, laser_position, Shapes1, properties, laserP
    )
    assert isinstance(result, jnp.ndarray)


def test_computeSourcesL3_runs():
    with open("tests/core/inputs/inputs_computeSourcesL3.pkl", "rb") as f:
        L3, laser_position, ne_nn, properties, laser_power = dill.load(f)

    result = computeSourcesL3(L3, laser_position, ne_nn, properties, laser_power)
    assert isinstance(result, jnp.ndarray)
    assert result.shape[0] == 567


def test_computeSources_runs():
    with open("tests/core/inputs/inputs_computeSources.pkl", "rb") as f:
        L3, v, Shapes, ne_nn, properties, laserP = dill.load(f)

    Fc, Fm, Ff = computeSources(L3, v, Shapes, ne_nn, properties, laserP)
    assert isinstance(Fc, jnp.ndarray)
    assert isinstance(Fm, jnp.ndarray)
    assert isinstance(Ff, jnp.ndarray)
    assert Fc.shape[0] == 726
    assert Fm.shape[0] == 847
    assert Ff.shape[0] == 567


if __name__ == "__main__":
    pytest.main([__file__])

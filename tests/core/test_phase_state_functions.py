import pytest
import jax.numpy as jnp
import dill

from go_melt.core.phase_state_functions import (
    computeStateProperties,
    updateStateProperties,
)


def test_computeStateProperties_bulk_and_fluid():
    properties = {
        "T_liquidus": 1000.0,
        "T_solidus": 800.0,
        "k_powder": 1.0,
        "k_bulk_coeff_a1": 0.01,
        "k_bulk_coeff_a0": 10.0,
        "k_fluid_coeff_a0": 5.0,
        "cp_solid_coeff_a1": 0.02,
        "cp_solid_coeff_a0": 1.0,
        "cp_mushy": 2.0,
        "cp_fluid": 3.0,
        "rho": 1000.0,
    }

    # Temperature array with values below solidus, between solidus/liquidus, and above liquidus
    T = jnp.array([700.0, 900.0, 1100.0])
    S1 = jnp.array([0.0, 0.0, 0.0])  # initial powder
    substrate_nodes = 1

    bulk, liquid, k, rhocp = computeStateProperties(T, S1, properties, substrate_nodes)

    # First node enforced as bulk due to substrate
    assert bulk[0] == 1.0
    # Third node should be liquid
    assert liquid[2] == 1.0
    # Thermal conductivity and rhocp should be positive
    assert jnp.all(k >= 0)
    assert jnp.all(rhocp >= 0)


def test_updateStateProperties():
    with open("tests/core/inputs/inputs_updateStateProperties.pkl", "rb") as f:
        Levels, properties, substrate = dill.load(f)

    updated_levels, Lk, Lrhocp = updateStateProperties(Levels, properties, substrate)

    assert updated_levels[1]["S1"].sum() == 484
    assert updated_levels[1]["S1"].size == 726
    # Thermal conductivity and rhocp lists should have correct length
    assert len(Lk) == 4
    assert len(Lrhocp) == 4
    # Values should be arrays
    assert isinstance(Lk[1], jnp.ndarray)
    assert isinstance(Lrhocp[1], jnp.ndarray)


if __name__ == "__main__":
    pytest.main([__file__])

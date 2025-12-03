import pytest
import dill
import jax.numpy as jnp
import jax
from go_melt.core.solution_functions import (
    stepGOMELTDwellTime,
    computeL1Temperature,
    computeSolutions,
    solveMatrixFreeFE,
)


def test_stepGOMELTDwellTime_updates_Level1_temperature():
    # Load test inputs from pickle file
    with open("tests/core/inputs/inputs_stepGOMELTDwellTime.pkl", "rb") as f:
        Levels, tmp_ne_nn, ne_nn, properties, dt, substrate, boundary_conditions = (
            dill.load(f)[0]
        )

    # Capture initial Level 1 temperature
    initial_T0 = Levels[1]["T0"].copy()

    # Convert inner tuples/lists to mutable lists
    bc0 = list(boundary_conditions[1][0])
    bc1 = list(boundary_conditions[1][1])
    bc2 = list(boundary_conditions[1][2])

    # Replace with Neumann
    bc0[0] = 1
    bc0[1] = 1
    # Replace functions with Convection (1) / Adiabatic (2)
    bc1[0] = 1
    bc1[1] = 2
    # Replace values with 1e-5 and nan
    bc2[0] = 1e-5
    bc2[1] = jnp.nan

    # Rebuild the outer structure
    boundary_conditions = (
        boundary_conditions[0],
        (tuple(bc0), tuple(bc1), tuple(bc2)) + boundary_conditions[1][3:],
    ) + boundary_conditions[2:]

    # Run the solver
    Levels = stepGOMELTDwellTime(
        Levels, tmp_ne_nn, ne_nn, properties, dt, substrate, boundary_conditions
    )

    # --- Assertions ---
    # 1. Output should be a list of dicts (Levels structure)
    assert isinstance(Levels, list)
    assert isinstance(Levels[1], dict)

    # 2. Updated Level 1 temperature should be a JAX array
    FinalL1 = Levels[1]["T0"]
    assert isinstance(FinalL1, jax.Array) or isinstance(FinalL1, jnp.ndarray)

    # 3. Shape should match the number of nodes in Level 1
    assert FinalL1.shape[0] == Levels[1]["nn"]

    # 4. No NaNs or Infs in the result
    assert jnp.all(jnp.isfinite(FinalL1))

    # 5. Values should be within a reasonable physical range
    assert jnp.all(FinalL1 < 1e6)
    assert jnp.all(FinalL1 > -1e6)

    # 6. Temperature field should be updated (not identical to initial)
    assert not jnp.allclose(initial_T0, FinalL1)


def test_computeL1Temperature_outputs_valid_temperature():
    # Load test inputs from pickle file
    with open("tests/core/inputs/inputs_computeL1Temperature.pkl", "rb") as f:
        (
            Levels,
            ne_nn,
            tmp_ne_nn,
            L1F,
            L1V,
            L1k,
            L1rhocp,
            dt,
            properties,
            boundary_conditions,
        ) = dill.load(f)[0]

    # Run the solver
    FinalL1 = computeL1Temperature(
        Levels,
        ne_nn,
        tmp_ne_nn,
        L1F,
        L1V,
        L1k,
        L1rhocp,
        dt,
        properties,
        boundary_conditions,
    )

    # --- Assertions ---
    # 1. Output should be a JAX array
    assert isinstance(FinalL1, jax.Array) or isinstance(FinalL1, jnp.ndarray)

    # 2. Shape should match the number of nodes in Level 1
    assert FinalL1.shape[0] == Levels[1]["nn"]

    # 3. No NaNs or Infs in the result
    assert jnp.all(jnp.isfinite(FinalL1))

    # 4. Values should be within a reasonable physical range
    assert jnp.all(FinalL1 < 1e6)
    assert jnp.all(FinalL1 > -1e6)

    # 5. Deterministic run: calling twice should give same result
    FinalL1b = computeL1Temperature(
        Levels,
        ne_nn,
        tmp_ne_nn,
        L1F,
        L1V,
        L1k,
        L1rhocp,
        dt,
        properties,
        boundary_conditions,
    )
    assert jnp.allclose(FinalL1, FinalL1b, rtol=1e-6, atol=1e-8)


def test_computeSolutions_outputs_three_levels():
    # Load test inputs from pickle file
    with open("tests/core/inputs/inputs_computeSolutions.pkl", "rb") as f:
        (
            Levels,
            ne_nn,
            tmp_ne_nn,
            LF,
            L1V,
            LInterp,
            Lk,
            Lrhocp,
            L2V,
            dt,
            properties,
            boundary_conditions,
        ) = dill.load(f)[0]

    # Run the solver
    FinalL1, FinalL2, FinalL3 = computeSolutions(
        Levels,
        ne_nn,
        tmp_ne_nn,
        LF,
        L1V,
        LInterp,
        Lk,
        Lrhocp,
        L2V,
        dt,
        properties,
        boundary_conditions,
    )

    # --- Assertions ---
    # 1. All outputs should be JAX arrays
    for arr in (FinalL1, FinalL2, FinalL3):
        assert isinstance(arr, jax.Array) or isinstance(arr, jnp.ndarray)

    # 2. Shapes should match the number of nodes in each level
    assert FinalL1.shape[0] == Levels[1]["nn"]
    assert FinalL2.shape[0] == Levels[2]["nn"]
    assert FinalL3.shape[0] == Levels[3]["nn"]

    # 3. No NaNs or Infs in any solution
    assert jnp.all(jnp.isfinite(FinalL1))
    assert jnp.all(jnp.isfinite(FinalL2))
    assert jnp.all(jnp.isfinite(FinalL3))

    # 4. Temperatures should remain within a reasonable physical range
    for arr in (FinalL1, FinalL2, FinalL3):
        assert jnp.all(arr < 1e6)
        assert jnp.all(arr > -1e6)

    # 5. Deterministic run: calling twice should give same results
    FinalL1b, FinalL2b, FinalL3b = computeSolutions(
        Levels,
        ne_nn,
        tmp_ne_nn,
        LF,
        L1V,
        LInterp,
        Lk,
        Lrhocp,
        L2V,
        dt,
        properties,
        boundary_conditions,
    )
    assert jnp.allclose(FinalL1, FinalL1b, rtol=1e-6, atol=1e-8)
    assert jnp.allclose(FinalL2, FinalL2b, rtol=1e-6, atol=1e-8)
    assert jnp.allclose(FinalL3, FinalL3b, rtol=1e-6, atol=1e-8)


def test_solveMatrixFreeFE_outputs_shape_and_values():
    # Load test inputs from pickle file
    with open("tests/core/inputs/inputs_solveMatrixFreeFE.pkl", "rb") as f:
        Level, nn, ne, k, rhocp, dt, T, Fc, Corr = dill.load(f)[0]

    # Taken from Level 1, to make sure finite, assume all substrate
    ne = jnp.prod(jnp.array(Level["elements"])).tolist()

    # Run the solver
    result = solveMatrixFreeFE(Level, nn, ne, k, rhocp, dt, T, Fc, Corr)

    # --- Assertions ---
    # 1. Output should be a JAX array
    assert isinstance(result, jax.Array) or isinstance(result, jnp.ndarray)

    # 2. Output length should match number of nodes
    assert result.shape[0] == nn

    # 3. No NaNs or Infs in the result
    assert jnp.all(jnp.isfinite(result[:]))

    # 4. Values should be within a reasonable physical range
    # (e.g., temperatures not exploding to absurd magnitudes)
    assert jnp.all(result < 1e6)
    assert jnp.all(result > -1e6)

    # 5. Deterministic run: calling twice should give same result
    result2 = solveMatrixFreeFE(Level, nn, ne, k, rhocp, dt, T, Fc, Corr)
    assert jnp.allclose(result, result2, rtol=1e-6, atol=1e-8)


if __name__ == "__main__":
    pytest.main([__file__])

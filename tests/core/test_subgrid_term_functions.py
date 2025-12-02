import pytest
import dill
import jax.numpy as jnp
import jax

from go_melt.core.subgrid_term_functions import (
    computeL2TprimeTerms_Part1,
    computeL2TprimeTerms_Part2,
    computeL1TprimeTerms_Part1,
    computeL1TprimeTerms_Part2,
    getNewTprime,
)
from go_melt.utils.interpolation_functions import interpolate_w_matrix


def test_computeL2TprimeTerms_Part1_shapes_and_types():
    with open("tests/core/inputs/inputs_computeL2TprimeTerms_Part1.pkl", "rb") as f:
        Levels, ne_nn, L3Tprime0, L3k, Shapes = dill.load(f)[0]

    key = jax.random.PRNGKey(0)  # seed for reproducibility
    L3Tprime0 = jax.random.uniform(key, shape=L3Tprime0.shape, dtype=L3Tprime0.dtype)

    # Run function
    Level2_Tprime_source = computeL2TprimeTerms_Part1(
        Levels=Levels,
        ne_nn=ne_nn,
        L3Tprime0=L3Tprime0,
        L3k=L3k,
        Shapes=Shapes,
    )

    # Assertions
    assert isinstance(Level2_Tprime_source, jnp.ndarray)
    # Should be 1D vector (projection to Level 2 correction space)
    assert Level2_Tprime_source.ndim == 1
    # Should not be all zeros
    assert not jnp.allclose(Level2_Tprime_source, 0.0)


def test_computeL2TprimeTerms_Part2_shapes_and_consistency():
    with open("tests/core/inputs/inputs_computeL2TprimeTerms_Part2.pkl", "rb") as f:
        (
            Levels,
            ne_nn,
            Level3_Tprime1,
            Level3_Tprime0,
            Level3_rhocp,
            dt,
            Shapes,
            Level2_Tprime_source_part1,
        ) = dill.load(f)[0]

    # Run Part2 using Part1 output as initial source
    Level2_Tprime_source_part2 = computeL2TprimeTerms_Part2(
        Levels=Levels,
        ne_nn=ne_nn,
        Level3_Tprime1=Level3_Tprime1,
        Level3_Tprime0=Level3_Tprime0,
        Level3_rhocp=Level3_rhocp,
        dt=dt,
        Shapes=Shapes,
        Level2_Tprime_Source=Level2_Tprime_source_part1.copy(),
    )

    # Assertions
    assert isinstance(Level2_Tprime_source_part2, jnp.ndarray)
    assert Level2_Tprime_source_part2.ndim == 1
    # Should differ from Part1 (since Part2 adds capacitance correction)
    assert not jnp.array_equal(Level2_Tprime_source_part2, Level2_Tprime_source_part1)

    # Check that update is additive
    diff = Level2_Tprime_source_part2 - Level2_Tprime_source_part1
    assert not jnp.allclose(diff, 0.0)


def test_computeL1TprimeTerms_Part1_shapes_and_types():
    with open("tests/core/inputs/inputs_computeL1TprimeTerms_Part1.pkl", "rb") as f:
        Levels, ne_nn, Level3_k, Shapes, Level2_k = dill.load(f)[0]
    key = jax.random.PRNGKey(0)  # seed for reproducibility
    Levels[2]["Tprime0"] = jax.random.uniform(
        key, shape=Levels[2]["Tprime0"].shape, dtype=Levels[2]["Tprime0"].dtype
    )
    Levels[3]["Tprime0"] = jax.random.uniform(
        key, shape=Levels[3]["Tprime0"].shape, dtype=Levels[3]["Tprime0"].dtype
    )

    # Run function
    Level1_Tprime_source = computeL1TprimeTerms_Part1(
        Levels=Levels,
        ne_nn=ne_nn,
        Level3_k=Level3_k,
        Shapes=Shapes,
        Level2_k=Level2_k,
    )

    # Assertions
    assert isinstance(Level1_Tprime_source, jnp.ndarray)
    # Should be 1D vector (projection to Level 2 correction space)
    assert Level1_Tprime_source.ndim == 1
    # Should not be all zeros
    assert not jnp.allclose(Level1_Tprime_source, 0.0)


def test_computeL1TprimeTerms_Part2_shapes_and_consistency():
    with open("tests/core/inputs/inputs_computeL1TprimeTerms_Part2.pkl", "rb") as f:
        (
            Levels,
            ne_nn,
            Level3_Tprime1,
            Level2_Tprime1,
            Level3_rhocp,
            Level2_rhocp,
            dt,
            Shapes,
            Level1_Tprime_source_part1,
        ) = dill.load(f)[0]
    Levels

    # Run Part2 using Part1 output as initial source
    Level1_Tprime_source_part2 = computeL1TprimeTerms_Part2(
        Levels=Levels,
        ne_nn=ne_nn,
        Level3_Tprime1=Level3_Tprime1,
        Level2_Tprime1=Level2_Tprime1,
        Level3_rhocp=Level3_rhocp,
        Level2_rhocp=Level2_rhocp,
        dt=dt,
        Shapes=Shapes,
        Level1_Tprime_source=Level1_Tprime_source_part1.copy(),
    )

    # Assertions
    assert isinstance(Level1_Tprime_source_part2, jnp.ndarray)
    assert Level1_Tprime_source_part2.ndim == 1
    # Should differ from Part1 (since Part2 adds capacitance correction)
    assert not jnp.array_equal(Level1_Tprime_source_part2, Level1_Tprime_source_part1)

    # Check that update is additive
    diff = Level1_Tprime_source_part2 - Level1_Tprime_source_part1
    assert not jnp.allclose(diff, 0.0)


def test_getNewTprime_outputs():
    # Load test inputs
    with open("tests/core/inputs/inputs_getNewTprime.pkl", "rb") as f:
        fine_level, fine_temp, coarse_temp, coarse_level, interpolate_coarse_to_fine = (
            dill.load(f)[0]
        )

    # Run function
    Tprime, new_coarse_temp = getNewTprime(
        fine_level=fine_level,
        fine_temp=fine_temp,
        coarse_temp=coarse_temp,
        coarse_level=coarse_level,
        interpolate_coarse_to_fine=interpolate_coarse_to_fine,
    )

    # Assertions
    # 1. Output types
    assert isinstance(Tprime, jnp.ndarray)
    assert isinstance(new_coarse_temp, jnp.ndarray)

    # 2. Shapes should match expectations
    assert Tprime.shape == fine_temp.shape
    assert new_coarse_temp.shape == coarse_temp.shape

    # 3. Coarse temp should be updated (not identical to input)
    assert not jnp.array_equal(new_coarse_temp, coarse_temp)

    # 4. Residual definition check: fine_temp ≈ Tprime + interpolated coarse
    reconstructed = Tprime + interpolate_w_matrix(
        interpolate_coarse_to_fine, new_coarse_temp
    )
    assert jnp.allclose(reconstructed, fine_temp, rtol=1e-5, atol=1e-8)


if __name__ == "__main__":
    pytest.main([__file__])

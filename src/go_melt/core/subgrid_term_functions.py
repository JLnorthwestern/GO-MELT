from jax.numpy import multiply
import jax.numpy as jnp
import jax
from functools import partial
from .mesh_functions import getSampleCoords
from go_melt.utils.gaussian_quadrature_functions import (
    computeQuad3dFemShapeFunctions_jax,
)
from go_melt.utils.helper_functions import (
    convert2XYZ,
    set_in_array,
    getOverlapRegion,
)
from go_melt.utils.interpolation_functions import (
    interpolate_w_matrix,
    interpolatePoints,
)


@partial(jax.jit, static_argnames=["level_num_elems"])
def computeL2TprimeTerms_Part1(
    Levels: dict[int, dict],
    level_num_elems: tuple,
    L3Tprime0: jnp.ndarray,
    L3k: jnp.ndarray,
    Shapes: tuple,
):
    """
    Computes subgrid conduction influence (K product terms) from Eqns. P3/C3 in GO-MELT
    paper by Leonor and Wagner (2024).
    """
    Level3_num_elems = level_num_elems[1]
    Level3_to_Level2_dNdx = Shapes[2][1]
    Level3_to_Level2_sum_operator = Shapes[2][2]

    fine_sample_coords = getSampleCoords(Levels[3])
    Nf, dNdxf, wqf = computeQuad3dFemShapeFunctions_jax(fine_sample_coords)

    _, _, _, fine_node_global_index = convert2XYZ(
        jnp.arange(Level3_num_elems),
        Levels[3]["elements"][0],
        Levels[3]["elements"][1],
        Levels[3]["nodes"][0],
        Levels[3]["nodes"][1],
    )
    _L3Tp0 = L3Tprime0[fine_node_global_index]
    L3kMean = jnp.matmul(Nf, L3k[fine_node_global_index]).mean(axis=0)

    # Compute ∇·(k ∇T') for Level 3
    level3_k_tprime_integrand = [
        multiply(L3kMean, (dNdxf[:, :, i] @ _L3Tp0)) for i in range(3)
    ]

    correction_level3_to_level2 = sum(
        multiply(
            multiply(
                -Level3_to_Level2_dNdx[i], level3_k_tprime_integrand[i].T[:, :, None]
            ),
            wqf[None, None, :],
        ).sum(axis=2)
        for i in range(3)
    )

    Level2_Tprime_source = (
        Level3_to_Level2_sum_operator @ correction_level3_to_level2.reshape(-1)
    )

    return Level2_Tprime_source


@partial(jax.jit, static_argnames=["level_num_elems"])
def computeL2TprimeTerms_Part2(
    Levels: dict[int, dict],
    level_num_elems: tuple,
    Level3_Tprime1: jnp.ndarray,
    Level3_Tprime0: jnp.ndarray,
    Level3_rhocp: jnp.ndarray,
    dt: float,
    Shapes: tuple,
    Level2_Tprime_Source: jnp.ndarray,
):
    """
    Computes subgrid capacitance influence (M product term) from Eqn. C3 in GO-MELT
    paper by Leonor and Wagner (2024).
    """
    Level3_num_elems = level_num_elems[1]
    Level3_to_Level2_N = Shapes[2][0]
    Level3_to_Level2_sum_operator = Shapes[2][2]

    # Compute change in subgrid temperature field
    delta_Level3_Tprime = Level3_Tprime1 - Level3_Tprime0

    # Get shape functions and weights for Level 3
    fine_sample_coords = getSampleCoords(Levels[3])
    Nf, _, wqf = computeQuad3dFemShapeFunctions_jax(fine_sample_coords)

    # Get global node indices for Level 3 elements
    _, _, _, fine_node_global_index = convert2XYZ(
        jnp.arange(Level3_num_elems),
        Levels[3]["elements"][0],
        Levels[3]["elements"][1],
        Levels[3]["nodes"][0],
        Levels[3]["nodes"][1],
    )

    # Integrate ρcp * T' over Level 3 elements
    level3_rhocp_tprime_integrand = multiply(
        Nf @ delta_Level3_Tprime[fine_node_global_index],
        jnp.matmul(Nf, Level3_rhocp[fine_node_global_index]).mean(axis=0),
    )

    # Compute weighted projection to Level 2
    correction_level3_to_level2 = multiply(
        multiply(-Level3_to_Level2_N, level3_rhocp_tprime_integrand.T[:, :, None]),
        (1 / dt) * wqf[None, None, :],
    ).sum(axis=2)

    # Update Level 2 correction vector
    Level2_Tprime_Source += (
        Level3_to_Level2_sum_operator @ correction_level3_to_level2.reshape(-1)
    )

    return Level2_Tprime_Source


@partial(jax.jit, static_argnames=["level_num_elems"])
def computeL1TprimeTerms_Part1(
    Levels: dict[int, dict],
    level_num_elems: tuple,
    Level3_k: jnp.ndarray,
    Shapes: tuple,
    Level2_k: jnp.ndarray,
) -> jnp.ndarray:
    """
    Computes subgrid conduction influence (K product terms) from Eqns. P1/C1 in GO-MELT
    paper by Leonor and Wagner (2024).
    """
    Level2_num_elems = level_num_elems[0]
    Level3_num_elems = level_num_elems[1]
    Level2_to_Level1_dNdx = Shapes[0][1]
    Level3_to_Level1_dNdx = Shapes[1][1]
    Level2_to_Level1_sum_operator = Shapes[0][2]
    Level3_to_Level1_sum_operator = Shapes[1][2]

    # --- Level 3 subgrid flux divergence ---
    fine_sample_coords = getSampleCoords(Levels[3])
    Nf, dNdxf, wqf = computeQuad3dFemShapeFunctions_jax(fine_sample_coords)

    _, _, _, fine_node_global_index = convert2XYZ(
        jnp.arange(Level3_num_elems),
        Levels[3]["elements"][0],
        Levels[3]["elements"][1],
        Levels[3]["nodes"][0],
        Levels[3]["nodes"][1],
    )
    _L3Tp0 = Levels[3]["Tprime0"][fine_node_global_index]
    L3kMean = jnp.matmul(Nf, Level3_k[fine_node_global_index]).mean(axis=0)

    # Compute ∇·(k ∇T') for Level 3
    level3_k_tprime_integrand = [
        multiply(L3kMean, (dNdxf[:, :, i] @ _L3Tp0)) for i in range(3)
    ]

    correction_level3_to_level1 = sum(
        multiply(
            multiply(
                -Level3_to_Level1_dNdx[i], level3_k_tprime_integrand[i].T[:, :, None]
            ),
            wqf[None, None, :],
        ).sum(axis=2)
        for i in range(3)
    )

    # --- Level 2 subgrid flux divergence ---
    meso_sample_coords = getSampleCoords(Levels[2])
    Nm, dNdxm, wqm = computeQuad3dFemShapeFunctions_jax(meso_sample_coords)

    _, _, _, meso_node_global_index = convert2XYZ(
        jnp.arange(Level2_num_elems),
        Levels[2]["elements"][0],
        Levels[2]["elements"][1],
        Levels[2]["nodes"][0],
        Levels[2]["nodes"][1],
    )
    _L2Tp0 = Levels[2]["Tprime0"][meso_node_global_index]
    L2kMean = jnp.matmul(Nm, Level2_k[meso_node_global_index]).mean(axis=0)

    # Compute ∇·(k ∇T') for Level 2
    level2_k_tprime_integrand = [
        multiply(L2kMean, (dNdxm[:, :, i] @ _L2Tp0)) for i in range(3)
    ]

    correction_level2_to_level1 = sum(
        multiply(
            multiply(
                -Level2_to_Level1_dNdx[i], level2_k_tprime_integrand[i].T[:, :, None]
            ),
            wqm[None, None, :],
        ).sum(axis=2)
        for i in range(3)
    )

    Level1_Tprime_source = (
        Level3_to_Level1_sum_operator @ correction_level3_to_level1.reshape(-1)
        + Level2_to_Level1_sum_operator @ correction_level2_to_level1.reshape(-1)
    )

    return Level1_Tprime_source


@partial(jax.jit, static_argnames=["level_num_elems"])
def computeL1TprimeTerms_Part2(
    Levels: dict[int, dict],
    level_num_elems: tuple,
    Level3_Tprime1: jnp.ndarray,
    Level2_Tprime1: jnp.ndarray,
    Level3_rhocp: jnp.ndarray,
    Level2_rhocp: jnp.ndarray,
    dt: float,
    Shapes: tuple,
    Level1_Tprime_source: jnp.ndarray,
) -> jnp.ndarray:
    """
    Computes subgrid capacitance influence (M product terms) from Eqn. C1 in GO-MELT
    paper by Leonor and Wagner (2024).
    """
    Level2_num_elems = level_num_elems[0]
    Level3_num_elems = level_num_elems[1]
    Level2_to_Level1_N = Shapes[0][0]
    Level3_to_Level1_N = Shapes[1][0]
    Level2_to_Level1_sum_operator = Shapes[0][2]
    Level3_to_Level1_sum_operator = Shapes[1][2]

    # Compute change in subgrid temperature fields
    delta_Level3_Tprime = Level3_Tprime1 - Levels[3]["Tprime0"]
    delta_Level2_Tprime = Level2_Tprime1 - Levels[2]["Tprime0"]

    # --- Level 3 contribution ---
    fine_sample_coords = getSampleCoords(Levels[3])
    Nf, _, wqf = computeQuad3dFemShapeFunctions_jax(fine_sample_coords)

    _, _, _, fine_node_global_index = convert2XYZ(
        jnp.arange(Level3_num_elems),
        Levels[3]["elements"][0],
        Levels[3]["elements"][1],
        Levels[3]["nodes"][0],
        Levels[3]["nodes"][1],
    )

    # Integrate ρcp * T' over Level 3 elements
    level3_rhocp_tprime_integrand = multiply(
        Nf @ delta_Level3_Tprime[fine_node_global_index],
        jnp.matmul(Nf, Level3_rhocp[fine_node_global_index]).mean(axis=0),
    )
    correction_level3_to_level1 = multiply(
        multiply(-Level3_to_Level1_N, level3_rhocp_tprime_integrand.T[:, :, None]),
        (1 / dt) * wqf[None, None, :],
    ).sum(axis=2)

    # --- Level 2 contribution ---
    meso_sample_coords = getSampleCoords(Levels[2])
    Nm, _, wqm = computeQuad3dFemShapeFunctions_jax(meso_sample_coords)

    _, _, _, meso_node_global_index = convert2XYZ(
        jnp.arange(Level2_num_elems),
        Levels[2]["elements"][0],
        Levels[2]["elements"][1],
        Levels[2]["nodes"][0],
        Levels[2]["nodes"][1],
    )

    # Integrate ρcp * T' over Level 2 elements
    level2_rhocp_tprime_integrand = multiply(
        Nm @ delta_Level2_Tprime[meso_node_global_index],
        jnp.matmul(Nm, Level2_rhocp[meso_node_global_index]).mean(axis=0),
    )
    correction_level2_to_level1 = multiply(
        multiply(-Level2_to_Level1_N, level2_rhocp_tprime_integrand.T[:, :, None]),
        (1 / dt) * wqm[None, None, :],
    ).sum(axis=2)

    # Project both contributions to Level 1 and update correction vector
    Level1_Tprime_source += (
        Level3_to_Level1_sum_operator @ correction_level3_to_level1.reshape(-1)
        + Level2_to_Level1_sum_operator @ correction_level2_to_level1.reshape(-1)
    )

    return Level1_Tprime_source


@jax.jit
def getNewTprime(
    fine_level: dict[str, list],
    fine_temp: jnp.ndarray,
    coarse_temp: jnp.ndarray,
    coarse_level: dict,
    interpolate_coarse_to_fine: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute the fine-level temperature subgrid term (T') and update the coarse-level
    temperature.
    """
    # Interpolate fine temperature at overlap coordinates
    values = interpolatePoints(fine_level, fine_temp, fine_level["overlapCoords"])

    # Get flattened index of overlap region in coarse mesh
    indices = getOverlapRegion(
        fine_level["overlapNodes"], coarse_level["nodes"][0], coarse_level["nodes"][1]
    )

    # Update coarse temperatures with fine temperatures at overlap region
    new_coarse_temp = set_in_array(coarse_temp, indices, values)

    # Compute residual between fine and interpolated coarse temperature
    Tprime = fine_temp - interpolate_w_matrix(
        interpolate_coarse_to_fine, new_coarse_temp
    )

    return Tprime, new_coarse_temp

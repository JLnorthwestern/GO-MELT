from functools import partial
import jax
import jax.numpy as jnp
from go_melt.utils.interpolation_functions import interpolatePoints
from go_melt.utils.helper_functions import getOverlapRegion
from .phase_state_functions import updateStateProperties, computeStateProperties
from .boundary_condition_functions import (
    computeConvRadBC,
)
from .solution_functions import (
    computeL1Temperature,
    computeL2Temperature,
    computeSolutions_L3,
    computeSolutions,
)
from .subgrid_term_functions import (
    computeL1TprimeTerms_Part1,
    computeL2TprimeTerms_Part1,
    getNewTprime,
    computeL1TprimeTerms_Part2,
    computeL2TprimeTerms_Part2,
)
from .heat_source_functions import computeSourcesL3, computeLevelSource, computeSources

# TFSP: Temporary fix for single precision


@partial(jax.jit, static_argnames=["ne_nn", "tmp_ne_nn", "substrate"])
def stepGOMELT(
    Levels, ne_nn, tmp_ne_nn, Shapes, LInterp, v, properties, dt, laserP, substrate
):
    """
    Perform a full explicit time step for the multilevel GOMELT simulation.

    This function executes the following sequence:
      1. Update material state and thermal properties.
      2. Compute source terms (laser, convection, radiation) for all levels.
      3. Compute volumetric correction terms from previous Tprime.
      4. Predictor step: solve temperature fields for all levels.
      5. Compute new Tprime fields and updated volumetric corrections.
      6. Corrector step: solve temperature fields again using updated corrections.
      7. Update Tprime and temperature fields for the next time step.

    Parameters:
    Levels (dict): Multilevel mesh and field data.
    ne_nn (list): Total number of elements and nodes for each level.
    tmp_ne_nn (list): Active element/node counts for Level 1 (based on layer).
    Shapes (list): Shape function data for interpolation between levels.
    LInterp (list): Interpolation matrices between levels.
    v (array): Current laser position.
    properties (dict): Material and simulation properties.
    dt (float): Time step size.
    laserP (float): Laser power.
    substrate (array): Node indices defining the substrate region.

    Returns:
    tuple:
        - Levels (dict): Updated multilevel data with new temperatures and Tprime fields.
        - _resetmask (array): Boolean mask indicating newly activated melt regions.
    """
    # Store previous melt state for comparison
    preS2 = Levels[3]["S2"]

    # Update material state and thermal properties
    Levels, Lk, Lrhocp = updateStateProperties(Levels, properties, substrate)

    # Unpack levels for clarity
    L1, L2, L3 = Levels[1], Levels[2], Levels[3]

    # --- Compute source terms ---
    Fc, Fm, Ff = computeSources(L3, v, Shapes, ne_nn, properties, laserP)
    Fc = computeConvRadBC(L1, L1["T0"], tmp_ne_nn[0], ne_nn[2], properties, Fc)
    Fm = computeConvRadBC(L2, L2["T0"], ne_nn[0], ne_nn[3], properties, Fm)
    Ff = computeConvRadBC(L3, L3["T0"], ne_nn[1], ne_nn[4], properties, Ff)
    F = [0, Fc, Fm, Ff]  # Source terms for Levels 1–3

    # --- Predictor step ---
    Vcu = computeL1TprimeTerms_Part1(Levels, ne_nn, Lk[3], Shapes, Lk[2])
    Vmu = computeL2TprimeTerms_Part1(Levels, ne_nn, Levels[3]["Tprime0"], Lk[3], Shapes)

    L1T, L2T, L3T = computeSolutions(
        Levels, ne_nn, tmp_ne_nn, F, Vcu, LInterp, Lk, Lrhocp, Vmu, dt, properties
    )

    # Enforce minimum temperature (TFSP)
    L1T = jnp.maximum(properties["T_amb"], L1T)
    L2T = jnp.maximum(properties["T_amb"], L2T)
    L3T = jnp.maximum(properties["T_amb"], L3T)

    # --- Compute new Tprime fields ---
    interpolate_L2_to_L3 = LInterp[1]
    interpolate_L1_to_L2 = LInterp[0]
    L3Tp, L2T = getNewTprime(Levels[3], L3T, L2T, Levels[2], interpolate_L2_to_L3)
    L2Tp, L1T = getNewTprime(Levels[2], L2T, L1T, Levels[1], interpolate_L1_to_L2)

    # --- Update volumetric correction terms ---
    Vcu = computeL1TprimeTerms_Part2(
        Levels, ne_nn, L3Tp, L2Tp, Lrhocp[3], Lrhocp[2], dt, Shapes, Vcu
    )
    Vmu = computeL2TprimeTerms_Part2(
        Levels, ne_nn, L3Tp, Levels[3]["Tprime0"], Lrhocp[3], dt, Shapes, Vmu
    )

    # --- Corrector step ---
    L1T, L2T, Levels[3]["T0"] = computeSolutions(
        Levels, ne_nn, tmp_ne_nn, F, Vcu, LInterp, Lk, Lrhocp, Vmu, dt, properties
    )

    # Enforce minimum temperature again (TFSP)
    L1T = jnp.maximum(properties["T_amb"], L1T)
    L2T = jnp.maximum(properties["T_amb"], L2T)
    Levels[3]["T0"] = jnp.maximum(properties["T_amb"], Levels[3]["T0"])

    # --- Final Tprime update for next time step ---
    Levels[3]["Tprime0"], Levels[2]["T0"] = getNewTprime(
        Levels[3], Levels[3]["T0"], L2T, Levels[2], interpolate_L2_to_L3
    )
    Levels[2]["Tprime0"], Levels[1]["T0"] = getNewTprime(
        Levels[2], Levels[2]["T0"], L1T, Levels[1], interpolate_L1_to_L2
    )

    # --- Update global state arrays ---
    Levels[0]["S1"] = Levels[0]["S1"].at[Levels[0]["idx"]].set(Levels[3]["S1"])
    Levels[0]["S2"] = Levels[0]["S2"].at[:].set(False)
    Levels[0]["S2"] = Levels[0]["S2"].at[Levels[0]["idx"]].set(Levels[3]["S2"])

    # Identify newly activated melt regions
    _resetmask = ((1 - 2 * preS2) * Levels[3]["S2"]) == 1

    return Levels, _resetmask


@partial(jax.jit, static_argnames=["ne_nn", "tmp_ne_nn", "substrate", "subcycle"])
def subcycleGOMELT(
    Levels: list[dict],
    ne_nn: tuple[int],
    Shapes: list[list],
    substrate: tuple[int],
    LInterp: list[list],
    tmp_ne_nn: tuple[int],
    laser_position: jnp.ndarray,
    properties: dict,
    laserP: jnp.ndarray,
    subcycle: tuple[int, int, int, float, float, float],
    max_accum_L3: jnp.ndarray,
    accum_L3: jnp.ndarray,
) -> tuple[list[dict], jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Perform a full predictor-corrector subcycling step for the GO-MELT model.

    This function executes the multilevel thermal update across Levels 1, 2, and 3,
    using nested subcycling and subgrid scale corrections. It includes both the
    predictor and corrector phases, updating temperature fields and phase states.
    """
    L3rhocp_L1, L2rhocp_L1, L1k, L1rhocp, L1F, L1V, L1T = computeLevel1predictor(
        Levels, substrate, properties, Shapes, ne_nn, tmp_ne_nn, laser_position, laserP
    )

    # --- Subcycle Level 2 ---
    def subcycleL2_Part1(_L2carry, _L2sub):
        (Lidx, L3rhocp_L2, L2S1, L2k, L2rhocp, L2F, L2V, _BC) = compute_Level2_step(
            _L2sub,
            subcycle,
            _L2carry,
            properties,
            substrate,
            Levels,
            ne_nn,
            laser_position,
            Shapes,
            laserP,
            L1T,
        )

        # --- Temperature Solve ---
        # Solve Level 2 temperature using matrix-free FEM
        L2T = computeL2Temperature(
            _BC,
            LInterp[0],
            Levels,
            ne_nn,
            _L2carry[0],
            L2F,
            L2V,
            L2k,
            L2rhocp,
            laser_position[Lidx, 5].sum(),
        )
        L2T = jnp.maximum(properties["T_amb"], L2T)  # TFSP

        # --- Subcycle Level 3 ---
        def subcycleL3_Part1(_L3carry, _L3sub):
            (L3S1, L3S2, LLidx, L3T) = compute_Level3_step(
                _L3carry,
                properties,
                substrate,
                _L3sub,
                _L2sub,
                subcycle,
                Levels,
                laser_position,
                ne_nn,
                laserP,
                L2T,
                _L2carry,
                LInterp,
            )

            return ([L3T, L3S1], [L3T, L3S1])

        # Run Level 3 subcycling loop
        [L3T, L3S1], _ = jax.lax.scan(
            subcycleL3_Part1,
            [_L2carry[2], _L2carry[4]],
            jnp.arange(subcycle[1]),
        )

        # Compute Updated Level 3 Tprime and update Level 2 Temperature
        L3Tp, L2T = getNewTprime(Levels[3], L3T, L2T, Levels[2], LInterp[1])

        return ([L2T, L2S1, L3T, L3Tp, L3S1], [L2T, L2S1, L3T, L3Tp, L3S1])

    # Run Level 2 subcycling loop
    [L2T, _, _, L3Tp, _], [_, _, _, L3Tp_L2, _] = jax.lax.scan(
        subcycleL2_Part1,
        [
            Levels[2]["T0"],
            Levels[2]["S1"],
            Levels[3]["T0"],
            Levels[3]["Tprime0"],
            Levels[3]["S1"],
        ],
        jnp.arange(subcycle[0]),
    )

    # --- Level 2 Tprime and Level 1 Corrector Update ---
    L2Tp, L1T = getNewTprime(Levels[2], L2T, L1T, Levels[1], LInterp[0])
    L1V = computeL1TprimeTerms_Part2(
        Levels,
        ne_nn,
        L3Tp,
        L2Tp,
        L3rhocp_L1,
        L2rhocp_L1,
        laser_position[:, 5].sum(),
        Shapes,
        L1V,
    )
    L1T = computeL1Temperature(
        Levels,
        ne_nn,
        tmp_ne_nn,
        L1F,
        L1V,
        L1k,
        L1rhocp,
        laser_position[:, 5].sum(),
        properties,
    )
    L1T = jnp.maximum(properties["T_amb"], L1T)  # TFSP

    # --- Subcycle Level 2 ---
    def subcycleL2_Part2(_L2carry, _L2sub):
        (Lidx, L3rhocp_L2, L2S1, L2k, L2rhocp, L2F, L2V, _BC) = compute_Level2_step(
            _L2sub,
            subcycle,
            _L2carry,
            properties,
            substrate,
            Levels,
            ne_nn,
            laser_position,
            Shapes,
            laserP,
            L1T,
        )

        # Add time derivative correction from Level 3 to Level 2
        L2V = computeL2TprimeTerms_Part2(
            Levels,
            ne_nn,
            L3Tp_L2[_L2sub],
            _L2carry[3],
            L3rhocp_L2,
            laser_position[Lidx, 5].sum(),
            Shapes,
            L2V,
        )

        # --- Temperature Solve ---
        # Solve Level 2 temperature using matrix-free FEM
        L2T = computeL2Temperature(
            _BC,
            LInterp[0],
            Levels,
            ne_nn,
            _L2carry[0],
            L2F,
            L2V,
            L2k,
            L2rhocp,
            laser_position[Lidx, 5].sum(),
        )
        L2T = jnp.maximum(properties["T_amb"], L2T)  # TFSP

        # --- Subcycle Level 3 ---
        def subcycleL3_Part2(_L3carry, _L3sub):
            (L3S1, L3S2, LLidx, L3T) = compute_Level3_step(
                _L3carry,
                properties,
                substrate,
                _L3sub,
                _L2sub,
                subcycle,
                Levels,
                laser_position,
                ne_nn,
                laserP,
                L2T,
                _L2carry,
                LInterp,
            )

            # --- Accumulated Melt Time Update ---
            # Reset accumulation if solid becomes liquid again
            _resetmask = ((1 - 2 * _L3carry[2]) * L3S2) == 1
            _resetaccumtime = _L3carry[4] * _resetmask

            # Update max accumulated melt time
            _max_check = jnp.maximum(_resetaccumtime, _L3carry[3])
            max_accum_L3 = _max_check

            # Update current accumulated melt time
            accum_L3 = _L3carry[4] + laser_position[LLidx, 5] * L3S2 - _resetaccumtime

            return (
                [L3T, L3S1, L3S2, max_accum_L3, accum_L3],
                [L3T, L3S1, L3S2, max_accum_L3, accum_L3],
            )

        # Run Level 3 subcycling loop
        [L3T, L3S1, L3S2, max_accum_L3, accum_L3], _ = jax.lax.scan(
            subcycleL3_Part2,
            [_L2carry[2], _L2carry[4], _L2carry[5], _L2carry[6], _L2carry[7]],
            jnp.arange(subcycle[1]),
        )

        # Compute updated Tprime for Level 3 and temperature for Level 2
        L3Tp, L2T = getNewTprime(Levels[3], L3T, L2T, Levels[2], LInterp[1])

        return (
            [L2T, L2S1, L3T, L3Tp, L3S1, L3S2, max_accum_L3, accum_L3],
            [L2T, L2S1, L3T, L3Tp, L3S1, L3S2, max_accum_L3, accum_L3],
        )

    [
        Levels[2]["T0"],
        Levels[2]["S1"],
        Levels[3]["T0"],
        Levels[3]["Tprime0"],
        Levels[3]["S1"],
        Levels[3]["S2"],
        max_accum_L3,
        accum_L3,
    ], [L2all, _, L3all, L3pall, _, _, _, _] = jax.lax.scan(
        subcycleL2_Part2,
        [
            Levels[2]["T0"],
            Levels[2]["S1"],
            Levels[3]["T0"],
            Levels[3]["Tprime0"],
            Levels[3]["S1"],
            Levels[3]["S2"],
            max_accum_L3,
            accum_L3,
        ],
        jnp.arange(subcycle[0]),
    )

    # --- Final Tprime and Phase Updates ---
    Levels[2]["Tprime0"], Levels[1]["T0"] = getNewTprime(
        Levels[2], Levels[2]["T0"], L1T, Levels[1], LInterp[0]
    )
    Levels[0]["S1"] = Levels[0]["S1"].at[Levels[0]["idx"]].set(Levels[3]["S1"])
    Levels[0]["S2"] = Levels[0]["S2"].at[:].set(False)
    Levels[0]["S2"] = Levels[0]["S2"].at[Levels[0]["idx"]].set(Levels[3]["S2"])

    return Levels, L2all, L3all, L3pall, max_accum_L3, accum_L3


def computeLevel1predictor(
    Levels, substrate, properties, Shapes, ne_nn, tmp_ne_nn, laser_position, laserP
) -> tuple:
    """
    Compute Level 1 temperature predictor and return all intermediate outputs
    as a tuple in the following order:
    """

    # --- Level 1 Material Properties ---
    _, _, L3k_L1, L3rhocp_L1 = computeStateProperties(
        Levels[3]["T0"], Levels[3]["S1"], properties, substrate[3]
    )
    _, _, L2k_L1, L2rhocp_L1 = computeStateProperties(
        Levels[2]["T0"], Levels[2]["S1"], properties, substrate[2]
    )

    # --- Update Level 1 S1 from Level 2 overlap ---
    _val = interpolatePoints(Levels[2], Levels[2]["S1"], Levels[2]["overlapCoords"])
    _idx = getOverlapRegion(
        Levels[2]["overlapNodes"], Levels[1]["nodes"][0], Levels[1]["nodes"][1]
    )
    Levels[1]["S1"] = Levels[1]["S1"].at[_idx].set(_val)
    Levels[1]["S1"] = Levels[1]["S1"].at[: substrate[1]].set(1)

    # --- Level 1 Material Properties (updated) ---
    _, _, L1k, L1rhocp = computeStateProperties(
        Levels[1]["T0"], Levels[1]["S1"], properties, substrate[1]
    )

    # --- Level 1 Source and Subgrid Terms ---
    L1F = computeLevelSource(
        Levels, ne_nn, laser_position, Shapes[1], properties, laserP
    )
    L1F = computeConvRadBC(
        Levels[1], Levels[1]["T0"], tmp_ne_nn[0], ne_nn[2], properties, L1F
    )
    L1V = computeL1TprimeTerms_Part1(Levels, ne_nn, L3k_L1, Shapes, L2k_L1)

    # --- Level 1 Temperature Predictor ---
    L1T = computeL1Temperature(
        Levels,
        ne_nn,
        tmp_ne_nn,
        L1F,
        L1V,
        L1k,
        L1rhocp,
        laser_position[:, 5].sum(),
        properties,
    )
    L1T = jnp.maximum(properties["T_amb"], L1T)  # TFSP

    return (L3rhocp_L1, L2rhocp_L1, L1k, L1rhocp, L1F, L1V, L1T)


def compute_Level2_step(
    _L2sub,
    subcycle,
    _L2carry,
    properties,
    substrate,
    Levels,
    ne_nn,
    laser_position,
    Shapes,
    laserP,
    L1T,
):
    # Compute interpolation weights for Level 2 boundary conditions
    alpha_L2 = (_L2sub + 1) / subcycle[3]
    beta_L2 = 1 - alpha_L2
    # Determine the laser substeps for this Level 2 subcycle
    Lidx = _L2sub * subcycle[1] + jnp.arange(subcycle[1])

    # --- Material Properties ---
    # Compute Level 3 properties using current Level 3 temperature and phase
    _, _, L3k_L2, L3rhocp_L2 = computeStateProperties(
        _L2carry[2], _L2carry[4], properties, substrate[3]
    )

    # Compute Level 2 properties using current Level 2 temperature and phase
    L2S1, _, L2k, L2rhocp = computeStateProperties(
        _L2carry[0], _L2carry[1], properties, substrate[2]
    )

    # --- Source Term ---
    # Compute laser source term for Level 2 using Level 3 mesh
    L2F = computeLevelSource(
        Levels, ne_nn, laser_position[Lidx, :], Shapes[2], properties, laserP[Lidx]
    )

    # Add convection, radiation, and evaporation boundary conditions
    L2F = computeConvRadBC(
        Levels[2],
        _L2carry[0],
        ne_nn[0],
        ne_nn[3],
        properties,
        L2F,
    )

    # --- Subgrid Correction ---
    # Compute divergence of subgrid heat flux from Level 3 to Level 2
    L2V = computeL2TprimeTerms_Part1(Levels, ne_nn, _L2carry[3], L3k_L2, Shapes)

    # Interpolate Level 1 temperature to Level 2 boundary using alpha-beta blend
    _BC = alpha_L2 * L1T + beta_L2 * Levels[1]["T0"]

    return (Lidx, L3rhocp_L2, L2S1, L2k, L2rhocp, L2F, L2V, _BC)


def compute_Level3_step(
    _L3carry,
    properties,
    substrate,
    _L3sub,
    _L2sub,
    subcycle,
    Levels,
    laser_position,
    ne_nn,
    laserP,
    L2T,
    _L2carry,
    LInterp,
):
    # Compute Level 3 material properties
    L3S1, L3S2, L3k, L3rhocp = computeStateProperties(
        _L3carry[0], _L3carry[1], properties, substrate[3]
    )

    # Determine laser index for this Level 3 substep
    LLidx = _L3sub + _L2sub * subcycle[1]

    # Compute laser source term for Level 3
    L3F = computeSourcesL3(
        Levels[3], laser_position[LLidx, :], ne_nn, properties, laserP[LLidx]
    )

    # Add convection, radiation, and evaporation boundary conditions
    L3F = computeConvRadBC(Levels[3], _L3carry[0], ne_nn[1], ne_nn[4], properties, L3F)

    # Interpolate Level 2 temperature to Level 3 boundary
    alpha_L3 = (_L3sub + 1) / subcycle[4]
    beta_L3 = 1 - alpha_L3
    _BC = alpha_L3 * L2T + beta_L3 * _L2carry[0]

    # Solve Level 3 temperature
    L3T = computeSolutions_L3(
        _BC,
        LInterp[1],
        Levels,
        ne_nn,
        _L3carry[0],
        L3F,
        L3k,
        L3rhocp,
        laser_position[LLidx, 5],
    )
    L3T = jnp.maximum(properties["T_amb"], L3T)  # TFSP

    return (L3S1, L3S2, LLidx, L3T)

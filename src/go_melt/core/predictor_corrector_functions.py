from functools import partial
import jax
import jax.numpy as jnp
import copy
from go_melt.utils.interpolation_functions import interpolatePoints
from go_melt.utils.helper_functions import getOverlapRegion
from .phase_state_functions import updateStateProperties, computeStateProperties
from .boundary_condition_functions import (
    computeConvRadBC,
    get_surface_faces,
)
from .move_mesh_functions import moveEverything
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
from .data_structures import (
    L2Carry_Predictor,
    L2Carry_Corrector,
    L3Carry_Predictor,
    L3Carry_Corrector,
    SubcycleContext_Predictor,
    SubcycleContext_Corrector,
)
from go_melt.io.save_results_functions import record_first_call

# TFSP: Temporary fix for single precision


# @record_first_call("stepGOMELT")
@partial(
    jax.jit,
    static_argnames=[
        "ne_nn",
        "tmp_ne_nn",
        "substrate",
        "record_accum",
        "LPBF_indicator",
    ],
)
def stepGOMELT(
    Levels: list[dict],
    ne_nn: tuple[int],
    tmp_ne_nn: tuple[int],
    Shapes: list[list],
    LInterp: list[list],
    laser_position: jnp.ndarray,
    properties: dict,
    dt: jnp.ndarray,
    laser_power: jnp.ndarray,
    substrate: tuple[int],
    max_accum_time: jnp.ndarray,
    accum_time: jnp.ndarray,
    record_accum: int,
    LPBF_indicator: bool,
) -> tuple[dict, jnp.ndarray, jnp.ndarray]:
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
    """
    L3T_prev = copy.deepcopy(Levels[3]["T0"])

    # Update material state and thermal properties
    Levels, Lk, Lrhocp = updateStateProperties(Levels, properties, substrate)
    if LPBF_indicator:
        Levels[1]["active"] = Levels[1]["active"].at[: tmp_ne_nn[1]].set(True)
        Levels[2]["active"] = Levels[2]["active"].at[:].set(True)
        Levels[3]["active"] = Levels[3]["active"].at[:].set(True)
    else:
        Levels[1]["active"] = Levels[1]["S1"] == 1
        Levels[2]["active"] = Levels[2]["S1"] == 1
        Levels[3]["active"] = Levels[3]["S1"] == 1

    for _ in range(1, 4):
        Levels[_]["S1ele"], Levels[_]["S1faces"] = get_surface_faces(
            Levels[_], Levels[_]["active"], ne_nn[0][_], ne_nn[2][_]
        )

    # Unpack levels for clarity
    L1, L2, L3 = Levels[1], Levels[2], Levels[3]

    # --- Compute source terms ---
    Fc, Fm, Ff = computeSources(
        L3, laser_position, Shapes, ne_nn, properties, laser_power
    )
    Fc = computeConvRadBC(L1, L1["T0"], ne_nn[0][1], ne_nn[1][1], properties, Fc)
    Fm = computeConvRadBC(L2, L2["T0"], ne_nn[0][2], ne_nn[1][2], properties, Fm)
    Ff = computeConvRadBC(L3, L3["T0"], ne_nn[0][3], ne_nn[1][3], properties, Ff)
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

    if record_accum == 1:
        accum_L3 = accum_time[Levels[0]["idx"]]
        max_accum_L3 = max_accum_time[Levels[0]["idx"]]

        accum_L3 += (
            jnp.minimum(
                jnp.maximum(
                    jnp.maximum(Levels[3]["T0"], L3T_prev) - properties["T_liquidus"],
                    0,
                )
                / (jnp.abs(Levels[3]["T0"] - L3T_prev) + 1e-6),
                1,
            )
            * dt
        )

        max_accum_L3 = jnp.maximum(max_accum_L3, accum_L3)
        # Reset where L3T_all < T_liquidus
        accum_L3 = jnp.where(Levels[3]["T0"] < properties["T_liquidus"], 0, accum_L3)
        max_accum_time = max_accum_time.at[Levels[0]["idx"]].set(max_accum_L3)
        accum_time = accum_time.at[Levels[0]["idx"]].set(accum_L3)

    return Levels, max_accum_time, accum_time


# @record_first_call("subcycleGOMELT")
@partial(
    jax.jit,
    static_argnames=["ne_nn", "tmp_ne_nn", "substrate", "subcycle", "record_accum"],
)
def subcycleGOMELT(
    Levels: list[dict],
    ne_nn: tuple[int],
    substrate: tuple[int],
    LInterp: list[list],
    tmp_ne_nn: tuple[int],
    laser_position_all: jnp.ndarray,
    properties: dict,
    subcycle: tuple[int, int, int, float, float, float, int],
    max_accum_time: jnp.ndarray,
    accum_time: jnp.ndarray,
    laser_start,
    move_hist,
    L1L2Eratio,
    L2L3Eratio,
    record_accum,
) -> tuple[list[dict], jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Perform a full predictor-corrector subcycling step for the GO-MELT model.

    This function executes the multilevel thermal update across Levels 1, 2, and 3,
    using nested subcycling and subgrid scale corrections. It includes both the
    predictor and corrector phases, updating temperature fields and phase states.
    """
    for _tool_path_loop in range(subcycle[-1]):
        _tool_path_idx = _tool_path_loop * subcycle[2] + jnp.arange(subcycle[2])
        laser_position = laser_position_all[_tool_path_idx, :]
        (Levels, Shapes, LInterp, move_hist) = moveEverything(
            laser_position[0, :],
            laser_start,
            Levels,
            move_hist,
            LInterp,
            L1L2Eratio,
            L2L3Eratio,
            properties["layer_height"],
        )

        laser_powers = laser_position[:, 6]
        if record_accum == 1:
            IC3 = Levels[3]["T0"][None, :]
        # Predictor Phase
        L3rhocp_L1, L2rhocp_L1, L1k, L1rhocp, L1F, L1V, L1T = computeLevel1predictor(
            Levels,
            substrate,
            properties,
            Shapes,
            ne_nn,
            tmp_ne_nn,
            laser_position,
            laser_powers,
        )

        # --- Subcycle Level 2 ---
        init_L2carry = L2Carry_Predictor(
            Levels[2]["T0"],
            Levels[2]["S1"],
            Levels[3]["T0"],
            Levels[3]["Tprime0"],
            Levels[3]["S1"],
        )

        ctx = SubcycleContext_Predictor(
            Levels,
            ne_nn,
            Shapes,
            substrate,
            LInterp,
            laser_position,
            laser_powers,
            subcycle,
            properties,
            L1T,
        )
        result_L2carry, history_L2carry = jax.lax.scan(
            lambda carry, sub: subcycleL2_Part1(carry, sub, ctx),
            init_L2carry,
            jnp.arange(subcycle[0]),
        )
        L2T = result_L2carry.T0
        L3Tp = result_L2carry.L3Tprime0
        L3Tp_L2 = history_L2carry.L3Tprime0
        L2Tp, L1T = getNewTprime(Levels[2], L2T, L1T, Levels[1], LInterp[0])

        # Corrector Phase
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

        init_L2carry = L2Carry_Corrector(
            Levels[2]["T0"],
            Levels[2]["S1"],
            Levels[3]["T0"],
            Levels[3]["Tprime0"],
            Levels[3]["S1"],
            Levels[3]["S2"],
        )

        ctx = SubcycleContext_Corrector(
            Levels,
            ne_nn,
            Shapes,
            substrate,
            LInterp,
            laser_position,
            laser_powers,
            subcycle,
            properties,
            L1T,
            L3Tp_L2,
        )
        final_L2carry, final_history_L2carry = jax.lax.scan(
            lambda carry, sub: subcycleL2_Part2(carry, sub, ctx),
            init_L2carry,
            jnp.arange(subcycle[0]),
        )

        # Final carry → update dictionaries
        Levels[2]["T0"] = final_L2carry.T0
        Levels[2]["S1"] = final_L2carry.S1
        Levels[3]["T0"] = final_L2carry.L3T0
        Levels[3]["Tprime0"] = final_L2carry.L3Tprime0
        Levels[3]["S1"] = final_L2carry.L3S1
        Levels[3]["S2"] = final_L2carry.L3S2

        L2all = final_history_L2carry.T0  # full time series of Level 2 temperatures
        L3all = final_history_L2carry.L3T0  # full time series of Level 3 temperatures
        L3pall = final_history_L2carry.L3Tprime0  # full time series of Level 3 Tprime

        Levels[2]["Tprime0"], Levels[1]["T0"] = getNewTprime(
            Levels[2], Levels[2]["T0"], L1T, Levels[1], LInterp[0]
        )
        Levels[0]["S1"] = Levels[0]["S1"].at[Levels[0]["idx"]].set(Levels[3]["S1"])
        Levels[0]["S2"] = Levels[0]["S2"].at[:].set(False)
        Levels[0]["S2"] = Levels[0]["S2"].at[Levels[0]["idx"]].set(Levels[3]["S2"])

        if record_accum == 1:
            L3all = jnp.vstack([IC3, L3all.reshape([-1, L3all.shape[-1]])])

            accum_L3 = accum_time[Levels[0]["idx"]]
            max_accum_L3 = max_accum_time[Levels[0]["idx"]]

            accum_L3 += (
                jnp.minimum(
                    jnp.maximum(
                        jnp.maximum(L3all[:-1, :], L3all[1:, :])
                        - properties["T_liquidus"],
                        0,
                    )
                    / (jnp.abs(jnp.diff(L3all, axis=0)) + 1e-6),
                    1,
                )
                * laser_position[:, 5, None]
            ).sum(axis=0)

            max_accum_L3 = jnp.maximum(max_accum_L3, accum_L3)

            # Reset where L3T_all < T_liquidus
            accum_L3 = jnp.where(
                Levels[3]["T0"] < properties["T_liquidus"], 0, accum_L3
            )

            max_accum_time = max_accum_time.at[Levels[0]["idx"]].set(max_accum_L3)
            accum_time = accum_time.at[Levels[0]["idx"]].set(accum_L3)

    return Levels, L2all, L3pall, move_hist, LInterp, max_accum_time, accum_time


# @record_first_call("computeLevel1predictor")
@partial(jax.jit, static_argnames=["ne_nn", "tmp_ne_nn", "substrate"])
def computeLevel1predictor(
    Levels,
    substrate,
    properties,
    Shapes,
    ne_nn,
    tmp_ne_nn,
    laser_position,
    laser_powers,
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
        Levels, ne_nn, laser_position, Shapes[1], properties, laser_powers
    )
    L1F = computeConvRadBC(
        Levels[1], Levels[1]["T0"], tmp_ne_nn[0], ne_nn[1][1], properties, L1F
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


@partial(jax.jit, static_argnames=["ne_nn", "substrate", "subcycle"])
def compute_Level2_step(
    _L2sub: int,
    subcycle: tuple[int, int, int, float, float, float, int],
    _L2carry: L2Carry_Predictor,
    properties: dict,
    substrate: tuple[int],
    Levels: list[dict],
    ne_nn: tuple[int],
    laser_position: jnp.ndarray,
    Shapes: list[list],
    laser_powers: jnp.ndarray,
    L1T: jnp.ndarray,
):
    # --- Boundary interpolation weights ---
    alpha_L2 = (_L2sub + 1) / subcycle[3]
    beta_L2 = 1 - alpha_L2

    # --- Laser substeps for this Level 2 subcycle ---
    Lidx = _L2sub * subcycle[1] + jnp.arange(subcycle[1])

    # --- Material Properties ---
    # Level 3 properties using current Level 3 temperature and phase
    _, _, L3k_L2, L3rhocp_L2 = computeStateProperties(
        _L2carry.L3T0, _L2carry.L3S1, properties, substrate[3]
    )

    # Level 2 properties using current Level 2 temperature and phase
    L2S1, _, L2k, L2rhocp = computeStateProperties(
        _L2carry.T0, _L2carry.S1, properties, substrate[2]
    )

    # --- Source Term ---
    L2F = computeLevelSource(
        Levels,
        ne_nn,
        laser_position[Lidx, :],
        Shapes[2],
        properties,
        laser_powers[Lidx],
    )
    L2F = computeConvRadBC(
        Levels[2],
        _L2carry.T0,
        ne_nn[0][2],
        ne_nn[1][2],
        properties,
        L2F,
    )

    # --- Subgrid Correction ---
    L2V = computeL2TprimeTerms_Part1(Levels, ne_nn, _L2carry.L3Tprime0, L3k_L2, Shapes)

    # --- Boundary condition interpolation ---
    _BC = alpha_L2 * L1T + beta_L2 * Levels[1]["T0"]

    return (Lidx, L3rhocp_L2, L2S1, L2k, L2rhocp, L2F, L2V, _BC)


@partial(jax.jit, static_argnames=["ne_nn", "substrate", "subcycle"])
def compute_Level3_step(
    _L3carry,
    properties: dict,
    substrate: tuple[int],
    _L3sub: int,
    _L2sub: int,
    subcycle: tuple[int, int, int, float, float, float, int],
    Levels: list[dict],
    laser_position: jnp.ndarray,
    ne_nn: tuple[int],
    laser_powers: jnp.ndarray,
    L2T: jnp.ndarray,
    _L2carry: L2Carry_Predictor,
    LInterp: list[list],
):
    # --- Material Properties ---
    L3S1, L3S2, L3k, L3rhocp = computeStateProperties(
        _L3carry[0], _L3carry[1], properties, substrate[3]
    )

    # --- Laser index for this Level 3 substep ---
    LLidx = _L3sub + _L2sub * subcycle[1]

    # --- Source Term ---
    L3F = computeSourcesL3(
        Levels[3], laser_position[LLidx, :], ne_nn, properties, laser_powers[LLidx]
    )
    L3F = computeConvRadBC(
        Levels[3], _L3carry[0], ne_nn[0][3], ne_nn[1][3], properties, L3F
    )

    # --- Boundary condition interpolation ---
    alpha_L3 = (_L3sub + 1) / subcycle[4]
    beta_L3 = 1 - alpha_L3
    _BC = alpha_L3 * L2T + beta_L3 * _L2carry.T0

    # --- Solve Level 3 temperature ---
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


def subcycleL2_Part1(
    _L2carry: L2Carry_Predictor, _L2sub: int, ctx: SubcycleContext_Predictor
):
    (
        Levels,
        ne_nn,
        Shapes,
        substrate,
        LInterp,
        laser_position,
        laser_powers,
        subcycle,
        properties,
        L1T,
    ) = ctx

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
        laser_powers,
        L1T,
    )

    # --- Temperature Solve ---
    # Solve Level 2 temperature using matrix-free FEM
    L2T = computeL2Temperature(
        _BC,
        LInterp[0],
        Levels,
        ne_nn,
        _L2carry.T0,
        L2F,
        L2V,
        L2k,
        L2rhocp,
        laser_position[Lidx, 5].sum(),
    )
    L2T = jnp.maximum(properties["T_amb"], L2T)  # TFSP

    # --- Subcycle Level 3 ---
    def subcycleL3_Part1(_L3carry: L3Carry_Predictor, _L3sub: int):
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
            laser_powers,
            L2T,
            _L2carry,
            LInterp,
        )

        return L3Carry_Predictor(L3T, L3S1), L3Carry_Predictor(L3T, L3S1)

    # Run Level 3 subcycling loop
    final_L3carry, _ = jax.lax.scan(
        subcycleL3_Part1,
        L3Carry_Predictor(_L2carry.L3T0, _L2carry.L3S1),
        jnp.arange(subcycle[1]),
    )

    # Compute Updated Level 3 Tprime and update Level 2 Temperature
    L3Tp, L2T = getNewTprime(Levels[3], final_L3carry.T0, L2T, Levels[2], LInterp[1])

    # Return updated L2 predictor carry
    next_L2 = L2Carry_Predictor(L2T, L2S1, final_L3carry.T0, L3Tp, final_L3carry.S1)
    return next_L2, next_L2


def subcycleL2_Part2(
    _L2carry: L2Carry_Corrector, _L2sub: int, ctx: SubcycleContext_Corrector
):
    (
        Levels,
        ne_nn,
        Shapes,
        substrate,
        LInterp,
        laser_position,
        laser_powers,
        subcycle,
        properties,
        L1T,
        L3Tp_L2,
    ) = ctx
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
        laser_powers,
        L1T,
    )

    # Add time derivative correction from Level 3 to Level 2
    L2V = computeL2TprimeTerms_Part2(
        Levels,
        ne_nn,
        L3Tp_L2[_L2sub],
        _L2carry.L3Tprime0,
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
        _L2carry.T0,
        L2F,
        L2V,
        L2k,
        L2rhocp,
        laser_position[Lidx, 5].sum(),
    )
    L2T = jnp.maximum(properties["T_amb"], L2T)  # TFSP

    # --- Subcycle Level 3 ---
    def subcycleL3_Part2(_L3carry: L3Carry_Corrector, _L3sub: int):
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
            laser_powers,
            L2T,
            _L2carry,
            LInterp,
        )

        return (
            L3Carry_Corrector(L3T, L3S1, L3S2),
            L3Carry_Corrector(L3T, L3S1, L3S2),
        )

    # Run Level 3 subcycling loop with structured init carry
    final_L3carry, final_history_L3carry = jax.lax.scan(
        subcycleL3_Part2,
        L3Carry_Corrector(_L2carry.L3T0, _L2carry.L3S1, _L2carry.L3S2),
        jnp.arange(subcycle[1]),
    )

    # Compute updated Tprime for Level 3 and temperature for Level 2
    L3Tp, L2T = getNewTprime(Levels[3], final_L3carry.T0, L2T, Levels[2], LInterp[1])

    # Return updated Level 2/3 state as L2Carry_Corrector
    next_L2 = L2Carry_Corrector(
        L2T,
        L2S1,
        final_L3carry.T0,
        L3Tp,
        final_L3carry.S1,
        final_L3carry.S2,
    )
    hist_L2 = L2Carry_Corrector(
        L2T,
        L2S1,
        final_history_L3carry.T0,
        L3Tp,
        final_history_L3carry.S1,
        final_history_L3carry.S2,
    )
    return next_L2, hist_L2

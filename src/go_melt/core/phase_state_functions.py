from functools import partial
import jax
import jax.numpy as jnp
from go_melt.utils.interpolation_functions import interpolatePoints
from go_melt.utils.helper_functions import getOverlapRegion


@partial(jax.jit, static_argnames=["substrate"])
def updateStateProperties(
    Levels: list[dict], properties: dict, substrate: tuple[int]
) -> tuple[list[dict], list[jnp.ndarray], list[jnp.ndarray]]:
    """
    Update material state fields (S1, S2) and compute thermal properties (k, rhocp)
    for all levels based on current temperature and substrate configuration.

    This function:
      • Updates melt state indicators (S1, S2) for Levels 1-3.
      • Computes temperature-dependent thermal conductivity and heat capacity.
      • Interpolates state from Level 2 to Level 1 to maintain consistency.
      • Applies substrate override to enforce solid state in substrate region.
    """
    # --- Level 3: Fine scale ---
    Levels[3]["S1"], Levels[3]["S2"], L3k, L3rhocp = computeStateProperties(
        Levels[3]["T0"], Levels[3]["S1"], properties, substrate[3]
    )

    # --- Level 2: Meso scale ---
    Levels[2]["S1"], L2S2, L2k, L2rhocp = computeStateProperties(
        Levels[2]["T0"], Levels[2]["S1"], properties, substrate[2]
    )

    # --- Interpolate S1 from Level 2 to Level 1 ---
    interpolated_S1 = interpolatePoints(
        Levels[2], Levels[2]["S1"], Levels[2]["overlapCoords"]
    )
    overlap_idx_L1 = getOverlapRegion(
        Levels[2]["overlapNodes"], Levels[1]["nodes"][0], Levels[1]["nodes"][1]
    )
    Levels[1]["S1"] = Levels[1]["S1"].at[overlap_idx_L1].set(interpolated_S1)

    # Enforce solid state in substrate region of Level 1
    Levels[1]["S1"] = Levels[1]["S1"].at[: substrate[1]].set(1)

    # --- Level 1: Coarse scale ---
    _, _, L1k, L1rhocp = computeStateProperties(
        Levels[1]["T0"], Levels[1]["S1"], properties, substrate[1]
    )

    # Return updated Levels and thermal properties (0-indexed for alignment)
    return Levels, [0, L1k, L2k, L3k], [0, L1rhocp, L2rhocp, L3rhocp]


def computeStateProperties(
    temperature: jnp.ndarray,
    bulk_phase_indicator: jnp.ndarray,
    properties: dict,
    Level_nodes_substrate: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Update phase states and compute temperature-dependent properties.
    """
    # Phase indicators
    liquid_phase_indicator = temperature >= properties["T_liquidus"]  # Fluid
    mushy_phase_indicator = (temperature > properties["T_solidus"]) & (
        temperature < properties["T_liquidus"]
    )  # Mushy

    # Update bulk_phase_indicator: bulk if already bulk or now fluid
    bulk_phase_indicator = 1.0 * (
        (bulk_phase_indicator > 0.499) | liquid_phase_indicator
    )

    # Enforce solid state in substrate region
    bulk_phase_indicator = bulk_phase_indicator.at[:Level_nodes_substrate].set(1)

    # Thermal conductivity (W/mm·K)
    k_powder = (
        (1 - bulk_phase_indicator)
        * (1 - liquid_phase_indicator)
        * properties["k_powder"]
    )
    k_bulk = (
        bulk_phase_indicator
        * (1 - liquid_phase_indicator)
        * (properties["k_bulk_coeff_a1"] * temperature + properties["k_bulk_coeff_a0"])
    )
    k_fluid = liquid_phase_indicator * properties["k_fluid_coeff_a0"]
    k = (k_powder + k_bulk + k_fluid) / 1000  # Convert from W/(m*K) to W/(mm*K)

    # Volumetric heat capacity (J/mm³·K)
    cp_solid = (
        (1 - liquid_phase_indicator)
        * (1 - mushy_phase_indicator)
        * (
            properties["cp_solid_coeff_a1"] * temperature
            + properties["cp_solid_coeff_a0"]
        )
    )
    cp_mushy = mushy_phase_indicator * properties["cp_mushy"]
    cp_fluid = liquid_phase_indicator * properties["cp_fluid"]
    rhocp = properties["rho"] * (cp_solid + cp_mushy + cp_fluid)

    return bulk_phase_indicator, liquid_phase_indicator, k, rhocp

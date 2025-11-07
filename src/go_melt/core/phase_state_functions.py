from functools import partial
import jax
from go_melt.utils.interpolation_functions import interpolatePoints
from go_melt.utils.helper_functions import getOverlapRegion


@partial(jax.jit, static_argnames=["substrate"])
def updateStateProperties(Levels, properties, substrate):
    """
    Update material state fields (S1, S2) and compute thermal properties (k, rhocp)
    for all levels based on current temperature and substrate configuration.

    This function:
      • Updates melt state indicators (S1, S2) for Levels 1-3.
      • Computes temperature-dependent thermal conductivity and heat capacity.
      • Interpolates state from Level 2 to Level 1 to maintain consistency.
      • Applies substrate override to enforce solid state in substrate region.

    Parameters:
    Levels (dict): Multilevel mesh and field data.
    properties (dict): Material and simulation properties.
    substrate (list): List of substrate node indices for each level.

    Returns:
    tuple:
        - Levels (dict): Updated with new S1, S2, and interpolated state.
        - Lk (list): Thermal conductivity for Levels 1-3 (index-aligned with Levels).
        - Lrhocp (list): Volumetric heat capacity for Levels 1-3 (index-aligned).
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


def computeStateProperties(T, S1, properties, Level_nodes_substrate):
    """
    Update phase states and compute temperature-dependent properties.

    Parameters:
    T (array): Temperature field.
    S1 (array): Solid/powder state (0 = powder, 1 = bulk).
    properties (dict): Material properties.
    Level_nodes_substrate (int): Number of substrate nodes (always solid).

    Returns:
    tuple:
        - S1 (array): Updated solid/powder state.
        - S2 (array): Fluid state (0 = not fluid, 1 = fluid).
        - k (array): Thermal conductivity (W/mm·K).
        - rhocp (array): Volumetric heat capacity (J/mm³·K).
    """
    # Phase indicators
    S2 = T >= properties["T_liquidus"]  # Fluid
    S3 = (T > properties["T_solidus"]) & (T < properties["T_liquidus"])  # Mushy

    # Update S1: bulk if already bulk or now fluid
    S1 = 1.0 * ((S1 > 0.499) | S2)

    # Enforce solid state in substrate region
    S1 = S1.at[:Level_nodes_substrate].set(1)

    # Thermal conductivity (W/mm·K)
    k_powder = (1 - S1) * (1 - S2) * properties["k_powder"]
    k_bulk = (
        S1
        * (1 - S2)
        * (properties["k_bulk_coeff_a1"] * T + properties["k_bulk_coeff_a0"])
    )
    k_fluid = S2 * properties["k_fluid_coeff_a0"]
    k = (k_powder + k_bulk + k_fluid) / 1000  # Convert from W/m·K

    # Volumetric heat capacity (J/mm³·K)
    cp_solid = (
        (1 - S2)
        * (1 - S3)
        * (properties["cp_solid_coeff_a1"] * T + properties["cp_solid_coeff_a0"])
    )
    cp_mushy = S3 * properties["cp_mushy"]
    cp_fluid = S2 * properties["cp_fluid"]
    rhocp = properties["rho"] * (cp_solid + cp_mushy + cp_fluid)

    return S1, S2, k, rhocp

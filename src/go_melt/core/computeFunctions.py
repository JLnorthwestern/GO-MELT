from functools import partial

import dill
import jax
import jax.numpy as jnp
import numpy as np
import sys
from go_melt.utils.interpolation_functions import (
    interpolatePoints,
)


def calcNumNodes(elements):
    """
    Increment each of the three input elements by 1.

    Parameters:
    elements (list or tuple of int): A sequence of three integers, typically
    representing node indices or positions.

    Returns:
    list of int: A list where each element is incremented by 1.
    """
    return [elements[0] + 1, elements[1] + 1, elements[2] + 1]


def createMesh3D(x, y, z):
    """
    Generate nodal coordinates and connectivity matrices for a 3D mesh.

    Parameters:
    x (list): [x_min, x_max, num_nodes_x] — bounds and number of nodes in x.
    y (list): [y_min, y_max, num_nodes_y] — bounds and number of nodes in y.
    z (list): [z_min, z_max, num_nodes_z] — bounds and number of nodes in z.

    Returns:
    tuple:
        node_coords (list of jnp.ndarray): Coordinates along x, y, and z axes.
        connect (list of jnp.ndarray): Connectivity matrices for x, y, and z.
    """
    # Node positions in x, y, z
    nx, ny, nz = [jnp.linspace(*axis) for axis in (x, y, z)]

    # Connectivity in x-direction
    cx0 = jnp.arange(0, x[2] - 1).reshape(-1, 1)
    cx1 = jnp.arange(1, x[2]).reshape(-1, 1)
    nconn_x = jnp.concatenate([cx0, cx1, cx1, cx0, cx0, cx1, cx1, cx0], axis=1)

    # Connectivity in y-direction
    cy0 = jnp.arange(0, y[2] - 1).reshape(-1, 1)
    cy1 = jnp.arange(1, y[2]).reshape(-1, 1)
    nconn_y = jnp.concatenate([cy0, cy0, cy1, cy1, cy0, cy0, cy1, cy1], axis=1)

    # Connectivity in z-direction
    cz0 = jnp.arange(0, z[2] - 1).reshape(-1, 1)
    cz1 = jnp.arange(1, z[2]).reshape(-1, 1)
    nconn_z = jnp.concatenate([cz0, cz0, cz0, cz0, cz1, cz1, cz1, cz1], axis=1)

    return [nx, ny, nz], [nconn_x, nconn_y, nconn_z]


def save_object(obj, filename):
    """
    Save a Python object to a file using the dill serialization library.

    This function overwrites any existing file with the same name.

    Parameters:
    obj (any): The Python object to serialize and save.
    filename (str): The path to the file where the object will be saved.
    """
    with open(filename, "wb") as outp:
        dill.dump(obj, outp, dill.HIGHEST_PROTOCOL)


def calcStaticTmpNodesAndElements(L, v):
    """
    Calculate temporary element and node counts in Level 1 up to a given z-value.

    Parameters:
    L (list): List of level dictionaries (e.g., from SetupLevels).
    v (list or array): A 3D coordinate [x, y, z] used to filter nodes by z-value.

    Returns:
    tuple:
        tmp_ne (int): Estimated number of elements below or at z = v[2].
        tmp_nn (int): Estimated number of nodes below or at z = v[2].
    """
    # Mask for nodes in Level 1 where z <= v[2] + tolerance
    Level1_mask = L[1]["node_coords"][2] <= v[2] + 1e-5
    Level1_nn = sum(Level1_mask)

    # Calculate temporary number of elements and nodes
    tmp_ne = L[1]["elements"][0] * L[1]["elements"][1] * (Level1_nn - 1)
    tmp_ne = tmp_ne.tolist()
    tmp_nn = (L[1]["nodes"][0] * L[1]["nodes"][1] * Level1_nn).tolist()

    return (tmp_ne, tmp_nn)


def getSubstrateNodes(Levels):
    """
    Calculate the number of substrate nodes (z ≤ 0) for Levels 1 to 3.

    Parameters:
    Levels (list): A list of level dictionaries, each containing:
        - "node_coords": list of arrays for x, y, z coordinates.
        - "nodes": list of node counts in x, y, z directions.

    Returns:
    tuple: A tuple of substrate node counts for Levels 0 to 3,
           where Level 0 is always 0.
    """
    substrate = [
        ((L["node_coords"][2] < 1e-5).sum() * L["nodes"][0] * L["nodes"][1]).tolist()
        for L in Levels[:4]
    ]
    return tuple(substrate)


def getCoarseNodesInFineRegion(xnf, xnc):
    """
    Identify coarse grid nodes that overlap with the fine grid region.

    This function determines which coarse grid nodes fall within the
    spatial extent of a given fine grid. It assumes uniform spacing
    in the coarse grid.

    Parameters:
    xnf (array): Coordinates of the fine grid nodes.
    xnc (array): Coordinates of the coarse grid nodes.

    Returns:
    array: Indices of coarse grid nodes that overlap with the fine grid.
    """
    xfmin = xnf.min()
    xfmax = xnf.max()
    xcmin = xnc.min()
    xcmax = xnc.max()

    nnc = xnc.size
    nec = nnc - 1
    hc = (xcmax - xcmin) / nec

    overlapMin = jnp.round((xfmin - xcmin) / hc)
    overlapMax = jnp.round((xfmax - xcmin) / hc) + 1

    overlap = jnp.arange(overlapMin, overlapMax).astype(int)

    return overlap


def getCoarseNodesInLargeFineRegion(xnc, xnf):
    """
    Identify fine grid indices that correspond to coarse grid nodes
    when the fine grid spans a larger domain than the coarse grid.

    This function computes the indices of fine grid nodes that align
    with the coarse grid node positions, assuming both grids are
    uniformly spaced.

    Parameters:
    xnc (array): Coordinates of the coarse grid nodes.
    xnf (array): Coordinates of the fine grid nodes.

    Returns:
    array: Indices of fine grid nodes that align with coarse grid nodes.
    """
    xfmin = xnf.min()
    xfmax = xnf.max()
    xcmin = xnc.min()
    xcmax = xnc.max()

    nnf = xnf.size
    nef = nnf - 1
    hf = (xfmax - xfmin) / nef

    nnc = xnc.size
    nec = nnc - 1
    hc = (xcmax - xcmin) / nec

    overlapMin = jnp.round((xcmin - xfmin) / hf)
    overlapMax = jnp.round((xcmax - xfmin) / hf) + 1

    step = int(jnp.round(hc / hf))
    overlap = jnp.arange(overlapMin, overlapMax, step).astype(int)

    return overlap


@partial(jax.jit, static_argnames=["nn"])
def bincount(N, D, nn):
    """
    Perform a weighted bin count operation.

    This function accumulates values from `D` into bins specified by `N`,
    producing a 1D array of length `nn`.

    Parameters:
    N (array): Bin indices (integer array).
    D (array): Values to accumulate (same shape as N).
    nn (int): Total number of bins (output length).

    Returns:
    array: Binned sum of values, shape (nn,).
    """
    return jnp.bincount(N, D, length=nn)


@jax.jit
def getOverlapRegion(node_coords, nx, ny):
    """
    Compute flattened global node indices for a structured 3D grid.

    This function generates a 1D array of global node indices based on
    the Cartesian product of x, y, and z coordinate arrays. It assumes
    a structured grid with dimensions (nx, ny, nz).

    Parameters:
    node_coords (list): List of 1D arrays [x, y, z] representing node coordinates.
    nx (int): Number of nodes in the x-direction.
    ny (int): Number of nodes in the y-direction.

    Returns:
    array: Flattened global node indices corresponding to the overlap region.
    """
    _x = jnp.tile(
        node_coords[0], node_coords[1].shape[0] * node_coords[2].shape[0]
    ).reshape(-1)

    _y = jnp.repeat(
        jnp.tile(node_coords[1], node_coords[2].shape[0]), node_coords[0].shape[0]
    ).reshape(-1)

    _z = jnp.repeat(node_coords[2], node_coords[0].shape[0] * node_coords[1].shape[0])

    return _x + _y * nx + _z * nx * ny


@partial(jax.jit, static_argnames=["_idx", "_val"])
def substitute_Tbar(Tbar, _idx, _val):
    """
    Replace a slice of the Tbar array starting at a given index with a new value.

    This function sets all elements from index `_idx` to the end of the array
    to the value `_val`. The index and value are treated as static arguments
    for JAX compilation efficiency.

    Parameters:
    Tbar (array): Input array to be modified.
    _idx (int): Starting index for substitution.
    _val (float or array): Value(s) to assign from _idx onward.

    Returns:
    array: Modified Tbar array with values substituted from _idx onward.
    """
    return Tbar.at[_idx:].set(_val)


@jax.jit
def substitute_Tbar2(Tbar, _idx, _val):
    """
    Replace a single element in the Tbar array at a given index.

    This function sets the element at index `_idx` to `_val`.

    Parameters:
    Tbar (array): Input array to be modified.
    _idx (int): Index of the element to be replaced.
    _val (float): New value to assign at the specified index.

    Returns:
    array: Modified Tbar array with the specified element updated.
    """
    return Tbar.at[_idx].set(_val)


def calc_length_h(A):
    """
    Compute domain lengths and element sizes in each spatial direction.

    This function calculates the physical length of the domain and the
    corresponding element size (grid spacing) in the x, y, and z directions.

    Parameters:
    A (object): Mesh object with attributes:
                - bounds.x, bounds.y, bounds.z: tuples (min, max)
                - elements: tuple (nx, ny, nz) representing number of elements

    Returns:
    tuple:
        - [Lx, Ly, Lz]: Physical domain lengths in x, y, z directions.
        - [hx, hy, hz]: Element sizes in x, y, z directions.
    """
    # Domain length
    Lx = A.bounds.x[1] - A.bounds.x[0]
    Ly = A.bounds.y[1] - A.bounds.y[0]
    Lz = A.bounds.z[1] - A.bounds.z[0]
    # Element length
    hx = Lx / A.elements[0]
    hy = Ly / A.elements[1]
    hz = Lz / A.elements[2]

    return [Lx, Ly, Lz], [hx, hy, hz]


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


def getSampleCoords(Level):
    """
    Extracts the nodal coordinates of the first element in the mesh.

    Parameters:
    Level (dict): Mesh data for a given level.

    Returns:
    array: (n_nodes, 3) array of x, y, z coordinates.
    """
    x = Level["node_coords"][0][Level["connect"][0][0, :]].reshape(-1, 1)
    y = Level["node_coords"][1][Level["connect"][1][0, :]].reshape(-1, 1)
    z = Level["node_coords"][2][Level["connect"][2][0, :]].reshape(-1, 1)
    return jnp.concatenate([x, y, z], axis=1)


def printLevelMaxMin(Ls, Lnames):
    """
    Print the min and max temperatures for each level and check for invalid values.
    Terminates the program if any temperature is NaN, too low, or unreasonably high.
    """
    print("Temps:", end=" ")
    flag = False

    for i in range(1, len(Ls)):
        T = Ls[i]["T0"]
        Lmin = np.min(T)
        Lmax = np.max(T)

        # Use vectorized checks for invalid values
        if not np.isfinite(Lmax) or Lmax <= 0 or Lmax > 1e5:
            print(
                f"\nTerminating program: Lmax for {Lnames[i-1]} is NaN, 0, or invalid."
            )
            flag = True

        if not np.isfinite(Lmin) or Lmin <= 0 or Lmin > 1e5:
            print(
                f"\nTerminating program: Lmin for {Lnames[i-1]} is NaN, 0, or invalid."
            )
            flag = True

        print(f"{Lnames[i-1]}: [{Lmin:.2f}, {Lmax:.2f}]", end=" ")

    if flag:
        sys.exit(1)
    print("")


def melting_temp(temps, delt_T, T_melt, accum_time, idx):
    """
    Update accumulated melt time for nodes above melting temperature.

    Parameters:
    temps (array): Current temperature field.
    delt_T (float): Time step duration.
    T_melt (float): Melting temperature threshold.
    accum_time (array): Accumulated melt time array.
    idx (array): Indices of nodes to update.

    Returns:
    array: Updated accumulated melt time.
    """
    T_above_threshold = np.array(temps > T_melt)
    accum_time = accum_time.at[idx].add(T_above_threshold * delt_T)
    return accum_time

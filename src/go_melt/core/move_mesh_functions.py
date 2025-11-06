import jax
import jax.numpy as jnp
from go_melt.utils.interpolation_functions import (
    interpolatePoints,
    interpolatePointsMatrix,
)
from go_melt.utils.shape_functions import computeCoarseFineShapeFunctions
from .computeFunctions import getOverlapRegion


@jax.jit
def moveEverything(v, vstart, Levels, move_v, LInterp, L1L2Eratio, L2L3Eratio, height):
    """
    Move the multilevel mesh system in response to laser motion.

    This function updates node coordinates, interpolates temperature and subgrid
    fields, and recalculates shape functions for Levels 1-3 based on the laser's
    movement vector.

    Parameters:
    v (array): Current laser position.
    vstart (array): Initial laser position.
    Levels (dict): Multilevel mesh and field data.
    move_v (array): Previous movement vector (used for scaling).
    LInterp (list): Interpolation matrices between levels.
    L1L2Eratio (list): Element ratio from Level 1 to Level 2 in [x, y, z].
    L2L3Eratio (list): Element ratio from Level 2 to Level 3 in [x, y, z].
    height (float): Layer height in mm.

    Returns:
    tuple:
        - Levels (dict): Updated mesh and field data.
        - Shapes (list): Updated shape function data.
        - LInterp (list): Updated interpolation matrices.
        - move_v (array): Updated movement vector.
    """
    # --- Move Level 3 (finest) ---
    vtot = v - vstart
    v_L3_constrained = jit_constrain_v(vtot, Levels[3])

    # Move mesh and update coordinates
    new_coords_L3, _ = move_fine_mesh(
        Levels[3]["init_node_coors"], Levels[2]["h"], v_L3_constrained
    )
    Levels[3] = update_overlap_nodes_coords(
        Levels[3], v_L3_constrained, Levels[2]["h"], [1, 1, 1]
    )

    # Interpolate temperature and subgrid fields
    Levels[3]["T0"] = interpolatePoints(Levels[1], Levels[1]["T0"], new_coords_L3)
    Tprime_L2 = interpolatePoints(Levels[2], Levels[2]["Tprime0"], new_coords_L3)
    Tprime_L3 = interpolatePoints(Levels[3], Levels[3]["Tprime0"], new_coords_L3)
    Levels[3]["T0"] += Tprime_L2 + Tprime_L3
    Levels[3]["Tprime0"] = Tprime_L3
    Levels[3]["node_coords"] = new_coords_L3

    # --- Move Level 2 ---
    v_L2_constrained = jit_constrain_v(vtot, Levels[2])
    h_L1 = Levels[1]["h"][:2]
    h_L1.append(jnp.array(height))
    new_coords_L2, move_v = move_fine_mesh(
        Levels[2]["init_node_coors"], h_L1, v_L2_constrained
    )
    move_v = [move_v[i] * L1L2Eratio[i] for i in range(3)]

    Levels[2] = update_overlap_nodes_coords_L1L2(
        Levels[2], v_L2_constrained, Levels[1]["h"], height
    )

    # Interpolate temperature and subgrid fields
    Levels[2]["T0"] = interpolatePoints(Levels[1], Levels[1]["T0"], new_coords_L2)
    Levels[2]["Tprime0"] = interpolatePoints(
        Levels[2], Levels[2]["Tprime0"], new_coords_L2
    )
    Levels[2]["T0"] += Levels[2]["Tprime0"]
    Levels[2]["node_coords"] = new_coords_L2

    # Recompute shape functions and interpolation matrix
    L2L1Shape = computeCoarseFineShapeFunctions(Levels[1], Levels[2])
    LInterp[0] = interpolatePointsMatrix(Levels[1], new_coords_L2)

    # --- Update Level 3 overlap nodes after Level 2 move ---
    Levels[3]["overlapNodes"] = [
        Levels[3]["overlapNodes"][i] - move_v[i] for i in range(3)
    ]

    # --- Update Level 0 (global) ---
    Levels[0] = update_overlap_nodes_coords(
        Levels[0], v_L3_constrained, Levels[2]["h"], L2L3Eratio
    )
    Levels[0] = update_overlap_nodes_coords_L2(
        Levels[0],
        v_L2_constrained,
        [Levels[1]["h"][0], Levels[1]["h"][1], Levels[2]["h"][2]],
        [L1L2Eratio[0] * L2L3Eratio[0], L1L2Eratio[1] * L2L3Eratio[1], L2L3Eratio[2]],
    )

    # Track z-direction movement only for Level 0
    Levels[0]["overlapNodes"][2] -= move_v[2] * L2L3Eratio[2]
    Levels[0]["overlapNodes_L2"][2] -= move_v[2] * L2L3Eratio[2]

    # Update overlap indices
    Levels[0]["idx"] = getOverlapRegion(
        Levels[0]["overlapNodes"], Levels[0]["nodes"][0], Levels[0]["nodes"][1]
    )
    Levels[0]["idx_L2"] = getOverlapRegion(
        Levels[0]["overlapNodes_L2"], Levels[0]["nodes"][0], Levels[0]["nodes"][1]
    )

    # Update subgrid state fields
    Levels[2]["S1"] = Levels[2]["S1"].at[:].set(Levels[0]["S1"][Levels[0]["idx_L2"]])
    Levels[3]["S1"] = Levels[3]["S1"].at[:].set(Levels[0]["S1"][Levels[0]["idx"]])
    Levels[3]["S2"] = Levels[3]["S2"].at[:].set(Levels[0]["S2"][Levels[0]["idx"]])

    # Recompute shape functions for updated meshes
    L3L1Shape = computeCoarseFineShapeFunctions(Levels[1], Levels[3])
    L3L2Shape = computeCoarseFineShapeFunctions(Levels[2], Levels[3])
    LInterp[1] = interpolatePointsMatrix(Levels[2], new_coords_L3)

    Shapes = [L2L1Shape, L3L1Shape, L3L2Shape]
    return Levels, Shapes, LInterp, move_v


@jax.jit
def move_fine_mesh(node_coords, element_size, v):
    """
    Shift a structured fine mesh in space based on a displacement vector.

    This function computes the new coordinates of a structured fine mesh
    by translating it in the x, y, and z directions according to the
    displacement vector `v`, scaled by the element size.

    Parameters:
    node_coords (list): Original node coordinates [x, y, z] as 1D arrays.
    element_size (list): Element sizes [hx, hy, hz] in each direction.
    v (array): Displacement vector [vx, vy, vz].

    Returns:
    list: New node coordinates [xnf_x, xnf_y, xnf_z] after translation.
    list: Integer shift indices [vx_, vy_, vz_] used for the translation.
    """
    vx_ = (v[0] / element_size[0] + 1e-2).astype(int)
    vy_ = (v[1] / element_size[1] + 1e-2).astype(int)
    vz_ = (v[2] / element_size[2] + 1e-2).astype(int)

    xnf_x = node_coords[0] + element_size[0] * vx_
    xnf_y = node_coords[1] + element_size[1] * vy_
    xnf_z = node_coords[2] + element_size[2] * vz_

    return [xnf_x, xnf_y, xnf_z], [vx_, vy_, vz_]


def find_max_const(CoarseLevel, FinerLevel):
    """
    Compute the maximum number of elements the finer level domain can move
    within the bounds of the coarser level domain in each direction.

    This function calculates how many elements the finer mesh can shift
    in the positive and negative directions (east/west, north/south, top/bottom)
    without exceeding the bounds of the coarser mesh.

    Parameters:
    CoarseLevel (object): Object with `bounds` attribute containing:
                          - bounds.x, bounds.y, bounds.z: tuples (min, max)
    FinerLevel (object): Object with `bounds` attribute structured similarly.

    Returns:
    tuple: Three lists representing allowable movement in:
           - x-direction: [west, east]
           - y-direction: [south, north]
           - z-direction: [bottom, top]
    """
    iE = CoarseLevel.bounds.x[1] - FinerLevel.bounds.x[1]  # Elements to east
    iN = CoarseLevel.bounds.y[1] - FinerLevel.bounds.y[1]  # Elements to north
    iT = CoarseLevel.bounds.z[1] - FinerLevel.bounds.z[1]  # Elements to top
    iW = CoarseLevel.bounds.x[0] - FinerLevel.bounds.x[0]  # Elements to west
    iS = CoarseLevel.bounds.y[0] - FinerLevel.bounds.y[0]  # Elements to south
    iB = CoarseLevel.bounds.z[0] - FinerLevel.bounds.z[0]  # Elements to bottom

    return [iW, iE], [iS, iN], [iB, iT]


@jax.jit
def jit_constrain_v(vtot, Level):
    """
    Constrain a 3D vector within the bounding box defined in the mesh level.

    This function clips each component of the input vector `vtot` to lie
    within the bounds specified in `Level["bounds"]`.

    Parameters:
    vtot (list): A list of 3 elements [vx, vy, vz] representing a 3D vector.
    Level (dict): Contains bounding box limits under:
                  - Level["bounds"]["ix"]: (xmin, xmax)
                  - Level["bounds"]["iy"]: (ymin, ymax)
                  - Level["bounds"]["iz"]: (zmin, zmax)

    Returns:
    list: Clipped 3D vector [vx_clipped, vy_clipped, vz_clipped].
    """
    vtot = [
        jnp.clip(vtot[0], Level["bounds"]["ix"][0], Level["bounds"]["ix"][1]),
        jnp.clip(vtot[1], Level["bounds"]["iy"][0], Level["bounds"]["iy"][1]),
        jnp.clip(vtot[2], Level["bounds"]["iz"][0], Level["bounds"]["iz"][1]),
    ]
    return vtot


@jax.jit
def update_overlap_nodes_coords(Level, vcon, element_size, ele_ratio):
    """
    Update the overlap node indices and coordinates based on a displacement vector.

    This function shifts the overlap region of a mesh by updating both the
    node indices and physical coordinates using the displacement vector `vcon`.

    Parameters:
    Level (dict): Mesh level containing original overlap node indices and coordinates.
                  - "orig_overlap_nodes": list of original node index arrays [x, y, z].
                  - "orig_overlap_coors": list of original coordinate arrays [x, y, z].
    vcon (array): Displacement vector [vx, vy, vz].
    element_size (array): Element sizes [hx, hy, hz].
    ele_ratio (array): Ratio of coarse-to-fine elements in each direction.

    Returns:
    dict: Updated Level dictionary with new "overlapNodes" and "overlapCoords".
    """
    shift = [
        (vcon[0] / element_size[0] + 1e-2).astype(int),
        (vcon[1] / element_size[1] + 1e-2).astype(int),
        (vcon[2] / element_size[2] + 1e-2).astype(int),
    ]

    Level["overlapNodes"] = [
        Level["orig_overlap_nodes"][0] + ele_ratio[0] * shift[0],
        Level["orig_overlap_nodes"][1] + ele_ratio[1] * shift[1],
        Level["orig_overlap_nodes"][2] + ele_ratio[2] * shift[2],
    ]

    Level["overlapCoords"] = [
        Level["orig_overlap_coors"][0] + element_size[0] * shift[0],
        Level["orig_overlap_coors"][1] + element_size[1] * shift[1],
        Level["orig_overlap_coors"][2] + element_size[2] * shift[2],
    ]

    return Level


@jax.jit
def update_overlap_nodes_coords_L1L2(Level, vcon, element_size, powder_layer):
    """
    Update overlap node indices and coordinates for Level 1/2 with powder layer offset.

    This function shifts the overlap region of a mesh by updating both the
    node indices and physical coordinates using the displacement vector `vcon`,
    accounting for an additional powder layer in the z-direction.

    Parameters:
    Level (dict): Mesh level containing original overlap node indices and coordinates.
                  - "orig_overlap_nodes": list of original node index arrays [x, y, z].
                  - "orig_overlap_coors": list of original coordinate arrays [x, y, z].
    vcon (array): Displacement vector [vx, vy, vz].
    element_size (array): Element sizes [hx, hy, hz].
    powder_layer (float): Additional offset in the z-direction.

    Returns:
    dict: Updated Level dictionary with new "overlapNodes" and "overlapCoords".
    """
    shift_x = (vcon[0] / element_size[0] + 1e-2).astype(int)
    shift_y = (vcon[1] / element_size[1] + 1e-2).astype(int)
    shift_z = (vcon[2] / element_size[2] + 1e-2).astype(int)
    shift_z_p = powder_layer * (vcon[2] / powder_layer + 1e-2).astype(int)

    Level["overlapNodes"] = [
        Level["orig_overlap_nodes"][0] + shift_x,
        Level["orig_overlap_nodes"][1] + shift_y,
        Level["orig_overlap_nodes"][2] + shift_z,
    ]

    Level["overlapCoords"] = [
        Level["orig_overlap_coors"][0] + element_size[0] * shift_x,
        Level["orig_overlap_coors"][1] + element_size[1] * shift_y,
        Level["orig_overlap_coors"][2] + shift_z_p,
    ]

    return Level


@jax.jit
def update_overlap_nodes_coords_L2(Level, vcon, element_size, ele_ratio):
    """
    Update overlap node indices and coordinates for Level 2 based on displacement.

    This function shifts the overlap region of Level 2 by updating both the
    node indices and physical coordinates using the displacement vector `vcon`
    and the element-to-node ratio `ele_ratio`.

    Parameters:
    Level (dict): Mesh level containing original overlap node indices and coordinates.
                - "orig_overlap_nodes_L2": list of original node index arrays [x, y, z].
                - "orig_overlap_coors_L2": list of original coordinate arrays [x, y, z].
    vcon (array): Displacement vector [vx, vy, vz].
    element_size (array): Element sizes [hx, hy, hz].
    ele_ratio (array): Ratio of coarse-to-fine elements in each direction.

    Returns:
    dict: Updated Level dictionary with new "overlapNodes_L2" and "overlapCoords_L2".
    """
    shift = [
        (vcon[0] / element_size[0] + 1e-2).astype(int),
        (vcon[1] / element_size[1] + 1e-2).astype(int),
        (vcon[2] / element_size[2] + 1e-2).astype(int),
    ]

    Level["overlapNodes_L2"] = [
        Level["orig_overlap_nodes_L2"][0] + ele_ratio[0] * shift[0],
        Level["orig_overlap_nodes_L2"][1] + ele_ratio[1] * shift[1],
        Level["orig_overlap_nodes_L2"][2] + ele_ratio[2] * shift[2],
    ]

    Level["overlapCoords_L2"] = [
        Level["orig_overlap_coors_L2"][0] + element_size[0] * shift[0],
        Level["orig_overlap_coors_L2"][1] + element_size[1] * shift[1],
        Level["orig_overlap_coors_L2"][2] + element_size[2] * shift[2],
    ]

    return Level

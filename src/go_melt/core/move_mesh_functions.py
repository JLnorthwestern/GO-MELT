import jax
import jax.numpy as jnp
import numpy as np
from go_melt.utils.interpolation_functions import (
    interpolatePoints,
    interpolatePointsMatrix,
)
from go_melt.utils.shape_functions import computeCoarseFineShapeFunctions
from go_melt.utils.helper_functions import getOverlapRegion


@jax.jit
def moveEverything(
    laser_pos_current: jnp.ndarray,
    laser_pos_initial: np.ndarray,
    Levels: list[dict],
    elem_shift_from_initial: list[jnp.ndarray],
    interlevel_interp_mats: list[list[jnp.ndarray]],
    elem_size_ratios_L1_L2: list[int],
    elem_size_ratios_L2_L3: list[int],
    layer_thickness: float,
) -> tuple[dict, list, list[list[jnp.ndarray]], list[jnp.ndarray]]:
    """
    Move the multilevel mesh system in response to laser motion.

    This function updates node coordinates, interpolates temperature and subgrid
    fields, and recalculates shape functions for Levels 1-3 based on the laser's
    movement vector.
    """
    # --- Move Level 3 (finest) ---
    laser_shift_from_initial = laser_pos_current - laser_pos_initial
    displacement_L3_constrained = jit_constrain_v(laser_shift_from_initial, Levels[3])

    # Move mesh and update coordinates
    new_coords_L3, _ = move_fine_mesh(
        Levels[3]["init_node_coors"], Levels[2]["h"], displacement_L3_constrained
    )
    Levels[3] = update_overlap_nodes_coords(
        Levels[3], displacement_L3_constrained, Levels[2]["h"], ele_ratio=[1, 1, 1]
    )

    # Interpolate temperature and subgrid fields
    Levels[3]["T0"] = interpolatePoints(Levels[1], Levels[1]["T0"], new_coords_L3)
    Tprime_L2 = interpolatePoints(Levels[2], Levels[2]["Tprime0"], new_coords_L3)
    Tprime_L3 = interpolatePoints(Levels[3], Levels[3]["Tprime0"], new_coords_L3)
    Levels[3]["T0"] += Tprime_L2 + Tprime_L3
    Levels[3]["Tprime0"] = Tprime_L3
    Levels[3]["node_coords"] = new_coords_L3

    # --- Move Level 2 ---
    displacement_L2_constrained = jit_constrain_v(laser_shift_from_initial, Levels[2])
    elem_sizes_L1 = Levels[1]["h"][:2]
    elem_sizes_L1.append(jnp.array(layer_thickness))
    new_coords_L2, elem_shift_from_initial = move_fine_mesh(
        Levels[2]["init_node_coors"], elem_sizes_L1, displacement_L2_constrained
    )
    elem_shift_from_initial = [
        elem_shift_from_initial[i] * elem_size_ratios_L1_L2[i] for i in range(3)
    ]

    Levels[2] = update_overlap_nodes_coords_L1L2(
        Levels[2], displacement_L2_constrained, Levels[1]["h"], layer_thickness
    )

    # Interpolate temperature and subgrid fields
    Levels[2]["T0"] = interpolatePoints(Levels[1], Levels[1]["T0"], new_coords_L2)
    Levels[2]["Tprime0"] = interpolatePoints(
        Levels[2], Levels[2]["Tprime0"], new_coords_L2
    )
    Levels[2]["T0"] += Levels[2]["Tprime0"]
    Levels[2]["node_coords"] = new_coords_L2

    # Recompute shape functions and interpolation matrix
    shape_L2_to_L1 = computeCoarseFineShapeFunctions(Levels[1], Levels[2])
    interlevel_interp_mats[0] = interpolatePointsMatrix(Levels[1], new_coords_L2)

    # --- Update Level 3 overlap nodes after Level 2 move ---
    Levels[3]["overlapNodes"] = [
        Levels[3]["overlapNodes"][i] - elem_shift_from_initial[i] for i in range(3)
    ]

    # --- Update Level 0 (global) ---
    Levels[0] = update_overlap_nodes_coords(
        Levels[0], displacement_L3_constrained, Levels[2]["h"], elem_size_ratios_L2_L3
    )
    Levels[0] = update_overlap_nodes_coords_L2(
        Levels[0],
        displacement_L2_constrained,
        [Levels[1]["h"][0], Levels[1]["h"][1], Levels[2]["h"][2]],
        [
            elem_size_ratios_L1_L2[0] * elem_size_ratios_L2_L3[0],
            elem_size_ratios_L1_L2[1] * elem_size_ratios_L2_L3[1],
            elem_size_ratios_L2_L3[2],
        ],
    )

    # Track z-direction movement only for Level 0
    Levels[0]["overlapNodes"][2] -= (
        elem_shift_from_initial[2] * elem_size_ratios_L2_L3[2]
    )
    Levels[0]["overlapNodes_L2"][2] -= (
        elem_shift_from_initial[2] * elem_size_ratios_L2_L3[2]
    )

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
    shape_L3_to_L1 = computeCoarseFineShapeFunctions(Levels[1], Levels[3])
    shape_L3_to_L2 = computeCoarseFineShapeFunctions(Levels[2], Levels[3])
    interlevel_interp_mats[1] = interpolatePointsMatrix(Levels[2], new_coords_L3)

    Shapes = [shape_L2_to_L1, shape_L3_to_L1, shape_L3_to_L2]
    return (
        Levels,
        Shapes,
        interlevel_interp_mats,
        elem_shift_from_initial,
    )


@jax.jit
def move_fine_mesh(node_coords, element_size, laser_pos_current):
    """
    Shift a structured fine mesh in space based on a displacement vector.

    This function computes the new coordinates of a structured fine mesh
    by translating it in the x, y, and z directions according to the
    displacement vector `laser_pos_current`, scaled by the element size.

    Parameters:
    node_coords (list): Original node coordinates [x, y, z] as 1D arrays.
    element_size (list): Element sizes [hx, hy, hz] in each direction.
    laser_pos_current (array): Displacement vector [vx, vy, vz].

    Returns:
    list: New node coordinates [xnf_x, xnf_y, xnf_z] after translation.
    list: Integer shift indices [vx_, vy_, vz_] used for the translation.
    """
    vx_ = (laser_pos_current[0] / element_size[0] + 1e-2).astype(int)
    vy_ = (laser_pos_current[1] / element_size[1] + 1e-2).astype(int)
    vz_ = (laser_pos_current[2] / element_size[2] + 1e-2).astype(int)

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
def jit_constrain_v(laser_shift_from_initial, Level):
    """
    Constrain a 3D vector within the bounding box defined in the mesh level.

    This function clips each component of the input vector `laser_shift_from_initial` to lie
    within the bounds specified in `Level["bounds"]`.

    Parameters:
    laser_shift_from_initial (list): A list of 3 elements [vx, vy, vz] representing a 3D vector.
    Level (dict): Contains bounding box limits under:
                  - Level["bounds"]["ix"]: (xmin, xmax)
                  - Level["bounds"]["iy"]: (ymin, ymax)
                  - Level["bounds"]["iz"]: (zmin, zmax)

    Returns:
    list: Clipped 3D vector [vx_clipped, vy_clipped, vz_clipped].
    """
    laser_shift_from_initial = [
        jnp.clip(
            laser_shift_from_initial[0],
            Level["bounds"]["ix"][0],
            Level["bounds"]["ix"][1],
        ),
        jnp.clip(
            laser_shift_from_initial[1],
            Level["bounds"]["iy"][0],
            Level["bounds"]["iy"][1],
        ),
        jnp.clip(
            laser_shift_from_initial[2],
            Level["bounds"]["iz"][0],
            Level["bounds"]["iz"][1],
        ),
    ]
    return laser_shift_from_initial


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

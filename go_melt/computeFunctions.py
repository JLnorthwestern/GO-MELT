import copy
import json
import os
from functools import partial

import dill
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse
from jax.numpy import multiply
from pyevtk.hl import gridToVTK
import sys

# TFSP: Temporary fix for single precision


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


class obj:
    """
    A simple wrapper class to convert a dictionary into an object
    with attributes accessible via dot notation.

    Attributes are dynamically created from the keys and values
    of the input dictionary.

    Example:
        data = {'a': 1, 'b': 2}
        o = obj(data)
        print(o.a)  # Outputs: 1
    """

    def __init__(self, dict1):
        self.__dict__.update(dict1)


def dict2obj(dict1):
    """
    Convert a dictionary into an object with attribute-style access.

    This function serializes the dictionary to JSON and then deserializes
    it using a custom object hook to create an instance of the `obj` class.

    Parameters:
    dict1 (dict): The dictionary to convert.

    Returns:
    obj: An object with attributes corresponding to the dictionary keys.
    """
    return json.loads(json.dumps(dict1), object_hook=obj)


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


def SetupLevels(solver_input, properties):
    """
    Set up multilevel mesh structures and initialize physical fields.

    This function constructs four hierarchical mesh levels (Level0-Level3),
    computes geometric and physical properties, and identifies overlapping
    regions between levels for multiscale simulations. Level0 is fine-scale
    top layer. Level 1 is part-scale coarse mesh. Level 2 is meso-scale.
    Level 3 is fine-scale near the melt pool.

    Parameters:
    solver_input (dict): Dictionary containing configuration for Level1-Level3.
    properties (dict): Dictionary of physical and simulation parameters.

    Returns:
    list: A list of dictionaries representing Level0 to Level3 configurations.
    """

    # Parse solver input into Level1–Level3 objects
    Level1 = dict2obj(solver_input.get("Level1", {}))
    Level2 = dict2obj(solver_input.get("Level2", {}))
    Level3 = dict2obj(solver_input.get("Level3", {}))

    # Common setup for all three levels
    for level in [Level1, Level2, Level3]:
        level.length, level.h = calc_length_h(level)
        level.nodes = calcNumNodes(level.elements)
        level.ne = level.elements[0] * level.elements[1] * level.elements[2]
        level.nn = level.nodes[0] * level.nodes[1] * level.nodes[2]
        level.BC = getBCindices(level)
        level.node_coords, level.connect = createMesh3D(
            (level.bounds.x[0], level.bounds.x[1], level.nodes[0]),
            (level.bounds.y[0], level.bounds.y[1], level.nodes[1]),
            (level.bounds.z[0], level.bounds.z[1], level.nodes[2]),
        )
        level.T = properties["T_amb"] * jnp.ones(level.nn)
        level.T0 = properties["T_amb"] * jnp.ones(level.nn)
        level.S1 = jnp.zeros(level.nn)
        level.S2 = jnp.zeros(level.nn, dtype=bool)
        level.k = properties["k_powder"] * jnp.ones(level.nn)
        level.rhocp = (
            properties["cp_solid_coeff_a0"] * properties["rho"] * jnp.ones(level.nn)
        )

    # Special storage for Level1
    Level1.S1_storage = jnp.zeros(
        [int(round(Level1.h[2] / properties["layer_height"])), Level1.nn]
    )

    # Additional setup for Level2 and Level3
    for level in [Level2, Level3]:
        (
            level.bounds.ix,
            level.bounds.iy,
            level.bounds.iz,
        ) = find_max_const(Level1, level)
        level.init_node_coors = copy.deepcopy(level.node_coords)
        level.Tprime = jnp.zeros(level.nn)
        level.Tprime0 = copy.deepcopy(level.Tprime)

    # Adjust Level1 z-coordinates to align with Level2
    Level1.orig_node_coords = copy.deepcopy(Level1.node_coords)
    tmp_coords = copy.deepcopy(Level1.orig_node_coords)
    while True:
        if jnp.isclose(tmp_coords[2] - Level2.node_coords[-1][-1], 0, atol=1e-4).any():
            break
        tmp_coords[2] += properties["layer_height"]
    Level1.node_coords = copy.deepcopy(tmp_coords)

    # Overlap regions between levels
    def get_overlap(level_fine, level_coarse):
        nodes = [
            getCoarseNodesInFineRegion(
                level_fine.node_coords[i], level_coarse.node_coords[i]
            )
            for i in range(3)
        ]
        coors = [
            jnp.array([level_coarse.node_coords[i][j] for j in nodes[i]])
            for i in range(3)
        ]
        return nodes, coors

    Level2.orig_overlap_nodes, Level2.orig_overlap_coors = get_overlap(Level2, Level1)
    Level3.orig_overlap_nodes, Level3.orig_overlap_coors = get_overlap(Level3, Level2)

    Level2.overlapNodes = copy.deepcopy(Level2.orig_overlap_nodes)
    Level2.overlapCoords = copy.deepcopy(Level2.orig_overlap_coors)
    Level3.overlapNodes = copy.deepcopy(Level3.orig_overlap_nodes)
    Level3.overlapCoords = copy.deepcopy(Level3.orig_overlap_coors)

    # Create Level0 for high-resolution state tracking
    Level0 = obj({})
    Level0.elements = [
        round(Level1.length[0] / Level3.h[0]),
        round(Level1.length[1] / Level3.h[1]),
        round(Level2.length[2] / Level3.h[2]),
    ]
    Level0.nodes = calcNumNodes(Level0.elements)
    Level0.ne = Level0.elements[0] * Level0.elements[1] * Level0.elements[2]
    Level0.nn = Level0.nodes[0] * Level0.nodes[1] * Level0.nodes[2]
    Level0.node_coords, Level0.connect = createMesh3D(
        (Level1.bounds.x[0], Level1.bounds.x[1], Level0.nodes[0]),
        (Level1.bounds.y[0], Level1.bounds.y[1], Level0.nodes[1]),
        (Level2.bounds.z[0], Level2.bounds.z[1], Level0.nodes[2]),
    )
    Level0.orig_node_coords = copy.deepcopy(Level0.node_coords)

    Level0.orig_overlap_nodes, Level0.orig_overlap_coors = get_overlap(Level3, Level0)

    Level0.overlapNodes = copy.deepcopy(Level0.orig_overlap_nodes)
    Level0.overlapCoords = copy.deepcopy(Level0.orig_overlap_coors)

    Level0.orig_overlap_nodes_L2 = [
        getCoarseNodesInLargeFineRegion(Level2.node_coords[i], Level0.node_coords[i])
        for i in range(3)
    ]
    Level0.orig_overlap_coors_L2 = [
        jnp.array([Level0.node_coords[i][j] for j in Level0.orig_overlap_nodes_L2[i]])
        for i in range(3)
    ]
    Level0.overlapNodes_L2 = copy.deepcopy(Level0.orig_overlap_nodes_L2)
    Level0.overlapCoords_L2 = copy.deepcopy(Level0.orig_overlap_coors_L2)

    Level0.S1 = jnp.zeros(Level0.nn)
    Level0.S2 = jnp.zeros(Level0.nn, dtype=bool)
    Level0.idx = getOverlapRegion(Level0.overlapNodes, Level0.nodes[0], Level0.nodes[1])
    Level0.idx_L2 = getOverlapRegion(
        Level0.overlapNodes_L2, Level0.nodes[0], Level0.nodes[1]
    )

    Level0.layer_idx_delta = int(round(properties["layer_height"] / Level3.h[2]))

    # Convert all numeric fields to jnp arrays
    for level in [Level0, Level1, Level2, Level3]:
        for attr in ["elements", "h", "length", "nodes"]:
            if hasattr(level, attr):
                setattr(level, attr, [jnp.array(v) for v in getattr(level, attr)])
        for attr in ["ne", "nn", "layer_idx_delta"]:
            if hasattr(level, attr):
                setattr(level, attr, jnp.array(getattr(level, attr)))

    # Return all levels as dictionaries
    Levels = [
        structure_to_dict(Level0),
        structure_to_dict(Level1),
        structure_to_dict(Level2),
        structure_to_dict(Level3),
    ]
    return Levels


def SetupProperties(prop_obj):
    """
    Initialize and return a dictionary of material and simulation properties.

    This function reads user-defined or default values from the input
    dictionary and assigns them to a structured object for use in
    simulations. It also computes derived coefficients used in
    evaporation and heat transfer models.

    Parameters:
    prop_obj (dict): Dictionary containing material and simulation parameters.

    Returns:
    dict: A dictionary of structured properties with default fallbacks.
    """
    properties = dict2obj(prop_obj)

    # --- Thermal conductivity (W/m·K) ---
    # Powder: constant
    properties.k_powder = prop_obj.get("thermal_conductivity_powder", 0.4)
    # Bulk: linear temperature dependence (a1 * T + a0)
    properties.k_bulk_coeff_a0 = prop_obj.get("thermal_conductivity_bulk_a0", 4.23)
    properties.k_bulk_coeff_a1 = prop_obj.get("thermal_conductivity_bulk_a1", 0.016)
    # Fluid: constant
    properties.k_fluid_coeff_a0 = prop_obj.get("thermal_conductivity_fluid_a0", 29.0)

    # --- Heat capacity (J/kg·K) ---
    # Solid: linear temperature dependence
    properties.cp_solid_coeff_a0 = prop_obj.get("heat_capacity_solid_a0", 383.1)
    properties.cp_solid_coeff_a1 = prop_obj.get("heat_capacity_solid_a1", 0.174)
    # Mushy and fluid phases: constant
    properties.cp_mushy = prop_obj.get("heat_capacity_mushy", 3235.0)
    properties.cp_fluid = prop_obj.get("heat_capacity_fluid", 769.0)

    # --- Density (kg/mm³) ---
    properties.rho = prop_obj.get("density", 8.0e-6)

    # --- Laser parameters ---
    properties.laser_radius = prop_obj.get("laser_radius", 0.110)  # mm
    properties.laser_depth = prop_obj.get("laser_depth", 0.05)  # mm
    properties.laser_power = prop_obj.get("laser_power", 300.0)  # W
    properties.laser_eta = prop_obj.get("laser_absorptivity", 0.25)  # unitless
    properties.laser_center = prop_obj.get("laser_center", [])  # mm

    # --- Temperature thresholds (K) ---
    properties.T_amb = prop_obj.get("T_amb", 353.15)
    properties.T_solidus = prop_obj.get("T_solidus", 1554.0)
    properties.T_liquidus = prop_obj.get("T_liquidus", 1625.0)
    properties.T_boiling = prop_obj.get("T_boiling", 3038.0)

    # --- Heat transfer and radiation ---
    properties.h_conv = prop_obj.get("h_conv", 1.473e-5)  # W/mm²·K
    properties.vareps = prop_obj.get("emissivity", 0.600)  # unitless
    properties.evc = prop_obj.get("evaporation_coefficient", 0.82)  # unitless

    # --- Physical constants ---
    properties.kb = prop_obj.get("boltzmann_constant", 1.38e-23)  # J/K
    properties.mA = prop_obj.get("atomic_mass", 7.9485017e-26)  # kg
    properties.Lev = prop_obj.get("latent_heat_evap", 4.22e6)  # J/kg
    properties.molar_mass = prop_obj.get("molar_mass", 58.69) * 1e-3  # g/mol → kg/mol

    # --- Layer height (mm) ---
    properties.layer_height = prop_obj.get("layer_height", 0.04)

    # --- Universal constants ---
    properties.sigma_sb = 5.67e-8  # Stefan–Boltzmann constant (W/m²·K⁴)
    properties.gas_const = 8.314  # Molar gas constant (J/mol·K)
    properties.atmospheric_pressure = 101325  # Pa

    # --- Derived evaporation model coefficients ---
    # CM_coeff: Heat loss temperature factor (K·s²/m²)
    properties.CM_coeff = properties.molar_mass / (2.0 * jnp.pi * properties.gas_const)
    # CT_coeff: Recoil pressure temperature factor (K)
    properties.CT_coeff = properties.Lev * properties.molar_mass / properties.gas_const
    # CP_coeff: Recoil pressure factor (Pa)
    properties.CP_coeff = 0.54 * properties.atmospheric_pressure

    return structure_to_dict(properties)


def SetupNonmesh(nonmesh_input):
    """
    Initialize and return a dictionary of non-mesh simulation parameters.

    Parameters:
    nonmesh_input (dict): Dictionary containing time-stepping, output,
                          and laser path configuration.

    Returns:
    dict: A dictionary of structured non-mesh parameters.
    """
    Nonmesh = dict2obj(nonmesh_input)

    # Timestep for finest mesh (s)
    Nonmesh.timestep_L3 = nonmesh_input.get("timestep_L3", 1e-5)

    # Subcycle numbers (unitless)
    Nonmesh.subcycle_num_L2 = nonmesh_input.get("subcycle_num_L2", 1)
    Nonmesh.subcycle_num_L3 = nonmesh_input.get("subcycle_num_L3", 1)

    # Dwell time duration (s)
    Nonmesh.dwell_time = nonmesh_input.get("dwell_time", 0.1)

    # Coarse-domain recording frequency (steps)
    Nonmesh.Level1_record_step = nonmesh_input.get("Level1_record_step", 1)

    # Output directory
    Nonmesh.save_path = nonmesh_input.get("save_path", "results/")

    # Output flags
    Nonmesh.output_files = nonmesh_input.get("output_files", 1)  # Save VTK/VTR

    # Toolpath file location
    Nonmesh.toolpath = nonmesh_input.get("toolpath", "laserPath.txt")

    # Wait time before increasing timestep in dwell (steps)
    Nonmesh.wait_time = nonmesh_input.get("wait_time", 500.0)

    # Layer control
    Nonmesh.layer_num = nonmesh_input.get("layer_num", 0)
    Nonmesh.restart_layer_num = nonmesh_input.get("restart_layer_num", 10000)

    # Output verbosity
    Nonmesh.info_T = nonmesh_input.get("info_T", 0)

    # Laser velocity (mm/s)
    Nonmesh.laser_velocity = nonmesh_input.get("laser_velocity", 500)

    # Wait time after each track (s)
    Nonmesh.wait_track = nonmesh_input.get("wait_track", 0.0)

    # Recording frequency (steps)
    Nonmesh.record_step = nonmesh_input.get(
        "record_step", Nonmesh.subcycle_num_L2 * Nonmesh.subcycle_num_L3
    )

    # G-code file path
    Nonmesh.gcode = nonmesh_input.get(
        "gcode", "./examples/gcodefiles/defaultName.gcode"
    )

    # Dwell time multiplier (unitless)
    Nonmesh.dwell_time_multiplier = nonmesh_input.get("dwell_time_multiplier", 1)

    # Use TXT format for toolpath (flag)
    Nonmesh.use_txt = nonmesh_input.get("use_txt", 0)

    # Create output directory if it doesn't exist
    if not os.path.exists(Nonmesh.save_path):
        os.makedirs(Nonmesh.save_path)

    return structure_to_dict(Nonmesh)


def structure_to_dict(struct):
    """
    Recursively convert a nested object with attributes into a dictionary.

    This function handles nested objects by checking if each attribute
    has a __dict__ (i.e., is an object) and recursively converting it,
    unless the object has a 'tolist' method (e.g., NumPy or JAX arrays),
    in which case it is left as-is.

    Parameters:
    struct (object): An object with attributes to convert.

    Returns:
    dict: A dictionary representation of the object's attributes.
    """
    return {
        k: (
            v
            if (not hasattr(v, "__dict__") or hasattr(v, "tolist"))
            else structure_to_dict(v)
        )
        for k, v in struct.__dict__.items()
    }


def getStaticNodesAndElements(L):
    """
    Extract static element and node counts from a list of level dictionaries.

    Parameters:
    L (list): A list of level dictionaries (e.g., from SetupLevels),
              where each dictionary contains 'ne' (number of elements)
              and 'nn' (number of nodes) as JAX arrays.

    Returns:
    tuple: A tuple containing:
        - L[2]["ne"] as int
        - L[3]["ne"] as int
        - L[1]["nn"] as int
        - L[2]["nn"] as int
        - L[3]["nn"] as int
    """
    return (
        L[2]["ne"].tolist(),
        L[3]["ne"].tolist(),
        L[1]["nn"].tolist(),
        L[2]["nn"].tolist(),
        L[3]["nn"].tolist(),
    )


def getStaticSubcycle(N):
    """
    Extract and compute subcycle values from the non-mesh configuration.

    Parameters:
    N (dict): Dictionary containing subcycle numbers for Level 2 and Level 3.

    Returns:
    tuple: A tuple containing:
        - N2 (int): Subcycle count for Level 2
        - N3 (int): Subcycle count for Level 3
        - N23 (int): Total subcycles (N2 * N3)
        - fN2 (float): N2 as float
        - fN3 (float): N3 as float
        - fN23 (float): N23 as float
    """
    N2, N3 = N["subcycle_num_L2"], N["subcycle_num_L3"]
    N23 = N2 * N3
    fN2, fN3, fN23 = float(N2), float(N3), float(N23)
    return (N2, N3, N23, fN2, fN3, fN23)


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


def getBCindices(x):
    """
    Compute boundary condition indices for a structured 3D mesh.

    Parameters:
    x (object): An object with attributes:
        - nodes (list of int): Number of nodes in x, y, z directions.
        - nn (int): Total number of nodes.

    Returns:
    list of jnp.ndarray: A list containing indices for:
        [west, east, south, north, bottom, top] boundaries.
    """
    nx, ny, nz = x.nodes[0], x.nodes[1], x.nodes[2]
    nn = x.nn

    # Bottom face (z = 0)
    bidx = jnp.arange(0, nx * ny)

    # Top face (z = nz - 1)
    tidx = jnp.arange(nx * ny * (nz - 1), nn)

    # West face (x = 0)
    widx = jnp.arange(0, nn, nx)

    # East face (x = nx - 1)
    eidx = jnp.arange(nx - 1, nn, nx)

    # South face (y = 0)
    sidx = jnp.arange(0, nx)[:, None] + (nx * ny * jnp.arange(0, nz))[None, :]
    sidx = sidx.reshape(-1)

    # North face (y = ny - 1)
    nidx = (
        jnp.arange(nx * (ny - 1), nx * ny)[:, None]
        + (nx * ny * jnp.arange(0, nz))[None, :]
    )
    nidx = nidx.reshape(-1)

    return [widx, eidx, sidx, nidx, bidx, tidx]


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


@partial(jax.jit, static_argnames=["ne", "nn"])
def solveMatrixFreeFE(Level, nn, ne, k, rhocp, dt, T, Fc, Corr):
    """
    Perform an explicit matrix-free finite element thermal solve.

    This function computes the temperature update for a single timestep
    using a matrix-free approach, avoiding global matrix assembly.

    Parameters:
    Level (dict): Mesh level containing connectivity and geometry.
    nn (int): Total number of nodes.
    ne (int): Total number of elements.
    k (array): Thermal conductivity at each node.
    rhocp (array): Volumetric density times heat capacity at each node.
    dt (float): Time step size.
    T (array): Temperature at the previous timestep.
    Fc (array): Integrated right-hand side (e.g., heat source + surface BC).
    Corr (array): Integrated T' correction terms.

    Returns:
    array: Updated temperature field for the next timestep.
    """
    nen = jnp.size(Level["connect"][0], 1)  # Nodes per element
    ndim = 3  # 3D problem

    coords = getSampleCoords(Level)
    N, dNdx, wq = computeQuad3dFemShapeFunctions_jax(coords)
    wq = wq[0][0]  # Quadrature weight

    # Precompute shape function matrices
    NTN = jnp.matmul(N.T, N)
    BTB = jnp.zeros((nen, nen))
    for idim in range(ndim):
        BTB += jnp.matmul(dNdx[:, :, idim].T, dNdx[:, :, idim])

    def calcVal(i):
        _, _, _, idx = convert2XYZ(
            i,
            Level["elements"][0],
            Level["elements"][1],
            Level["nodes"][0],
            Level["nodes"][1],
        )
        kvec = k[idx].mean()
        mvec = rhocp[idx].mean() / dt

        Me = jnp.sum(NTN * wq * mvec, axis=0)
        Ke = BTB * kvec * wq
        LHSe = jnp.diag(Me) - Ke

        return jnp.matmul(LHSe, T[idx]), Me, idx

    # Vectorized element-wise computation
    vcalcVal = jax.vmap(calcVal)
    aT, aMe, aidx = vcalcVal(jnp.arange(ne))

    # Assemble global temperature and mass contributions
    newT = jnp.bincount(aidx.reshape(-1), aT.reshape(-1), length=nn)
    newM = jnp.bincount(aidx.reshape(-1), aMe.reshape(-1), length=nn)

    return (newT + Fc + Corr) / newM


@jax.jit
def convert2XYZ(i, ne_x, ne_y, nn_x, nn_y):
    """
    Compute local element indices and global node connectivity in 3D.

    Parameters:
    i (int): Element index (flattened).
    ne_x (int): Number of elements in the x-direction.
    ne_y (int): Number of elements in the y-direction.
    nn_x (int): Number of nodes in the x-direction.
    nn_y (int): Number of nodes in the y-direction.

    Returns:
    tuple:
        ix (int): Element index in x-direction.
        iy (int): Element index in y-direction.
        iz (int): Element index in z-direction.
        idx (jnp.ndarray): Global node indices for the 8-node hexahedral element.
    """
    ne_xy = ne_x * ne_y
    nn_xy = nn_x * nn_y

    iz = i // ne_xy
    iy = (i // ne_x) - iz * ne_y
    ix = i % ne_x

    base = ix + iy * nn_x + iz * nn_xy
    dx = 1
    dy = nn_x
    dz = nn_xy

    idx = jnp.array(
        [
            base,
            base + dx,
            base + dx + dy,
            base + dy,
            base + dz,
            base + dx + dz,
            base + dx + dy + dz,
            base + dy + dz,
        ]
    )

    return ix, iy, iz, idx


@jax.jit
def computeQuad3dFemShapeFunctions_jax(coords):
    """
    Compute shape functions, their derivatives, and quadrature weights
    for an 8-node hexahedral element using 3D Gaussian quadrature.

    This function evaluates the shape function matrix (N), its spatial
    derivatives (dNdx), and the quadrature weights (wq) at 8 integration
    points in the reference element.

    Parameters:
    coords (array): Nodal coordinates of the hexahedral element, shape (8, 3).

    Returns:
    N (array): Shape function values at each quadrature point, shape (8, 8).
    dNdx (array): Derivatives of shape functions w.r.t. global coordinates,
                  shape (8, 8, 3).
    wq (array): Quadrature weights for each integration point, shape (8, 1).
    """
    ngp = 8  # Number of Gauss points
    ndim = 3  # Number of spatial dimensions

    # Isoparametric coordinates for 8-node hexahedral element
    ksi_i = jnp.array([-1, 1, 1, -1, -1, 1, 1, -1])
    eta_i = jnp.array([-1, -1, 1, 1, -1, -1, 1, 1])
    zeta_i = jnp.array([-1, -1, -1, -1, 1, 1, 1, 1])

    # Gauss points in reference space (±1/√3)
    ksi_q = (1 / jnp.sqrt(3)) * ksi_i
    eta_q = (1 / jnp.sqrt(3)) * eta_i
    zeta_q = (1 / jnp.sqrt(3)) * zeta_i

    # Uniform quadrature weights for 2-point Gauss rule
    tmp_wq = jnp.ones(ngp)

    # Evaluate shape functions at Gauss points
    _ksi = 1 + ksi_q[:, None] @ ksi_i[None, :]
    _eta = 1 + eta_q[:, None] @ eta_i[None, :]
    _zeta = 1 + zeta_q[:, None] @ zeta_i[None, :]
    N = (1 / 8) * _ksi * _eta * _zeta
    # Derivatives w.r.t. reference coordinates
    dNdksi = (1 / 8) * ksi_i[None, :] * _eta * _zeta
    dNdeta = (1 / 8) * eta_i[None, :] * _ksi * _zeta
    dNdzeta = (1 / 8) * zeta_i[None, :] * _ksi * _eta

    # Compute Jacobian components (diagonal for structured mesh)
    dxdksi = jnp.matmul(dNdksi, coords[:, 0])
    dydeta = jnp.matmul(dNdeta, coords[:, 1])
    dzdzeta = jnp.matmul(dNdzeta, coords[:, 2])

    # Manually construct Jacobian and its inverse
    J = jnp.array(
        [
            [dxdksi[0], 0.0, 0.0],
            [0.0, dydeta[0], 0.0],
            [0.0, 0.0, dzdzeta[0]],
        ]
    )
    Jinv = jnp.array(
        [
            [1.0 / dxdksi[0], 0.0, 0.0],
            [0.0, 1.0 / dydeta[0], 0.0],
            [0.0, 0.0, 1.0 / dzdzeta[0]],
        ]
    )

    # Allocate arrays for shape function derivatives and weights
    dNdx = jnp.zeros((ngp, ngp, ndim))
    wq = jnp.zeros((ngp, 1))

    # Loop over Gauss points to compute global derivatives and weights
    for q in range(ngp):
        dN_dxi = jnp.concatenate(
            [
                dNdksi[q, :, None],
                dNdeta[q, :, None],
                dNdzeta[q, :, None],
            ],
            axis=1,
        )
        dNdx = dNdx.at[q, :, :].set(dN_dxi @ Jinv)
        wq = wq.at[q].set(jnp.linalg.det(J) * tmp_wq[q])

    return jnp.array(N), jnp.array(dNdx), jnp.array(wq)


@jax.jit
def computeQuad2dFemShapeFunctions_jax(coords):
    """
    Compute shape functions, their derivatives, and quadrature weights
    for a 4-node quadrilateral element using 2D Gaussian quadrature.

    This function evaluates the shape function matrix (N), its spatial
    derivatives (dNdx), and the quadrature weights (wq) at 4 integration
    points in the reference element.

    Parameters:
    coords (array): Nodal coordinates of the quadrilateral element, shape (8, 2). Only
                    the last 4 coordinates (top surface of element) are used.

    Returns:
    N (array): Shape function values at each quadrature point, shape (4, 4).
    dNdx (array): Derivatives of shape functions w.r.t. global coordinates,
                  shape (4, 4, 2).
    wq (array): Quadrature weights for each integration point, shape (4, 1).
    """
    ngp = 4  # Number of Gauss points
    ndim = 2  # Number of spatial dimensions

    # Isoparametric coordinates for 4-node quadrilateral element
    ksi_i = jnp.array([-1, 1, 1, -1])
    eta_i = jnp.array([-1, -1, 1, 1])

    # Gauss points in reference space (±1/√3)
    ksi_q = (1 / jnp.sqrt(3)) * ksi_i
    eta_q = (1 / jnp.sqrt(3)) * eta_i

    # Uniform quadrature weights for 2-point Gauss rule
    tmp_wq = jnp.ones(ngp)

    # Evaluate shape functions at Gauss points
    _ksi = 1 + ksi_q[:, None] @ ksi_i[None, :]
    _eta = 1 + eta_q[:, None] @ eta_i[None, :]
    N = (1 / 4) * _ksi * _eta

    # Derivatives w.r.t. reference coordinates
    dNdksi = (1 / 4) * ksi_i[None, :] * _eta
    dNdeta = (1 / 4) * eta_i[None, :] * _ksi

    # Compute Jacobian components (diagonal for structured mesh)
    dxdksi = jnp.matmul(dNdksi, coords[4:, 0])
    dydeta = jnp.matmul(dNdeta, coords[4:, 1])

    # Manually construct Jacobian and its inverse
    J = jnp.array(
        [
            [dxdksi[0], 0.0],
            [0.0, dydeta[0]],
        ]
    )
    Jinv = jnp.array(
        [
            [1.0 / dxdksi[0], 0.0],
            [0.0, 1.0 / dydeta[0]],
        ]
    )

    # Allocate arrays for shape function derivatives and weights
    dNdx = jnp.zeros((ngp, ngp, ndim))
    wq = jnp.zeros((ngp, 1))

    # Loop over Gauss points to compute global derivatives and weights
    for q in range(ngp):
        dN_dxi = jnp.concatenate(
            [
                dNdksi[q, :, None],
                dNdeta[q, :, None],
            ],
            axis=1,
        )
        dNdx = dNdx.at[q, :, :].set(dN_dxi @ Jinv)
        wq = wq.at[q].set(jnp.linalg.det(J) * tmp_wq[q])

    return jnp.array(N), jnp.array(dNdx), jnp.array(wq)


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


@partial(jax.jit, static_argnames=["ne_nn"])
def computeSources(Level, v, Shapes, ne_nn, properties, laserP):
    """
    Compute integrated source terms for all three levels using Level 3 mesh.

    This function evaluates the laser-induced heat source at quadrature
    points of the fine mesh (Level 3), integrates it using shape functions,
    and projects the result to coarser levels using precomputed transfer
    operators.

    Parameters:
    Level (dict): Mesh level containing connectivity and geometry for Level 3.
    v (array): Current laser position.
    Shapes (list): Shape transfer operators between Level 3 and Levels 1 & 2.
                   Shapes[1][0]: L3 → L1 interpolation matrix.
                   Shapes[2][0]: L3 → L2 interpolation matrix.
                   Shapes[1][2], Shapes[2][2]: Projection matrices.
    ne_nn (tuple): Element/node counts for each level.
                   ne_nn[1]: Number of elements in Level 3.
                   ne_nn[4]: Number of nodes in Level 3.
    properties (dict): Material and laser properties.
    laserP (float): Laser power.

    Returns:
    Fc (array): Integrated source term for Level 1.
    Fm (array): Integrated source term for Level 2.
    Ff (array): Integrated source term for Level 3.
    """
    # Get shape functions and quadrature weights for Level 3 elements
    coords = getSampleCoords(Level)
    Nf, _, wqf = computeQuad3dFemShapeFunctions_jax(coords)

    def stepcomputeCoarseSource(ieltf):
        # Get nodal indices for the fine element
        ix, iy, iz, idx = convert2XYZ(
            ieltf,
            Level["elements"][0],
            Level["elements"][1],
            Level["nodes"][0],
            Level["nodes"][1],
        )
        # Evaluate coordinates at quadrature points
        x, y, z = getQuadratureCoords(Level, ix, iy, iz, Nf)
        # Evaluate source function at quadrature points
        Q = computeSourceFunction_jax(x, y, z, v, properties, laserP)
        # Return raw source, integrated source, and node indices
        return Q * wqf, Nf @ Q * wqf, idx

    # Vectorized computation over all Level 3 elements
    vstepcomputeCoarseSource = jax.vmap(stepcomputeCoarseSource)
    _data, _data3, nodes3 = vstepcomputeCoarseSource(jnp.arange(ne_nn[1]))

    # Project source terms to Levels 1 and 2 using transfer operators
    _data1 = multiply(Shapes[1][0], _data).sum(axis=1)
    _data2 = multiply(Shapes[2][0], _data).sum(axis=1)

    Fc = Shapes[1][2] @ _data1.reshape(-1)
    Fm = Shapes[2][2] @ _data2.reshape(-1)
    Ff = bincount(nodes3.reshape(-1), _data3.reshape(-1), ne_nn[4])

    return Fc, Fm, Ff


@jax.jit
def computeSourceFunction_jax(x, y, z, v, properties, P):
    """
    Compute a 3D Gaussian heat source term for laser-material interaction.

    This function evaluates a separable Gaussian distribution in x, y, and z
    directions centered at the laser position `v`, scaled by laser power and
    material properties.

    Parameters:
    x, y, z (array): Coordinates of quadrature points.
    v (array): Laser center position [vx, vy, vz].
    properties (dict): Material and laser properties.
                       Required keys:
                       - "laser_eta": laser absorption efficiency.
                       - "laser_radius": standard deviation in x/y.
                       - "laser_depth": standard deviation in z.
    P (float): Laser power.

    Returns:
    array: Evaluated source term at each quadrature point.
    """
    # Precompute constants
    _pcoeff = 6 * jnp.sqrt(3) * P * properties["laser_eta"]
    _rcoeff = 1 / (properties["laser_radius"] * jnp.sqrt(jnp.pi))
    _dcoeff = 1 / (properties["laser_depth"] * jnp.sqrt(jnp.pi))
    _rsq = properties["laser_radius"] ** 2
    _dsq = properties["laser_depth"] ** 2

    # Evaluate separable Gaussian in each direction
    Qx = _rcoeff * jnp.exp(-3 * (x - v[0]) ** 2 / _rsq)
    Qy = _rcoeff * jnp.exp(-3 * (y - v[1]) ** 2 / _rsq)
    Qz = _dcoeff * jnp.exp(-3 * (z - v[2]) ** 2 / _dsq)

    return _pcoeff * Qx * Qy * Qz


@jax.jit
def interpolatePointsMatrix(Level, node_coords_new):
    """
    Compute interpolation shape functions and node indices for mapping
    values from a coarse mesh to a new set of coordinates.

    Parameters:
    Level (dict): Contains mesh information:
                  - "node_coords": list of 1D arrays for x, y, z coordinates.
                  - "connect": list of connectivity arrays in x, y, z.
    node_coords_new (list): New nodal coordinates [x_new, y_new, z_new],
                            each as a 2D array.

    Returns:
    list: [_Nc, _node]
          _Nc (array): Shape function values for interpolation.
          _node (array): Indices of coarse nodes contributing to each point.
    """
    ne_x = Level["connect"][0].shape[0]
    ne_y = Level["connect"][1].shape[0]
    ne_z = Level["connect"][2].shape[0]
    nn_x, nn_y = ne_x + 1, ne_y + 1

    nn_xn = len(node_coords_new[0])
    nn_yn = len(node_coords_new[1])
    nn_zn = len(node_coords_new[2])
    total_points = nn_xn * nn_yn * nn_zn

    h_x = Level["node_coords"][0][1] - Level["node_coords"][0][0]
    h_y = Level["node_coords"][1][1] - Level["node_coords"][1][0]
    h_z = Level["node_coords"][2][1] - Level["node_coords"][2][0]

    def stepInterpolatePoints(ielt):
        izn, rem = jnp.divmod(ielt, nn_xn * nn_yn)
        iyn, ixn = jnp.divmod(rem, nn_xn)

        _x = node_coords_new[0][ixn]
        _y = node_coords_new[1][iyn]
        _z = node_coords_new[2][izn]

        # Determine coarse element indices
        ielt_x = jnp.clip(
            jnp.floor((_x - Level["node_coords"][0][0]) / h_x).astype(int), 0, ne_x - 1
        )
        ielt_y = jnp.clip(
            jnp.floor((_y - Level["node_coords"][1][0]) / h_y).astype(int), 0, ne_y - 1
        )
        ielt_z = jnp.clip(
            jnp.floor((_z - Level["node_coords"][2][0]) / h_z).astype(int), 0, ne_z - 1
        )

        # Get node indices
        nodex = Level["connect"][0][ielt_x, :]
        nodey = Level["connect"][1][ielt_y, :]
        nodez = Level["connect"][2][ielt_z, :]
        node = nodex + nodey * nn_x + nodez * (nn_x * nn_y)

        # Get coordinates of the coarse element nodes
        xx = Level["node_coords"][0][nodex]
        yy = Level["node_coords"][1][nodey]
        zz = Level["node_coords"][2][nodez]

        # Bounding box corners
        xc0, xc1 = xx[0], xx[1]
        yc0, yc3 = yy[0], yy[3]
        zc0, zc5 = zz[0], zz[5]

        # Compute shape functions
        Nc = compute3DN(
            [_x, _y, _z], [xc0, xc1], [yc0, yc3], [zc0, zc5], [h_x, h_y, h_z]
        )

        # Clip and mask invalid values
        valid = jnp.logical_and((Nc >= -1e-2).all(), (Nc <= 1 + 1e-2).all())
        Nc = jax.lax.select(valid, jnp.clip(Nc, 0.0, 1.0), jnp.zeros_like(Nc))

        return Nc, node

    _Nc, _node = jax.vmap(stepInterpolatePoints)(jnp.arange(total_points))
    return [_Nc, _node]


@jax.jit
def interpolate_w_matrix(C2F, T):
    """
    Interpolate a solution field to new nodal coordinates using shape functions.

    This function applies precomputed shape functions and node indices
    (from `interpolatePointsMatrix`) to interpolate the solution `T`
    from a source mesh to a new set of points.

    Parameters:
    C2F (tuple): Interpolation data.
                 - C2F[0]: Shape function weights (array of shape [n_new, n_basis]).
                 - C2F[1]: Indices of source nodes (array of shape [n_new, n_basis]).
    T (array): Source solution values at coarse nodes.

    Returns:
    array: Interpolated solution at new nodal coordinates.
    """
    return multiply(C2F[0], T[C2F[1]]).sum(axis=1)


@jax.jit
def interpolatePoints(Level, u, node_coords_new):
    """
    Interpolate a scalar field from a structured mesh to new coordinates.

    This function evaluates shape functions at new nodal coordinates and
    uses them to interpolate values from the source field `u` defined on
    Level["node_coords"] and Level["connect"].

    Parameters:
    Level (dict): Contains mesh information:
                  - "node_coords": list of 1D arrays for x, y, z coordinates.
                  - "connect": list of connectivity arrays in x, y, z.
    u (array): Field values at the source mesh nodes.
    node_coords_new (list): New nodal coordinates [x_new, y_new, z_new].

    Returns:
    array: Interpolated values at the new nodal coordinates.
    """
    ne_x = Level["connect"][0].shape[0]
    ne_y = Level["connect"][1].shape[0]
    ne_z = Level["connect"][2].shape[0]
    nn_x, nn_y = ne_x + 1, ne_y + 1

    nn_xn = len(node_coords_new[0])
    nn_yn = len(node_coords_new[1])
    nn_zn = len(node_coords_new[2])
    total_points = nn_xn * nn_yn * nn_zn

    h_x = Level["node_coords"][0][1] - Level["node_coords"][0][0]
    h_y = Level["node_coords"][1][1] - Level["node_coords"][1][0]
    h_z = Level["node_coords"][2][1] - Level["node_coords"][2][0]

    def stepInterpolatePoints(ielt):
        izn, rem = jnp.divmod(ielt, nn_xn * nn_yn)
        iyn, ixn = jnp.divmod(rem, nn_xn)

        _x = node_coords_new[0][ixn]
        _y = node_coords_new[1][iyn]
        _z = node_coords_new[2][izn]

        # Determine coarse element indices
        ielt_x = jnp.clip(
            jnp.floor((_x - Level["node_coords"][0][0]) / h_x).astype(int), 0, ne_x - 1
        )
        ielt_y = jnp.clip(
            jnp.floor((_y - Level["node_coords"][1][0]) / h_y).astype(int), 0, ne_y - 1
        )
        ielt_z = jnp.clip(
            jnp.floor((_z - Level["node_coords"][2][0]) / h_z).astype(int), 0, ne_z - 1
        )

        # Get node indices
        nodex = Level["connect"][0][ielt_x, :]
        nodey = Level["connect"][1][ielt_y, :]
        nodez = Level["connect"][2][ielt_z, :]
        node = nodex + nodey * nn_x + nodez * (nn_x * nn_y)

        # Get coordinates of the coarse element nodes
        xx = Level["node_coords"][0][nodex]
        yy = Level["node_coords"][1][nodey]
        zz = Level["node_coords"][2][nodez]

        # Bounding box corners
        xc0, xc1 = xx[0], xx[1]
        yc0, yc3 = yy[0], yy[3]
        zc0, zc5 = zz[0], zz[5]

        # Compute shape functions
        Nc = compute3DN(
            [_x, _y, _z], [xc0, xc1], [yc0, yc3], [zc0, zc5], [h_x, h_y, h_z]
        )

        # Clip and mask invalid values
        valid = jnp.logical_and((Nc >= -1e-2).all(), (Nc <= 1 + 1e-2).all())
        Nc = jax.lax.select(valid, jnp.clip(Nc, 0.0, 1.0), jnp.zeros_like(Nc))

        return Nc @ u[node]

    return jax.vmap(stepInterpolatePoints)(jnp.arange(total_points))


@jax.jit
def computeCoarseFineShapeFunctions(Coarse, Fine):
    """
    Compute coarse shape functions and their derivatives at fine-scale
    quadrature points, and return a sparse projection matrix.

    Parameters:
    Coarse (dict): Coarse mesh data.
                   - "node_coords": list of 1D arrays for x, y, z coordinates.
                   - "connect": list of connectivity arrays in x, y, z.
    Fine (dict): Fine mesh data.
                 - "node_coords": list of 1D arrays for x, y, z coordinates.
                 - "connect": list of connectivity arrays in x, y, z.

    Returns:
    Nc (array): Coarse shape functions at fine quadrature points,
                shape (n_fine_elem, 8, 8).
    dNcdx, dNcdy, dNcdz (arrays): Derivatives of coarse shape functions,
                                  each of shape (n_fine_elem, 8, 8).
    test (BCOO): Sparse projection matrix from coarse nodes to fine quadrature.
    """
    # Mesh sizes
    nec_x, nec_y, nec_z = [Coarse["connect"][i].shape[0] for i in range(3)]
    nnc_x, nnc_y, nnc_z = [Coarse["node_coords"][i].shape[0] for i in range(3)]
    nnc = nnc_x * nnc_y * nnc_z
    nef_x, nef_y, nef_z = [Fine["connect"][i].shape[0] for i in range(3)]
    nef = nef_x * nef_y * nef_z
    nnf_x, nnf_y = [Fine["node_coords"][i].shape[0] for i in range(2)]

    # Coarse mesh spacing
    hc_x = Coarse["node_coords"][0][1] - Coarse["node_coords"][0][0]
    hc_y = Coarse["node_coords"][1][1] - Coarse["node_coords"][1][0]
    hc_z = Coarse["node_coords"][2][1] - Coarse["node_coords"][2][0]
    hc_xyz = hc_x * hc_y * hc_z
    xminc_x, xminc_y, xminc_z = [Coarse["node_coords"][i][0] for i in range(3)]

    # Reference shape functions for fine elements
    coords = jnp.stack(
        [
            Fine["node_coords"][0][Fine["connect"][0][0, :]],
            Fine["node_coords"][1][Fine["connect"][1][0, :]],
            Fine["node_coords"][2][Fine["connect"][2][0, :]],
        ],
        axis=1,
    )
    Nf, _, _ = computeQuad3dFemShapeFunctions_jax(coords)

    def stepComputeCoarseFineTerm(ieltf):
        ix, iy, iz, _ = convert2XYZ(ieltf, nef_x, nef_y, nnf_x, nnf_y)
        coords_x = Fine["node_coords"][0][Fine["connect"][0][ix, :]].reshape(-1, 1)
        coords_y = Fine["node_coords"][1][Fine["connect"][1][iy, :]].reshape(-1, 1)
        coords_z = Fine["node_coords"][2][Fine["connect"][2][iz, :]].reshape(-1, 1)

        x = (Nf @ coords_x).reshape(-1)
        y = (Nf @ coords_y).reshape(-1)
        z = (Nf @ coords_z).reshape(-1)

        # Determine coarse element indices
        ieltc_x = jnp.clip(jnp.floor((x - xminc_x) / hc_x).astype(int), 0, nec_x - 1)
        ieltc_y = jnp.clip(jnp.floor((y - xminc_y) / hc_y).astype(int), 0, nec_y - 1)
        ieltc_z = jnp.clip(jnp.floor((z - xminc_z) / hc_z).astype(int), 0, nec_z - 1)

        def iqLoop(iq):
            # Coarse element node indices
            nodec_x = Coarse["connect"][0][ieltc_x[iq], :]
            nodec_y = Coarse["connect"][1][ieltc_y[iq], :]
            nodec_z = Coarse["connect"][2][ieltc_z[iq], :]
            nodes = nodec_x + nodec_y * nnc_x + nodec_z * nnc_x * nnc_y

            # Bounding box corners
            xc0 = Coarse["node_coords"][0][nodec_x[0]]
            xc1 = Coarse["node_coords"][0][nodec_x[1]]
            yc0 = Coarse["node_coords"][1][nodec_y[0]]
            yc3 = Coarse["node_coords"][1][nodec_y[3]]
            zc0 = Coarse["node_coords"][2][nodec_z[0]]
            zc5 = Coarse["node_coords"][2][nodec_z[5]]

            _x, _y, _z = x[iq], y[iq], z[iq]

            Nc = compute3DN(
                [_x, _y, _z], [xc0, xc1], [yc0, yc3], [zc0, zc5], [hc_x, hc_y, hc_z]
            )

            dNcdx = (
                jnp.array(
                    [
                        (-1 * (yc3 - _y) * (zc5 - _z)),
                        (1 * (yc3 - _y) * (zc5 - _z)),
                        (1 * (_y - yc0) * (zc5 - _z)),
                        (-1 * (_y - yc0) * (zc5 - _z)),
                        (-1 * (yc3 - _y) * (_z - zc0)),
                        (1 * (yc3 - _y) * (_z - zc0)),
                        (1 * (_y - yc0) * (_z - zc0)),
                        (-1 * (_y - yc0) * (_z - zc0)),
                    ]
                )
                / hc_xyz
            )

            dNcdy = (
                jnp.array(
                    [
                        ((xc1 - _x) * -1 * (zc5 - _z)),
                        ((_x - xc0) * -1 * (zc5 - _z)),
                        ((_x - xc0) * 1 * (zc5 - _z)),
                        ((xc1 - _x) * 1 * (zc5 - _z)),
                        ((xc1 - _x) * -1 * (_z - zc0)),
                        ((_x - xc0) * -1 * (_z - zc0)),
                        ((_x - xc0) * 1 * (_z - zc0)),
                        ((xc1 - _x) * 1 * (_z - zc0)),
                    ]
                )
                / hc_xyz
            )

            dNcdz = (
                jnp.array(
                    [
                        ((xc1 - _x) * (yc3 - _y) * -1),
                        ((_x - xc0) * (yc3 - _y) * -1),
                        ((_x - xc0) * (_y - yc0) * -1),
                        ((xc1 - _x) * (_y - yc0) * -1),
                        ((xc1 - _x) * (yc3 - _y) * 1),
                        ((_x - xc0) * (yc3 - _y) * 1),
                        ((_x - xc0) * (_y - yc0) * 1),
                        ((xc1 - _x) * (_y - yc0) * 1),
                    ]
                )
                / hc_xyz
            )

            return Nc, dNcdx, dNcdy, dNcdz, nodes

        return jax.vmap(iqLoop)(jnp.arange(8))

    Nc, dNcdx, dNcdy, dNcdz, _nodes = jax.vmap(stepComputeCoarseFineTerm)(
        jnp.arange(nef)
    )
    _nodes = _nodes[:, 0, :]  # Only need one set of node indices per element

    # Construct sparse projection matrix
    indices = jnp.stack([_nodes.reshape(-1), jnp.arange(_nodes.size)], axis=1)
    test = jax.experimental.sparse.BCOO(
        (jnp.ones(_nodes.size), indices), shape=(nnc, _nodes.size)
    )
    return [Nc, [dNcdx, dNcdy, dNcdz], test]


def compute3DN(q, x, y, z, h):
    """
    Compute trilinear shape functions for a hexahedral element at a given point.

    Parameters:
    q (list or array): Evaluation point [xq, yq, zq].
    x (list): x-coordinates of the element's bounding box [x0, x1].
    y (list): y-coordinates of the element's bounding box [y0, y1].
    z (list): z-coordinates of the element's bounding box [z0, z1].
    h (list): Element sizes in x, y, z directions [hx, hy, hz].

    Returns:
    array: Shape function values at point q, shape (8,).
    """
    inv_vol = 1.0 / (h[0] * h[1] * h[2])

    N = (
        jnp.array(
            [
                (x[1] - q[0]) * (y[1] - q[1]) * (z[1] - q[2]),
                (q[0] - x[0]) * (y[1] - q[1]) * (z[1] - q[2]),
                (q[0] - x[0]) * (q[1] - y[0]) * (z[1] - q[2]),
                (x[1] - q[0]) * (q[1] - y[0]) * (z[1] - q[2]),
                (x[1] - q[0]) * (y[1] - q[1]) * (q[2] - z[0]),
                (q[0] - x[0]) * (y[1] - q[1]) * (q[2] - z[0]),
                (q[0] - x[0]) * (q[1] - y[0]) * (q[2] - z[0]),
                (x[1] - q[0]) * (q[1] - y[0]) * (q[2] - z[0]),
            ]
        )
        * inv_vol
    )

    return N


@jax.jit
def computeCoarseTprimeMassTerm_jax(
    Levels, Tprimef, Tprimem, L3rhocp, L2rhocp, dt, Shapes, Vcu, Vmu
):
    """
    Compute coarse-scale mass terms from fine and medium-scale T' fields.

    This function integrates the T' correction terms from Level 3 (fine)
    and Level 2 (medium) and projects them to Level 1 and Level 2 using
    precomputed shape transfer operators.

    Parameters:
    Levels (dict): Multilevel mesh hierarchy with keys [1, 2, 3].
    Tprimef (array): Fine-scale T' field (Level 3).
    Tprimem (array): Medium-scale T' field (Level 2).
    L3rhocp (array): rhocp values at Level 3 nodes.
    L2rhocp (array): rhocp values at Level 2 nodes.
    dt (float): Time step size.
    Shapes (list): Shape transfer operators between levels.
                   Shapes[0]: L2 → L1
                   Shapes[1]: L3 → L1
                   Shapes[2]: L3 → L2
    Vcu (array): Accumulator for Level 1 correction.
    Vmu (array): Accumulator for Level 2 correction.

    Returns:
    Vcu (array): Updated Level 1 correction term.
    Vmu (array): Updated Level 2 correction term.
    """
    # Subtract initial T' values
    Tprimef_new = Tprimef - Levels[3]["Tprime0"]
    Tprimem_new = Tprimem - Levels[2]["Tprime0"]

    # Level 3 setup
    nef_x, nef_y, nef_z = [Levels[3]["connect"][i].shape[0] for i in range(3)]
    nef = nef_x * nef_y * nef_z
    nnf_x = Levels[3]["node_coords"][0].shape[0]
    nnf_y = Levels[3]["node_coords"][1].shape[0]

    coordsf = getSampleCoords(Levels[3])
    Nf, _, wqf = computeQuad3dFemShapeFunctions_jax(coordsf)

    # Level 3
    _, _, _, idxf = convert2XYZ(jnp.arange(nef), nef_x, nef_y, nnf_x, nnf_y)
    _Tprimef = multiply(
        Nf @ Tprimef_new[idxf], jnp.matmul(Nf, L3rhocp[idxf]).mean(axis=0)
    )

    _data1 = multiply(
        multiply(-Shapes[1][0], _Tprimef.T[:, :, None]), (1 / dt) * wqf[None, None, :]
    ).sum(axis=2)

    _data2 = multiply(
        multiply(-Shapes[2][0], _Tprimef.T[:, :, None]), (1 / dt) * wqf[None, None, :]
    ).sum(axis=2)

    # Level 2 setup
    nem_x, nem_y, nem_z = [Levels[2]["connect"][i].shape[0] for i in range(3)]
    nem = nem_x * nem_y * nem_z
    nnm_x = Levels[2]["node_coords"][0].shape[0]
    nnm_y = Levels[2]["node_coords"][1].shape[0]

    coordsm = getSampleCoords(Levels[2])
    Nm, _, wqm = computeQuad3dFemShapeFunctions_jax(coordsm)

    _, _, _, idxm = convert2XYZ(jnp.arange(nem), nem_x, nem_y, nnm_x, nnm_y)
    _Tprimem = multiply(
        Nm @ Tprimem_new[idxm], jnp.matmul(Nm, L2rhocp[idxm]).mean(axis=0)
    )

    _data3 = multiply(
        multiply(-Shapes[0][0], _Tprimem.T[:, :, None]), (1 / dt) * wqm[None, None, :]
    ).sum(axis=2)

    # Project to coarse levels
    Vcu += Shapes[1][2] @ _data1.reshape(-1) + Shapes[0][2] @ _data3.reshape(-1)
    Vmu += Shapes[2][2] @ _data2.reshape(-1)

    return Vcu, Vmu


@jax.jit
def computeCoarseTprimeTerm_jax(Levels, L3k, L2k, Shapes):
    """
    Compute coarse-scale projection of T' gradient terms from fine and medium levels.

    This function computes the gradient of the T' field at Level 3 and Level 2,
    multiplies it by thermal conductivity, and projects the result to coarser
    levels using precomputed shape transfer operators.

    Parameters:
    Levels (dict): Multilevel mesh hierarchy with keys [1, 2, 3].
                   Each level contains "connect", "node_coords", and "Tprime0".
    L3k (array): Thermal conductivity at Level 3 nodes.
    L2k (array): Thermal conductivity at Level 2 nodes.
    Shapes (list): Shape transfer operators between levels.
                   Shapes[0]: L2 → L1
                   Shapes[1]: L3 → L1
                   Shapes[2]: L3 → L2

    Returns:
    Vcu (array): Coarse-scale correction term for Level 1.
    Vmu (array): Coarse-scale correction term for Level 2.
    """
    # Level 3 setup
    nef_x, nef_y, nef_z = [Levels[3]["connect"][i].shape[0] for i in range(3)]
    nef = nef_x * nef_y * nef_z
    nnf_x = Levels[3]["node_coords"][0].shape[0]
    nnf_y = Levels[3]["node_coords"][1].shape[0]

    coordsf = getSampleCoords(Levels[3])
    Nf, dNdxf, wqf = computeQuad3dFemShapeFunctions_jax(coordsf)

    _, _, _, idxf = convert2XYZ(jnp.arange(nef), nef_x, nef_y, nnf_x, nnf_y)
    Tprimef = Levels[3]["Tprime0"][idxf]
    L3kMean = jnp.matmul(Nf, L3k[idxf]).mean(axis=0)

    dTprimefdx = multiply(L3kMean, dNdxf[:, :, 0] @ Tprimef)
    dTprimefdy = multiply(L3kMean, dNdxf[:, :, 1] @ Tprimef)
    dTprimefdz = multiply(L3kMean, dNdxf[:, :, 2] @ Tprimef)

    # Project to Level 1 and Level 2
    _data1 = sum(
        [
            multiply(
                multiply(-Shapes[1][1][i], d.T[:, :, None]), wqf[None, None, :]
            ).sum(axis=2)
            for i, d in enumerate([dTprimefdx, dTprimefdy, dTprimefdz])
        ]
    )
    _data2 = sum(
        [
            multiply(
                multiply(-Shapes[2][1][i], d.T[:, :, None]), wqf[None, None, :]
            ).sum(axis=2)
            for i, d in enumerate([dTprimefdx, dTprimefdy, dTprimefdz])
        ]
    )

    # Level 2 setup
    nem_x, nem_y, nem_z = [Levels[2]["connect"][i].shape[0] for i in range(3)]
    nem = nem_x * nem_y * nem_z
    nnm_x = Levels[2]["node_coords"][0].shape[0]
    nnm_y = Levels[2]["node_coords"][1].shape[0]

    coordsm = getSampleCoords(Levels[2])
    Nm, dNdxm, wqm = computeQuad3dFemShapeFunctions_jax(coordsm)

    _, _, _, idxm = convert2XYZ(jnp.arange(nem), nem_x, nem_y, nnm_x, nnm_y)
    Tprimem = Levels[2]["Tprime0"][idxm]
    L2kMean = jnp.matmul(Nm, L2k[idxm]).mean(axis=0)

    dTprimemdx = multiply(L2kMean, dNdxm[:, :, 0] @ Tprimem)
    dTprimemdy = multiply(L2kMean, dNdxm[:, :, 1] @ Tprimem)
    dTprimemdz = multiply(L2kMean, dNdxm[:, :, 2] @ Tprimem)

    _data3 = sum(
        [
            multiply(
                multiply(-Shapes[0][1][i], d.T[:, :, None]), wqm[None, None, :]
            ).sum(axis=2)
            for i, d in enumerate([dTprimemdx, dTprimemdy, dTprimemdz])
        ]
    )

    # Final projection to coarse levels
    Vcu = Shapes[1][2] @ _data1.reshape(-1) + Shapes[0][2] @ _data3.reshape(-1)
    Vmu = Shapes[2][2] @ _data2.reshape(-1)

    return Vcu, Vmu


@jax.jit
def assignBCs(RHS, Levels):
    """
    Apply Dirichlet boundary conditions to the right-hand side vector.

    This function sets prescribed values at boundary nodes for Level 1
    based on the boundary condition indices and values stored in the
    Levels dictionary.

    Parameters:
    RHS (array): Right-hand side vector to be modified.
    Levels (dict): Multilevel mesh hierarchy containing:
                   - Levels[1]["BC"]: list of boundary node index arrays
                     [x0, x1, y0, y1, z0]
                   - Levels[1]["conditions"]: dict with keys "x", "y", "z"
                     containing prescribed values.

    Returns:
    array: Modified RHS vector with boundary conditions applied.
    """
    _RHS = RHS
    _RHS = _RHS.at[Levels[1]["BC"][2]].set(Levels[1]["conditions"]["y"][0])  # y-min
    _RHS = _RHS.at[Levels[1]["BC"][3]].set(Levels[1]["conditions"]["y"][1])  # y-max
    _RHS = _RHS.at[Levels[1]["BC"][0]].set(Levels[1]["conditions"]["x"][0])  # x-min
    _RHS = _RHS.at[Levels[1]["BC"][1]].set(Levels[1]["conditions"]["x"][1])  # x-max
    _RHS = _RHS.at[Levels[1]["BC"][4]].set(Levels[1]["conditions"]["z"][0])  # z-min

    return _RHS


@jax.jit
def assignBCsFine(RHS, TfAll, BC):
    """
    Apply Dirichlet boundary conditions to the fine-level RHS vector.

    This function sets the RHS values at boundary nodes using the
    corresponding values from the full fine-scale solution `TfAll`.

    Parameters:
    RHS (array): Right-hand side vector to be modified.
    TfAll (array): Full fine-scale solution field.
    BC (list): List of boundary node index arrays [x0, x1, y0, y1, z0].

    Returns:
    array: Modified RHS vector with boundary values applied.
    """
    _RHS = RHS
    _RHS = _RHS.at[BC[2]].set(TfAll[BC[2]])  # y-min
    _RHS = _RHS.at[BC[3]].set(TfAll[BC[3]])  # y-max
    _RHS = _RHS.at[BC[0]].set(TfAll[BC[0]])  # x-min
    _RHS = _RHS.at[BC[1]].set(TfAll[BC[1]])  # x-max
    _RHS = _RHS.at[BC[4]].set(TfAll[BC[4]])  # z-min
    return _RHS


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
    shift_z = ((vcon[2] + powder_layer) / element_size[2] + 1e-2).astype(int)

    Level["overlapNodes"] = [
        Level["orig_overlap_nodes"][0] + shift_x,
        Level["orig_overlap_nodes"][1] + shift_y,
        Level["orig_overlap_nodes"][2] + shift_z,
    ]

    Level["overlapCoords"] = [
        Level["orig_overlap_coors"][0] + element_size[0] * shift_x,
        Level["orig_overlap_coors"][1] + element_size[1] * shift_y,
        Level["orig_overlap_coors"][2] + element_size[2] * (vcon[2] / element_size[2]),
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


@partial(jax.jit, static_argnames=["_idx"])
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


def saveResult(Level, save_str, record_lab, save_path, zoffset):
    """
    Save the temperature and state fields of a mesh level to a VTK file.

    This function exports the structured grid data for visualization,
    including temperature and material state fields, using the VTK format.

    Parameters:
    Level (dict): Dictionary containing mesh and field data:
                  - "node_coords": [x, y, z] coordinate arrays
                  - "T0": temperature field (flattened)
                  - "S1": state field (flattened)
                  - "nodes": [nx, ny, nz] number of nodes in each direction
    save_str (str): Prefix for the output filename.
    record_lab (int): Frame or timestep label for file naming.
    save_path (str): Directory path to save the output file.
    zoffset (float): Offset applied to z-coordinates for rendering purposes.

    Returns:
    None
    """
    # List coordinates in each direction for structured save
    vtkcx = np.array(Level["node_coords"][0])
    vtkcy = np.array(Level["node_coords"][1])
    vtkcz = np.array(Level["node_coords"][2] - zoffset)

    # Reshape the temperature and state fields for correct rendering
    vtkT = np.array(
        Level["T0"].reshape(Level["nodes"][2], Level["nodes"][1], Level["nodes"][0])
    ).transpose((2, 1, 0))
    vtkS = np.array(
        Level["S1"].reshape(Level["nodes"][2], Level["nodes"][1], Level["nodes"][0])
    ).transpose((2, 1, 0))

    # Save a VTK rectilinear grid file
    pointData = {"Temperature (K)": vtkT, "State (Powder/Solid)": vtkS}
    vtkSave = f"{save_path}{save_str}{record_lab:08}"
    gridToVTK(vtkSave, vtkcx, vtkcy, vtkcz, pointData=pointData)


def saveFinalResult(Level, save_str, save_path, zoffset):
    """
    Save the final temperature and state fields of a mesh level to a VTK file.

    This function exports the final structured grid data for visualization,
    including temperature and material state fields, using the VTK format.

    Parameters:
    Level (dict): Dictionary containing mesh and field data:
                  - "node_coords": [x, y, z] coordinate arrays
                  - "T0": temperature field (flattened)
                  - "S1": state field (flattened)
                  - "nodes": [nx, ny, nz] number of nodes in each direction
    save_str (str): Prefix for the output filename.
    save_path (str): Directory path to save the output file.
    zoffset (float): Offset applied to z-coordinates for rendering purposes.

    Returns:
    None
    """
    # List coordinates in each direction for structured save
    vtkcx = np.array(Level["node_coords"][0])
    vtkcy = np.array(Level["node_coords"][1])
    vtkcz = np.array(Level["node_coords"][2] - zoffset)

    # Reshape the temperature and state fields for correct rendering
    vtkT = np.array(
        Level["T0"].reshape(Level["nodes"][2], Level["nodes"][1], Level["nodes"][0])
    ).transpose((2, 1, 0))
    vtkS = np.array(
        Level["S1"].reshape(Level["nodes"][2], Level["nodes"][1], Level["nodes"][0])
    ).transpose((2, 1, 0))

    # Save a VTK rectilinear grid file
    pointData = {"Temperature (K)": vtkT, "State (Powder/Solid)": vtkS}
    vtkSave = f"{save_path}{save_str}Final"
    gridToVTK(vtkSave, vtkcx, vtkcy, vtkcz, pointData=pointData)


def saveState(Level, save_str, record_lab, save_path, zoffset):
    """
    Save the current state field of a mesh level to a VTK file.

    This function exports the structured grid data for visualization,
    including only the material state field (e.g., powder or solid).

    Parameters:
    Level (dict): Dictionary containing mesh and field data:
                  - "node_coords": [x, y, z] coordinate arrays
                  - "S1": state field (flattened)
                  - "nodes": [nx, ny, nz] number of nodes in each direction
    save_str (str): Prefix for the output filename.
    record_lab (int): Frame or timestep label for file naming.
    save_path (str): Directory path to save the output file.
    zoffset (float): Offset applied to z-coordinates for rendering purposes.

    Returns:
    None
    """
    # List coordinates in each direction for structured save
    vtkcx = np.array(Level["node_coords"][0])
    vtkcy = np.array(Level["node_coords"][1])
    vtkcz = np.array(Level["node_coords"][2] - zoffset)

    # Reshape the state field for correct rendering
    vtkS = np.array(
        Level["S1"].reshape(Level["nodes"][2], Level["nodes"][1], Level["nodes"][0])
    ).transpose((2, 1, 0))

    # Save a VTK rectilinear grid file
    pointData = {"State (Powder/Solid)": vtkS}
    vtkSave = f"{save_path}{save_str}{record_lab:08}"
    gridToVTK(vtkSave, vtkcx, vtkcy, vtkcz, pointData=pointData)


@jax.jit
def getNewTprime(Fine, FineT0, CoarseT, Coarse, C2F):
    """
    Compute the fine-level temperature residual (T') and update the coarse-level
    temperature.

    This function interpolates the fine-level temperature onto the overlap region,
    updates the coarse-level temperature at the overlap, and computes the temperature
    difference (residual) between the fine level and the interpolated coarse level.

    Parameters:
    Fine (dict): Fine level mesh and metadata, including:
                 - "overlapCoords": coordinates for interpolation
                 - "overlapNodes": node indices for overlap region
    FineT0 (array): Fine level temperature field.
    CoarseT (array): Coarse level temperature field (to be updated).
    Coarse (dict): Coarse level mesh and metadata, including:
                   - "nodes": [nx, ny, nz] node counts
    C2F (array): Coarse-to-fine interpolation matrix.

    Returns:
    tuple:
        - Tprime (array): Fine-level temperature residual.
        - CoarseT (array): Updated coarse-level temperature field.
    """
    # Interpolate fine temperature at overlap coordinates
    _val = interpolatePoints(Fine, FineT0, Fine["overlapCoords"])

    # Get flattened index of overlap region in coarse mesh
    _idx = getOverlapRegion(
        Fine["overlapNodes"], Coarse["nodes"][0], Coarse["nodes"][1]
    )

    # Update coarse temperature at overlap region
    CoarseT = substitute_Tbar2(CoarseT, _idx, _val)

    # Compute residual between fine and interpolated coarse temperature
    Tprime = FineT0 - interpolate_w_matrix(C2F, CoarseT)

    return Tprime, CoarseT


@jax.jit
def getBothNewTprimes(Levels, FineT, MesoT, M2F, CoarseT, C2M):
    """
    Compute temperature residuals at two nested levels in a multilevel hierarchy.

    This function computes the temperature residuals (T') for both the fine and
    meso levels by updating and interpolating across the hierarchy:
    Fine → Meso → Coarse.

    Parameters:
    Levels (dict): Dictionary of mesh levels:
                   - Levels[1]: Coarse
                   - Levels[2]: Meso
                   - Levels[3]: Fine
    FineT (array): Fine level temperature field.
    MesoT (array): Meso level temperature field.
    M2F (array): Meso-to-fine interpolation matrix.
    CoarseT (array): Coarse level temperature field.
    C2M (array): Coarse-to-meso interpolation matrix.

    Returns:
    tuple:
        - lTprime (array): Fine-level temperature residual.
        - mTprime (array): Meso-level temperature residual.
        - mT0 (array): Updated meso-level temperature field.
        - uT0 (array): Updated coarse-level temperature field.
    """
    lTprime, mT0 = getNewTprime(Levels[3], FineT, MesoT, Levels[2], M2F)
    mTprime, uT0 = getNewTprime(Levels[2], mT0, CoarseT, Levels[1], C2M)

    return lTprime, mTprime, mT0, uT0


@partial(jax.jit, static_argnames=["ne_nn", "tmp_ne_nn"])
def computeSolutions(
    Levels, ne_nn, tmp_ne_nn, LF, L1V, LInterp, Lk, Lrhocp, L2V, dt, properties
):
    """
    Compute temperature solutions across three mesh levels using a matrix-free FEM solver.

    This function performs a multiscale temperature update by solving the heat equation
    at coarse, meso, and fine levels. It uses matrix-free finite element methods and
    interpolates between levels to apply boundary conditions and source terms.

    Parameters:
    Levels (dict): Dictionary of mesh levels:
                   - Levels[1]: Coarse
                   - Levels[2]: Meso
                   - Levels[3]: Fine
    ne_nn (list): List of neighbor-element and node-element mappings for each level.
    tmp_ne_nn (list): Temporary mappings and boundary indices for Level 1.
    LF (dict): Source terms for each level.
    L1V (array): Volume weights for Level 1.
    LInterp (list): Interpolation matrices:
                    - LInterp[0]: Level 1 → Level 2
                    - LInterp[1]: Level 2 → Level 3
    Lk (dict): Thermal conductivity for each level.
    Lrhocp (dict): Volumetric heat capacity for each level.
    L2V (array): Volume weights for Level 2.
    dt (float): Time step size.
    properties (dict): Material properties, including:
                       - "T_amb": Ambient temperature for Dirichlet BCs.

    Returns:
    tuple:
        - FinalL1 (array): Updated temperature field at Level 1.
        - FinalL2 (array): Updated temperature field at Level 2.
        - FinalL3 (array): Updated temperature field at Level 3.
    """
    # Solve coarse-level problem (Level 1)
    L1T = solveMatrixFreeFE(
        Levels[1],
        ne_nn[2],
        tmp_ne_nn[0],
        Lk[1],
        Lrhocp[1],
        dt,
        Levels[1]["T0"],
        LF[1],
        L1V,
    )
    L1T = substitute_Tbar(L1T, tmp_ne_nn[1], properties["T_amb"])
    FinalL1 = assignBCs(L1T, Levels)

    # Interpolate Level 1 solution to Level 2 for source term
    TfAll = interpolate_w_matrix(LInterp[0], FinalL1)

    # Solve meso-level problem (Level 2)
    L2T = solveMatrixFreeFE(
        Levels[2], ne_nn[3], ne_nn[0], Lk[2], Lrhocp[2], dt, Levels[2]["T0"], LF[2], L2V
    )
    FinalL2 = assignBCsFine(L2T, TfAll, Levels[2]["BC"])

    # Interpolate Level 2 solution to Level 3 for boundary conditions
    TfAll = interpolate_w_matrix(LInterp[1], FinalL2)

    # Solve fine-level problem (Level 3)
    FinalL3 = solveMatrixFreeFE(
        Levels[3], ne_nn[4], ne_nn[1], Lk[3], Lrhocp[3], dt, Levels[3]["T0"], LF[3], 0
    )
    FinalL3 = assignBCsFine(FinalL3, TfAll, Levels[3]["BC"])

    return FinalL1, FinalL2, FinalL3


@partial(jax.jit, static_argnames=["ne", "nn"])
def computeConvRadBC(Level, LevelT0, ne, nn, properties, F):
    """
    Compute Neumann boundary conditions on the top surface due to:
      • Convection (Newton's law of cooling)
      • Radiation (Stefan-Boltzmann law)
      • Evaporation (empirical model)

    The resulting heat flux is integrated using 2D quadrature and added
    to the global force vector.

    Parameters:
    Level (dict): Mesh data with:
                  - "node_coords": [x, y] coordinate arrays
                  - "connect": [cx, cy] element connectivity arrays
    LevelT0 (array): Temperature field at current time step.
    ne (int): Total number of elements.
    nn (int): Total number of nodes.
    properties (dict): Material and physical constants, including:
                       - h_conv, sigma_sb, T_amb, T_boiling, evc, Lev,
                         cp_fluid, vareps, CM_coeff, CT_coeff, CP_coeff
    F (array): Global force vector to be updated.

    Returns:
    array: Updated force vector with top-surface Neumann BCs applied.
    """
    # Extract physical constants
    T_amb = properties["T_amb"]
    T_boiling = properties["T_boiling"]
    vareps = properties["vareps"]
    evc = properties["evc"]
    Lev = properties["Lev"]
    cp_fluid = properties["cp_fluid"]
    h_conv = properties["h_conv"]
    sigma_sb = properties["sigma_sb"]

    # Evaporation model coefficients (precomputed in SetupProperties)
    invT_b = 1.0 / T_boiling
    CM_coeff = properties["CM_coeff"]
    CT_coeff = properties["CT_coeff"]
    CP_coeff = properties["CP_coeff"]

    # Mesh geometry
    x, y = Level["node_coords"][0], Level["node_coords"][1]
    cx, cy = Level["connect"][0], Level["connect"][1]
    ne_x = jnp.size(cx, 0)
    ne_y = jnp.size(cy, 0)
    top_ne = ne - ne_x * ne_y
    nn_x, nn_y = ne_x + 1, ne_y + 1

    # Coordinates of a representative top-surface element
    coords = jnp.stack([x[cx[0, :]], y[cy[0, :]]], axis=1)

    # Precompute shape functions and quadrature weights
    N, _, wq = computeQuad2dFemShapeFunctions_jax(coords)

    def calcCR(i):
        """
        Compute the integrated heat flux vector for a single top-surface element.
        """
        _, _, _, idx = convert2XYZ(i, ne_x, ne_y, nn_x, nn_y)

        # Interpolate nodal temperatures to quadrature points
        Tq = jnp.matmul(N, LevelT0[idx[4:]])
        Tq = jnp.minimum(Tq, T_boiling + 1000)

        invT = 1.0 / Tq

        # Energy required for phase change and heating of vapor
        # Includes latent heat and sensible heat from ambient to surface temperature
        E_pv = Lev + cp_fluid * (Tq - T_amb)  # J/kg

        # Molecular motion term from kinetic theory
        # Proportional to sqrt(m / (2πRT)), affects evaporation rate
        MolMot = jnp.sqrt(CM_coeff * invT)  # s/m

        # Evaporative heat loss (W/m²)
        S = evc * CP_coeff * jnp.exp(-CT_coeff * (invT - invT_b)) * MolMot * E_pv

        # Total heat flux (positive into the domain)
        q_flux = h_conv * (T_amb - Tq) + sigma_sb * vareps * (T_amb**4 - Tq**4) - S

        # Convert from W/m² to W/mm²
        q_flux *= 1e-6

        # Integrate over the element
        return jnp.matmul(N.T, q_flux * wq.reshape(-1)), idx[4:]

    # Vectorized computation over all top-surface elements
    aT, aidx = jax.vmap(calcCR)(jnp.arange(top_ne, ne))

    # Assemble into global force vector
    NeumannBC = jnp.bincount(aidx.reshape(-1), aT.reshape(-1), length=nn)

    return F + NeumannBC


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
    Vcu, Vmu = computeCoarseTprimeTerm_jax(Levels, Lk[3], Lk[2], Shapes)
    L1T, L2T, L3T = computeSolutions(
        Levels, ne_nn, tmp_ne_nn, F, Vcu, LInterp, Lk, Lrhocp, Vmu, dt, properties
    )

    # Enforce minimum temperature (TFSP)
    L1T = jnp.maximum(properties["T_amb"], L1T)
    L2T = jnp.maximum(properties["T_amb"], L2T)
    L3T = jnp.maximum(properties["T_amb"], L3T)

    # --- Compute new Tprime fields ---
    L3Tp, L2Tp, L2T, L1T = getBothNewTprimes(
        Levels, L3T, L2T, LInterp[1], L1T, LInterp[0]
    )

    # --- Update volumetric correction terms ---
    Vcu, Vmu = computeCoarseTprimeMassTerm_jax(
        Levels, L3Tp, L2Tp, Lrhocp[3], Lrhocp[2], dt, Shapes, Vcu, Vmu
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
    Levels[3]["Tprime0"], Levels[2]["Tprime0"], Levels[2]["T0"], Levels[1]["T0"] = (
        getBothNewTprimes(Levels, Levels[3]["T0"], L2T, LInterp[1], L1T, LInterp[0])
    )

    # --- Update global state arrays ---
    Levels[0]["S1"] = Levels[0]["S1"].at[Levels[0]["idx"]].set(Levels[3]["S1"])
    Levels[0]["S2"] = Levels[0]["S2"].at[:].set(False)
    Levels[0]["S2"] = Levels[0]["S2"].at[Levels[0]["idx"]].set(Levels[3]["S2"])

    # Identify newly activated melt regions
    _resetmask = ((1 - 2 * preS2) * Levels[3]["S2"]) == 1

    return Levels, _resetmask


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


@partial(jax.jit, static_argnames=["ne_nn", "tmp_ne_nn", "substrate"])
def stepGOMELTDwellTime(Levels, tmp_ne_nn, ne_nn, properties, dt, substrate):
    """
    Perform a dwell-time update for Level 1 in the GO-MELT framework.

    This function updates the temperature field of Level 1 during laser dwell time,
    assuming no motion of finer levels. It includes convection, radiation, and
    evaporation boundary conditions, and enforces inactive nodes above the powder line.

    Parameters:
    Levels (list of dict): Multilevel simulation state. Only Levels[1] is updated.
    tmp_ne_nn (tuple): Level 1 metadata.
        - tmp_ne_nn[0] (int): Number of elements in Level 1.
        - tmp_ne_nn[1] (int): Index above which nodes are inactive (above surface).
    ne_nn (tuple): Neighbor and node connectivity data.
        - ne_nn[2]: Connectivity for Level 1.
    properties (dict): Material properties.
    dt (float): Time step size (s).
    substrate (tuple): Substrate information.
        - substrate[1]: Number of substrate nodes (always solid).

    Returns:
    list of dict: Updated Levels list with modified temperature field in Levels[1].
    """
    L1 = Levels[1]
    num_elements_L1 = tmp_ne_nn[0]
    inactive_start_idx = tmp_ne_nn[1]

    # Compute boundary conditions (includes convection, radiation, evaporation)
    Fc = computeConvRadBC(L1, L1["T0"], num_elements_L1, ne_nn[2], properties, F=0)

    # Compute temperature-dependent properties
    _, _, k, rhocp = computeStateProperties(
        L1["T0"], L1["S1"], properties, substrate[1]
    )

    # Solve temperature field using matrix-free FEM
    T_new = solveMatrixFreeFE(
        L1, ne_nn[2], num_elements_L1, k, rhocp, dt, L1["T0"], Fc, 0
    )

    # Enforce ambient temperature in inactive region above surface
    T_new = substitute_Tbar(T_new, inactive_start_idx, properties["T_amb"])

    # Apply boundary conditions and update Level 1 temperature
    Levels[1]["T0"] = assignBCs(T_new, Levels)

    return Levels


@partial(jax.jit, static_argnames=["ne_nn"])
def computeLevelSource(Levels, ne_nn, laser_position, LevelShape, properties, laserP):
    """
    Compute the volumetric heat source for Level 1 or Level 2 using Level 3 mesh.

    This function evaluates the laser heat source on the Level 3 mesh using
    quadrature integration, then projects the result onto the target level
    (Level 1 or Level 2) using the provided shape function matrices.

    Parameters:
    Levels (list of dict): Multilevel simulation state. Level 3 is used for integration.
    ne_nn (tuple): Mesh metadata.
        - ne_nn[1] (int): Number of elements in Level 3.
    laser_position (array): Laser center positions for each time step.
    LevelShape (tuple): FEM shape function data for the target level.
        - LevelShape[0] (array): Shape function values at quadrature points.
        - LevelShape[2] (array): Projection matrix to target level's global vector.
    properties (dict): Material and process properties.
    laserP (array): Laser power or intensity at each time step.

    Returns:
    array: Assembled global heat source vector for the target level.
    """
    # Get quadrature shape functions and weights for Level 3
    coords = getSampleCoords(Levels[3])
    Nf, _, wqf = computeQuad3dFemShapeFunctions_jax(coords)

    def stepLaserPosition(ilaser):
        def stepcomputeCoarseSource(ieltf):
            # Convert element index to 3D indices and global node index
            ix, iy, iz, idx = convert2XYZ(
                ieltf,
                Levels[3]["elements"][0],
                Levels[3]["elements"][1],
                Levels[3]["nodes"][0],
                Levels[3]["nodes"][1],
            )
            # Get quadrature point coordinates for this element
            x, y, z = getQuadratureCoords(Levels[3], ix, iy, iz, Nf)
            w = wqf

            # Evaluate laser source at quadrature points
            Q = computeSourceFunction_jax(
                x, y, z, laser_position[ilaser], properties, laserP[ilaser]
            )
            return Q * w  # Weighted source term

        # Vectorize over all Level 3 elements
        vstepcomputeCoarseSource = jax.vmap(stepcomputeCoarseSource)
        _data = vstepcomputeCoarseSource(jnp.arange(ne_nn[1]))

        # Integrate over elements using shape functions
        _data1tmp = multiply(LevelShape[0], _data).sum(axis=1)
        return _data1tmp

    # Vectorize over all laser positions
    vstepLaserPosition = jax.vmap(stepLaserPosition)

    # Average source over all laser positions
    lshape = laser_position.shape[0]
    _data1 = vstepLaserPosition(jnp.arange(lshape)).sum(axis=0) / lshape

    # Project integrated source to global vector of target level
    return LevelShape[2] @ _data1.reshape(-1)


@partial(jax.jit, static_argnames=["ne_nn"])
def computeL1TprimeTerms_Part1(Levels, ne_nn, L3k, Shapes, L2k):
    """
    Compute the subgrid correction terms for Level 1 from Levels 2 and 3.

    This function calculates the divergence of the subgrid heat fluxes from
    Level 2 and Level 3 and projects them onto the Level 1 mesh. These terms
    are used in the predictor step of the GO-MELT predictor-corrector scheme.

    Parameters:
    Levels (list of dict): Multilevel simulation state.
        - Levels[2]["Tprime0"]: Subgrid temperature field from Level 2.
        - Levels[3]["Tprime0"]: Subgrid temperature field from Level 3.
    ne_nn (tuple): Mesh metadata.
        - ne_nn[0]: Number of elements in Level 2.
        - ne_nn[1]: Number of elements in Level 3.
    L3k (array): Thermal conductivity field for Level 3.
    Shapes (tuple): Shape function data for projection.
        - Shapes[0][1]: ∇N₂ᵀ ∇N₁ mapping for Level 2 to Level 1.
        - Shapes[0][2]: Projection matrix from Level 2 to Level 1.
        - Shapes[1][1]: ∇N₃ᵀ ∇N₁ mapping for Level 3 to Level 1.
        - Shapes[1][2]: Projection matrix from Level 3 to Level 1.
    L2k (array): Thermal conductivity field for Level 2.

    Returns:
    array: Combined subgrid correction term projected onto Level 1.
    """

    # --- Level 3 subgrid flux divergence ---
    coordsf = getSampleCoords(Levels[3])
    Nf, dNdxf, wqf = computeQuad3dFemShapeFunctions_jax(coordsf)

    _, _, _, idxf = convert2XYZ(
        jnp.arange(ne_nn[1]),
        Levels[3]["elements"][0],
        Levels[3]["elements"][1],
        Levels[3]["nodes"][0],
        Levels[3]["nodes"][1],
    )
    _L3Tp0 = Levels[3]["Tprime0"][idxf]
    L3kMean = jnp.matmul(Nf, L3k[idxf]).mean(axis=0)

    # Compute ∇·(k ∇T') for Level 3
    dL3Tp0dX = [multiply(L3kMean, (dNdxf[:, :, i] @ _L3Tp0)) for i in range(3)]
    _wqf = wqf[None, None, :]
    _dL3L1dX = Shapes[1][1]

    _1 = sum(
        multiply(multiply(-_dL3L1dX[i], dL3Tp0dX[i].T[:, :, None]), _wqf).sum(axis=2)
        for i in range(3)
    )

    # --- Level 2 subgrid flux divergence ---
    coordsm = getSampleCoords(Levels[2])
    Nm, dNdxm, wqm = computeQuad3dFemShapeFunctions_jax(coordsm)

    _, _, _, idxm = convert2XYZ(
        jnp.arange(ne_nn[0]),
        Levels[2]["elements"][0],
        Levels[2]["elements"][1],
        Levels[2]["nodes"][0],
        Levels[2]["nodes"][1],
    )
    _L2Tp0 = Levels[2]["Tprime0"][idxm]
    L2kMean = jnp.matmul(Nm, L2k[idxm]).mean(axis=0)

    # Compute ∇·(k ∇T') for Level 2
    dL2Tp0dX = [multiply(L2kMean, (dNdxm[:, :, i] @ _L2Tp0)) for i in range(3)]
    _wqm = wqm[None, None, :]
    _dL2L1dX = Shapes[0][1]

    _2 = sum(
        multiply(multiply(-_dL2L1dX[i], dL2Tp0dX[i].T[:, :, None]), _wqm).sum(axis=2)
        for i in range(3)
    )

    # Project both subgrid terms to Level 1
    return Shapes[1][2] @ _1.reshape(-1) + Shapes[0][2] @ _2.reshape(-1)


@partial(jax.jit, static_argnames=["ne_nn", "tmp_ne_nn"])
def computeL1Temperature(
    Levels, ne_nn, tmp_ne_nn, L1F, L1V, L1k, L1rhocp, dt, properties
):
    """
    Compute the updated temperature field for Level 1.

    This function performs the temperature update for Level 1 using a matrix-free
    finite element solver. It includes source terms, subgrid corrections, and
    enforces boundary and ambient conditions.

    Parameters:
    Levels (list of dict): Multilevel simulation state.
        - Levels[1]["T0"]: Current temperature field for Level 1.
    ne_nn (tuple): Mesh connectivity data.
        - ne_nn[2]: Connectivity for Level 1.
    tmp_ne_nn (tuple): Level 1 metadata.
        - tmp_ne_nn[0]: Number of elements in Level 1.
        - tmp_ne_nn[1]: Index above which nodes are inactive (above surface).
    L1F (array): Heat source vector for Level 1.
    L1V (array): Subgrid correction vector for Level 1.
    L1k (array): Thermal conductivity field for Level 1.
    L1rhocp (array): Volumetric heat capacity field for Level 1.
    dt (float): Time step size (s).
    properties (dict): Material and process properties.
        - properties["T_amb"]: Ambient temperature (K).

    Returns:
    array: Updated temperature field for Level 1.
    """
    # Solve temperature field using matrix-free FEM
    L1T = solveMatrixFreeFE(
        Levels[1], ne_nn[2], tmp_ne_nn[0], L1k, L1rhocp, dt, Levels[1]["T0"], L1F, L1V
    )

    # Apply ambient temperature to inactive region above surface
    L1T = substitute_Tbar(L1T, tmp_ne_nn[1], properties["T_amb"])

    # Enforce Dirichlet boundary conditions
    FinalL1 = assignBCs(L1T, Levels)

    return FinalL1


@partial(jax.jit, static_argnames=["ne_nn"])
def computeL2TprimeTerms_Part1(Levels, ne_nn, L3Tprime0, L3k, Shapes):
    """
    Compute the subgrid correction term for Level 2 from Level 3.

    This function calculates the divergence of the subgrid heat flux from Level 3
    and projects it onto the Level 2 mesh. This term is used in the predictor step
    of the GO-MELT predictor-corrector scheme.

    Parameters:
    Levels (list of dict): Multilevel simulation state.
        - Levels[3]["elements"], ["nodes"]: Mesh structure for Level 3.
    ne_nn (tuple): Mesh metadata.
        - ne_nn[1]: Number of elements in Level 3.
    L3Tprime0 (array): Subgrid temperature field from Level 3.
    L3k (array): Thermal conductivity field for Level 3.
    Shapes (tuple): Shape function data.
        - Shapes[2][1]: ∇N₃ᵀ ∇N₂ mapping for Level 3 to Level 2.
        - Shapes[2][2]: Projection matrix from Level 3 to Level 2.

    Returns:
    array: Subgrid correction term projected onto Level 2.
    """
    # Get quadrature shape functions and weights for Level 3
    coordsf = getSampleCoords(Levels[3])
    Nf, dNdxf, wqf = computeQuad3dFemShapeFunctions_jax(coordsf)

    # Get global node indices for all Level 3 elements
    _, _, _, idxf = convert2XYZ(
        jnp.arange(ne_nn[1]),
        Levels[3]["elements"][0],
        Levels[3]["elements"][1],
        Levels[3]["nodes"][0],
        Levels[3]["nodes"][1],
    )

    # Extract subgrid temperature and conductivity
    _L3Tp0 = L3Tprime0[idxf]
    L3kMean = jnp.matmul(Nf, L3k[idxf]).mean(axis=0)

    # Compute ∇·(k ∇T') for Level 3
    dL3Tp0dx = multiply(L3kMean, (dNdxf[:, :, 0] @ _L3Tp0))
    dL3Tp0dy = multiply(L3kMean, (dNdxf[:, :, 1] @ _L3Tp0))
    dL3Tp0dz = multiply(L3kMean, (dNdxf[:, :, 2] @ _L3Tp0))

    # Weighted divergence projection to Level 2
    _1 = multiply(
        multiply(-Shapes[2][1][0], dL3Tp0dx.T[:, :, None]), wqf[None, None, :]
    ).sum(axis=2)
    _1 += multiply(
        multiply(-Shapes[2][1][1], dL3Tp0dy.T[:, :, None]), wqf[None, None, :]
    ).sum(axis=2)
    _1 += multiply(
        multiply(-Shapes[2][1][2], dL3Tp0dz.T[:, :, None]), wqf[None, None, :]
    ).sum(axis=2)

    # Project to global Level 2 vector
    return Shapes[2][2] @ _1.reshape(-1)


@partial(jax.jit, static_argnames=["ne_nn"])
def computeL2Temperature(
    L1T, L1L2Interp, Levels, ne_nn, L2T0, L2F, L2V, L2k, L2rhocp, dt
):
    """
    Compute the updated temperature field for Level 2.

    This function solves the meso-scale (Level 2) temperature field using
    matrix-free FEM, incorporating subgrid corrections and boundary conditions
    interpolated from the coarse-scale (Level 1) solution.

    Parameters:
    L1T (array): Temperature field from Level 1.
    L1L2Interp (array): Interpolation matrix from Level 1 to Level 2 boundary.
    Levels (list of dict): Multilevel simulation state.
        - Levels[2]: Contains Level 2 mesh and boundary condition metadata.
    ne_nn (tuple): Mesh metadata.
        - ne_nn[0]: Number of elements in Level 2.
        - ne_nn[3]: Connectivity for Level 2.
    L2T0 (array): Initial temperature field for Level 2.
    L2F (array): Heat source vector for Level 2.
    L2V (array): Subgrid correction vector for Level 2.
    L2k (array): Thermal conductivity field for Level 2.
    L2rhocp (array): Volumetric heat capacity field for Level 2.
    dt (float): Time step size (s).

    Returns:
    array: Updated temperature field for Level 2.
    """
    # Interpolate Level 1 temperature to Level 2 boundary
    TfAll = interpolate_w_matrix(L1L2Interp, L1T)

    # Solve Level 2 temperature using matrix-free FEM
    L2T = solveMatrixFreeFE(
        Levels[2], ne_nn[3], ne_nn[0], L2k, L2rhocp, dt, L2T0, L2F, L2V
    )

    # Apply Dirichlet boundary conditions from interpolated Level 1 solution
    FinalL2 = assignBCsFine(L2T, TfAll, Levels[2]["BC"])

    return FinalL2


@partial(jax.jit, static_argnames=["ne_nn"])
def computeSourcesL3(Level, v, ne_nn, properties, laserP):
    """
    Compute the integrated volumetric heat source for Level 3.

    This function evaluates the laser heat source at quadrature points
    using the Level 3 mesh and assembles the global source vector.

    Parameters:
    Level (dict): Level 3 mesh and field data.
        - Level["elements"]: Element dimensions (nx, ny, nz).
        - Level["nodes"]: Node dimensions (nx, ny, nz).
    v (array): Current laser position.
    ne_nn (tuple): Mesh metadata.
        - ne_nn[1]: Number of elements in Level 3.
        - ne_nn[4]: Total number of nodes in Level 3.
    properties (dict): Material and process properties.
    laserP (float): Laser power at the current position.

    Returns:
    array: Assembled global source vector for Level 3.
    """
    # Get shape functions and quadrature weights
    coords = getSampleCoords(Level)
    Nf, _, wqf = computeQuad3dFemShapeFunctions_jax(coords)

    def stepcomputeCoarseSource(ieltf):
        # Convert element index to 3D indices and global node indices
        ix, iy, iz, idx = convert2XYZ(
            ieltf,
            Level["elements"][0],
            Level["elements"][1],
            Level["nodes"][0],
            Level["nodes"][1],
        )
        # Get quadrature point coordinates for this element
        x, y, z = getQuadratureCoords(Level, ix, iy, iz, Nf)
        w = wqf

        # Evaluate laser source at quadrature points
        Q = computeSourceFunction_jax(x, y, z, v, properties, laserP)

        # Integrate source over element using shape functions
        return Nf @ Q * w, idx

    # Vectorize over all Level 3 elements
    vstepcomputeCoarseSource = jax.vmap(stepcomputeCoarseSource)
    _data3, nodes3 = vstepcomputeCoarseSource(jnp.arange(ne_nn[1]))

    # Assemble global source vector using node indices
    Ff = bincount(nodes3.reshape(-1), _data3.reshape(-1), ne_nn[4])

    return Ff


@partial(jax.jit, static_argnames=["ne_nn"])
def computeSolutions_L3(
    FinalL2, L2L3Interp, Levels, ne_nn, L3T0, L3F, L3k, L3rhocp, dt
):
    """
    Compute the updated temperature field for Level 3.

    This function solves the fine-scale (Level 3) temperature field using
    matrix-free FEM, applying Dirichlet boundary conditions interpolated
    from the meso-scale (Level 2) solution.

    Parameters:
    FinalL2 (array): Updated temperature field from Level 2.
    L2L3Interp (array): Interpolation matrix from Level 2 to Level 3 boundary.
    Levels (list of dict): Multilevel simulation state.
        - Levels[3]: Contains Level 3 mesh and boundary condition metadata.
    ne_nn (tuple): Mesh metadata.
        - ne_nn[1]: Number of elements in Level 3.
        - ne_nn[4]: Connectivity for Level 3.
    L3T0 (array): Initial temperature field for Level 3.
    L3F (array): Heat source vector for Level 3.
    L3k (array): Thermal conductivity field for Level 3.
    L3rhocp (array): Volumetric heat capacity field for Level 3.
    dt (float): Time step size (s).

    Returns:
    array: Updated temperature field for Level 3.
    """
    # Interpolate Level 2 temperature to Level 3 boundary
    TfAll = interpolate_w_matrix(L2L3Interp, FinalL2)

    # Solve Level 3 temperature using matrix-free FEM
    FinalL3 = solveMatrixFreeFE(
        Levels[3], ne_nn[4], ne_nn[1], L3k, L3rhocp, dt, L3T0, L3F, 0
    )

    # Apply Dirichlet boundary conditions from interpolated Level 2 solution
    FinalL3 = assignBCsFine(FinalL3, TfAll, Levels[3]["BC"])

    return FinalL3


@partial(jax.jit, static_argnames=["ne_nn"])
def computeL1TprimeTerms_Part2(
    Levels, ne_nn, L3Tp, L2Tp, L3rhocp, L2rhocp, dt, Shapes, Vcu
):
    """
    Compute the time derivative subgrid correction terms for Level 1.

    This function calculates the contribution of the time derivative of the
    subgrid temperature fields from Levels 2 and 3, and projects them onto
    the Level 1 mesh. These terms are added to the Level 1 correction vector.

    Parameters:
    Levels (list of dict): Multilevel simulation state.
        - Levels[2]["Tprime0"], Levels[3]["Tprime0"]: Previous subgrid fields.
    ne_nn (tuple): Mesh metadata.
        - ne_nn[0]: Number of elements in Level 2.
        - ne_nn[1]: Number of elements in Level 3.
    L3Tp (array): Updated subgrid temperature field for Level 3.
    L2Tp (array): Updated subgrid temperature field for Level 2.
    L3rhocp (array): Volumetric heat capacity for Level 3.
    L2rhocp (array): Volumetric heat capacity for Level 2.
    dt (float): Time step size (s).
    Shapes (tuple): Shape function data.
        - Shapes[0][0]: N₂ᵀ N₁ mapping for Level 2 to Level 1.
        - Shapes[0][2]: Projection matrix from Level 2 to Level 1.
        - Shapes[1][0]: N₃ᵀ N₁ mapping for Level 3 to Level 1.
        - Shapes[1][2]: Projection matrix from Level 3 to Level 1.
    Vcu (array): Level 1 correction vector to be updated.

    Returns:
    array: Updated Level 1 correction vector including time derivative terms.
    """
    # Compute change in subgrid temperature fields
    L3Tp_new = L3Tp - Levels[3]["Tprime0"]
    L2Tp_new = L2Tp - Levels[2]["Tprime0"]

    # --- Level 3 contribution ---
    coordsf = getSampleCoords(Levels[3])
    Nf, _, wqf = computeQuad3dFemShapeFunctions_jax(coordsf)

    _, _, _, idxf = convert2XYZ(
        jnp.arange(ne_nn[1]),
        Levels[3]["elements"][0],
        Levels[3]["elements"][1],
        Levels[3]["nodes"][0],
        Levels[3]["nodes"][1],
    )

    # Integrate ρcp * T' over Level 3 elements
    _L3Tp = multiply(Nf @ L3Tp_new[idxf], jnp.matmul(Nf, L3rhocp[idxf]).mean(axis=0))
    _1 = multiply(
        multiply(-Shapes[1][0], _L3Tp.T[:, :, None]), (1 / dt) * wqf[None, None, :]
    ).sum(axis=2)

    # --- Level 2 contribution ---
    coordsm = getSampleCoords(Levels[2])
    Nm, _, wqm = computeQuad3dFemShapeFunctions_jax(coordsm)

    _, _, _, idxm = convert2XYZ(
        jnp.arange(ne_nn[0]),
        Levels[2]["elements"][0],
        Levels[2]["elements"][1],
        Levels[2]["nodes"][0],
        Levels[2]["nodes"][1],
    )

    # Integrate ρcp * T' over Level 2 elements
    _L2Tp = multiply(Nm @ L2Tp_new[idxm], jnp.matmul(Nm, L2rhocp[idxm]).mean(axis=0))
    _2 = multiply(
        multiply(-Shapes[0][0], _L2Tp.T[:, :, None]), (1 / dt) * wqm[None, None, :]
    ).sum(axis=2)

    # Project both contributions to Level 1 and update correction vector
    Vcu += Shapes[1][2] @ _1.reshape(-1) + Shapes[0][2] @ _2.reshape(-1)

    return Vcu


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


def getQuadratureCoords(Level, ix, iy, iz, Nf):
    """
    Computes the quadrature point coordinates for a given element.

    Parameters:
    Level (dict): Mesh data for a given level.
    ix, iy, iz (int): Element indices in x, y, z directions.
    Nf (array): Shape function matrix.

    Returns:
    tuple: x, y, z coordinates at quadrature points.
    """
    coords_x = Level["node_coords"][0][Level["connect"][0][ix, :]].reshape(-1, 1)
    coords_y = Level["node_coords"][1][Level["connect"][1][iy, :]].reshape(-1, 1)
    coords_z = Level["node_coords"][2][Level["connect"][2][iz, :]].reshape(-1, 1)
    return Nf @ coords_x, Nf @ coords_y, Nf @ coords_z


@partial(jax.jit, static_argnames=["ne_nn"])
def computeL2TprimeTerms_Part2(Levels, ne_nn, L3Tp, L3Tp0, L3rhocp, dt, Shapes, L2V):
    """
    Compute the time derivative subgrid correction term for Level 2 from Level 3.

    This function calculates the contribution of the time derivative of the
    subgrid temperature field from Level 3 and projects it onto the Level 2 mesh.

    Parameters:
    Levels (list of dict): Multilevel simulation state.
        - Levels[3]["Tprime0"]: Previous subgrid temperature field.
    ne_nn (tuple): Mesh metadata.
        - ne_nn[1]: Number of elements in Level 3.
    L3Tp (array): Updated subgrid temperature field for Level 3.
    L3Tp0 (array): Previous subgrid temperature field for Level 3.
    L3rhocp (array): Volumetric heat capacity for Level 3.
    dt (float): Time step size (s).
    Shapes (tuple): Shape function data.
        - Shapes[2][0]: N₃ᵀ N₂ mapping for Level 3 to Level 2.
        - Shapes[2][2]: Projection matrix from Level 3 to Level 2.
    L2V (array): Level 2 correction vector to be updated.

    Returns:
    array: Updated Level 2 correction vector including time derivative terms.
    """
    # Compute change in subgrid temperature field
    L3Tp_new = L3Tp - L3Tp0

    # Get shape functions and weights for Level 3
    coordsf = getSampleCoords(Levels[3])
    Nf, _, wqf = computeQuad3dFemShapeFunctions_jax(coordsf)

    # Get global node indices for Level 3 elements
    _, _, _, idxf = convert2XYZ(
        jnp.arange(ne_nn[1]),
        Levels[3]["elements"][0],
        Levels[3]["elements"][1],
        Levels[3]["nodes"][0],
        Levels[3]["nodes"][1],
    )

    # Integrate ρcp * T' over Level 3 elements
    _L3Tp = multiply(Nf @ L3Tp_new[idxf], jnp.matmul(Nf, L3rhocp[idxf]).mean(axis=0))

    # Compute weighted projection to Level 2
    _data2 = multiply(
        multiply(-Shapes[2][0], _L3Tp.T[:, :, None]), (1 / dt) * wqf[None, None, :]
    ).sum(axis=2)

    # Update Level 2 correction vector
    L2V += Shapes[2][2] @ _data2.reshape(-1)

    return L2V


@partial(jax.jit, static_argnames=["ne_nn", "tmp_ne_nn", "substrate", "subcycle"])
def subcycleGOMELT(
    Levels,
    ne_nn,
    Shapes,
    substrate,
    LInterp,
    tmp_ne_nn,
    laser_position,
    properties,
    laserP,
    subcycle,
    max_accum_L3,
    accum_L3,
):
    """
    Perform a full predictor-corrector subcycling step for the GO-MELT model.

    This function executes the multilevel thermal update across Levels 1, 2, and 3,
    using nested subcycling and subgrid scale corrections. It includes both the
    predictor and corrector phases, updating temperature fields and phase states.

    Parameters:
    Levels (list of dict): Multilevel simulation state.
    ne_nn (tuple): Mesh metadata for all levels.
    Shapes (tuple): Shape function data for subgrid projections.
    substrate (tuple): Substrate node indices for each level.
    LInterp (tuple): Interpolation matrices between levels.
    tmp_ne_nn (tuple): Level 1 metadata (element count, inactive node index).
    laser_position (array): Laser positions for each substep.
    properties (dict): Material and process properties.
    laserP (array): Laser power at each substep.
    subcycle (tuple): Subcycling parameters (L2 steps, L3 steps per L2, etc.).
    max_accum_L3 (array): Max accumulated melt time for Level 3 nodes.
    accum_L3 (array): Current accumulated melt time for Level 3 nodes.

    Returns:
    tuple: Updated Levels, Level 2 and 3 temperature histories, Level 3 Tprime history,
           updated max_accum_L3 and accum_L3.
    """
    # --- Level 1 Material Properties ---
    _, _, L3k_L1, L3rhocp_L1 = computeStateProperties(
        Levels[3]["T0"], Levels[3]["S1"], properties, substrate[3]
    )
    _, _, L2k_L1, L2rhocp_L1 = computeStateProperties(
        Levels[2]["T0"], Levels[2]["S1"], properties, substrate[2]
    )

    # Update Level 1 S1 from Level 2 overlap
    _val = interpolatePoints(Levels[2], Levels[2]["S1"], Levels[2]["overlapCoords"])
    _idx = getOverlapRegion(
        Levels[2]["overlapNodes"], Levels[1]["nodes"][0], Levels[1]["nodes"][1]
    )
    Levels[1]["S1"] = Levels[1]["S1"].at[_idx].set(_val)
    Levels[1]["S1"] = Levels[1]["S1"].at[: substrate[1]].set(1)

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

    # --- Subcycle Level 2 ---
    def subcycleL2_Part1(_L2carry, _L2sub):
        # Compute interpolation weights for Level 2 boundary conditions
        alpha_L2 = (_L2sub + 1) / subcycle[3]
        beta_L2 = 1 - alpha_L2

        # Determine the laser substeps for this Level 2 subcycle
        Lidx = _L2sub * subcycle[1] + jnp.arange(subcycle[1])

        # --- Material Properties ---
        # Compute Level 3 properties using current Level 3 temperature and phase
        _, _, L3k_L2, _ = computeStateProperties(
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

        # --- Temperature Solve ---
        # Interpolate Level 1 temperature to Level 2 boundary using alpha-beta blend
        _BC = alpha_L2 * L1T + beta_L2 * Levels[1]["T0"]

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
            # Compute Level 3 material properties
            L3S1, _, L3k, L3rhocp = computeStateProperties(
                _L3carry[0], _L3carry[1], properties, substrate[3]
            )

            # Determine laser index for this Level 3 substep
            LLidx = _L3sub + _L2sub * subcycle[1]

            # Compute laser source term for Level 3
            L3F = computeSourcesL3(
                Levels[3], laser_position[LLidx, :], ne_nn, properties, laserP[LLidx]
            )

            # Add convection, radiation, and evaporation boundary conditions
            L3F = computeConvRadBC(
                Levels[3], _L3carry[0], ne_nn[1], ne_nn[4], properties, L3F
            )

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
        # Interpolate Level 1 temperature to Level 2 boundary using alpha-beta blend
        _BC = alpha_L2 * L1T + beta_L2 * Levels[1]["T0"]

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
            L3F = computeConvRadBC(
                Levels[3], _L3carry[0], ne_nn[1], ne_nn[4], properties, L3F
            )

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


def saveResults(Levels, Nonmesh, savenum):
    """
    Save temperature results for Levels 1-3 based on save frequency and flags.
    """
    if Nonmesh["output_files"] == 1:
        if savenum == 1 or (
            np.mod(savenum, Nonmesh["Level1_record_step"]) == 1
            or Nonmesh["Level1_record_step"] == 1
        ):
            saveResult(Levels[1], "Level1_", savenum, Nonmesh["save_path"], 2e-3)
            # saveState(Levels[0], "Level0_", savenum, Nonmesh["save_path"], 0)

        saveResult(Levels[2], "Level2_", savenum, Nonmesh["save_path"], 1e-3)
        saveResult(Levels[3], "Level3_", savenum, Nonmesh["save_path"], 0)
        print(f"Saved Levels_{savenum:08}")


def saveResultsFinal(Levels, Nonmesh):
    """
    Save final temperature results for Levels 1-3.
    """
    if Nonmesh["output_files"] == 1:
        saveFinalResult(Levels[1], "Level1_", Nonmesh["save_path"], 2e-3)
        saveFinalResult(Levels[2], "Level2_", Nonmesh["save_path"], 1e-3)
        saveFinalResult(Levels[3], "Level3_", Nonmesh["save_path"], 0)
        print("Saved Final Results")


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

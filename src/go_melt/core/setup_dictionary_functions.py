import json
import os
import jax.numpy as jnp
import copy
from .computeFunctions import *
from .boundary_condition_functions import getBCindices
from .move_mesh_functions import find_max_const


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
    properties.h_conv *= 1e6  # Quick fix: later assumes in m for surface BC calc
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


def SetupStaticNodesAndElements(L):
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


def SetupStaticSubcycle(N):
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

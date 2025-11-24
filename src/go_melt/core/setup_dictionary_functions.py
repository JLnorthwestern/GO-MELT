import json
import os
import jax.numpy as jnp
import copy
from .boundary_condition_functions import getBCindices
from .mesh_functions import calc_length_h, calcNumNodes, createMesh3D
from .data_structures import obj
from go_melt.utils.helper_functions import (
    getCoarseNodesInLargeFineRegion,
    getCoarseNodesInFineRegion,
    getOverlapRegion,
)
from typing import Union
import pint


def SetupLevels(solver_input: dict, properties: dict) -> list[dict]:
    """
    Set up multilevel mesh structures and initialize physical fields.

    This function constructs four hierarchical mesh levels (Level0-Level3),
    computes geometric and physical properties, and identifies overlapping
    regions between levels for multiscale simulations. Level0 is fine-scale
    top layer. Level 1 is part-scale coarse mesh. Level 2 is meso-scale.
    Level 3 is fine-scale near the melt pool.
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
        level = getBCindices(level)
        level = convert_boundary_types(level)
        level.node_coords, level.connect = createMesh3D(
            (level.bounds.x[0], level.bounds.x[1], level.nodes[0]),
            (level.bounds.y[0], level.bounds.y[1], level.nodes[1]),
            (level.bounds.z[0], level.bounds.z[1], level.nodes[2]),
        )
        level.T = properties["T_amb"] * jnp.ones(level.nn)
        level.T0 = properties["T_amb"] * jnp.ones(level.nn)
        level.S1 = jnp.zeros(level.nn)
        level.S2 = jnp.zeros(level.nn, dtype=bool)
        level.active = jnp.zeros(level.nn, dtype=bool)
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


def SetupProperties(prop_obj: dict) -> dict:
    """
    Initialize and return a dictionary of material and simulation properties.

    This function reads user-defined or default values from the input
    dictionary and assigns them to a structured object for use in
    simulations. It also computes derived coefficients used in
    evaporation and heat transfer models.
    """
    properties = dict2obj({})

    # --- Thermal conductivity ---
    properties.k_powder = get_with_units(
        prop_obj, "thermal_conductivity_powder", 0.4e-3, "W/(mm*K)"
    )
    properties.k_bulk_coeff_a0 = get_with_units(
        prop_obj, "thermal_conductivity_bulk_a0", 4.23e-3, "W/(mm*K)"
    )
    properties.k_bulk_coeff_a1 = get_with_units(
        prop_obj, "thermal_conductivity_bulk_a1", 0.016e-3, "W/(mm*K^2)"
    )
    properties.k_fluid_coeff_a0 = get_with_units(
        prop_obj, "thermal_conductivity_fluid_a0", 29.0e-3, "W/(mm*K)"
    )

    # --- Heat capacity ---
    properties.cp_solid_coeff_a0 = get_with_units(
        prop_obj, "heat_capacity_solid_a0", 383.1, "J/(kg*K)"
    )
    properties.cp_solid_coeff_a1 = get_with_units(
        prop_obj, "heat_capacity_solid_a1", 0.174, "J/(kg*K^2)"
    )
    properties.cp_mushy = get_with_units(
        prop_obj, "heat_capacity_mushy", 3235.0, "J/(kg*K)"
    )
    properties.cp_fluid = get_with_units(
        prop_obj, "heat_capacity_fluid", 769.0, "J/(kg*K)"
    )

    # --- Density ---
    properties.rho = get_with_units(prop_obj, "density", 8.0e-6, "kg/mm^3")

    # --- Laser parameters ---
    properties.laser_radius = get_with_units(prop_obj, "laser_radius", 0.110, "mm")
    properties.laser_depth = get_with_units(prop_obj, "laser_depth", 0.05, "mm")
    properties.laser_power = get_with_units(prop_obj, "laser_power", 300.0, "W")
    properties.laser_eta = prop_obj.get("laser_absorptivity", 0.25)  # unitless

    # --- Temperature thresholds ---
    properties.T_amb = get_with_units(prop_obj, "T_amb", 353.15, "K")
    properties.T_solidus = get_with_units(prop_obj, "T_solidus", 1554.0, "K")
    properties.T_liquidus = get_with_units(prop_obj, "T_liquidus", 1625.0, "K")
    properties.T_boiling = get_with_units(prop_obj, "T_boiling", 3038.0, "K")

    # --- Heat transfer and radiation ---
    properties.h_conv = get_with_units(prop_obj, "h_conv", 1.0e1, "W/(m^2*K)")
    properties.vareps = prop_obj.get("emissivity", 0.600)  # unitless
    properties.evc = prop_obj.get("evaporation_coefficient", 0.82)  # unitless

    # --- Physical constants ---
    properties.mA = get_with_units(prop_obj, "atomic_mass", 7.9485017e-26, "kg")
    properties.Lev = get_with_units(prop_obj, "latent_heat_evap", 4.22e6, "J/kg")
    properties.molar_mass = get_with_units(prop_obj, "molar_mass", 0.04, "kg/mol")

    # --- Layer height ---
    properties.layer_height = get_with_units(prop_obj, "layer_height", 0.04, "mm")

    # --- Universal constants ---
    properties.kb = 1.380649e-23  # Boltzmann constant (J/K)
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


def get_with_units(prop_obj, key, default_value, default_unit):
    """
    Extracts a property with units, converts to SI, returns float.
    """
    ureg = pint.UnitRegistry()
    entry = prop_obj.get(key, {"value": default_value, "unit": default_unit})
    value = entry["value"]
    unit = entry["unit"]
    result = (value * ureg(unit)).to(default_unit).magnitude
    return result


def SetupNonmesh(nonmesh_input: dict) -> dict:
    """
    Initialize and return a dictionary of non-mesh simulation parameters.
    """
    Nonmesh = dict2obj({})

    # Timestep for finest mesh (s)
    Nonmesh.timestep_L3 = get_with_units(nonmesh_input, "timestep_L3", 1e-5, "s")

    # Subcycle numbers (unitless)
    Nonmesh.subcycle_num_L2 = nonmesh_input.get("subcycle_num_L2", 1)
    Nonmesh.subcycle_num_L3 = nonmesh_input.get("subcycle_num_L3", 1)

    # Dwell time duration (s)
    Nonmesh.dwell_time = get_with_units(nonmesh_input, "dwell_time", 0.1, "s")

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
    Nonmesh.laser_velocity = get_with_units(
        nonmesh_input, "laser_velocity", 500, "mm/s"
    )

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

    # Laser center
    Nonmesh.laser_center = nonmesh_input.get("laser_center", [])

    # Hybrid Algorithm for Surrogate Temperature Evaluation
    Nonmesh.haste = nonmesh_input.get("haste", 0)

    # Number of subcycled predictor-corrector loops to execute on GPU before CPU return
    Nonmesh.loop_GOMELT = nonmesh_input.get("loop_GOMELT", 1)

    # Recording data for Time-Above-Melting (TAM)
    Nonmesh.record_TAM = nonmesh_input.get("record_TAM", 0)

    # If running for LPBF or not (only LPBF supported)
    Nonmesh.LPBF = nonmesh_input.get("LPBF", True)

    # Temperature probe locations
    Nonmesh.probe_locations = nonmesh_input.get("probe_locations", [])

    # Create output directory if it doesn't exist
    if not os.path.exists(Nonmesh.save_path):
        os.makedirs(Nonmesh.save_path)

    return structure_to_dict(Nonmesh)


def SetupStaticNodesAndElements(Levels: list[dict]) -> tuple[tuple[int]]:
    """
    Extract static element and node counts from a list of level dictionaries.
    """
    return (
        (
            Levels[0]["ne"].tolist(),
            Levels[1]["ne"].tolist(),
            Levels[2]["ne"].tolist(),
            Levels[3]["ne"].tolist(),
        ),
        (
            Levels[0]["nn"].tolist(),
            Levels[1]["nn"].tolist(),
            Levels[2]["nn"].tolist(),
            Levels[3]["nn"].tolist(),
        ),
        (
            tuple([Levels[0]["elements"][_].tolist() for _ in range(3)]),
            tuple([Levels[1]["elements"][_].tolist() for _ in range(3)]),
            tuple([Levels[2]["elements"][_].tolist() for _ in range(3)]),
            tuple([Levels[3]["elements"][_].tolist() for _ in range(3)]),
        ),
    )


def SetupStaticSubcycle(
    nonmesh: dict[str, int],
) -> tuple[int, int, int, float, float, float, int]:
    """
    Extract and compute static subcycle values from the non-mesh configuration.
    """
    subcycles_L2 = nonmesh["subcycle_num_L2"]
    subcycles_L3 = nonmesh["subcycle_num_L3"]
    total_subcycles = subcycles_L2 * subcycles_L3

    # Float versions for downstream calculations
    subcycles_L2_float = float(subcycles_L2)
    subcycles_L3_float = float(subcycles_L3)
    total_subcycles_float = float(total_subcycles)

    return (
        subcycles_L2,
        subcycles_L3,
        total_subcycles,
        subcycles_L2_float,
        subcycles_L3_float,
        total_subcycles_float,
        nonmesh["loop_GOMELT"],
    )


def dict2obj(dict1: dict) -> obj:
    """
    Convert a dictionary into an object with attribute-style access.

    This function serializes the dictionary to JSON and then deserializes
    it using a custom object hook to create an instance of the `obj` class.
    """
    return json.loads(json.dumps(dict1), object_hook=obj)


def structure_to_dict(struct: obj) -> dict:
    """
    Recursively convert a nested object with attributes into a dictionary.

    This function handles nested objects by checking if each attribute
    has a __dict__ (i.e., is an object) and recursively converting it,
    unless the object has a 'tolist' method (e.g., NumPy or JAX arrays),
    in which case it is left as-is.
    """
    return {
        k: (
            v
            if (not hasattr(v, "__dict__") or hasattr(v, "tolist"))
            else structure_to_dict(v)
        )
        for k, v in struct.__dict__.items()
    }


def calcStaticTmpNodesAndElements(
    Levels: list[dict],
    toolpath_input: Union[list[float], jnp.ndarray],
) -> tuple[int, int]:
    """
    Calculate active elements and nodes in Level 1. Active elements are below the top
    surface of the active layer.
    """
    # Mask for nodes in Level 1 where z <= toolpath_input[2] + tolerance
    Level1_mask = Levels[1]["node_coords"][2] <= toolpath_input[2] + 1e-5
    Level1_nn = sum(Level1_mask)

    # Calculate temporary number of elements and nodes
    active_Level1_elements = (
        Levels[1]["elements"][0] * Levels[1]["elements"][1] * (Level1_nn - 1)
    )
    active_Level1_elements = active_Level1_elements.tolist()
    active_Level1_nodes = (
        Levels[1]["nodes"][0] * Levels[1]["nodes"][1] * Level1_nn
    ).tolist()

    return (active_Level1_elements, active_Level1_nodes)


def find_max_const(
    coarse_level: obj, finer_level: obj
) -> tuple[list[int], list[int], list[int]]:
    """
    Compute the maximum number of elements the finer level domain can move
    within the bounds of the coarser level domain in each direction.

    This function calculates how many elements the finer mesh can shift
    in the positive and negative directions (east/west, north/south, top/bottom)
    without exceeding the bounds of the coarser mesh.
    """
    east_shift = coarse_level.bounds.x[1] - finer_level.bounds.x[1]
    north_shift = coarse_level.bounds.y[1] - finer_level.bounds.y[1]
    top_shift = coarse_level.bounds.z[1] - finer_level.bounds.z[1]
    west_shift = coarse_level.bounds.x[0] - finer_level.bounds.x[0]
    south_shift = coarse_level.bounds.y[0] - finer_level.bounds.y[0]
    bottom_shift = coarse_level.bounds.z[0] - finer_level.bounds.z[0]

    return (
        [west_shift, east_shift],
        [south_shift, north_shift],
        [bottom_shift, top_shift],
    )


def convert_boundary_types(level: dict) -> dict:
    for face in ["west", "east", "north", "south", "top", "bottom"]:
        bc = getattr(level.conditions, face)
        if bc.type == "Dirichlet":
            bc.type = 0
            if bc.function == "Constant":
                bc.function = 0
            elif bc.function == "Interpolate":
                bc.function = 1
        elif bc.type == "Neumann":
            bc.type = 1
            if bc.function == "Surface":
                bc.function = 0
            elif bc.function == "Convection":
                bc.function = 1
            elif bc.function == "Adiabatic":
                bc.function = 2
    return level


def SetupStaticBoundaryConditions(Levels: list[dict]) -> tuple:
    boundary_conditions = [()]
    for level in range(1, len(Levels)):
        level_types = []
        level_functions = []
        level_value = []
        level_global_indices = []
        level_local_indices = []
        for face in ["west", "east", "south", "north", "bottom", "top"]:
            level_types.append(Levels[level]["conditions"][face]["type"])
            level_functions.append(Levels[level]["conditions"][face]["function"])
            if "value" in Levels[level]["conditions"][face].keys():
                level_value.append(Levels[level]["conditions"][face]["value"])
            else:
                level_value.append(jnp.nan)
            level_global_indices.append(
                tuple(Levels[level]["conditions"][face]["indices"].tolist())
            )
            if face == "west":
                level_local_indices.append((0, 4, 7, 3))
            elif face == "east":
                level_local_indices.append((1, 2, 6, 5))
            elif face == "south":
                level_local_indices.append((0, 1, 5, 4))
            elif face == "north":
                level_local_indices.append((3, 7, 6, 2))
            elif face == "bottom":
                level_local_indices.append((0, 3, 2, 1))
            elif face == "top":
                level_local_indices.append((4, 5, 6, 7))
        boundary_conditions.append(
            (
                tuple(level_types),
                tuple(level_functions),
                tuple(level_value),
                tuple(level_global_indices),
                tuple(level_local_indices),
            )
        )
    return tuple(boundary_conditions)

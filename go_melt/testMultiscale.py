import copy
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from computeFunctions import *
from jax.config import config

# True is for convergence (double precision), False is single precision
config.update("jax_enable_x64", False)


def testMultiscale(solver_input: dict):
    """testMultiscale is where the majority of the thermal solver takes place. It
    is called by some run function that will provide a dictionary from a JSON
    input file. testMultiscale inherently takes place on the CPU, but it calls
    many functions from computeFunctions.py that take place on the GPU. Nothing
    is directly outputted from testMultiscale. Instead, most of the results
    are saved in the vtk file format. In testMultiscale.py, three levels of
    refinement are solved using the multilevel method and saved according
    to input parameters.
    :param solver_input: JSON file output after processing
    """

    tstart = time.time()

    ### Parse inputs into multiple objects (START) ###
    class obj:
        # Constructor
        def __init__(self, dict1):
            self.__dict__.update(dict1)

    def dict2obj(dict1):
        return json.loads(json.dumps(dict1), object_hook=obj)

    # Coarse domain
    _ = solver_input.get("Level1", {})
    Level1 = dict2obj(_)
    # Finer domain
    _ = solver_input.get("Level2", {})
    Level2 = dict2obj(_)
    # Finest domain
    _ = solver_input.get("Level3", {})
    Level3 = dict2obj(_)
    ### Parse inputs into multiple objects (END) ###

    ### User-defined parameters (START) ###
    # Timestep
    dt = solver_input.get("nonmesh", {}).get("timestep", 1e-5)
    # Whether to use Forward Euler method (default is yes)
    explicit = solver_input.get("nonmesh", {}).get("explicit", 1)
    # Whether to solve for the steady-state solution (default is no)
    steady = solver_input.get("nonmesh", {}).get("steady", 0)
    # How frequently to record and/or check to move the fine domain
    record_step = solver_input.get("nonmesh", {}).get("record_step", 1)
    # How frequently (factor) to record coarse-domain
    Level1_record_inc = solver_input.get("nonmesh", {}).get("Level1_record_step", 1)
    # Where to save
    save_path = solver_input.get("nonmesh", {}).get("save", "results/")
    # Whether to save
    output_files = solver_input.get("nonmesh", {}).get("output_files", 1)
    # Save timing results
    save_time = solver_input.get("nonmesh", {}).get("savetime", 0)
    # Which toolpath
    tool_path_input = solver_input.get("nonmesh", {}).get("toolpath", "laserPath.txt")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if save_time:
        timing_file = open(save_path + "timing.csv", "w")

    # Define the material properties
    # Conductivity (W/mmK)
    k = solver_input.get("properties", {}).get("thermal_conductivity", 0.022)
    # Heat capacity (J/kgK)
    cp = solver_input.get("properties", {}).get("heat_capacity", 745)
    # Density (kg/mm^3)
    rho = solver_input.get("properties", {}).get("density", 4.27e-6)
    # Laser radius (sigma, mm)
    laser_r = solver_input.get("properties", {}).get("laser_radius", 0.110)
    # Laser depth (d, mm)
    laser_d = solver_input.get("properties", {}).get("laser_depth", 0.05)
    # Laser power (P, W)
    laser_P = solver_input.get("properties", {}).get("laser_power", 200)
    # Laser absorptivity (eta, unitless)
    laser_eta = solver_input.get("properties", {}).get("laser_absorptivity", 0.25)
    # Ambient Temperature (K)
    T_amb = solver_input.get("properties", {}).get("T_amb", 298.15)
    # Convection coefficient (h_conv, W/mm^2K)
    h_conv = solver_input.get("properties", {}).get("h_conv", 1.473e-5)
    # Emissivity (vareps, unitless)
    vareps = solver_input.get("properties", {}).get("emissivity", 0.600)
    ### User-defined parameters (END) ###

    ### Calculated parameters (START) ###
    # Domain and element lengths
    Level1.length, Level1.h = calc_length_h(Level1)
    Level2.length, Level2.h = calc_length_h(Level2)
    Level3.length, Level3.h = calc_length_h(Level3)
    # Constrain fine mesh based on initial conditions
    (
        Level2.bounds.ix,
        Level2.bounds.iy,
        Level2.bounds.iz,
    ) = find_max_const(Level1, Level2)
    (
        Level3.bounds.ix,
        Level3.bounds.iy,
        Level3.bounds.iz,
    ) = find_max_const(Level1, Level3)

    # Calculate number of nodes in each direction
    Level1.nodes = calcNumNodes(Level1.elements)
    Level2.nodes = calcNumNodes(Level2.elements)
    Level3.nodes = calcNumNodes(Level3.elements)
    # Calculate total number of nodes (nn) and elements (ne)
    Level1.ne = Level1.elements[0] * Level1.elements[1] * Level1.elements[2]
    Level1.nn = Level1.nodes[0] * Level1.nodes[1] * Level1.nodes[2]
    Level2.ne = Level2.elements[0] * Level2.elements[1] * Level2.elements[2]
    Level2.nn = Level2.nodes[0] * Level2.nodes[1] * Level2.nodes[2]
    Level3.ne = Level3.elements[0] * Level3.elements[1] * Level3.elements[2]
    Level3.nn = Level3.nodes[0] * Level3.nodes[1] * Level3.nodes[2]
    # Get BC indices
    Level1.BC = getBCindices(Level1)
    Level2.BC = getBCindices(Level2)
    Level3.BC = getBCindices(Level3)
    ### Calculated parameters (END) ###

    # Find starting position and how often position changes (get initial position)
    tool_path_file = open(tool_path_input, "r")
    _ = tool_path_file.readline()
    vstart = np.array([float(_i) for _i in _.split(",")])
    vtot = vstart - vstart
    v = vtot + vstart
    _ = tool_path_file.readline()
    if steady == 1:
        Level3step = int(1e8)
    else:
        vnext = np.array([float(_i) for _i in _.split(",")])
        if np.linalg.norm(v - vnext) == 0:
            Level3step = int(1e8)
        else:
            Level3step = int(
                np.ceil(np.min(np.array(Level2.h)) / np.linalg.norm(v - vnext))
            )
    tool_path_file.close()

    # Set up meshes for all three levels
    Level1.node_coords, Level1.connect = createMesh3D(
        (Level1.bounds.x[0], Level1.bounds.x[1], Level1.nodes[0]),
        (Level1.bounds.y[0], Level1.bounds.y[1], Level1.nodes[1]),
        (Level1.bounds.z[0], Level1.bounds.z[1], Level1.nodes[2]),
    )
    Level2.node_coords, Level2.connect = createMesh3D(
        (Level2.bounds.x[0], Level2.bounds.x[1], Level2.nodes[0]),
        (Level2.bounds.y[0], Level2.bounds.y[1], Level2.nodes[1]),
        (Level2.bounds.z[0], Level2.bounds.z[1], Level2.nodes[2]),
    )
    Level3.node_coords, Level3.connect = createMesh3D(
        (Level3.bounds.x[0], Level3.bounds.x[1], Level3.nodes[0]),
        (Level3.bounds.y[0], Level3.bounds.y[1], Level3.nodes[1]),
        (Level3.bounds.z[0], Level3.bounds.z[1], Level3.nodes[2]),
    )

    # Deep copy initial coordinates for updating mesh position
    Level2.init_node_coors = copy.deepcopy(Level2.node_coords)
    Level3.init_node_coors = copy.deepcopy(Level3.node_coords)

    # Preallocate
    Level2.T = T_amb * jnp.ones(Level2.nn)
    Level2.T0 = T_amb * jnp.ones(Level2.nn)
    Level2.Tprime = jnp.zeros(Level2.nn)
    Level2.Tprime0 = copy.deepcopy(Level2.Tprime)
    Level3.T = T_amb * jnp.ones(Level3.nn)
    Level3.T0 = T_amb * jnp.ones(Level3.nn)
    Level3.Tprime = jnp.zeros(Level3.nn)
    Level3.Tprime0 = copy.deepcopy(Level3.Tprime)
    Level1.T = T_amb * jnp.ones(Level1.nn)
    Level1.T0 = T_amb * jnp.ones(Level1.nn)

    # Get overlapping nodes in x, y, and z directions (separately)
    Level2.orig_overlap_nodes = [
        getCoarseNodesInFineRegion(Level2.node_coords[0], Level1.node_coords[0]),
        getCoarseNodesInFineRegion(Level2.node_coords[1], Level1.node_coords[1]),
        getCoarseNodesInFineRegion(Level2.node_coords[2], Level1.node_coords[2]),
    ]

    Level2.orig_overlap_coors = [
        jnp.array([Level1.node_coords[0][_] for _ in Level2.orig_overlap_nodes[0]]),
        jnp.array([Level1.node_coords[1][_] for _ in Level2.orig_overlap_nodes[1]]),
        jnp.array([Level1.node_coords[2][_] for _ in Level2.orig_overlap_nodes[2]]),
    ]

    Level3.orig_overlap_nodes = [
        getCoarseNodesInFineRegion(Level3.node_coords[0], Level2.node_coords[0]),
        getCoarseNodesInFineRegion(Level3.node_coords[1], Level2.node_coords[1]),
        getCoarseNodesInFineRegion(Level3.node_coords[2], Level2.node_coords[2]),
    ]

    Level3.orig_overlap_coors = [
        jnp.array([Level2.node_coords[0][_] for _ in Level3.orig_overlap_nodes[0]]),
        jnp.array([Level2.node_coords[1][_] for _ in Level3.orig_overlap_nodes[1]]),
        jnp.array([Level2.node_coords[2][_] for _ in Level3.orig_overlap_nodes[2]]),
    ]

    # Identify coarse nodes inside fine-scale region
    Level2.overlapNodes = copy.deepcopy(Level2.orig_overlap_nodes)
    Level2.overlapCoords = copy.deepcopy(Level2.orig_overlap_coors)
    Level3.overlapNodes = copy.deepcopy(Level3.orig_overlap_nodes)
    Level3.overlapCoords = copy.deepcopy(Level3.orig_overlap_coors)

    # Initialize time and record counters
    time_inc = 0
    record_inc = 0

    # Get the initial coarse shape functions for correction terms
    # Level2NcLevel1: (Level2.ne, 8, 8), shape functions connecting Level2 to Level1
    # Level2dNcdxLevel1: (Level2.ne, 8, 8)
    # Level2dNcdyLevel1: (Level2.ne, 8, 8)
    # Level2dNcdzLevel1: (Level2.ne, 8, 8)
    # Level2nodesLevel1: (Level2.ne * 8 * 8), indexing into Level2
    (
        Level2NcLevel1,
        Level2dNcdxLevel1,
        Level2dNcdyLevel1,
        Level2dNcdzLevel1,
        Level2nodesLevel1,
    ) = computeCoarseFineShapeFunctions(
        Level1.node_coords[0],
        Level1.node_coords[1],
        Level1.node_coords[2],
        Level1.connect[0],
        Level1.connect[1],
        Level1.connect[2],
        Level2.node_coords[0],
        Level2.node_coords[1],
        Level2.node_coords[2],
        Level2.connect[0],
        Level2.connect[1],
        Level2.connect[2],
    )
    # Level3NcLevel1: (Level3.ne, 8, 8), shape functions connecting Level3 to Level1
    # Level3dNcdxLevel1: (Level3.ne, 8, 8)
    # Level3dNcdyLevel1: (Level3.ne, 8, 8)
    # Level3dNcdzLevel1: (Level3.ne, 8, 8)
    # Level3nodesLevel1: (Level3.ne * 8 * 8), indexing into Level3
    (
        Level3NcLevel1,
        Level3dNcdxLevel1,
        Level3dNcdyLevel1,
        Level3dNcdzLevel1,
        Level3nodesLevel1,
    ) = computeCoarseFineShapeFunctions(
        Level1.node_coords[0],
        Level1.node_coords[1],
        Level1.node_coords[2],
        Level1.connect[0],
        Level1.connect[1],
        Level1.connect[2],
        Level3.node_coords[0],
        Level3.node_coords[1],
        Level3.node_coords[2],
        Level3.connect[0],
        Level3.connect[1],
        Level3.connect[2],
    )

    # Level3NcLevel2: (Level3.ne, 8, 8), shape functions connecting Level3 to Level2
    # Level3dNcdxLevel2: (Level3.ne, 8, 8)
    # Level3dNcdyLevel2: (Level3.ne, 8, 8)
    # Level3dNcdzLevel2: (Level3.ne, 8, 8)
    # Level3nodesLevel2: (Level3.ne * 8 * 8), indexing into Level3
    (
        Level3NcLevel2,
        Level3dNcdxLevel2,
        Level3dNcdyLevel2,
        Level3dNcdzLevel2,
        Level3nodesLevel2,
    ) = computeCoarseFineShapeFunctions(
        Level2.node_coords[0],
        Level2.node_coords[1],
        Level2.node_coords[2],
        Level2.connect[0],
        Level2.connect[1],
        Level2.connect[2],
        Level3.node_coords[0],
        Level3.node_coords[1],
        Level3.node_coords[2],
        Level3.connect[0],
        Level3.connect[1],
        Level3.connect[2],
    )

    # Get the interpolation matrices
    # Level1Level2_intmat: (Level2.nn, 8), from Level1 to Level2
    # Level1Level2_node: (Level2.nn, 8), nodes indexing into Level1
    Level1Level2_intmat, Level1Level2_node = interpolatePointsMatrix(
        Level1.node_coords[0],
        Level1.node_coords[1],
        Level1.node_coords[2],
        Level1.connect[0],
        Level1.connect[1],
        Level1.connect[2],
        Level2.node_coords[0],
        Level2.node_coords[1],
        Level2.node_coords[2],
    )
    # Level2Level3_intmat: (Level3.nn, 8), from Level2 to Level3
    # Level2Level3_node: (Level3.nn, 8), nodes indexing into Level2
    Level2Level3_intmat, Level2Level3_node = interpolatePointsMatrix(
        Level2.node_coords[0],
        Level2.node_coords[1],
        Level2.node_coords[2],
        Level2.connect[0],
        Level2.connect[1],
        Level2.connect[2],
        Level3.node_coords[0],
        Level3.node_coords[1],
        Level3.node_coords[2],
    )
    # Level2Level1_intmat: (Level1.nn, 8), from Level2 to Level1
    # Level2Level1_node: (Level1.nn, 8), nodes indexing into Level2
    Level2Level1_intmat, Level2Level1_node = interpolatePointsMatrix(
        Level2.node_coords[0],
        Level2.node_coords[1],
        Level2.node_coords[2],
        Level2.connect[0],
        Level2.connect[1],
        Level2.connect[2],
        Level1.node_coords[0],
        Level1.node_coords[1],
        Level1.node_coords[2],
    )
    # Level3Level2_intmat: (Level2.nn, 8), from Level3 to Level2
    # Level3Level2_node: (Level2.nn, 8), nodes indexing into Level3
    Level3Level2_intmat, Level3Level2_node = interpolatePointsMatrix(
        Level3.node_coords[0],
        Level3.node_coords[1],
        Level3.node_coords[2],
        Level3.connect[0],
        Level3.connect[1],
        Level3.connect[2],
        Level2.node_coords[0],
        Level2.node_coords[1],
        Level2.node_coords[2],
    )

    ### START OF THE TIME LOOP ###
    record_lab = int(time_inc / record_step) + 1
    if output_files == 1:
        save_result(Level1, "Level1_", record_lab, save_path, 2e-3)
        save_result(Level2, "Level2_", record_lab, save_path, 1e-3)
        save_result(Level3, "Level3_", record_lab, save_path, 0)

    _vx, _vy, _vz = 0, 0, 0

    tool_path_file = open(tool_path_input, "r")

    vprev = 1e6
    vprev1 = 1e6
    force_move = False
    for _v in tool_path_file:
        tstart_loop = time.time()
        v = np.array([float(_) for _ in _v.split(",")])

        # Get time at t^(n+1) and t^(n)
        tn1, tn = (time_inc + 1) * dt, time_inc * dt
        _t = (1 - explicit) * tn1 + explicit * tn

        # Prep LHS to apply Dirichlet BCs
        if v[2] != vprev:
            Level1_mask = Level1.node_coords[2] <= v[2] + 1e-5
            Level1_nn = sum(Level1_mask)
            Level1.T = Level1.T.at[jnp.isnan(Level1.T)].set(0)
            tmp_ne = Level1.elements[0] * Level1.elements[1] * (Level1_nn - 1)
            tmp_ne = tmp_ne.tolist()
            tmp_nn = (Level1.nodes[0] * Level1.nodes[1] * Level1_nn).tolist()
            vprev = v[2]
            Level1.idT = jnp.zeros(Level1.nn)
            Level1.idT = Level1.idT.at[tmp_nn:].set(1) == 1
            force_move = True
        if v[0] != vprev1:
            force_move = True
            vprev1 = v[0]

        if ((time_inc % Level3step) == 0) | force_move:
            force_move = False
            (
                Level3.node_coords,
                Level3.overlapNodes,
                Level3.overlapCoords,
                Level3.T0,
                Level3.Tprime0,
                vtot,
                Level2.node_coords,
                Level2.overlapNodes,
                Level2.overlapCoords,
                Level2.T0,
                Level2.Tprime0,
                Level2NcLevel1,
                Level2dNcdxLevel1,
                Level2dNcdyLevel1,
                Level2dNcdzLevel1,
                Level2nodesLevel1,
                Level1Level2_intmat,
                Level1Level2_node,
                Level2Level1_intmat,
                Level2Level1_node,
                Level3NcLevel1,
                Level3dNcdxLevel1,
                Level3dNcdyLevel1,
                Level3dNcdzLevel1,
                Level3nodesLevel1,
                Level3NcLevel2,
                Level3dNcdxLevel2,
                Level3dNcdyLevel2,
                Level3dNcdzLevel2,
                Level3nodesLevel2,
                Level2Level3_intmat,
                Level2Level3_node,
                Level3Level2_intmat,
                Level3Level2_node,
                _vx,
                _vy,
                _vz,
            ) = moveEverything(
                v,
                vstart,
                Level3.node_coords,
                Level3.connect,
                Level3.init_node_coors,
                Level3.orig_overlap_nodes,
                Level3.orig_overlap_coors,
                Level3.bounds.ix,
                Level3.bounds.iy,
                Level3.bounds.iz,
                Level3.Tprime0,
                Level2.node_coords,
                Level2.connect,
                Level2.h,
                Level2.Tprime0,
                Level1.node_coords,
                Level1.connect,
                Level1.T0,
                Level2.init_node_coors,
                Level2.orig_overlap_nodes,
                Level2.orig_overlap_coors,
                Level2.bounds.ix,
                Level2.bounds.iy,
                Level2.bounds.iz,
                Level1.h,
                _vx,
                _vy,
                _vz,
            )

        if explicit:
            (
                Level3.T0,
                Level2.T0,
                Level1.T0,
                Level3.Tprime0,
                Level2.Tprime0,
            ) = doExplicitTimestep(
                Level3.node_coords,
                Level3.connect,
                Level3.ne,
                Level3.overlapCoords,
                Level3.overlapNodes,
                Level2.node_coords,
                Level2.connect,
                Level2.ne,
                Level2.nodes,
                Level2.overlapCoords,
                Level2.overlapNodes,
                Level1.node_coords,
                Level1.connect,
                Level1.nodes,
                Level1.conditions.x,
                Level1.conditions.y,
                Level1.conditions.z,
                Level3.BC,
                Level2.BC,
                Level1.BC,
                tmp_ne,
                tmp_nn,
                Level1.nn,
                Level2.nn,
                Level3.nn,
                Level3.Tprime0,
                Level2.Tprime0,
                Level3.T0,
                Level2.T0,
                Level1.T0,
                Level3NcLevel1,
                Level3NcLevel2,
                Level2NcLevel1,
                Level3dNcdxLevel1,
                Level3dNcdyLevel1,
                Level3dNcdzLevel1,
                Level3nodesLevel1,
                Level3dNcdxLevel2,
                Level3dNcdyLevel2,
                Level3dNcdzLevel2,
                Level3nodesLevel2,
                Level2dNcdxLevel1,
                Level2dNcdyLevel1,
                Level2dNcdzLevel1,
                Level2nodesLevel1,
                Level2Level3_intmat,
                Level2Level3_node,
                Level3Level2_intmat,
                Level3Level2_node,
                Level1Level2_intmat,
                Level1Level2_node,
                Level2Level1_intmat,
                Level2Level1_node,
                _t,
                v,
                k,
                rho,
                cp,
                dt,
                laser_r,
                laser_d,
                laser_P,
                laser_eta,
                T_amb,
                h_conv,
                vareps,
            )
            # Update parameters
            time_inc += 1
            record_inc += 1
            ###---------------------END OF EXPLICIT---------------------###
        if record_inc == record_step:
            record_inc = 0
            record_lab = int(time_inc / record_step) + 1
            if output_files == 1:
                if np.mod(record_lab, Level1_record_inc) == 0:
                    save_result(Level1, "Level1_", record_lab, save_path, 2e-3)
                save_result(Level2, "Level2_", record_lab, save_path, 1e-3)
                save_result(Level3, "Level3_", record_lab, save_path, 0)

        tend = time.time()
        if save_time:
            timing_file.write("%.9f\n" % (tend - tstart))

        # Timestep, Iter, dTbar, elapsed time
        print(
            "%13.0f,%14.6f seconds, %13.6f ms/timestep"
            % (time_inc, tend - tstart, 1000 * (tend - tstart_loop))
        )
    if save_time:
        timing_file.close()
    print("Level1.T0.max: %f" % (Level1.T0.max()))
    tool_path_file.close()

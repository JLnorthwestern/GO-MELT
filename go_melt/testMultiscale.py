import numpy as np
import jax
import jax.numpy as jnp
from computeFunctions import *
import time
import json
from jax.config import config
import copy
import os

# True is for convergence (double precision), False is single precision
config.update("jax_enable_x64", False)

def testMultiscale(solver_input: dict):
    """ testMultiscale is where the majority of the thermal solver takes place. It
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
    _ = solver_input.get('full', {})
    full = dict2obj(_)
    # Finer domain
    _ = solver_input.get('sub1', {})
    sub1 = dict2obj(_)
    # Finest domain
    _ = solver_input.get('sub2', {})
    sub2 = dict2obj(_)
    ### Parse inputs into multiple objects (END) ###

    ### User-defined parameters (START) ###
    # Timestep (implicit)
    dt = solver_input.get("nonmesh", {}).get("timestep", 1e-5)
    # Whether to use Forward Euler method (default is yes)
    explicit = solver_input.get("nonmesh", {}).get("explicit", 1)
    # Whether to solve for the steady-state solution (default is no)
    steady = solver_input.get("nonmesh", {}).get("steady", 0)
    # How frequently to record and/or check to move the fine domain
    record_step = solver_input.get("nonmesh", {}).get("record_step", 1)
    # How frequently (factor) to record coarse-domain
    full_record_inc = solver_input.get("nonmesh", {}).get("full_record_step", 1)
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
    full.length, full.h = calc_length_h(full)
    sub1.length, sub1.h = calc_length_h(sub1)
    sub2.length, sub2.h = calc_length_h(sub2)
    # Constrain fine mesh based on initial conditions
    sub1.bounds.ix, sub1.bounds.iy, sub1.bounds.iz = find_max_const(full, sub1)
    sub2.bounds.ix, sub2.bounds.iy, sub2.bounds.iz = find_max_const(full, sub2)
    # Calculate number of nodes in each direction
    full.nodes = calcNumNodes(full.elements)
    sub1.nodes = calcNumNodes(sub1.elements)
    sub2.nodes = calcNumNodes(sub2.elements)
    # Calculate total number of nodes (nn) and elements (ne)
    full.ne = full.elements[0] * full.elements[1] * full.elements[2]
    full.nn = full.nodes[0] * full.nodes[1] * full.nodes[2]
    sub1.ne = sub1.elements[0] * sub1.elements[1] * sub1.elements[2]
    sub1.nn = sub1.nodes[0] * sub1.nodes[1] * sub1.nodes[2]
    sub2.ne = sub2.elements[0] * sub2.elements[1] * sub2.elements[2]
    sub2.nn = sub2.nodes[0] * sub2.nodes[1] * sub2.nodes[2]
    # Get BC indices
    full.widx, full.eidx, full.sidx, full.nidx, full.bidx, full.tidx = getBCindices(full)
    sub1.widx, sub1.eidx, sub1.sidx, sub1.nidx, sub1.bidx, sub1.tidx = getBCindices(sub1)
    sub2.widx, sub2.eidx, sub2.sidx, sub2.nidx, sub2.bidx, sub2.tidx = getBCindices(sub2)
    ### Calculated parameters (END) ###
    
    # Find starting position and how often position changes (get initial position)
    tool_path_file = open(tool_path_input,'r')
    _ = tool_path_file.readline()
    vstart = np.array([float(_i) for _i in _.split(',')])
    vtot = vstart - vstart
    v = vtot + vstart
    _ = tool_path_file.readline()
    if steady == 1:
        sub2step = int(1e8)
    else:
        vnext = np.array([float(_i) for _i in _.split(',')])
        if np.linalg.norm(v - vnext) == 0:
            sub2step = int(1e8)
        else:
            sub2step = int(np.ceil(np.min(np.array(sub1.h))/np.linalg.norm(v - vnext)))
    tool_path_file.close()

    # Set up meshes for all three levels
    full.node_coords, full.connect = createMesh3D(
        (full.bounds.x[0], full.bounds.x[1], full.nodes[0]),
        (full.bounds.y[0], full.bounds.y[1], full.nodes[1]),
        (full.bounds.z[0], full.bounds.z[1], full.nodes[2]))
    sub1.node_coords, sub1.connect = createMesh3D(
        (sub1.bounds.x[0], sub1.bounds.x[1], sub1.nodes[0]),
        (sub1.bounds.y[0], sub1.bounds.y[1], sub1.nodes[1]),
        (sub1.bounds.z[0], sub1.bounds.z[1], sub1.nodes[2]))
    sub2.node_coords, sub2.connect = createMesh3D(
        (sub2.bounds.x[0], sub2.bounds.x[1], sub2.nodes[0]),
        (sub2.bounds.y[0], sub2.bounds.y[1], sub2.nodes[1]),
        (sub2.bounds.z[0], sub2.bounds.z[1], sub2.nodes[2]))

    # Deep copy initial coordinates for updating mesh position
    sub1.init_node_coors = copy.deepcopy(sub1.node_coords)
    sub2.init_node_coors = copy.deepcopy(sub2.node_coords)

    # Preallocate
    sub1.T = T_amb * jnp.ones(sub1.nn)
    sub1.T0 = T_amb * jnp.ones(sub1.nn)
    sub1.Tprime = jnp.zeros(sub1.nn)
    sub1.Tprime0 = copy.deepcopy(sub1.Tprime)
    sub2.T = T_amb * jnp.ones(sub2.nn)
    sub2.T0 = T_amb * jnp.ones(sub2.nn)
    sub2.Tprime = jnp.zeros(sub2.nn)
    sub2.Tprime0 = copy.deepcopy(sub2.Tprime)
    full.T = T_amb * jnp.ones(full.nn)
    full.T0 = T_amb * jnp.ones(full.nn)

    # Get overlapping nodes in x, y, and z directions (separately)
    sub1.orig_overlap_nodes = [getCoarseNodesInFineRegion(sub1.node_coords[0], full.node_coords[0]),
                               getCoarseNodesInFineRegion(sub1.node_coords[1], full.node_coords[1]),
                               getCoarseNodesInFineRegion(sub1.node_coords[2], full.node_coords[2])]

    sub1.orig_overlap_coors = [jnp.array([full.node_coords[0][_] for _ in sub1.orig_overlap_nodes[0]]),
                               jnp.array([full.node_coords[1][_] for _ in sub1.orig_overlap_nodes[1]]),
                               jnp.array([full.node_coords[2][_] for _ in sub1.orig_overlap_nodes[2]])]

    sub2.orig_overlap_nodes = [getCoarseNodesInFineRegion(sub2.node_coords[0], sub1.node_coords[0]),
                               getCoarseNodesInFineRegion(sub2.node_coords[1], sub1.node_coords[1]),
                               getCoarseNodesInFineRegion(sub2.node_coords[2], sub1.node_coords[2])]

    sub2.orig_overlap_coors = [jnp.array([sub1.node_coords[0][_] for _ in sub2.orig_overlap_nodes[0]]),
                               jnp.array([sub1.node_coords[1][_] for _ in sub2.orig_overlap_nodes[1]]),
                               jnp.array([sub1.node_coords[2][_] for _ in sub2.orig_overlap_nodes[2]])]

    # Identify coarse nodes inside fine-scale region
    sub1.overlapNodes = copy.deepcopy(sub1.orig_overlap_nodes)
    sub1.overlapCoords = copy.deepcopy(sub1.orig_overlap_coors)
    sub2.overlapNodes = copy.deepcopy(sub2.orig_overlap_nodes)
    sub2.overlapCoords = copy.deepcopy(sub2.orig_overlap_coors)

    # Initialize time and record counters
    time_inc = 0
    record_inc = 0

    # Get the initial coarse shape functions for correction terms
    # sub1NcLevel1: (sub1.ne, 8, 8), shape functions connecting sub1 to full
    # sub1dNcdxLevel1: (sub1.ne, 8, 8)
    # sub1dNcdyLevel1: (sub1.ne, 8, 8)
    # sub1dNcdzLevel1: (sub1.ne, 8, 8)
    # sub1nodesLevel1: (sub1.ne * 8 * 8), indexing into sub1
    sub1NcLevel1, sub1dNcdxLevel1, sub1dNcdyLevel1,\
        sub1dNcdzLevel1, sub1nodesLevel1 =\
        computeCoarseFineShapeFunctions(full.node_coords[0],
                                        full.node_coords[1],
                                        full.node_coords[2],
                                        full.connect[0],
                                        full.connect[1],
                                        full.connect[2],
                                        sub1.node_coords[0],
                                        sub1.node_coords[1],
                                        sub1.node_coords[2],
                                        sub1.connect[0],
                                        sub1.connect[1],
                                        sub1.connect[2])
    # sub2NcLevel1: (sub2.ne, 8, 8), shape functions connecting sub2 to full
    # sub2dNcdxLevel1: (sub2.ne, 8, 8)
    # sub2dNcdyLevel1: (sub2.ne, 8, 8)
    # sub2dNcdzLevel1: (sub2.ne, 8, 8)
    # sub2nodesLevel1: (sub2.ne * 8 * 8), indexing into sub2
    sub2NcLevel1, sub2dNcdxLevel1, sub2dNcdyLevel1,\
        sub2dNcdzLevel1, sub2nodesLevel1 =\
        computeCoarseFineShapeFunctions(full.node_coords[0],
                                        full.node_coords[1],
                                        full.node_coords[2],
                                        full.connect[0],
                                        full.connect[1],
                                        full.connect[2],
                                        sub2.node_coords[0],
                                        sub2.node_coords[1],
                                        sub2.node_coords[2],
                                        sub2.connect[0],
                                        sub2.connect[1],
                                        sub2.connect[2])

    # sub2NcLevel2: (sub2.ne, 8, 8), shape functions connecting sub2 to sub1
    # sub2dNcdxLevel2: (sub2.ne, 8, 8)
    # sub2dNcdyLevel2: (sub2.ne, 8, 8)
    # sub2dNcdzLevel2: (sub2.ne, 8, 8)
    # sub2nodesLevel2: (sub2.ne * 8 * 8), indexing into sub2
    sub2NcLevel2, sub2dNcdxLevel2, sub2dNcdyLevel2,\
        sub2dNcdzLevel2, sub2nodesLevel2 =\
        computeCoarseFineShapeFunctions(sub1.node_coords[0],
                                        sub1.node_coords[1],
                                        sub1.node_coords[2],
                                        sub1.connect[0],
                                        sub1.connect[1],
                                        sub1.connect[2],
                                        sub2.node_coords[0],
                                        sub2.node_coords[1],
                                        sub2.node_coords[2],
                                        sub2.connect[0],
                                        sub2.connect[1],
                                        sub2.connect[2])
    
    # Get the interpolation matrices
    # fullsub1_intmat: (sub1.nn, 8), from full to sub1
    # fullsub1_node: (sub1.nn, 8), nodes indexing into full
    fullsub1_intmat, fullsub1_node = interpolatePointsMatrix(
                                            full.node_coords[0],
                                            full.node_coords[1],
                                            full.node_coords[2],
                                            full.connect[0],
                                            full.connect[1],
                                            full.connect[2],
                                            sub1.node_coords[0],
                                            sub1.node_coords[1],
                                            sub1.node_coords[2])
    # sub1sub2_intmat: (sub2.nn, 8), from sub1 to sub2
    # sub1sub2_node: (sub2.nn, 8), nodes indexing into sub1
    sub1sub2_intmat, sub1sub2_node = interpolatePointsMatrix(
                                            sub1.node_coords[0],
                                            sub1.node_coords[1],
                                            sub1.node_coords[2],
                                            sub1.connect[0],
                                            sub1.connect[1],
                                            sub1.connect[2],
                                            sub2.node_coords[0],
                                            sub2.node_coords[1],
                                            sub2.node_coords[2])
    # sub1full_intmat: (full.nn, 8), from sub1 to full
    # sub1full_node: (full.nn, 8), nodes indexing into sub1
    sub1full_intmat, sub1full_node = interpolatePointsMatrix(
                                            sub1.node_coords[0],
                                            sub1.node_coords[1],
                                            sub1.node_coords[2],
                                            sub1.connect[0],
                                            sub1.connect[1],
                                            sub1.connect[2],
                                            full.node_coords[0],
                                            full.node_coords[1],
                                            full.node_coords[2])
    # sub2sub1_intmat: (sub1.nn, 8), from sub2 to sub1
    # sub2sub1_node: (sub1.nn, 8), nodes indexing into sub2
    sub2sub1_intmat, sub2sub1_node = interpolatePointsMatrix(
                                            sub2.node_coords[0],
                                            sub2.node_coords[1],
                                            sub2.node_coords[2],
                                            sub2.connect[0],
                                            sub2.connect[1],
                                            sub2.connect[2],
                                            sub1.node_coords[0],
                                            sub1.node_coords[1],
                                            sub1.node_coords[2])

    ### START OF THE TIME LOOP ###
    record_lab = int(time_inc / record_step) + 1
    if output_files == 1:
        save_result(full, 'full_', record_lab, save_path)
        save_result(sub1, 'sub1_', record_lab, save_path)
        save_result(sub2, 'sub2_', record_lab, save_path)

    _vx, _vy, _vz = 0, 0, 0

    tool_path_file = open(tool_path_input,'r')

    vprev = 1e6
    vprev1 = 1e6
    force_move = False
    for _v in tool_path_file:
        tstart_loop = time.time()
        v = np.array([float(_) for _ in _v.split(',')])
        
        # Get time at t^(n+1) and t^(n)
        tn1, tn = (time_inc + 1) * dt, time_inc * dt
        _t = (1-explicit)*tn1 + explicit*tn

        # Prep LHS to apply Dirichlet BCs
        if v[2] != vprev:
            full_mask = full.node_coords[2] <= v[2] + 1e-5
            full_nn = sum(full_mask)
            full.T = full.T.at[jnp.isnan(full.T)].set(0)
            tmp_ne = full.elements[0] * full.elements[1] * (full_nn - 1)
            tmp_ne = tmp_ne.tolist()
            tmp_nn = (full.nodes[0] * full.nodes[1] * full_nn).tolist()
            vprev = v[2]
            full.idT = jnp.zeros(full.nn)
            full.idT = full.idT.at[tmp_nn:].set(1) == 1
            force_move = True
        if v[0] != vprev1:
            force_move = True
            vprev1 = v[0]

        if ((time_inc % sub2step) == 0) | force_move:
            force_move = False
            sub2.node_coords, sub2.overlapNodes, sub2.overlapCoords,\
            sub2.T0, sub2.Tprime0, vtot,\
            sub1.node_coords, sub1.overlapNodes, sub1.overlapCoords,\
            sub1.T0, sub1.Tprime0, sub1NcLevel1, sub1dNcdxLevel1, sub1dNcdyLevel1,\
            sub1dNcdzLevel1, sub1nodesLevel1, fullsub1_intmat, fullsub1_node,\
            sub1full_intmat, sub1full_node, sub2NcLevel1, sub2dNcdxLevel1, sub2dNcdyLevel1,\
            sub2dNcdzLevel1, sub2nodesLevel1,\
            sub2NcLevel2, sub2dNcdxLevel2, sub2dNcdyLevel2,\
            sub2dNcdzLevel2, sub2nodesLevel2,\
            sub1sub2_intmat, sub1sub2_node,\
            sub2sub1_intmat, sub2sub1_node, _vx, _vy, _vz = moveEverything(v,
                                                            vstart,
                                                            sub2.node_coords,
                                                            sub2.connect,
                                                            sub2.init_node_coors,
                                                            sub2.orig_overlap_nodes,
                                                            sub2.orig_overlap_coors,
                                                            sub2.bounds.ix,
                                                            sub2.bounds.iy,
                                                            sub2.bounds.iz,
                                                            sub2.Tprime0,
                                                            sub1.node_coords,
                                                            sub1.connect,
                                                            sub1.h,
                                                            sub1.Tprime0,
                                                            full.node_coords,
                                                            full.connect,
                                                            full.T0,
                                                            sub1.init_node_coors,
                                                            sub1.orig_overlap_nodes,
                                                            sub1.orig_overlap_coors,
                                                            sub1.bounds.ix,
                                                            sub1.bounds.iy,
                                                            sub1.bounds.iz,
                                                            full.h,
                                                            _vx,
                                                            _vy,
                                                            _vz)

        if explicit:
            sub2.T0, sub1.T0, full.T0, sub2.Tprime0, sub1.Tprime0 = doExplicitTimestep(
                sub2.node_coords, sub2.connect, sub2.ne, sub2.overlapCoords, sub2.overlapNodes,
                sub1.node_coords, sub1.connect, sub1.ne, sub1.nodes, sub1.overlapCoords, sub1.overlapNodes,
                full.node_coords, full.connect, full.nodes, full.conditions.x, full.conditions.y, full.conditions.z,
                sub2.widx, sub2.eidx, sub2.sidx, sub2.nidx, sub2.bidx, sub2.nidx,
                sub1.widx, sub1.eidx, sub1.sidx, sub1.nidx, sub1.bidx, sub1.nidx,
                full.widx, full.eidx, full.sidx, full.nidx, full.bidx, full.nidx,
                tmp_ne, tmp_nn,
                full.nn, sub1.nn, sub2.nn,
                sub2.Tprime0, sub1.Tprime0,
                sub2.T0, sub1.T0, full.T0,
                sub2NcLevel1, sub2NcLevel2, sub1NcLevel1,
                sub2dNcdxLevel1, sub2dNcdyLevel1, sub2dNcdzLevel1, sub2nodesLevel1,
                sub2dNcdxLevel2, sub2dNcdyLevel2, sub2dNcdzLevel2, sub2nodesLevel2,
                sub1dNcdxLevel1, sub1dNcdyLevel1, sub1dNcdzLevel1, sub1nodesLevel1,
                sub1sub2_intmat, sub1sub2_node, sub2sub1_intmat, sub2sub1_node,
                fullsub1_intmat, fullsub1_node, sub1full_intmat, sub1full_node,
                _t, v, k, rho, cp, dt, laser_r, laser_d, laser_P, laser_eta,
                T_amb, h_conv, vareps)
            # Update parameters
            time_inc += 1
            record_inc += 1
            ###---------------------END OF EXPLICIT---------------------###
        if record_inc == record_step:
            record_inc = 0
            record_lab = int(time_inc / record_step) + 1
            if output_files == 1:
                if np.mod(record_lab,full_record_inc)==0:
                    save_result(full, 'full_', record_lab, save_path)
                save_result(sub1, 'sub1_', record_lab, save_path)
                save_result(sub2, 'sub2_', record_lab, save_path)

        tend = time.time()
        if save_time:
            timing_file.write('%.9f\n' % (tend - tstart))

        # Timestep, Iter, dTbar, elapsed time
        print('%13.0f,%14.6f seconds, %13.6f ms/timestep' % (time_inc,
                                                             tend - tstart,
                                                             1000*(tend - tstart_loop)))
    if save_time:
        timing_file.close()
    print('full.T0.max: %f' % (full.T0.max()))
    tool_path_file.close()

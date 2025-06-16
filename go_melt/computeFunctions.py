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
from jax.numpy import newaxis as newax
from pyevtk.hl import gridToVTK
import sys
import math

# TFSP: Temporary fix for single precision


def calcNumNodes(elements):
    return [elements[0] + 1, elements[1] + 1, elements[2] + 1]


def createMesh3D(x, y, z):
    """createMesh3D finds nodal coordinates and mesh
    connectivity matrices.
    :param x: list containing (bounds.x[0], bounds.x[1], nodes[0])
    :param y: list containing (bounds.y[0], bounds.y[1], nodes[1])
    :param z: list containing (bounds.z[0], bounds.z[1], nodes[2])
    :return node_coords, connect
    :return node_coords: list of nodes coordinates along axis (3D)
    :return connect: list of connectivity matrices along axis (3D)
    """
    # Node positions in x, y, z
    nx, ny, nz = [jnp.linspace(*axis) for axis in (x, y, z)]

    # Connectivity in x, y, and z
    cx0 = jnp.arange(0, x[2] - 1).reshape(-1, 1)
    cx1 = jnp.arange(1, x[2]).reshape(-1, 1)
    nconn_x = jnp.concatenate([cx0, cx1, cx1, cx0, cx0, cx1, cx1, cx0], axis=1)

    cy0 = jnp.arange(0, y[2] - 1).reshape(-1, 1)
    cy1 = jnp.arange(1, y[2]).reshape(-1, 1)
    nconn_y = jnp.concatenate([cy0, cy0, cy1, cy1, cy0, cy0, cy1, cy1], axis=1)

    cz0 = jnp.arange(0, z[2] - 1).reshape(-1, 1)
    cz1 = jnp.arange(1, z[2]).reshape(-1, 1)
    nconn_z = jnp.concatenate([cz0, cz0, cz0, cz0, cz1, cz1, cz1, cz1], axis=1)

    return [nx, ny, nz], [nconn_x, nconn_y, nconn_z]


### Parse inputs into multiple objects (START) ###
class obj:
    # Constructor
    def __init__(self, dict1):
        self.__dict__.update(dict1)


def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)


def save_object(obj, filename):
    with open(filename, "wb") as outp:  # Overwrites any existing file.
        dill.dump(obj, outp, dill.HIGHEST_PROTOCOL)


def SetupLevels(solver_input, properties):
    # Coarse domain
    _ = solver_input.get("Level1", {})
    Level1 = dict2obj(_)
    # Finer domain
    _ = solver_input.get("Level2", {})
    Level2 = dict2obj(_)
    # Finest domain
    _ = solver_input.get("Level3", {})
    Level3 = dict2obj(_)

    # Steps to do for all three levels
    for level in [Level1, Level2, Level3]:
        level.length, level.h = calc_length_h(level)
        # Calculate number of nodes in each direction
        level.nodes = calcNumNodes(level.elements)
        # Calculate total number of nodes (nn) and elements (ne)
        level.ne = level.elements[0] * level.elements[1] * level.elements[2]
        level.nn = level.nodes[0] * level.nodes[1] * level.nodes[2]
        # Get BC indices
        level.BC = getBCindices(level)
        # Set up meshes for all three levels
        level.node_coords, level.connect = createMesh3D(
            (level.bounds.x[0], level.bounds.x[1], level.nodes[0]),
            (level.bounds.y[0], level.bounds.y[1], level.nodes[1]),
            (level.bounds.z[0], level.bounds.z[1], level.nodes[2]),
        )

        # Preallocate
        level.T = properties["T_amb"] * jnp.ones(level.nn)
        level.T0 = properties["T_amb"] * jnp.ones(level.nn)
        level.S1 = jnp.zeros(level.nn)
        level.S2 = jnp.zeros(level.nn, dtype=bool)
        level.k = properties["k_powder"] * jnp.ones(level.nn)
        level.rhocp = (
            properties["cp_solid_coeff_a0"] * properties["rho"] * jnp.ones(level.nn)
        )

    Level1.S1_storage = jnp.zeros(
        [int(round(Level1.h[2] / properties["layer_height"])), Level1.nn]
    )

    # Steps for sublevels only
    for level in [Level2, Level3]:
        # Constrain fine mesh based on initial conditions
        (
            level.bounds.ix,
            level.bounds.iy,
            level.bounds.iz,
        ) = find_max_const(Level1, level)
        # Deep copy initial coordinates for updating mesh position
        level.init_node_coors = copy.deepcopy(level.node_coords)

        # Preallocate
        level.Tprime = jnp.zeros(level.nn)
        level.Tprime0 = copy.deepcopy(level.Tprime)

    Level1.orig_node_coords = copy.deepcopy(Level1.node_coords)
    trying_flag = True
    tmp_coords = copy.deepcopy(Level1.orig_node_coords)
    while trying_flag:
        if jnp.isclose(tmp_coords[2] - Level2.node_coords[-1][-1], 0, atol=1e-4).any():
            trying_flag = False
        else:
            tmp_coords[2] = tmp_coords[2] + properties["layer_height"]
    Level1.node_coords = copy.deepcopy(tmp_coords)

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

    # Create Level0 to save high-resolution state information
    Level0 = obj({})
    Level0.elements = [
        round(Level1.length[0] / Level3.h[0]),
        round(Level1.length[1] / Level3.h[1]),
        round(Level2.length[2] / Level3.h[2]),
    ]

    # Calculate number of nodes in each direction
    Level0.nodes = calcNumNodes(Level0.elements)
    # Calculate total number of nodes (nn) and elements (ne)
    Level0.ne = Level0.elements[0] * Level0.elements[1] * Level0.elements[2]
    Level0.nn = Level0.nodes[0] * Level0.nodes[1] * Level0.nodes[2]
    # Set up meshes for all three levels
    Level0.node_coords, Level0.connect = createMesh3D(
        (Level1.bounds.x[0], Level1.bounds.x[1], Level0.nodes[0]),
        (Level1.bounds.y[0], Level1.bounds.y[1], Level0.nodes[1]),
        (Level2.bounds.z[0], Level2.bounds.z[1], Level0.nodes[2]),
    )

    Level0.orig_node_coords = copy.deepcopy(Level0.node_coords)

    Level0.orig_overlap_nodes = [
        getCoarseNodesInFineRegion(Level3.node_coords[0], Level0.node_coords[0]),
        getCoarseNodesInFineRegion(Level3.node_coords[1], Level0.node_coords[1]),
        getCoarseNodesInFineRegion(Level3.node_coords[2], Level0.node_coords[2]),
    ]

    Level0.orig_overlap_coors = [
        jnp.array([Level0.node_coords[0][_] for _ in Level0.orig_overlap_nodes[0]]),
        jnp.array([Level0.node_coords[1][_] for _ in Level0.orig_overlap_nodes[1]]),
        jnp.array([Level0.node_coords[2][_] for _ in Level0.orig_overlap_nodes[2]]),
    ]

    Level0.overlapNodes = copy.deepcopy(Level0.orig_overlap_nodes)
    Level0.overlapCoords = copy.deepcopy(Level0.orig_overlap_coors)

    Level0.orig_overlap_nodes_L2 = [
        getCoarseNodesInLargeFineRegion(Level2.node_coords[0], Level0.node_coords[0]),
        getCoarseNodesInLargeFineRegion(Level2.node_coords[1], Level0.node_coords[1]),
        getCoarseNodesInLargeFineRegion(Level2.node_coords[2], Level0.node_coords[2]),
    ]

    Level0.orig_overlap_coors_L2 = [
        jnp.array([Level0.node_coords[0][_] for _ in Level0.orig_overlap_nodes_L2[0]]),
        jnp.array([Level0.node_coords[1][_] for _ in Level0.orig_overlap_nodes_L2[1]]),
        jnp.array([Level0.node_coords[2][_] for _ in Level0.orig_overlap_nodes_L2[2]]),
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

    # To convert it to jax for type comparison later
    for level in [Level0, Level1, Level2, Level3]:
        if hasattr(level, "elements"):
            level.elements = [jnp.array(level.elements[i]) for i in range(3)]
        if hasattr(level, "h"):
            level.h = [jnp.array(level.h[i]) for i in range(3)]
        if hasattr(level, "length"):
            level.length = [jnp.array(level.length[i]) for i in range(3)]
        if hasattr(level, "ne"):
            level.ne = jnp.array(level.ne)
        if hasattr(level, "nn"):
            level.nn = jnp.array(level.nn)
        if hasattr(level, "nodes"):
            level.nodes = [jnp.array(level.nodes[i]) for i in range(3)]
        if hasattr(level, "layer_idx_delta"):
            level.layer_idx_delta = jnp.array(level.layer_idx_delta)

    Levels = [
        structure_to_dict(Level0),
        structure_to_dict(Level1),
        structure_to_dict(Level2),
        structure_to_dict(Level3),
    ]
    return Levels


def SetupProperties(prop_obj):
    properties = dict2obj(prop_obj)
    # Define the material properties
    # Conductivity (W/mK), (Temperature-dependence a1 * T + a0)
    properties.k_powder = prop_obj.get("thermal_conductivity_powder", 0.4)
    properties.k_bulk_coeff_a0 = prop_obj.get("thermal_conductivity_bulk_a0", 4.23)
    properties.k_bulk_coeff_a1 = prop_obj.get("thermal_conductivity_bulk_a1", 0.016)
    properties.k_fluid_coeff_a0 = prop_obj.get("thermal_conductivity_fluid_a0", 29.0)
    # Heat capacity (J/kgK), (Temperature-dependence a1 * T + a0)
    properties.cp_solid_coeff_a0 = prop_obj.get("heat_capacity_solid_a0", 383.1)
    properties.cp_solid_coeff_a1 = prop_obj.get("heat_capacity_solid_a1", 0.174)
    properties.cp_mushy = prop_obj.get("heat_capacity_mushy", 3235.0)
    properties.cp_fluid = prop_obj.get("heat_capacity_fluid", 769.0)
    # Density (kg/mm^3)
    properties.rho = prop_obj.get("density", 8.0e-6)
    # Laser radius (sigma, mm)
    properties.laser_r = prop_obj.get("laser_radius", 0.110)
    # Laser radius (sigma, mm)
    properties.laser_r_bot = prop_obj.get("laser_radius_bottom", 0.110)
    # Laser depth (d, mm)
    properties.laser_d = prop_obj.get("laser_depth", 0.05)
    # Laser power (P, W)
    properties.laser_P = prop_obj.get("laser_power", 300.0)
    # Laser absorptivity (eta, unitless)
    properties.laser_eta = prop_obj.get("laser_absorptivity", 0.25)
    # Starting laser center (read from tool path file is default)
    properties.laser_center = prop_obj.get("laser_center", [])
    # Ambient Temperature (properties.k)
    properties.T_amb = prop_obj.get("T_amb", 353.15)
    properties.T_solidus = prop_obj.get("T_solidus", 1554.0)
    properties.T_liquidus = prop_obj.get("T_liquidus", 1625.0)
    properties.T_boiling = prop_obj.get("T_boiling", 3038.0)
    # Convection coefficient (properties.h_conv, W/mm^2K)
    properties.h_conv = prop_obj.get("h_conv", 1.473e-5)
    # Emissivity (unitless)
    properties.vareps = prop_obj.get("emissivity", 0.600)
    # Evaporation coefficient (unitless)
    properties.evc = prop_obj.get("evaporation_coefficient", 0.82)
    # Boltzmann's constant (J/properties.k)
    properties.kb = prop_obj.get("boltzmann_constant", 1.38e-23)
    # Atomic mass (kg)
    properties.mA = prop_obj.get("atomic_mass", 7.9485017e-26)
    # Saturation pressure (Pa)
    properties.Patm = prop_obj.get("saturation_prussure", 101.0e3)
    # Latent heat of evaporation (J/kg)
    properties.Lev = prop_obj.get("latent_heat_evap", 4.22e6)
    # Layer height (micrometers)
    properties.layer_height = prop_obj.get("layer_height", 0.04)
    properties.sigma_sb = 5.67e-8
    ### User-defined parameters (END) ###
    properties_dict = structure_to_dict(properties)
    return properties_dict


def SetupNonmesh(nonmesh_input):
    Nonmesh = dict2obj(nonmesh_input)
    ### User-defined parameters (START) ###
    # Timestep (for finest mesh)
    Nonmesh.timestep_L3 = nonmesh_input.get("timestep_L3", 1e-5)
    # Subcycle numbers
    Nonmesh.subcycle_num_L2 = nonmesh_input.get("subcycle_num_L2", 1)
    Nonmesh.subcycle_num_L3 = nonmesh_input.get("subcycle_num_L3", 1)
    # Dwell time timestep (explicit)
    Nonmesh.dwell_time = nonmesh_input.get("dwell_time", 0.1)

    # How frequently (factor) to record coarse-domain
    Nonmesh.Level1_record_step = nonmesh_input.get("Level1_record_step", 1)
    # Where to save
    Nonmesh.save_path = nonmesh_input.get("save_path", "results/")
    # Where to save
    Nonmesh.npz_folder = nonmesh_input.get("npz_folder", "npz_folder/")
    # Flag to save vtk/vtr files
    Nonmesh.output_files = nonmesh_input.get("output_files", 1)
    # Flag to save timing results
    Nonmesh.savetime = nonmesh_input.get("savetime", 0)
    # Flag to save valid results
    Nonmesh.savevalid = nonmesh_input.get("savevalid", 0)
    # Location of toolpath file
    Nonmesh.toolpath = nonmesh_input.get("toolpath", "laserPath.txt")
    # Number of timesteps to wait before larger dt in dwell time
    Nonmesh.wait_time = nonmesh_input.get("wait_time", 500.0)
    # Load layer
    Nonmesh.layer_num = nonmesh_input.get("layer_num", 0)
    # Output info
    Nonmesh.info_T = nonmesh_input.get("info_T", 0)
    # Laser velocity (mm/s)
    Nonmesh.laser_velocity = nonmesh_input.get("laser_velocity", 500)
    # How long to wait after each track finishes before moving on
    Nonmesh.wait_track = nonmesh_input.get("wait_track", 0.0)

    # How frequently to record and/or check to move the fine domain
    Nonmesh.record_step = nonmesh_input.get(
        "record_step", Nonmesh.subcycle_num_L2 * Nonmesh.subcycle_num_L3
    )
    Nonmesh.gcode = nonmesh_input.get(
        "gcode", "./examples/gcodefiles/defaultName.gcode"
    )
    Nonmesh.dwell_time_multiplier = nonmesh_input.get("dwell_time_multiplier", 1)
    Nonmesh.use_txt = nonmesh_input.get("use_txt", 0)
    # How long to wait after each track finishes before moving on
    Nonmesh.restart_layer_num = nonmesh_input.get("restart_layer_num", 10000)

    if not os.path.exists(Nonmesh.save_path):
        os.makedirs(Nonmesh.save_path)

    ### User-defined parameters (END) ###
    Nonmesh_dict = structure_to_dict(Nonmesh)
    return Nonmesh_dict


def structure_to_dict(struct):
    return {
        k: (
            v
            if (not hasattr(v, "__dict__") or hasattr(v, "tolist"))
            else structure_to_dict(v)
        )
        for k, v in struct.__dict__.items()
    }


def getStaticNodesAndElements(L):
    # L is Levels
    return (
        L[2]["ne"].tolist(),
        L[3]["ne"].tolist(),
        L[1]["nn"].tolist(),
        L[2]["nn"].tolist(),
        L[3]["nn"].tolist(),
    )


def getStaticSubcycle(N):
    # N is Nonmesh
    N2, N3 = N["subcycle_num_L2"], N["subcycle_num_L3"]
    N23 = N2 * N3
    fN2, fN3, fN23 = float(N2), float(N3), float(N23)
    return (N2, N3, N23, fN2, fN3, fN23)


def calcStaticTmpNodesAndElements(L, v):
    # L is Levels
    Level1_mask = L[1]["node_coords"][2] <= v[2] + 1e-5
    Level1_nn = sum(Level1_mask)
    tmp_ne = L[1]["elements"][0] * L[1]["elements"][1] * (Level1_nn - 1)
    tmp_ne = tmp_ne.tolist()
    tmp_nn = (L[1]["nodes"][0] * L[1]["nodes"][1] * Level1_nn).tolist()
    return (tmp_ne, tmp_nn)


def getBCindices(x):
    nx, ny, nz = x.nodes[0], x.nodes[1], x.nodes[2]
    nn = x.nn

    bidx = jnp.arange(0, nx * ny)
    tidx = jnp.arange(nx * ny * (nz - 1), nn)
    widx = jnp.arange(0, nn, nx)
    eidx = jnp.arange(nx - 1, nn, nx)
    sidx = jnp.arange(0, nx)[:, newax] + (nx * ny * jnp.arange(0, nz))[newax, :]
    sidx = sidx.reshape(-1)
    nidx = (
        jnp.arange(nx * (ny - 1), nx * ny)[:, newax]
        + (nx * ny * jnp.arange(0, nz))[newax, :]
    )
    nidx = nidx.reshape(-1)
    return [widx, eidx, sidx, nidx, bidx, tidx]


def getSubstrateNodes(Levels):
    substrate = [
        ((L["node_coords"][2] < 1e-5).sum() * L["nodes"][0] * L["nodes"][1]).tolist()
        for L in Levels[1:5]
    ]
    return (0, *substrate)


@partial(jax.jit, static_argnames=["ne", "nn"])
def solveMatrixFreeFE(Level, nn, ne, k, rhocp, dt, T, Fc, Corr):
    """solveMatrixFreeFE computes the thermal solve for the explicit timestep.
    :param Level: Mesh level
    :param nn, ne: number of nodes, number of elements
    :param dt: timestep
    :param T: previous temperature for mesh
    :param Fc: integrated RHS values (including heat source)
    :param Corr: integrated correction terms
    :return (newT + Fc + Corr) / newM
    :return newT: Temperture for next timestep
    """

    nen = jnp.size(Level["connect"][0], 1)
    ndim = 3

    coords = getSampleCoords(Level)
    N, dNdx, wq = computeQuad3dFemShapeFunctions_jax(coords)
    wq = wq[0][0]
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
        kvec = (k[idx]).mean()
        mvec = (rhocp[idx]).mean() / dt

        Me = jnp.sum(NTN * wq * mvec, 0)
        Ke = BTB * kvec * wq
        LHSe = jnp.diag(Me) - Ke

        return jnp.matmul(LHSe, T[idx]), Me, idx

    vcalcVal = jax.vmap(calcVal)
    aT, aMe, aidx = vcalcVal(jnp.arange(ne))
    newT = jnp.bincount(aidx.reshape(-1), aT.reshape(-1), length=nn)
    newM = jnp.bincount(aidx.reshape(-1), aMe.reshape(-1), length=nn)
    return (newT + Fc + Corr) / newM


@jax.jit
def convert2XYZ(i, ne_x, ne_y, nn_x, nn_y):
    """convert2XYZ computes the indices for each node w.r.t. each axis.
    It also computes the connectivity matrix in terms of global indices.
    :param i: element id
    :param ne_x, ne_y: number of elements in x and y directions
    :param nn_x, nn_y: number of nodes in x and y directions
    :return ix, iy, iz, idx
    :return ix, iy, iz: Node id in either x, y, or z axis
    :return idx: connectivity vector of global node ids
    """

    iz, _ = jnp.divmod(i, (ne_x) * (ne_y))
    iy, _ = jnp.divmod(i, ne_x)
    iy -= iz * ne_y
    ix = jnp.mod(i, ne_x)

    idx = jnp.array(
        [
            ix + iy * (nn_x) + iz * (nn_x * nn_y),
            (ix + 1) + iy * (nn_x) + iz * (nn_x * nn_y),
            (ix + 1) + (iy + 1) * (nn_x) + iz * (nn_x * nn_y),
            ix + (iy + 1) * (nn_x) + iz * (nn_x * nn_y),
            ix + iy * (nn_x) + (iz + 1) * (nn_x * nn_y),
            (ix + 1) + iy * (nn_x) + (iz + 1) * (nn_x * nn_y),
            (ix + 1) + (iy + 1) * (nn_x) + (iz + 1) * (nn_x * nn_y),
            ix + (iy + 1) * (nn_x) + (iz + 1) * (nn_x * nn_y),
        ]
    )

    return ix, iy, iz, idx


@jax.jit
def computeQuad3dFemShapeFunctions_jax(coords):
    """def computeQuad3dFemShapeFunctions_jax calculates the 3D shape functions
    and shape function derivatives for a given element when integrating using
    Gaussian quadrature. The quadrature weights are also returned.
    :param coords: nodal coordinates of element
    :return N, dNdx, wq
    :return N: shape function
    :return dNdx: derivative of shape function (3D)
    :return wq: quadrature weights for each of the eight quadrature points
    """

    ngp = 8  # Total number of quadrature points
    ndim = 3  # Total number of spatial dimensions

    # Define isoparametric coordinates in 3D space
    ksi_i = jnp.array([-1, 1, 1, -1, -1, 1, 1, -1])
    eta_i = jnp.array([-1, -1, 1, 1, -1, -1, 1, 1])
    zeta_i = jnp.array([-1, -1, -1, -1, 1, 1, 1, 1])

    # Define quadrature coordinates
    ksi_q = (1 / jnp.sqrt(3)) * ksi_i
    eta_q = (1 / jnp.sqrt(3)) * eta_i
    zeta_q = (1 / jnp.sqrt(3)) * zeta_i

    # Preallocate quadrature weights
    tmp_wq = jnp.ones(ngp)

    # Calculate shape function and derivative of shape function for quadrature points
    _ksi = 1 + ksi_q[:, newax] @ ksi_i[newax, :]
    _eta = 1 + eta_q[:, newax] @ eta_i[newax, :]
    _zeta = 1 + zeta_q[:, newax] @ zeta_i[newax, :]
    N = (1 / 8) * (_ksi) * (_eta) * (_zeta)
    dNdksi = (1 / 8) * ksi_i[newax, :] * (_eta) * (_zeta)
    dNdeta = (1 / 8) * eta_i[newax, :] * (_ksi) * (_zeta)
    dNdzeta = (1 / 8) * zeta_i[newax, :] * (_ksi) * (_eta)

    # Find derivative of parent coordinates w.r.t. isoparametric space
    dxdksi = jnp.matmul(dNdksi, coords[:, 0])
    dydeta = jnp.matmul(dNdeta, coords[:, 1])
    dzdzeta = jnp.matmul(dNdzeta, coords[:, 2])

    # Find Jacobian matrices and calculate quadrature weights and dNdx
    J = jnp.array([[dxdksi[0], 0, 0], [0, dydeta[0], 0], [0, 0, dzdzeta[0]]])
    Jinv = jnp.array(
        [[1 / dxdksi[0], 0, 0], [0, 1 / dydeta[0], 0], [0, 0, 1 / dzdzeta[0]]]
    )
    dNdx = jnp.zeros([ngp, ngp, ndim])
    wq = jnp.zeros([ngp, 1])
    for q in range(ngp):
        dNdx = dNdx.at[q, :, :].set(
            jnp.concatenate(
                [dNdksi[q, :, newax], dNdeta[q, :, newax], dNdzeta[q, :, newax]], axis=1
            )
            @ Jinv
        )
        wq = wq.at[q].set(jnp.linalg.det(J) * tmp_wq[q])
    return jnp.array(N), jnp.array(dNdx), jnp.array(wq)


@jax.jit
def computeQuad2dFemShapeFunctions_jax(coords):
    """def computeQuad2dFemShapeFunctions_jax calculates the 2D shape functions
    and shape function derivatives for a given element when integrating using
    Gaussian quadrature. The quadrature weights are also returned.
    :param coords: nodal coordinates of element
    :return N, dNdx, wq
    :return N: shape function
    :return dNdx: derivative of shape function (2D)
    :return wq: quadrature weights for each of the four quadrature points
    """
    ngp = 4  # Total number of quadrature points
    ndim = 2  # Total number of dimensions

    # Define isoparametric coordinates in 3D space
    ksi_i = jnp.array([-1, 1, 1, -1])
    eta_i = jnp.array([-1, -1, 1, 1])

    # Define quadrature coordinates
    ksi_q = (1 / jnp.sqrt(3)) * ksi_i
    eta_q = (1 / jnp.sqrt(3)) * eta_i

    # Preallocate quadrature weights
    tmp_wq = jnp.ones(ngp)

    # Calculate using local coordinate system
    _ksi = 1 + ksi_q[:, newax] @ ksi_i[newax, :]
    _eta = 1 + eta_q[:, newax] @ eta_i[newax, :]
    N = (1 / 4) * _ksi * _eta
    dNdksi = (1 / 4) * ksi_i[newax, :] * _eta
    dNdeta = (1 / 4) * eta_i[newax, :] * _ksi

    dxdksi = jnp.matmul(dNdksi, coords[4:, 0])
    dydeta = jnp.matmul(dNdeta, coords[4:, 1])

    J = jnp.array([[dxdksi[0], 0], [0, dydeta[0]]])
    Jinv = jnp.array([[1 / dxdksi[0], 0], [0, 1 / dydeta[0]]])
    dNdx = jnp.zeros([ngp, ngp, ndim])
    wq = jnp.zeros([ngp, 1])
    for q in range(ngp):
        dNdx = dNdx.at[q, :, :].set(
            jnp.concatenate([dNdksi[q, :, newax], dNdeta[q, :, newax]], axis=1) @ Jinv
        )
        wq = wq.at[q].set(jnp.linalg.det(J) * tmp_wq[q])
    return jnp.array(N), jnp.array(dNdx), jnp.array(wq)


def getCoarseNodesInFineRegion(xnf, xnc):
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
    overlap = jnp.arange(overlapMin, overlapMax, int(jnp.round(hc / hf))).astype(int)

    return overlap


@partial(jax.jit, static_argnames=["ne_nn"])
def computeSources(Level, v, Shapes, ne_nn, properties, laserP):
    """computeSources computes the integrated source term for all three levels using
    the mesh from Level 3.
    :param Levels: multilevel container
    :param v: current position of laser (from reading file)
    :param Shapes[1]: shape/node between Level 3 and Level 1 (symmetric)
    :param Shapes[2]: shape/node between Level 3 and Level 2 (symmetric)
    :param ne_nn: total number of elements/nodes (active and deactive)
    :param properties: material/laser properties
    :param laserP: laser power
    :return Fc, Fm, Ff
    :return Fc: integrated source term for Levels[1]
    :return Fm: integrated source term for Levels[2]
    :return Ff: integrated source term for Levels[3]
    """
    # Get shape functions and weights
    coords = getSampleCoords(Level)
    Nf, _, wqf = computeQuad3dFemShapeFunctions_jax(coords)

    def stepcomputeCoarseSource(ieltf):
        # Get the nodal indices for that element
        ix, iy, iz, idx = convert2XYZ(
            ieltf,
            Level["elements"][0],
            Level["elements"][1],
            Level["nodes"][0],
            Level["nodes"][1],
        )
        # Get nodal coordinates for the fine element
        x, y, z = getQuadratureCoords(Level, ix, iy, iz, Nf)
        # Compute the source at the quadrature point location
        Q = computeSourceFunction_jax(x, y, z, v, properties, laserP)
        return Q * wqf, Nf @ Q * wqf, idx

    vstepcomputeCoarseSource = jax.vmap(stepcomputeCoarseSource)

    # Returns data for Shapes[1][0]/Shapes[2][0], data premultiplied by Nf, and nodes for Level 3
    _data, _data3, nodes3 = vstepcomputeCoarseSource(jnp.arange(ne_nn[1]))

    # This will be equivalent to a transverse matrix operation for each fine element
    _data1 = multiply(Shapes[1][0], _data).sum(axis=1)
    _data2 = multiply(Shapes[2][0], _data).sum(axis=1)
    Fc = Shapes[1][2] @ _data1.reshape(-1)
    Fm = Shapes[2][2] @ _data2.reshape(-1)
    Ff = bincount(nodes3.reshape(-1), _data3.reshape(-1), ne_nn[4])
    return Fc, Fm, Ff


@jax.jit
def computeSourceFunction_jax(x, y, z, v, properties, P):
    """computeSourceFunction_jax computes a 3D Gaussian term.
    :param x, y, z: nodal coordinates
    :param v: laser center
    :param properties
    :param P: laser power
    :param properties["laser_eta"]: laser emissivity
    :return F: output of source equation
    """
    # Precompute some constants
    _pcoeff = 6 * jnp.sqrt(3) * P * properties["laser_eta"]
    _rcoeff = 1 / (properties["laser_r"] * jnp.sqrt(jnp.pi))
    _dcoeff = 1 / (properties["laser_d"] * jnp.sqrt(jnp.pi))
    _rsq = properties["laser_r"] ** 2
    _dsq = properties["laser_d"] ** 2

    # Assume each source is independent, multiply afterwards
    Qx = _rcoeff * jnp.exp(-3 * (x - v[0]) ** 2 / _rsq)
    Qy = _rcoeff * jnp.exp(-3 * (y - v[1]) ** 2 / _rsq)
    Qz = _dcoeff * jnp.exp(-3 * (z - v[2]) ** 2 / _dsq)

    return _pcoeff * Qx * Qy * Qz


@jax.jit
def interpolatePointsMatrix(Level, node_coords_new):
    """interpolatePointsMatrix computes shape functions and node indices
    to interpolate solutions located on Level["node_coords"] and connected with
    Level["connect"] to new coordinates node_coords_new. This function outputs
    shape functions for interpolation that is between levels
    :param Level["node_coords"]: source nodal coordinates
    :param Level["connect"]: connectivity matrix
    :param node_coords_new: output nodal coordinates
    :return _Nc, _node
    :return _Nc: shape function connecting source to output
    :return _node: coarse node indices
    """
    ne_x = Level["connect"][0].shape[0]
    ne_y = Level["connect"][1].shape[0]
    ne_z = Level["connect"][2].shape[0]
    nn_x, nn_y = ne_x + 1, ne_y + 1

    nn_xn = len(node_coords_new[0])
    nn_yn = len(node_coords_new[1])
    nn_zn = len(node_coords_new[2])
    tmp_ne_nn = nn_xn * nn_yn * nn_zn
    h_x = Level["node_coords"][0][1] - Level["node_coords"][0][0]
    h_y = Level["node_coords"][1][1] - Level["node_coords"][1][0]
    h_z = Level["node_coords"][2][1] - Level["node_coords"][2][0]

    def stepInterpolatePoints(ielt):
        # Get nodal indices
        izn, _ = jnp.divmod(ielt, (nn_xn) * (nn_yn))
        iyn, _ = jnp.divmod(ielt, nn_xn)
        iyn -= izn * nn_yn
        ixn = jnp.mod(ielt, nn_xn)

        _x = node_coords_new[0][ixn, newax]
        _y = node_coords_new[1][iyn, newax]
        _z = node_coords_new[2][izn, newax]

        x_comp = (ne_x - 1) * jnp.ones_like(_x)
        y_comp = (ne_y - 1) * jnp.ones_like(_y)
        z_comp = (ne_z - 1) * jnp.ones_like(_z)

        x_comp2 = jnp.zeros_like(_x)
        y_comp2 = jnp.zeros_like(_y)
        z_comp2 = jnp.zeros_like(_z)

        # Figure out which coarse element we are in
        _floorx = jnp.floor((_x - Level["node_coords"][0][0]) / h_x)
        _conx = jnp.concatenate((_floorx, x_comp))
        _ielt_x = jnp.min(_conx)
        _conx = jnp.concatenate((_ielt_x[newax], x_comp2))
        ielt_x = jnp.max(_conx).T.astype(int)

        _floory = jnp.floor((_y - Level["node_coords"][1][0]) / h_y)
        _cony = jnp.concatenate((_floory, y_comp))
        _ielt_y = jnp.min(_cony)
        _cony = jnp.concatenate((_ielt_y[newax], y_comp2))
        ielt_y = jnp.max(_cony).T.astype(int)

        _floorz = jnp.floor((_z - Level["node_coords"][2][0]) / h_z)
        _conz = jnp.concatenate((_floorz, z_comp))
        _ielt_z = jnp.min(_conz).T.astype(int)
        _conz = jnp.concatenate((_ielt_z[newax], z_comp2))
        ielt_z = jnp.max(_conz).T.astype(int)

        nodex = Level["connect"][0][ielt_x, :]
        nodey = Level["connect"][1][ielt_y, :]
        nodez = Level["connect"][2][ielt_z, :]
        node = nodex + nodey * nn_x + nodez * (nn_x * nn_y)

        xx = Level["node_coords"][0][nodex]
        yy = Level["node_coords"][1][nodey]
        zz = Level["node_coords"][2][nodez]

        xc0, xc1 = xx[0], xx[1]
        yc0, yc3 = yy[0], yy[3]
        zc0, zc5 = zz[0], zz[5]

        # Evaluate shape functions associated with coarse nodes
        Nc = jnp.concatenate(
            compute3DN(
                [_x, _y, _z], [xc0, xc1], [yc0, yc3], [zc0, zc5], [h_x, h_y, h_z]
            )
        )
        # The where makes it zero outside the interpolation range
        Nc = ((Nc >= -1e-2).all() & (Nc <= 1 + 1e-2).all()) * Nc
        Nc = jnp.clip(Nc, 0.0, 1.0)
        return Nc, node

    vstepInterpolatePoints = jax.vmap(stepInterpolatePoints)
    _Nc, _node = vstepInterpolatePoints(jnp.arange(tmp_ne_nn))
    return [_Nc, _node]


@jax.jit
def interpolate_w_matrix(C2F, T):
    """interpolate_w_matrix uses shape functions from interpolatePointsMatrix
    to interpolate the solution to the new nodal coordinates
    :param C2F[0]: shape functions for interpolation
    :param C2F[1]: nodal indices of source solution
    :param T: source solution
    :return T_new: interpolated solution at new nodal coordinates
    """
    return multiply(C2F[0], T[C2F[1]]).sum(axis=1)


@jax.jit
def interpolatePoints(Level, u, node_coords_new):
    """interpolatePoints interpolate solutions located on Level["node_coords"]
    and connected with Level["connect"] to new coordinates node_coords_new. Values
    that are later bin counted are the output
    :param Level["node_coords"]: source nodal coordinates
    :param Level["connect"]: connectivity matrix
    :param node_coords_new: output nodal coordinates
    :return _val: nodal values that need to be bincounted
    """
    ne_x = Level["connect"][0].shape[0]
    ne_y = Level["connect"][1].shape[0]
    ne_z = Level["connect"][2].shape[0]
    nn_x, nn_y = ne_x + 1, ne_y + 1

    nn_xn = len(node_coords_new[0])
    nn_yn = len(node_coords_new[1])
    nn_zn = len(node_coords_new[2])
    tmp_ne_nn = nn_xn * nn_yn * nn_zn
    h_x = Level["node_coords"][0][1] - Level["node_coords"][0][0]
    h_y = Level["node_coords"][1][1] - Level["node_coords"][1][0]
    h_z = Level["node_coords"][2][1] - Level["node_coords"][2][0]

    def stepInterpolatePoints(ielt):
        # Get nodal indices
        izn, _ = jnp.divmod(ielt, (nn_xn) * (nn_yn))
        iyn, _ = jnp.divmod(ielt, nn_xn)
        iyn -= izn * nn_yn
        ixn = jnp.mod(ielt, nn_xn)

        _x = node_coords_new[0][ixn, newax]
        _y = node_coords_new[1][iyn, newax]
        _z = node_coords_new[2][izn, newax]

        x_comp = (ne_x - 1) * jnp.ones_like(_x)
        y_comp = (ne_y - 1) * jnp.ones_like(_y)
        z_comp = (ne_z - 1) * jnp.ones_like(_z)

        x_comp2 = jnp.zeros_like(_x)
        y_comp2 = jnp.zeros_like(_y)
        z_comp2 = jnp.zeros_like(_z)

        # Figure out which coarse element we are in
        _floorx = jnp.floor((_x - Level["node_coords"][0][0]) / h_x)
        _conx = jnp.concatenate((_floorx, x_comp))
        _ielt_x = jnp.min(_conx)
        _conx = jnp.concatenate((_ielt_x[newax], x_comp2))
        ielt_x = jnp.max(_conx).T.astype(int)

        _floory = jnp.floor((_y - Level["node_coords"][1][0]) / h_y)
        _cony = jnp.concatenate((_floory, y_comp))
        _ielt_y = jnp.min(_cony)
        _cony = jnp.concatenate((_ielt_y[newax], y_comp2))
        ielt_y = jnp.max(_cony).T.astype(int)

        _floorz = jnp.floor((_z - Level["node_coords"][2][0]) / h_z)
        _conz = jnp.concatenate((_floorz, z_comp))
        _ielt_z = jnp.min(_conz).T.astype(int)
        _conz = jnp.concatenate((_ielt_z[newax], z_comp2))
        ielt_z = jnp.max(_conz).T.astype(int)

        nodex = Level["connect"][0][ielt_x, :]
        nodey = Level["connect"][1][ielt_y, :]
        nodez = Level["connect"][2][ielt_z, :]
        node = nodex + nodey * nn_x + nodez * (nn_x * nn_y)

        xx = Level["node_coords"][0][nodex]
        yy = Level["node_coords"][1][nodey]
        zz = Level["node_coords"][2][nodez]

        xc0, xc1 = xx[0], xx[1]
        yc0, yc3 = yy[0], yy[3]
        zc0, zc5 = zz[0], zz[5]

        # Evaluate shape functions associated with coarse nodes
        Nc = jnp.concatenate(
            compute3DN(
                [_x, _y, _z], [xc0, xc1], [yc0, yc3], [zc0, zc5], [h_x, h_y, h_z]
            )
        )
        # The where makes it zero outside the interpolation range
        Nc = ((Nc >= -1e-2).all() & (Nc <= 1 + 1e-2).all()) * Nc
        Nc = jnp.clip(Nc, 0.0, 1.0)
        return Nc @ u[node]

    vstepInterpolatePoints = jax.vmap(stepInterpolatePoints)
    return vstepInterpolatePoints(jnp.arange(tmp_ne_nn))


@jax.jit
def computeCoarseFineShapeFunctions(Coarse, Fine):
    """computeCoarseFineShapeFunctions finds the shape functions of
    the fine scale quadrature points for the coarse element
    :param Coarse["node_coords"]: nodal coordinates of global coarse
    :param Coarse["connect"]: indices to get coordinates of nodes of coarse element
    :param Fine["node_coords"]: nodal coordinates of global fine
    :param Fine["connect"]: indices to get x coordinates of nodes of fine element
    :return Nc, dNcdx, dNcdy, dNcdz, _nodes.reshape(-1)
    :return Nc: (Num fine elements, 8 quadrature, 8), coarse shape function for fine element
    :return dNcdx: (Num fine elements, 8 quadrature, 8), coarse x-derivate shape function for fine element
    :return dNcdy: (Num fine elements, 8 quadrature, 8), coarse y-derivate shape function for fine element
    :return dNcdz: (Num fine elements, 8 quadrature, 8), coarse z-derivate shape function for fine element
    :return _nodes: (Num fine elements * 8 * 8), coarse nodal indices
    """
    # Get number of elements and nodes for both coarse and fine
    nec_x, nec_y, nec_z = [Coarse["connect"][i].shape[0] for i in range(3)]
    nnc_x, nnc_y, nnc_z = [Coarse["node_coords"][i].shape[0] for i in range(3)]
    nnc = nnc_x * nnc_y * nnc_z
    nef_x, nef_y, nef_z = [Fine["connect"][i].shape[0] for i in range(3)]
    nef = nef_x * nef_y * nef_z
    nnf_x, nnf_y = [Fine["node_coords"][i].shape[0] for i in range(2)]

    # Assume constant mesh sizes
    hc_x = Coarse["node_coords"][0][1] - Coarse["node_coords"][0][0]
    hc_y = Coarse["node_coords"][1][1] - Coarse["node_coords"][1][0]
    hc_z = Coarse["node_coords"][2][1] - Coarse["node_coords"][2][0]

    # Get lower bounds of meshes
    xminc_x, xminc_y, xminc_z = [Coarse["node_coords"][i][0] for i in range(3)]

    # Get shape functions and weights
    coords_x = Fine["node_coords"][0][Fine["connect"][0][0, :]].reshape(-1, 1)
    coords_y = Fine["node_coords"][1][Fine["connect"][1][0, :]].reshape(-1, 1)
    coords_z = Fine["node_coords"][2][Fine["connect"][2][0, :]].reshape(-1, 1)
    coords = jnp.concatenate([coords_x, coords_y, coords_z], axis=1)
    Nf, _, _ = computeQuad3dFemShapeFunctions_jax(coords)

    def stepComputeCoarseFineTerm(ieltf):
        ix, iy, iz, _ = convert2XYZ(ieltf, nef_x, nef_y, nnf_x, nnf_y)
        coords_x = Fine["node_coords"][0][Fine["connect"][0][ix, :]].reshape(-1, 1)
        coords_y = Fine["node_coords"][1][Fine["connect"][1][iy, :]].reshape(-1, 1)
        coords_z = Fine["node_coords"][2][Fine["connect"][2][iz, :]].reshape(-1, 1)

        # Do all of the quadrature points simultaneously
        x = Nf @ coords_x
        y = Nf @ coords_y
        z = Nf @ coords_z

        x_comp = (nec_x - 1) * jnp.ones_like(x)
        y_comp = (nec_y - 1) * jnp.ones_like(y)
        z_comp = (nec_z - 1) * jnp.ones_like(z)

        # Figure out which coarse element we are in
        _floorx = jnp.floor((x - xminc_x) / hc_x)
        _conx = jnp.concatenate((_floorx, x_comp), axis=1)
        ieltc_x = jnp.min(_conx, axis=1).T.astype(int)
        _floory = jnp.floor((y - xminc_y) / hc_y)
        _cony = jnp.concatenate((_floory, y_comp), axis=1)
        ieltc_y = jnp.min(_cony, axis=1).T.astype(int)
        _floorz = jnp.floor((z - xminc_z) / hc_z)
        _conz = jnp.concatenate((_floorz, z_comp), axis=1)
        ieltc_z = jnp.min(_conz, axis=1).T.astype(int)

        x = x.reshape(-1)
        y = y.reshape(-1)
        z = z.reshape(-1)

        def iqLoopMass(iq):
            nodec_x = Coarse["connect"][0][ieltc_x[iq], :].astype(int)
            nodec_y = Coarse["connect"][1][ieltc_y[iq], :].astype(int)
            nodec_z = Coarse["connect"][2][ieltc_z[iq], :].astype(int)
            nodes = nodec_x + nodec_y * nnc_x + nodec_z * nnc_x * nnc_y

            _x = x[iq]
            _y = y[iq]
            _z = z[iq]

            xc0 = Coarse["node_coords"][0][Coarse["connect"][0][ieltc_x[iq], 0]]
            xc1 = Coarse["node_coords"][0][Coarse["connect"][0][ieltc_x[iq], 1]]
            yc0 = Coarse["node_coords"][1][Coarse["connect"][1][ieltc_y[iq], 0]]
            yc3 = Coarse["node_coords"][1][Coarse["connect"][1][ieltc_y[iq], 3]]
            zc0 = Coarse["node_coords"][2][Coarse["connect"][2][ieltc_z[iq], 0]]
            zc5 = Coarse["node_coords"][2][Coarse["connect"][2][ieltc_z[iq], 5]]

            # Evaluate shape functions associated with coarse nodes
            Nc = compute3DN(
                [_x, _y, _z], [xc0, xc1], [yc0, yc3], [zc0, zc5], [hc_x, hc_y, hc_z]
            )

            # Evaluate shape functions associated with coarse nodes
            dNcdx = jnp.array(
                [
                    ((-1) / hc_x * (yc3 - _y) / hc_y * (zc5 - _z) / hc_z),
                    ((1) / hc_x * (yc3 - _y) / hc_y * (zc5 - _z) / hc_z),
                    ((1) / hc_x * (_y - yc0) / hc_y * (zc5 - _z) / hc_z),
                    ((-1) / hc_x * (_y - yc0) / hc_y * (zc5 - _z) / hc_z),
                    ((-1) / hc_x * (yc3 - _y) / hc_y * (_z - zc0) / hc_z),
                    ((1) / hc_x * (yc3 - _y) / hc_y * (_z - zc0) / hc_z),
                    ((1) / hc_x * (_y - yc0) / hc_y * (_z - zc0) / hc_z),
                    ((-1) / hc_x * (_y - yc0) / hc_y * (_z - zc0) / hc_z),
                ]
            )
            # Evaluate shape functions associated with coarse nodes
            dNcdy = jnp.array(
                [
                    ((xc1 - _x) / hc_x * (-1) / hc_y * (zc5 - _z) / hc_z),
                    ((_x - xc0) / hc_x * (-1) / hc_y * (zc5 - _z) / hc_z),
                    ((_x - xc0) / hc_x * (1) / hc_y * (zc5 - _z) / hc_z),
                    ((xc1 - _x) / hc_x * (1) / hc_y * (zc5 - _z) / hc_z),
                    ((xc1 - _x) / hc_x * (-1) / hc_y * (_z - zc0) / hc_z),
                    ((_x - xc0) / hc_x * (-1) / hc_y * (_z - zc0) / hc_z),
                    ((_x - xc0) / hc_x * (1) / hc_y * (_z - zc0) / hc_z),
                    ((xc1 - _x) / hc_x * (1) / hc_y * (_z - zc0) / hc_z),
                ]
            )
            # Evaluate shape functions associated with coarse nodes
            dNcdz = jnp.array(
                [
                    ((xc1 - _x) / hc_x * (yc3 - _y) / hc_y * (-1) / hc_z),
                    ((_x - xc0) / hc_x * (yc3 - _y) / hc_y * (-1) / hc_z),
                    ((_x - xc0) / hc_x * (_y - yc0) / hc_y * (-1) / hc_z),
                    ((xc1 - _x) / hc_x * (_y - yc0) / hc_y * (-1) / hc_z),
                    ((xc1 - _x) / hc_x * (yc3 - _y) / hc_y * (1) / hc_z),
                    ((_x - xc0) / hc_x * (yc3 - _y) / hc_y * (1) / hc_z),
                    ((_x - xc0) / hc_x * (_y - yc0) / hc_y * (1) / hc_z),
                    ((xc1 - _x) / hc_x * (_y - yc0) / hc_y * (1) / hc_z),
                ]
            )
            return Nc, dNcdx, dNcdy, dNcdz, nodes

        viqLoopMass = jax.vmap(iqLoopMass)
        return viqLoopMass(jnp.arange(8))

    vstepComputeCoarseFineTerm = jax.vmap(stepComputeCoarseFineTerm)
    Nc, dNcdx, dNcdy, dNcdz, _nodes = vstepComputeCoarseFineTerm(jnp.arange(nef))
    _nodes = _nodes[:, 0, :]
    indices = jnp.concatenate(
        [_nodes.reshape(-1, 1), jnp.arange(_nodes.size).reshape(-1, 1)], axis=1
    )
    test = jax.experimental.sparse.BCOO(
        [jnp.ones(_nodes.size), indices], shape=(nnc, _nodes.size)
    )
    return [Nc, [dNcdx, dNcdy, dNcdz], test]


def compute3DN(q, x, y, z, h):
    N = jnp.array(
        [
            ((x[1] - q[0]) / h[0] * (y[1] - q[1]) / h[1] * (z[1] - q[2]) / h[2]),
            ((q[0] - x[0]) / h[0] * (y[1] - q[1]) / h[1] * (z[1] - q[2]) / h[2]),
            ((q[0] - x[0]) / h[0] * (q[1] - y[0]) / h[1] * (z[1] - q[2]) / h[2]),
            ((x[1] - q[0]) / h[0] * (q[1] - y[0]) / h[1] * (z[1] - q[2]) / h[2]),
            ((x[1] - q[0]) / h[0] * (y[1] - q[1]) / h[1] * (q[2] - z[0]) / h[2]),
            ((q[0] - x[0]) / h[0] * (y[1] - q[1]) / h[1] * (q[2] - z[0]) / h[2]),
            ((q[0] - x[0]) / h[0] * (q[1] - y[0]) / h[1] * (q[2] - z[0]) / h[2]),
            ((x[1] - q[0]) / h[0] * (q[1] - y[0]) / h[1] * (q[2] - z[0]) / h[2]),
        ]
    )
    return N


@jax.jit
def computeCoarseTprimeMassTerm_jax(
    Levels, Tprimef, Tprimem, L3rhocp, L2rhocp, dt, Shapes, Vcu, Vmu
):

    Tprimef_new = Tprimef - Levels[3]["Tprime0"]
    Tprimem_new = Tprimem - Levels[2]["Tprime0"]

    # Level 3
    nef_x = Levels[3]["connect"][0].shape[0]
    nef_y = Levels[3]["connect"][1].shape[0]
    nef_z = Levels[3]["connect"][2].shape[0]
    nef = nef_x * nef_y * nef_z
    nnf_x = Levels[3]["node_coords"][0].shape[0]
    nnf_y = Levels[3]["node_coords"][1].shape[0]

    # Level 3 Get shape functions and weights
    coordsf = getSampleCoords(Levels[3])
    Nf, _, wqf = computeQuad3dFemShapeFunctions_jax(coordsf)

    # Level 3
    _, _, _, idxf = convert2XYZ(jnp.arange(nef), nef_x, nef_y, nnf_x, nnf_y)
    _Tprimef = multiply(
        Nf @ Tprimef_new[idxf], jnp.matmul(Nf, L3rhocp[idxf]).mean(axis=0)
    )
    _data1 = multiply(
        multiply(-Shapes[1][0], _Tprimef.T[:, :, newax]),
        (1 / dt) * wqf[newax, newax, :],
    ).sum(axis=2)
    _data2 = multiply(
        multiply(-Shapes[2][0], _Tprimef.T[:, :, newax]),
        (1 / dt) * wqf[newax, newax, :],
    ).sum(axis=2)

    # Level 2
    nem_x = Levels[2]["connect"][0].shape[0]
    nem_y = Levels[2]["connect"][1].shape[0]
    nem_z = Levels[2]["connect"][2].shape[0]
    nem = nem_x * nem_y * nem_z
    nnm_x = Levels[2]["node_coords"][0].shape[0]
    nnm_y = Levels[2]["node_coords"][1].shape[0]

    # Level 2 Get shape functions and weights
    coordsm = getSampleCoords(Levels[2])
    Nm, _, wqm = computeQuad3dFemShapeFunctions_jax(coordsm)

    # Level 2
    _, _, _, idxm = convert2XYZ(jnp.arange(nem), nem_x, nem_y, nnm_x, nnm_y)
    _Tprimem = multiply(
        Nm @ Tprimem_new[idxm], jnp.matmul(Nm, L2rhocp[idxm]).mean(axis=0)
    )
    _data3 = multiply(
        multiply(-Shapes[0][0], _Tprimem.T[:, :, newax]),
        (1 / dt) * wqm[newax, newax, :],
    ).sum(axis=2)

    Vcu += Shapes[1][2] @ _data1.reshape(-1) + Shapes[0][2] @ _data3.reshape(-1)
    Vmu += Shapes[2][2] @ _data2.reshape(-1)

    return Vcu, Vmu


@jax.jit
def computeCoarseTprimeTerm_jax(Levels, L3k, L2k, Shapes):
    # Level 3
    nef_x = Levels[3]["connect"][0].shape[0]
    nef_y = Levels[3]["connect"][1].shape[0]
    nef_z = Levels[3]["connect"][2].shape[0]
    nef = nef_x * nef_y * nef_z
    nnf_x = Levels[3]["node_coords"][0].shape[0]
    nnf_y = Levels[3]["node_coords"][1].shape[0]

    # Level 3 Get shape functions and weights
    coordsf = getSampleCoords(Levels[3])
    Nf, dNdxf, wqf = computeQuad3dFemShapeFunctions_jax(coordsf)

    # Level 3
    # idxf: (8, nef), indexing in Levels[3]["Tprime0"] for later shape function use
    _, _, _, idxf = convert2XYZ(jnp.arange(nef), nef_x, nef_y, nnf_x, nnf_y)
    _Tprimef = Levels[3]["Tprime0"][idxf]
    L3kMean = jnp.matmul(Nf, L3k[idxf]).mean(axis=0)
    dTprimefdx = multiply(L3kMean, (dNdxf[:, :, 0] @ _Tprimef))
    dTprimefdy = multiply(L3kMean, (dNdxf[:, :, 1] @ _Tprimef))
    dTprimefdz = multiply(L3kMean, (dNdxf[:, :, 2] @ _Tprimef))

    _data1 = multiply(
        multiply(-Shapes[1][1][0], dTprimefdx.T[:, :, newax]),
        wqf[newax, newax, :],
    ).sum(axis=2)
    _data1 += multiply(
        multiply(-Shapes[1][1][1], dTprimefdy.T[:, :, newax]),
        wqf[newax, newax, :],
    ).sum(axis=2)
    _data1 += multiply(
        multiply(-Shapes[1][1][2], dTprimefdz.T[:, :, newax]),
        wqf[newax, newax, :],
    ).sum(axis=2)

    _data2 = multiply(
        multiply(-Shapes[2][1][0], dTprimefdx.T[:, :, newax]),
        wqf[newax, newax, :],
    ).sum(axis=2)
    _data2 += multiply(
        multiply(-Shapes[2][1][1], dTprimefdy.T[:, :, newax]),
        wqf[newax, newax, :],
    ).sum(axis=2)
    _data2 += multiply(
        multiply(-Shapes[2][1][2], dTprimefdz.T[:, :, newax]),
        wqf[newax, newax, :],
    ).sum(axis=2)

    # Level 2
    nem_x = Levels[2]["connect"][0].shape[0]
    nem_y = Levels[2]["connect"][1].shape[0]
    nem_z = Levels[2]["connect"][2].shape[0]
    nem = nem_x * nem_y * nem_z
    nnm_x = Levels[2]["node_coords"][0].shape[0]
    nnm_y = Levels[2]["node_coords"][1].shape[0]

    # Level 2 Get shape functions and weights
    coordsm = getSampleCoords(Levels[2])
    Nm, dNdxm, wqm = computeQuad3dFemShapeFunctions_jax(coordsm)

    # Level 2
    _, _, _, idxm = convert2XYZ(jnp.arange(nem), nem_x, nem_y, nnm_x, nnm_y)
    _Tprimem = Levels[2]["Tprime0"][idxm]
    L2kMean = jnp.matmul(Nm, L2k[idxm]).mean(axis=0)
    dTprimemdx = multiply(L2kMean, (dNdxm[:, :, 0] @ _Tprimem))
    dTprimemdy = multiply(L2kMean, (dNdxm[:, :, 1] @ _Tprimem))
    dTprimemdz = multiply(L2kMean, (dNdxm[:, :, 2] @ _Tprimem))

    _data3 = multiply(
        multiply(-Shapes[0][1][0], dTprimemdx.T[:, :, newax]),
        wqm[newax, newax, :],
    ).sum(axis=2)
    _data3 += multiply(
        multiply(-Shapes[0][1][1], dTprimemdy.T[:, :, newax]),
        wqm[newax, newax, :],
    ).sum(axis=2)
    _data3 += multiply(
        multiply(-Shapes[0][1][2], dTprimemdz.T[:, :, newax]),
        wqm[newax, newax, :],
    ).sum(axis=2)

    Vcu = Shapes[1][2] @ _data1.reshape(-1) + Shapes[0][2] @ _data3.reshape(-1)
    Vmu = Shapes[2][2] @ _data2.reshape(-1)

    return Vcu, Vmu


@jax.jit
def assignBCs(RHS, Levels):
    _RHS = RHS
    _RHS = _RHS.at[Levels[1]["BC"][2]].set(Levels[1]["conditions"]["y"][0])
    _RHS = _RHS.at[Levels[1]["BC"][3]].set(Levels[1]["conditions"]["y"][1])
    _RHS = _RHS.at[Levels[1]["BC"][0]].set(Levels[1]["conditions"]["x"][0])
    _RHS = _RHS.at[Levels[1]["BC"][1]].set(Levels[1]["conditions"]["x"][1])
    _RHS = _RHS.at[Levels[1]["BC"][4]].set(Levels[1]["conditions"]["z"][0])
    # _RHS = _RHS.at[Levels[1]["BC"][5]].set(Levels[1]["conditions"]["z"][1])
    return _RHS


@jax.jit
def assignBCsFine(RHS, TfAll, BC):
    _RHS = RHS
    _RHS = _RHS.at[BC[2]].set(TfAll[BC[2]])
    _RHS = _RHS.at[BC[3]].set(TfAll[BC[3]])
    _RHS = _RHS.at[BC[0]].set(TfAll[BC[0]])
    _RHS = _RHS.at[BC[1]].set(TfAll[BC[1]])
    _RHS = _RHS.at[BC[4]].set(TfAll[BC[4]])
    # _RHS = _RHS.at[BC[5]].set(TfAll[BC[5]])
    return _RHS


@partial(jax.jit, static_argnames=["nn"])
def bincount(N, D, nn):
    return jnp.bincount(N, D, length=nn)


@jax.jit
def getOverlapRegion(node_coords, nx, ny):
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
    # jnp.clip is equivalent to jnp.minimum(a_max, np.maximum(a, a_min))
    vtot = [
        jnp.clip(vtot[0], Level["bounds"]["ix"][0], Level["bounds"]["ix"][1]),
        jnp.clip(vtot[1], Level["bounds"]["iy"][0], Level["bounds"]["iy"][1]),
        jnp.clip(vtot[2], Level["bounds"]["iz"][0], Level["bounds"]["iz"][1]),
    ]
    return vtot


@jax.jit
def move_fine_mesh(node_coords, element_size, v):
    vx_ = (v[0] / element_size[0] + 1e-2).astype(int)
    vy_ = (v[1] / element_size[1] + 1e-2).astype(int)
    vz_ = (v[2] / element_size[2] + 1e-2).astype(int)
    xnf_x = node_coords[0] + element_size[0] * vx_
    xnf_y = node_coords[1] + element_size[1] * vy_
    xnf_z = node_coords[2] + element_size[2] * vz_
    return [xnf_x, xnf_y, xnf_z], [vx_, vy_, vz_]


@jax.jit
def update_overlap_nodes_coords(Level, vcon, element_size, ele_ratio):
    Level["overlapNodes"] = [
        Level["orig_overlap_nodes"][0]
        + ele_ratio[0] * (vcon[0] / element_size[0] + 1e-2).astype(int),
        Level["orig_overlap_nodes"][1]
        + ele_ratio[1] * (vcon[1] / element_size[1] + 1e-2).astype(int),
        Level["orig_overlap_nodes"][2]
        + ele_ratio[2] * (vcon[2] / element_size[2] + 1e-2).astype(int),
    ]
    Level["overlapCoords"] = [
        Level["orig_overlap_coors"][0]
        + element_size[0] * (vcon[0] / element_size[0] + 1e-2).astype(int),
        Level["orig_overlap_coors"][1]
        + element_size[1] * (vcon[1] / element_size[1] + 1e-2).astype(int),
        Level["orig_overlap_coors"][2]
        + element_size[2] * (vcon[2] / element_size[2] + 1e-2).astype(int),
    ]
    return Level


@jax.jit
def update_overlap_nodes_coords_L1L2(Level, vcon, element_size, powder_layer):
    Level["overlapNodes"] = [
        Level["orig_overlap_nodes"][0] + (vcon[0] / element_size[0] + 1e-2).astype(int),
        Level["orig_overlap_nodes"][1] + (vcon[1] / element_size[1] + 1e-2).astype(int),
        Level["orig_overlap_nodes"][2]
        + ((vcon[2] + powder_layer) / element_size[2] + 1e-2).astype(int),
    ]
    Level["overlapCoords"] = [
        Level["orig_overlap_coors"][0]
        + element_size[0] * (vcon[0] / element_size[0] + 1e-2).astype(int),
        Level["orig_overlap_coors"][1]
        + element_size[1] * (vcon[1] / element_size[1] + 1e-2).astype(int),
        Level["orig_overlap_coors"][2] + element_size[2] * (vcon[2] / element_size[2]),
    ]
    return Level


@jax.jit
def update_overlap_nodes_coords_L2(Level, vcon, element_size, ele_ratio):
    Level["overlapNodes_L2"] = [
        Level["orig_overlap_nodes_L2"][0]
        + ele_ratio[0] * (vcon[0] / element_size[0] + 1e-2).astype(int),
        Level["orig_overlap_nodes_L2"][1]
        + ele_ratio[1] * (vcon[1] / element_size[1] + 1e-2).astype(int),
        Level["orig_overlap_nodes_L2"][2]
        + ele_ratio[2] * (vcon[2] / element_size[2] + 1e-2).astype(int),
    ]
    Level["overlapCoords_L2"] = [
        Level["orig_overlap_coors_L2"][0]
        + element_size[0] * (vcon[0] / element_size[0] + 1e-2).astype(int),
        Level["orig_overlap_coors_L2"][1]
        + element_size[1] * (vcon[1] / element_size[1] + 1e-2).astype(int),
        Level["orig_overlap_coors_L2"][2]
        + element_size[2] * (vcon[2] / element_size[2] + 1e-2).astype(int),
    ]
    return Level


@partial(jax.jit, static_argnames=["_idx", "_val"])
def substitute_Tbar(Tbar, _idx, _val):
    return Tbar.at[_idx:].set(_val)


@jax.jit
def substitute_Tbar2(Tbar, _idx, _val):
    return Tbar.at[_idx].set(_val)


def find_max_const(CoarseLevel, FinerLevel):
    # Used to find the maximum number of elements the finer level domain can move
    iE = CoarseLevel.bounds.x[1] - FinerLevel.bounds.x[1]  # Number of elements to east
    iN = CoarseLevel.bounds.y[1] - FinerLevel.bounds.y[1]  # Number of elements to north
    iT = CoarseLevel.bounds.z[1] - FinerLevel.bounds.z[1]  # Number of elements to top
    iW = CoarseLevel.bounds.x[0] - FinerLevel.bounds.x[0]  # Number of elements to west
    iS = CoarseLevel.bounds.y[0] - FinerLevel.bounds.y[0]  # Number of elements to south
    iB = CoarseLevel.bounds.z[0] - FinerLevel.bounds.z[0]  # Number of elements to bot
    return [iW, iE], [iS, iN], [iB, iT]


def calc_length_h(A):
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
    """saveResult saves a vtk for the current level's temperature field
    :param Level: structure of Level
    :param save_str: prefix of save string
    :param record_lab: recording label that is incremented after each save
    :param save_path: folder where file is saved
    :param zoffset: used for rendering purposes, no effect on model itself
    """
    # List coordinates in each direction for structured save
    vtkcx = np.array(Level["node_coords"][0])
    vtkcy = np.array(Level["node_coords"][1])
    vtkcz = np.array(Level["node_coords"][2] - zoffset)
    # Reshape the temperature field for correct rendering later
    vtkT = np.array(
        Level["T0"].reshape(Level["nodes"][2], Level["nodes"][1], Level["nodes"][0])
    ).transpose((2, 1, 0))
    vtkS = np.array(
        Level["S1"].reshape(Level["nodes"][2], Level["nodes"][1], Level["nodes"][0])
    ).transpose((2, 1, 0))
    # Save a vtr
    pointData = {"Temperature (K)": vtkT, "State (Powder/Solid)": vtkS}
    vtkSave = f"{save_path}{save_str}{record_lab:08}"
    gridToVTK(vtkSave, vtkcx, vtkcy, vtkcz, pointData=pointData)


def saveFinalResult(Level, save_str, save_path, zoffset):
    """saveResult saves a vtk for the current level's temperature field
    :param Level: structure of Level
    :param save_str: prefix of save string
    :param record_lab: recording label that is incremented after each save
    :param save_path: folder where file is saved
    :param zoffset: used for rendering purposes, no effect on model itself
    """
    # List coordinates in each direction for structured save
    vtkcx = np.array(Level["node_coords"][0])
    vtkcy = np.array(Level["node_coords"][1])
    vtkcz = np.array(Level["node_coords"][2] - zoffset)
    # Reshape the temperature field for correct rendering later
    vtkT = np.array(
        Level["T0"].reshape(Level["nodes"][2], Level["nodes"][1], Level["nodes"][0])
    ).transpose((2, 1, 0))
    vtkS = np.array(
        Level["S1"].reshape(Level["nodes"][2], Level["nodes"][1], Level["nodes"][0])
    ).transpose((2, 1, 0))
    # Save a vtr
    pointData = {"Temperature (K)": vtkT, "State (Powder/Solid)": vtkS}
    vtkSave = f"{save_path}{save_str}Final"
    gridToVTK(vtkSave, vtkcx, vtkcy, vtkcz, pointData=pointData)


def saveState(Level, save_str, record_lab, save_path, zoffset):
    """saveResult saves a vtk for the current level's temperature field
    :param Level: structure of Level
    :param save_str: prefix of save string
    :param record_lab: recording label that is incremented after each save
    :param save_path: folder where file is saved
    :param zoffset: used for rendering purposes, no effect on model itself
    """
    # List coordinates in each direction for structured save
    vtkcx = np.array(Level["node_coords"][0])
    vtkcy = np.array(Level["node_coords"][1])
    vtkcz = np.array(Level["node_coords"][2] - zoffset)
    # Reshape the temperature field for correct rendering later
    vtkS = np.array(
        Level["S1"].reshape(Level["nodes"][2], Level["nodes"][1], Level["nodes"][0])
    ).transpose((2, 1, 0))
    # Save a vtr
    pointData = {"State (Powder/Solid)": vtkS}
    vtkSave = f"{save_path}{save_str}{record_lab:08}"
    gridToVTK(vtkSave, vtkcx, vtkcy, vtkcz, pointData=pointData)


@jax.jit
def getNewTprime(Fine, FineT0, CoarseT, Coarse, C2F):
    # Fine: Fine Level information
    # FineT0: Fine Level Temperature
    # CoarseT: Coarse Level Temperature
    # Coarse: Coarse Level information
    # C2F: Coarse-to-Fine matrix/node list

    # Find new T
    _val = interpolatePoints(Fine, FineT0, Fine["overlapCoords"])
    _idx = getOverlapRegion(
        Fine["overlapNodes"], Coarse["nodes"][0], Coarse["nodes"][1]
    )
    # Directly substitute into T0 to save deepcopy
    CoarseT = substitute_Tbar2(CoarseT, _idx, _val)
    # Subtract coarser from finer
    Tprime = FineT0 - interpolate_w_matrix(C2F, CoarseT)

    return Tprime, CoarseT


@jax.jit
def getBothNewTprimes(Levels, FineT, MesoT, M2F, CoarseT, C2M):
    lTprime, mT0 = getNewTprime(Levels[3], FineT, MesoT, Levels[2], M2F)
    mTprime, uT0 = getNewTprime(Levels[2], mT0, CoarseT, Levels[1], C2M)
    return lTprime, mTprime, mT0, uT0


@partial(jax.jit, static_argnames=["ne_nn", "tmp_ne_nn"])
def computeSolutions(
    Levels, ne_nn, tmp_ne_nn, LF, L1V, LInterp, Lk, Lrhocp, L2V, dt, properties
):
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

    # Compute source term for medium scale problem using fine mesh
    TfAll = interpolate_w_matrix(LInterp[0], FinalL1)
    # Avoids assembling LHS matrix
    L2T = solveMatrixFreeFE(
        Levels[2], ne_nn[3], ne_nn[0], Lk[2], Lrhocp[2], dt, Levels[2]["T0"], LF[2], L2V
    )
    FinalL2 = assignBCsFine(L2T, TfAll, Levels[2]["BC"])

    # Use Levels[2].T to get Dirichlet BCs for fine-scale solution
    TfAll = interpolate_w_matrix(LInterp[1], FinalL2)
    FinalL3 = solveMatrixFreeFE(
        Levels[3], ne_nn[4], ne_nn[1], Lk[3], Lrhocp[3], dt, Levels[3]["T0"], LF[3], 0
    )
    FinalL3 = assignBCsFine(FinalL3, TfAll, Levels[3]["BC"])
    return FinalL1, FinalL2, FinalL3


@partial(jax.jit, static_argnames=["ne", "nn"])
def computeConvRadBC(Level, LevelT0, ne, nn, properties, F):
    h_conv = properties["h_conv"] / 1e6  # convert to W/mm^2K
    sigma_sb = properties["sigma_sb"] / 1e6  # convert to W/mm^2/K^4

    x, y = Level["node_coords"][0], Level["node_coords"][1]
    cx, cy = Level["connect"][0], Level["connect"][1]

    ne_x = jnp.size(cx, 0)
    ne_y = jnp.size(cy, 0)
    top_ne = ne - ne_x * ne_y

    nn_x = ne_x + 1
    nn_y = ne_y + 1

    coords_x = x[cx[0, :]].reshape(-1, 1)
    coords_y = y[cy[0, :]].reshape(-1, 1)

    coords = jnp.concatenate([coords_x, coords_y], axis=1)
    N, dNdx, wq = computeQuad2dFemShapeFunctions_jax(coords)

    def calcCR(i):
        ix, iy, iz, idx = convert2XYZ(i, ne_x, ne_y, nn_x, nn_y)
        iT = jnp.matmul(N, LevelT0[idx[4:]])
        iT = jnp.minimum(iT, properties["T_boiling"] + 1000)
        S = (
            1e-6
            * properties["evc"]
            * 54.0e3
            * jnp.exp(-50000 * ((1 / iT) - (1 / properties["T_boiling"])))
            * jnp.sqrt(0.001 / iT)
            * (properties["Lev"] + properties["cp_fluid"] * (iT - properties["T_amb"]))
        )
        iT = (
            h_conv * (properties["T_amb"] - iT)
            + sigma_sb * properties["vareps"] * (properties["T_amb"] ** 4 - iT**4)
            - S
        )
        return jnp.matmul(N.T, multiply(iT, wq.reshape(-1))), idx[4:]

    vcalcCR = jax.vmap(calcCR)
    aT, aidx = vcalcCR(jnp.arange(top_ne, ne))
    NeumannBC = jnp.bincount(aidx.reshape(-1), aT.reshape(-1), length=nn)

    # Returns properties["k"] grad(T) integral, which is Neumann BC (expect ambient < body)
    return F + NeumannBC


@partial(jax.jit, static_argnames=["ne_nn", "tmp_ne_nn", "substrate"])
def stepGOMELT(
    Levels, ne_nn, tmp_ne_nn, Shapes, LInterp, v, properties, dt, laserP, substrate
):
    """stepGOMELT computes a Levels[1] explicit timestep starting by
    computing the source terms for all three levels, computing the convection/radiation
    terms for all three levels, computing the previous step volumetric correction terms,
    computing solutions for all three levels (predictor step), calculating new Tprime terms
    and their volumetric correction terms, computing solutions for all three levels (corrector step),
    and finally updating the Tprime terms for the next explicit time step
    :param Levels: contains all multilevel information
    :param ne_nn: total number of elements/nodes (active and deactive)
    :param tmp_ne_nn: number of elements/nodes active on Levels[1] mesh (based on layer)
    :param Shapes[1]: shape functions/derivates/nodes from Levels[3] to Levels[1]
    :param LInterp[1]: interpolation matrix/nodes from Levels[2] to Levels[3]
    :param v: current position of laser (from reading file)
    :param properties: all material properties
    :param dt: timestep
    :param laserP: laser Power
    :param substrate: nodes that define substrate
    :return Levels: update temperature and subgrid terms
    """
    preS2 = Levels[3]["S2"]
    (Levels, Lk, Lrhocp) = updateStateProperties(Levels, properties, substrate)

    L1, L2, L3 = [Levels[_] for _ in range(1, 4)]

    Fc, Fm, Ff = computeSources(L3, v, Shapes, ne_nn, properties, laserP)
    Fc = computeConvRadBC(L1, L1["T0"], tmp_ne_nn[0], ne_nn[2], properties, Fc)
    Fm = computeConvRadBC(L2, L2["T0"], ne_nn[0], ne_nn[3], properties, Fm)
    Ff = computeConvRadBC(L3, L3["T0"], ne_nn[1], ne_nn[4], properties, Ff)
    F = [0, Fc, Fm, Ff]

    Vcu, Vmu = computeCoarseTprimeTerm_jax(Levels, Lk[3], Lk[2], Shapes)
    L1T, L2T, L3T = computeSolutions(
        Levels, ne_nn, tmp_ne_nn, F, Vcu, LInterp, Lk, Lrhocp, Vmu, dt, properties
    )
    L1T = jnp.maximum(properties["T_amb"], L1T)  # TFSP
    L2T = jnp.maximum(properties["T_amb"], L2T)  # TFSP
    L3T = jnp.maximum(properties["T_amb"], L3T)  # TFSP
    L3Tp, L2Tp, L2T, L1T = getBothNewTprimes(
        Levels, L3T, L2T, LInterp[1], L1T, LInterp[0]
    )
    Vcu, Vmu = computeCoarseTprimeMassTerm_jax(
        Levels, L3Tp, L2Tp, Lrhocp[3], Lrhocp[2], dt, Shapes, Vcu, Vmu
    )
    L1T, L2T, Levels[3]["T0"] = computeSolutions(
        Levels, ne_nn, tmp_ne_nn, F, Vcu, LInterp, Lk, Lrhocp, Vmu, dt, properties
    )
    L1T = jnp.maximum(properties["T_amb"], L1T)  # TFSP
    L2T = jnp.maximum(properties["T_amb"], L2T)  # TFSP
    Levels[3]["T0"] = jnp.maximum(properties["T_amb"], Levels[3]["T0"])  # TFSP
    Levels[3]["Tprime0"], Levels[2]["Tprime0"], Levels[2]["T0"], Levels[1]["T0"] = (
        getBothNewTprimes(Levels, Levels[3]["T0"], L2T, LInterp[1], L1T, LInterp[0])
    )

    Levels[0]["S1"] = Levels[0]["S1"].at[Levels[0]["idx"]].set(Levels[3]["S1"])
    Levels[0]["S2"] = Levels[0]["S2"].at[:].set(False)
    Levels[0]["S2"] = Levels[0]["S2"].at[Levels[0]["idx"]].set(Levels[3]["S2"])
    _resetmask = ((1 - 2 * preS2) * Levels[3]["S2"]) == 1
    return Levels, _resetmask


@jax.jit
def moveEverything(v, vstart, Levels, move_v, LInterp, L1L2Eratio, L2L3Eratio, height):
    ###### moveL3Mesh #####
    vtot = v - vstart
    _L3v_tot_con = jit_constrain_v(vtot, Levels[3])

    ### Correction step (fine) ###
    Levels3newcoords, _a = move_fine_mesh(
        Levels[3]["init_node_coors"], Levels[2]["h"], _L3v_tot_con
    )

    # Element ratios are [1,1,1] since they are updated later
    Levels[3] = update_overlap_nodes_coords(
        Levels[3], _L3v_tot_con, Levels[2]["h"], [1, 1, 1]
    )

    Levels[3]["T0"] = interpolatePoints(Levels[1], Levels[1]["T0"], Levels3newcoords)
    _3T0 = interpolatePoints(Levels[2], Levels[2]["Tprime0"], Levels3newcoords)
    Levels[3]["Tprime0"] = interpolatePoints(
        Levels[3], Levels[3]["Tprime0"], Levels3newcoords
    )
    Levels[3]["T0"] += _3T0 + Levels[3]["Tprime0"]
    Levels[3]["node_coords"] = Levels3newcoords
    ###### moveL3Mesh #####

    ###### prepL2Move #####
    _v_tot_con = jit_constrain_v(vtot, Levels[2])
    _tmp = Levels[1]["h"][:2]
    _tmp.append(jnp.array(height))

    Levels2newcoords, move_v = move_fine_mesh(
        Levels[2]["init_node_coors"], _tmp, _v_tot_con
    )

    move_v = [move_v[i] * L1L2Eratio[i] for i in range(3)]

    Levels[2] = update_overlap_nodes_coords_L1L2(
        Levels[2], _v_tot_con, Levels[1]["h"], height
    )
    ###### prepL2Move #####

    ###### updateL2objects #####
    Levels[2]["T0"] = interpolatePoints(Levels[1], Levels[1]["T0"], Levels2newcoords)
    Levels[2]["Tprime0"] = interpolatePoints(
        Levels[2], Levels[2]["Tprime0"], Levels2newcoords
    )
    Levels[2]["T0"] += Levels[2]["Tprime0"]
    Levels[2]["node_coords"] = Levels2newcoords

    # If mesh moves, recalculate shape functions
    L2L1Shape = computeCoarseFineShapeFunctions(Levels[1], Levels[2])
    LInterp[0] = interpolatePointsMatrix(Levels[1], Levels2newcoords)
    ###### updateL2objects #####

    ###### updateL3AfterMove #####
    Levels[3]["overlapNodes"] = [
        Levels[3]["overlapNodes"][i] - move_v[i] for i in range(3)
    ]
    ###### updateL3AfterMove #####
    # Update Level 0
    Levels[0] = update_overlap_nodes_coords(
        Levels[0], _L3v_tot_con, Levels[2]["h"], L2L3Eratio
    )
    Levels[0] = update_overlap_nodes_coords_L2(
        Levels[0],
        _v_tot_con,
        [Levels[1]["h"][0], Levels[1]["h"][1], Levels[2]["h"][2]],
        [L1L2Eratio[0] * L2L3Eratio[0], L1L2Eratio[1] * L2L3Eratio[1], L2L3Eratio[2]],
    )
    # Track movement of Level 0 in z-direction with Level 3, but nothing else.
    Levels[0]["overlapNodes"][2] = (
        Levels[0]["overlapNodes"][2] - move_v[2] * L2L3Eratio[2]
    )
    Levels[0]["overlapNodes_L2"][2] = (
        Levels[0]["overlapNodes_L2"][2] - move_v[2] * L2L3Eratio[2]
    )
    Levels[0]["idx"] = getOverlapRegion(
        Levels[0]["overlapNodes"], Levels[0]["nodes"][0], Levels[0]["nodes"][1]
    )
    Levels[0]["idx_L2"] = getOverlapRegion(
        Levels[0]["overlapNodes_L2"], Levels[0]["nodes"][0], Levels[0]["nodes"][1]
    )
    Levels[2]["S1"] = Levels[2]["S1"].at[:].set(Levels[0]["S1"][Levels[0]["idx_L2"]])
    Levels[3]["S1"] = Levels[3]["S1"].at[:].set(Levels[0]["S1"][Levels[0]["idx"]])
    Levels[3]["S2"] = Levels[3]["S2"].at[:].set(Levels[0]["S2"][Levels[0]["idx"]])

    # If mesh moves, recalculate shape functions
    L3L1Shape = computeCoarseFineShapeFunctions(Levels[1], Levels[3])

    # Move Levels[3] with respect to Levels[2]
    L3L2Shape = computeCoarseFineShapeFunctions(Levels[2], Levels[3])

    LInterp[1] = interpolatePointsMatrix(Levels[2], Levels3newcoords)
    ###### updateL3AfterMove #####

    Shapes = [L2L1Shape, L3L1Shape, L3L2Shape]
    return (Levels, Shapes, LInterp, move_v)


@partial(jax.jit, static_argnames=["substrate"])
def updateStateProperties(Levels, properties, substrate):
    Levels[3]["S1"], Levels[3]["S2"], L3k, L3rhocp = computeStateProperties(
        Levels[3]["T0"], Levels[3]["S1"], properties, substrate[3]
    )
    Levels[2]["S1"], L2S2, L2k, L2rhocp = computeStateProperties(
        Levels[2]["T0"], Levels[2]["S1"], properties, substrate[2]
    )
    _val = interpolatePoints(Levels[2], Levels[2]["S1"], Levels[2]["overlapCoords"])
    _idx = getOverlapRegion(
        Levels[2]["overlapNodes"], Levels[1]["nodes"][0], Levels[1]["nodes"][1]
    )
    # Directly substitute into T0 to save deepcopy
    Levels[1]["S1"] = Levels[1]["S1"].at[_idx].set(_val)
    Levels[1]["S1"] = Levels[1]["S1"].at[: substrate[1]].set(1)
    _, _, L1k, L1rhocp = computeStateProperties(
        Levels[1]["T0"], Levels[1]["S1"], properties, substrate[1]
    )

    # 0 added to beginning of list so index matches levels
    return (Levels, [0, L1k, L2k, L3k], [0, L1rhocp, L2rhocp, L3rhocp])


# S1 = 0 -> powder, S1 = 1 -> bulk
# S2 = 0 -> not fluid, S2 = 1 -> fluid
def computeStateProperties(T, S1, properties, Level_nodes_substrate):
    # This is using temperature and phase dependent material properties
    S2 = T >= properties["T_liquidus"]
    S3 = (T > properties["T_solidus"]) & (T < properties["T_liquidus"])
    S1 = 1.0 * ((S1 > 0.499) | S2)
    S1 = S1.at[:Level_nodes_substrate].set(1)
    k = (
        (1 - S1) * (1 - S2) * properties["k_powder"]
        + S1
        * (1 - S2)
        * (properties["k_bulk_coeff_a1"] * T + properties["k_bulk_coeff_a0"])
        + S2 * properties["k_fluid_coeff_a0"]
    ) / 1000
    rhocp = properties["rho"] * (
        (1 - S2)
        * (1 - S3)
        * (properties["cp_solid_coeff_a1"] * T + properties["cp_solid_coeff_a0"])
        + S3 * properties["cp_mushy"]
        + S2 * properties["cp_fluid"]
    )

    return S1, S2, k, rhocp


@partial(jax.jit, static_argnames=["ne_nn", "tmp_ne_nn", "substrate"])
def stepGOMELTDwellTime(Levels, tmp_ne_nn, ne_nn, properties, dt, substrate):
    L1, L1ne = Levels[1], tmp_ne_nn[0]
    Fc = computeConvRadBC(L1, L1["T0"], L1ne, ne_nn[2], properties, F=0)
    _, _, L1k, L1rhocp = computeStateProperties(
        L1["T0"], L1["S1"], properties, substrate[1]
    )

    L1T = solveMatrixFreeFE(L1, ne_nn[2], L1ne, L1k, L1rhocp, dt, L1["T0"], Fc, 0)
    L1T = substitute_Tbar(L1T, tmp_ne_nn[1], properties["T_amb"])
    Levels[1]["T0"] = assignBCs(L1T, Levels)
    return Levels


################## Nested Subcycle Items #####################
@partial(jax.jit, static_argnames=["ne_nn"])
def computeLevelSource(Levels, ne_nn, laser_position, LevelShape, properties, laserP):
    # Get shape functions and weights
    coords = getSampleCoords(Levels[3])
    Nf, _, wqf = computeQuad3dFemShapeFunctions_jax(coords)

    def stepLaserPosition(ilaser):
        def stepcomputeCoarseSource(ieltf):
            # Get the nodal indices for that element
            ix, iy, iz, idx = convert2XYZ(
                ieltf,
                Levels[3]["elements"][0],
                Levels[3]["elements"][1],
                Levels[3]["nodes"][0],
                Levels[3]["nodes"][1],
            )
            # Get nodal coordinates for the fine element
            x, y, z = getQuadratureCoords(Levels[3], ix, iy, iz, Nf)
            w = wqf
            # Compute the source at the quadrature point location
            Q = computeSourceFunction_jax(
                x, y, z, laser_position[ilaser], properties, laserP[ilaser]
            )
            return Q * w

        vstepcomputeCoarseSource = jax.vmap(stepcomputeCoarseSource)

        _data = vstepcomputeCoarseSource(jnp.arange(ne_nn[1]))
        _data1tmp = multiply(LevelShape[0], _data).sum(axis=1)
        return _data1tmp

    vstepLaserPosition = jax.vmap(stepLaserPosition)
    # Find the average heat source value
    lshape = laser_position.shape[0]
    _data1 = vstepLaserPosition(jnp.arange(lshape)).sum(axis=0) / lshape

    return LevelShape[2] @ _data1.reshape(-1)


def computeAllSources(Levels, ne_nn, laser_position, Shapes, properties, laserP):
    # Get shape functions and weights
    coords = getSampleCoords(Levels[3])
    Nf, _, wqf = computeQuad3dFemShapeFunctions_jax(coords)

    def stepLaserPosition(ilaser):
        def stepcomputeCoarseSource(ieltf):
            # Get the nodal indices for that element
            ix, iy, iz, idx = convert2XYZ(
                ieltf,
                Levels[3]["elements"][0],
                Levels[3]["elements"][1],
                Levels[3]["nodes"][0],
                Levels[3]["nodes"][1],
            )
            # Get nodal coordinates for the fine element
            x, y, z = getQuadratureCoords(Levels[3], ix, iy, iz, Nf)
            # Compute the source at the quadrature point location
            Q = computeSourceFunction_jax(
                x, y, z, laser_position[ilaser], properties, laserP[ilaser]
            )
            return Q * wqf, Nf @ Q * wqf, idx

        vstepcomputeCoarseSource = jax.vmap(stepcomputeCoarseSource)

        _data, _data3, nodes3 = vstepcomputeCoarseSource(jnp.arange(ne_nn[1]))
        _data1tmp = multiply(Shapes[1][0], _data).sum(axis=1)
        _data2tmp = multiply(Shapes[2][0], _data).sum(axis=1)
        Fc = Shapes[1][2] @ _data1tmp.reshape(-1)
        Fm = Shapes[2][2] @ _data2tmp.reshape(-1)
        Ff = bincount(nodes3.reshape(-1), _data3.reshape(-1), ne_nn[4])
        return Fc, Fm, Ff

    vstepLaserPosition = jax.vmap(stepLaserPosition)
    # Find the average heat source value
    lshape = laser_position.shape[0]
    Fc, Fm, Ff = vstepLaserPosition(jnp.arange(lshape))
    return Fc, Fm, Ff


@partial(jax.jit, static_argnames=["ne_nn"])
def computeL1TprimeTerms_Part1(Levels, ne_nn, L3k, Shapes, L2k):
    # Level 3 calculations
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
    dL3Tp0dX = [multiply(L3kMean, (dNdxf[:, :, i] @ _L3Tp0)) for i in range(3)]

    _wqf = wqf[newax, newax, :]
    _dL3L1dX = Shapes[1][1]
    _1 = multiply(multiply(-_dL3L1dX[0], dL3Tp0dX[0].T[:, :, newax]), _wqf).sum(axis=2)
    _1 += multiply(multiply(-_dL3L1dX[1], dL3Tp0dX[1].T[:, :, newax]), _wqf).sum(axis=2)
    _1 += multiply(multiply(-_dL3L1dX[2], dL3Tp0dX[2].T[:, :, newax]), _wqf).sum(axis=2)

    # Level 2 calculations
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
    dL2Tp0dX = [multiply(L2kMean, (dNdxm[:, :, i] @ _L2Tp0)) for i in range(3)]

    _wqm = wqm[newax, newax, :]
    _dL2L1dX = Shapes[0][1]
    _2 = multiply(multiply(-_dL2L1dX[0], dL2Tp0dX[0].T[:, :, newax]), _wqm).sum(axis=2)
    _2 += multiply(multiply(-_dL2L1dX[1], dL2Tp0dX[1].T[:, :, newax]), _wqm).sum(axis=2)
    _2 += multiply(multiply(-_dL2L1dX[2], dL2Tp0dX[2].T[:, :, newax]), _wqm).sum(axis=2)

    return Shapes[1][2] @ _1.reshape(-1) + Shapes[0][2] @ _2.reshape(-1)


@partial(jax.jit, static_argnames=["ne_nn", "tmp_ne_nn"])
def computeL1Temperature(
    Levels, ne_nn, tmp_ne_nn, L1F, L1V, L1k, L1rhocp, dt, properties
):
    L1T = solveMatrixFreeFE(
        Levels[1], ne_nn[2], tmp_ne_nn[0], L1k, L1rhocp, dt, Levels[1]["T0"], L1F, L1V
    )
    L1T = substitute_Tbar(L1T, tmp_ne_nn[1], properties["T_amb"])
    FinalL1 = assignBCs(L1T, Levels)

    return FinalL1


@partial(jax.jit, static_argnames=["ne_nn"])
def computeL2TprimeTerms_Part1(Levels, ne_nn, L3Tprime0, L3k, Shapes):
    # Level 3 calculations
    coordsf = getSampleCoords(Levels[3])
    Nf, dNdxf, wqf = computeQuad3dFemShapeFunctions_jax(coordsf)

    _, _, _, idxf = convert2XYZ(
        jnp.arange(ne_nn[1]),
        Levels[3]["elements"][0],
        Levels[3]["elements"][1],
        Levels[3]["nodes"][0],
        Levels[3]["nodes"][1],
    )
    _L3Tp0 = L3Tprime0[idxf]
    L3kMean = jnp.matmul(Nf, L3k[idxf]).mean(axis=0)
    dL3Tp0dx = multiply(L3kMean, (dNdxf[:, :, 0] @ _L3Tp0))
    dL3Tp0dy = multiply(L3kMean, (dNdxf[:, :, 1] @ _L3Tp0))
    dL3Tp0dz = multiply(L3kMean, (dNdxf[:, :, 2] @ _L3Tp0))

    _1 = multiply(
        multiply(-Shapes[2][1][0], dL3Tp0dx.T[:, :, newax]),
        wqf[newax, newax, :],
    ).sum(axis=2)
    _1 += multiply(
        multiply(-Shapes[2][1][1], dL3Tp0dy.T[:, :, newax]),
        wqf[newax, newax, :],
    ).sum(axis=2)
    _1 += multiply(
        multiply(-Shapes[2][1][2], dL3Tp0dz.T[:, :, newax]),
        wqf[newax, newax, :],
    ).sum(axis=2)

    return Shapes[2][2] @ _1.reshape(-1)


@partial(jax.jit, static_argnames=["ne_nn"])
def computeL2Temperature(
    L1T, L1L2Interp, Levels, ne_nn, L2T0, L2F, L2V, L2k, L2rhocp, dt
):
    # Compute source term for medium scale problem using fine mesh
    TfAll = interpolate_w_matrix(L1L2Interp, L1T)
    # Avoids assembling LHS matrix
    L2T = solveMatrixFreeFE(
        Levels[2], ne_nn[3], ne_nn[0], L2k, L2rhocp, dt, L2T0, L2F, L2V
    )
    FinalL2 = assignBCsFine(L2T, TfAll, Levels[2]["BC"])
    return FinalL2


@partial(jax.jit, static_argnames=["ne_nn"])
def computeSourcesL3(Level, v, ne_nn, properties, laserP):
    """computeSources computes the integrated source term for all three levels using
    the mesh from Level 3.
    :param Levels: multilevel information
    :param v: current position of laser (from reading file)
    :param ne_nn: total number of elements/nodes (active and deactive)
    :param properties: material properties
    :param laserP: laser Power per position
    :return Ff
    :return Ff: integrated source term for Levels[3]
    """
    # Get shape functions and weights
    coords = getSampleCoords(Level)
    Nf, _, wqf = computeQuad3dFemShapeFunctions_jax(coords)

    def stepcomputeCoarseSource(ieltf):
        # Get the nodal indices for that element
        ix, iy, iz, idx = convert2XYZ(
            ieltf,
            Level["elements"][0],
            Level["elements"][1],
            Level["nodes"][0],
            Level["nodes"][1],
        )
        # Get nodal coordinates for the fine element
        x, y, z = getQuadratureCoords(Level, ix, iy, iz, Nf)
        w = wqf
        # Compute the source at the quadrature point location
        Q = computeSourceFunction_jax(x, y, z, v, properties, laserP)
        return Nf @ Q * w, idx

    vstepcomputeCoarseSource = jax.vmap(stepcomputeCoarseSource)
    _data3, nodes3 = vstepcomputeCoarseSource(jnp.arange(ne_nn[1]))
    Ff = bincount(nodes3.reshape(-1), _data3.reshape(-1), ne_nn[4])
    return Ff


@partial(jax.jit, static_argnames=["ne_nn"])
def computeSolutions_L3(
    FinalL2, L2L3Interp, Levels, ne_nn, L3T0, L3F, L3k, L3rhocp, dt
):
    # Use Levels[2].T to get Dirichlet BCs for fine-scale solution
    TfAll = interpolate_w_matrix(L2L3Interp, FinalL2)
    FinalL3 = solveMatrixFreeFE(
        Levels[3], ne_nn[4], ne_nn[1], L3k, L3rhocp, dt, L3T0, L3F, 0
    )
    FinalL3 = assignBCsFine(FinalL3, TfAll, Levels[3]["BC"])
    return FinalL3


@partial(jax.jit, static_argnames=["ne_nn"])
def computeL1TprimeTerms_Part2(
    Levels, ne_nn, L3Tp, L2Tp, L3rhocp, L2rhocp, dt, Shapes, Vcu
):

    L3Tp_new = L3Tp - Levels[3]["Tprime0"]
    L2Tp_new = L2Tp - Levels[2]["Tprime0"]

    # Level 3 Get shape functions and weights
    coordsf = getSampleCoords(Levels[3])
    Nf, _, wqf = computeQuad3dFemShapeFunctions_jax(coordsf)

    # Level 3
    _, _, _, idxf = convert2XYZ(
        jnp.arange(ne_nn[1]),
        Levels[3]["elements"][0],
        Levels[3]["elements"][1],
        Levels[3]["nodes"][0],
        Levels[3]["nodes"][1],
    )
    _L3Tp = multiply(Nf @ L3Tp_new[idxf], jnp.matmul(Nf, L3rhocp[idxf]).mean(axis=0))
    _1 = multiply(
        multiply(-Shapes[1][0], _L3Tp.T[:, :, newax]),
        (1 / dt) * wqf[newax, newax, :],
    ).sum(axis=2)

    # Level 2 Get shape functions and weights
    coordsm = getSampleCoords(Levels[2])
    Nm, _, wqm = computeQuad3dFemShapeFunctions_jax(coordsm)

    # Level 2
    _, _, _, idxm = convert2XYZ(
        jnp.arange(ne_nn[0]),
        Levels[2]["elements"][0],
        Levels[2]["elements"][1],
        Levels[2]["nodes"][0],
        Levels[2]["nodes"][1],
    )
    _L2Tp = multiply(Nm @ L2Tp_new[idxm], jnp.matmul(Nm, L2rhocp[idxm]).mean(axis=0))
    _2 = multiply(
        multiply(-Shapes[0][0], _L2Tp.T[:, :, newax]),
        (1 / dt) * wqm[newax, newax, :],
    ).sum(axis=2)

    Vcu += Shapes[1][2] @ _1.reshape(-1) + Shapes[0][2] @ _2.reshape(-1)

    return Vcu


def getSampleCoords(Level):
    x = Level["node_coords"][0][Level["connect"][0][0, :]].reshape(-1, 1)
    y = Level["node_coords"][1][Level["connect"][1][0, :]].reshape(-1, 1)
    z = Level["node_coords"][2][Level["connect"][2][0, :]].reshape(-1, 1)
    return jnp.concatenate([x, y, z], axis=1)


def getQuadratureCoords(Level, ix, iy, iz, Nf):
    # Get nodal coordinates for the fine element
    coords_x = Level["node_coords"][0][Level["connect"][0][ix, :]].reshape(-1, 1)
    coords_y = Level["node_coords"][1][Level["connect"][1][iy, :]].reshape(-1, 1)
    coords_z = Level["node_coords"][2][Level["connect"][2][iz, :]].reshape(-1, 1)
    # Do all of the quadrature points simultaneously
    return Nf @ coords_x, Nf @ coords_y, Nf @ coords_z


@partial(jax.jit, static_argnames=["ne_nn"])
def computeL2TprimeTerms_Part2(Levels, ne_nn, L3Tp, L3Tp0, L3rhocp, dt, Shapes, L2V):

    L3Tp_new = L3Tp - L3Tp0

    # Level 3 Get shape functions and weights
    coordsf = getSampleCoords(Levels[3])
    Nf, _, wqf = computeQuad3dFemShapeFunctions_jax(coordsf)

    # Level 3
    _, _, _, idxf = convert2XYZ(
        jnp.arange(ne_nn[1]),
        Levels[3]["elements"][0],
        Levels[3]["elements"][1],
        Levels[3]["nodes"][0],
        Levels[3]["nodes"][1],
    )
    _L3Tp = multiply(Nf @ L3Tp_new[idxf], jnp.matmul(Nf, L3rhocp[idxf]).mean(axis=0))

    _data2 = multiply(
        multiply(-Shapes[2][0], _L3Tp.T[:, :, newax]),
        (1 / dt) * wqf[newax, newax, :],
    ).sum(axis=2)

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
    # Compute Material Properties (Level 3, Level 2, Level 1) #
    _, _, L3k_L1, L3rhocp_L1 = computeStateProperties(
        Levels[3]["T0"], Levels[3]["S1"], properties, substrate[3]
    )
    _, _, L2k_L1, L2rhocp_L1 = computeStateProperties(
        Levels[2]["T0"], Levels[2]["S1"], properties, substrate[2]
    )
    _val = interpolatePoints(Levels[2], Levels[2]["S1"], Levels[2]["overlapCoords"])
    _idx = getOverlapRegion(
        Levels[2]["overlapNodes"], Levels[1]["nodes"][0], Levels[1]["nodes"][1]
    )
    Levels[1]["S1"] = Levels[1]["S1"].at[_idx].set(_val)
    Levels[1]["S1"] = Levels[1]["S1"].at[: substrate[1]].set(1)
    _, _, L1k, L1rhocp = computeStateProperties(
        Levels[1]["T0"], Levels[1]["S1"], properties, substrate[1]
    )

    # Compute Level 1 Source #
    L1F = computeLevelSource(
        Levels, ne_nn, laser_position, Shapes[1], properties, laserP
    )
    L1F = computeConvRadBC(
        Levels[1],
        Levels[1]["T0"],
        tmp_ne_nn[0],
        ne_nn[2],
        properties,
        L1F,
    )

    # Compute Level 1 Tprime #
    L1V = computeL1TprimeTerms_Part1(Levels, ne_nn, L3k_L1, Shapes, L2k_L1)

    # Solve Level 1 Temp #
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

    ## Subcycle Level 2 ##
    def subcycleL2_Part1(_L2carry, _L2sub):
        alpha_L2 = (_L2sub + 1) / subcycle[3]
        beta_L2 = 1 - alpha_L2
        Lidx = _L2sub * subcycle[1] + jnp.arange(subcycle[1])
        ## Compute Material Properties (Level 3, Level 2) ##
        _, _, L3k_L2, _ = computeStateProperties(
            _L2carry[2], _L2carry[4], properties, substrate[3]
        )
        L2S1, _, L2k, L2rhocp = computeStateProperties(
            _L2carry[0], _L2carry[1], properties, substrate[2]
        )
        ## Compute Level 2 Source ##
        L2F = computeLevelSource(
            Levels, ne_nn, laser_position[Lidx, :], Shapes[2], properties, laserP[Lidx]
        )
        L2F = computeConvRadBC(
            Levels[2],
            _L2carry[0],
            ne_nn[0],
            ne_nn[3],
            properties,
            L2F,
        )
        ## Compute Level 2 Tprime ##
        L2V = computeL2TprimeTerms_Part1(Levels, ne_nn, _L2carry[3], L3k_L2, Shapes)
        ## Solve Level 2 Temperature ##
        _BC = alpha_L2 * L1T + beta_L2 * Levels[1]["T0"]
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

        ### Subcycle Level 3 ###
        def subcycleL3_Part1(_L3carry, _L3sub):
            ## Compute Material Properties (Level 3, Level 2) ##
            L3S1, _, L3k, L3rhocp = computeStateProperties(
                _L3carry[0], _L3carry[1], properties, substrate[3]
            )
            ### Compute Level 3 Source ###
            LLidx = _L3sub + _L2sub * subcycle[1]
            L3F = computeSourcesL3(
                Levels[3], laser_position[LLidx, :], ne_nn, properties, laserP[LLidx]
            )
            L3F = computeConvRadBC(
                Levels[3], _L3carry[0], ne_nn[1], ne_nn[4], properties, L3F
            )
            ### Solve Level 3 Temperature ###
            alpha_L3 = (_L3sub + 1) / subcycle[4]
            beta_L3 = 1 - alpha_L3
            _BC = alpha_L3 * L2T + beta_L3 * _L2carry[0]
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
            ### End Subcycle Level 3 ###
            return ([L3T, L3S1], [L3T, L3S1])

        [L3T, L3S1], _ = jax.lax.scan(
            subcycleL3_Part1,
            [_L2carry[2], _L2carry[4]],
            jnp.arange(subcycle[1]),
        )

        ## Compute Updated Level 3 Tprime ##
        L3Tp, L2T = getNewTprime(Levels[3], L3T, L2T, Levels[2], LInterp[1])

        return ([L2T, L2S1, L3T, L3Tp, L3S1], [L2T, L2S1, L3T, L3Tp, L3S1])

    # Predictor state values are carried over, but they
    # are not used in output since they would incorrectly
    # affect corrector.
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

    # Calculate Updated Level 2 Tprime #
    L2Tp, L1T = getNewTprime(Levels[2], L2T, L1T, Levels[1], LInterp[0])

    ########### End of Predictor #############

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

    # Solve Updated Level 1 Temp #
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

    ## Subcycle Level 2 ##
    def subcycleL2_Part2(_L2carry, _L2sub):
        alpha_L2 = (_L2sub + 1) / subcycle[3]
        beta_L2 = 1 - alpha_L2
        Lidx = _L2sub * subcycle[1] + jnp.arange(subcycle[1])
        ## Compute Material Properties (Level 3, Level 2) ##
        _, _, L3k_L2, L3rhocp_L2 = computeStateProperties(
            _L2carry[2], _L2carry[4], properties, substrate[3]
        )
        L2S1, _, L2k, L2rhocp = computeStateProperties(
            _L2carry[0], _L2carry[1], properties, substrate[2]
        )
        ## Compute Level 2 Source ##
        L2F = computeLevelSource(
            Levels, ne_nn, laser_position[Lidx, :], Shapes[2], properties, laserP[Lidx]
        )
        L2F = computeConvRadBC(
            Levels[2],
            _L2carry[0],
            ne_nn[0],
            ne_nn[3],
            properties,
            L2F,
        )
        ## Compute Level 2 Tprime ##
        L2V = computeL2TprimeTerms_Part1(Levels, ne_nn, _L2carry[3], L3k_L2, Shapes)
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

        ## Solve Updated Level 2 Temperature ##
        _BC = alpha_L2 * L1T + beta_L2 * Levels[1]["T0"]
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

        ### Subcycle Level 3 ###
        def subcycleL3_Part2(_L3carry, _L3sub):
            ## Compute Material Properties (Level 3, Level 2) ##
            L3S1, L3S2, L3k, L3rhocp = computeStateProperties(
                _L3carry[0], _L3carry[1], properties, substrate[3]
            )
            ### Compute Level 3 Source ###
            LLidx = _L3sub + _L2sub * subcycle[1]
            L3F = computeSourcesL3(
                Levels[3], laser_position[LLidx, :], ne_nn, properties, laserP[LLidx]
            )
            L3F = computeConvRadBC(
                Levels[3], _L3carry[0], ne_nn[1], ne_nn[4], properties, L3F
            )
            ### Solve Level 3 Temperature ###
            alpha_L3 = (_L3sub + 1) / subcycle[4]
            beta_L3 = 1 - alpha_L3
            _BC = alpha_L3 * L2T + beta_L3 * _L2carry[0]
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
            # If changed from solid back to liquid, clear
            _resetmask = ((1 - 2 * _L3carry[2]) * L3S2) == 1
            _resetaccumtime = _L3carry[4] * _resetmask
            _max_check = jnp.maximum(_resetaccumtime, _L3carry[3])
            max_accum_L3 = _max_check
            accum_L3 = _L3carry[4] + laser_position[LLidx, 5] * L3S2 - _resetaccumtime
            ### End Subcycle Level 3 ###
            return (
                [L3T, L3S1, L3S2, max_accum_L3, accum_L3],
                [L3T, L3S1, L3S2, max_accum_L3, accum_L3],
            )

        [L3T, L3S1, L3S2, max_accum_L3, accum_L3], _ = jax.lax.scan(
            subcycleL3_Part2,
            [_L2carry[2], _L2carry[4], _L2carry[5], _L2carry[6], _L2carry[7]],
            jnp.arange(subcycle[1]),
        )

        ## Compute Updated Level 2 Tprime Term ##
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

    # Calculate Level 2 Tprime #
    Levels[2]["Tprime0"], Levels[1]["T0"] = getNewTprime(
        Levels[2], Levels[2]["T0"], L1T, Levels[1], LInterp[0]
    )

    Levels[0]["S1"] = Levels[0]["S1"].at[Levels[0]["idx"]].set(Levels[3]["S1"])
    Levels[0]["S2"] = Levels[0]["S2"].at[:].set(False)
    Levels[0]["S2"] = Levels[0]["S2"].at[Levels[0]["idx"]].set(Levels[3]["S2"])
    return Levels, L2all, L3all, L3pall, max_accum_L3, accum_L3


def printLevelMaxMin(Ls, Lnames):
    print("Temps:", end=" ")
    flag = False
    for i in range(1, len(Ls)):
        Lmax, Lmin = Ls[i]["T0"].max(), Ls[i]["T0"].min()
        # Check if Lmax or Lmin are NaN, 0, or negative
        if math.isnan(Lmax) or Lmax <= 0 or Lmax > 1e5:
            print(
                f"\nTerminating program: Lmax for {Lnames[i-1]} is NaN, 0, or negative."
            )
            flag = True

        if math.isnan(Lmin) or Lmin <= 0 or Lmin > 1e5:
            print(
                f"\nTerminating program: Lmin for {Lnames[i-1]} is NaN, 0, or negative."
            )
            flag = True

        print(f"{Lnames[i-1]}: [{Lmin:.2f}, {Lmax:.2f}]", end=" ")
    if flag:
        # print(Ls)
        sys.exit(1)
    print("")


def saveResults(Levels, Nonmesh, savenum):
    if Nonmesh["output_files"] == 1:
        if savenum == 1:
            saveResult(Levels[1], "Level1_", savenum, Nonmesh["save_path"], 2e-3)
            # saveState(Levels[0], "Level0_", savenum, Nonmesh["save_path"], 0)
        elif (np.mod(savenum, Nonmesh["Level1_record_step"]) == 1) or (
            Nonmesh["Level1_record_step"] == 1
        ):
            saveResult(Levels[1], "Level1_", savenum, Nonmesh["save_path"], 2e-3)
            # saveState(Levels[0], "Level0_", savenum, Nonmesh["save_path"], 0)
        saveResult(Levels[2], "Level2_", savenum, Nonmesh["save_path"], 1e-3)
        saveResult(Levels[3], "Level3_", savenum, Nonmesh["save_path"], 0)
        print(f"Saved Levels_{savenum:08}")


def saveResultsFinal(Levels, Nonmesh):
    if Nonmesh["output_files"] == 1:
        saveFinalResult(Levels[1], "Level1_", Nonmesh["save_path"], 2e-3)
        saveFinalResult(Levels[2], "Level2_", Nonmesh["save_path"], 1e-3)
        saveFinalResult(Levels[3], "Level3_", Nonmesh["save_path"], 0)
        print(f"Saved Final Results")


######################################################################################


def melting_temp(temps, delt_T, T_melt, accum_time, idx):
    T_above_threshold = np.array(temps > T_melt)
    accum_time = accum_time.at[idx].add(T_above_threshold * delt_T)
    return accum_time

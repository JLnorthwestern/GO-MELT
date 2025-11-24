from functools import partial

import jax
import jax.numpy as jnp
from .data_structures import obj
from go_melt.utils.gaussian_quadrature_functions import (
    computeQuad2dFemShapeFunctions_jax,
)
from go_melt.utils.helper_functions import convert2XYZ


def getBCindices(level: obj) -> list[jnp.ndarray]:
    """
    Compute boundary condition indices for a structured 3D mesh.
    """
    nodes_x, nodes_y, nodes_z = level.nodes[0], level.nodes[1], level.nodes[2]
    total_nodes = level.nn

    # Bottom face (z = 0)
    bidx = jnp.arange(0, nodes_x * nodes_y)

    # Top face (z = nodes_z - 1)
    tidx = jnp.arange(nodes_x * nodes_y * (nodes_z - 1), total_nodes)

    # West face (level = 0)
    widx = jnp.arange(0, total_nodes, nodes_x)

    # East face (level = nodes_x - 1)
    eidx = jnp.arange(nodes_x - 1, total_nodes, nodes_x)

    # South face (y = 0)
    sidx = (
        jnp.arange(0, nodes_x)[:, None]
        + (nodes_x * nodes_y * jnp.arange(0, nodes_z))[None, :]
    )
    sidx = sidx.reshape(-1)

    # North face (y = nodes_y - 1)
    nidx = (
        jnp.arange(nodes_x * (nodes_y - 1), nodes_x * nodes_y)[:, None]
        + (nodes_x * nodes_y * jnp.arange(0, nodes_z))[None, :]
    )
    nidx = nidx.reshape(-1)

    level.conditions.west.indices = widx
    level.conditions.east.indices = eidx
    level.conditions.south.indices = sidx
    level.conditions.north.indices = nidx
    level.conditions.bottom.indices = bidx
    level.conditions.top.indices = tidx

    return level


@jax.jit
def assignBCs(
    RHS: jnp.ndarray, global_indices: jnp.ndarray, value: float
) -> jnp.ndarray:
    """
    Apply Dirichlet boundary conditions to the right-hand side (RHS) vector.
    """
    RHS = RHS.at[global_indices].set(value)

    return RHS


@jax.jit
def assignBCsFine(RHS: jnp.ndarray, TfAll: jnp.ndarray, level: dict) -> jnp.ndarray:
    """
    Apply Dirichlet boundary conditions to the fine-level RHS vector.

    This function sets the RHS values at boundary nodes using the
    corresponding values from the full fine-scale solution `TfAll`.
    """
    _RHS = RHS
    _RHS = _RHS.at[level["conditions"]["west"]["indices"]].set(
        TfAll[level["conditions"]["west"]["indices"]]
    )
    _RHS = _RHS.at[level["conditions"]["east"]["indices"]].set(
        TfAll[level["conditions"]["east"]["indices"]]
    )
    _RHS = _RHS.at[level["conditions"]["south"]["indices"]].set(
        TfAll[level["conditions"]["south"]["indices"]]
    )
    _RHS = _RHS.at[level["conditions"]["north"]["indices"]].set(
        TfAll[level["conditions"]["north"]["indices"]]
    )
    _RHS = _RHS.at[level["conditions"]["bottom"]["indices"]].set(
        TfAll[level["conditions"]["bottom"]["indices"]]
    )
    return _RHS


@partial(jax.jit, static_argnames=["num_elems", "num_nodes", "bc_index"])
def computeConvRadBC(
    Level: dict,
    temperature: jnp.ndarray,
    num_elems: int,
    num_nodes: int,
    properties: dict,
    flux_vector: jnp.ndarray,
    local_indices: jnp.ndarray,
    bc_index: int,
) -> jnp.ndarray:
    """
    Compute Neumann boundary conditions on the surface due to:
      • Convection (Newton's law of cooling)
      • Radiation (Stefan-Boltzmann law)
      • Evaporation (empirical model)

    The resulting heat flux is integrated using 2D quadrature and added
    to the global force vector.
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
    nn_x, nn_y = ne_x + 1, ne_y + 1

    # Coordinates of a representative top-surface element
    coords = jnp.stack([x[cx[0, :]], y[cy[0, :]]], axis=1)

    # Precompute shape functions and quadrature weights
    coords = jnp.array([[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]]).T
    if bc_index in [0, 1]:
        coords = coords.at[:, 0].multiply(Level["h"][1])
        coords = coords.at[:, 1].multiply(Level["h"][2])
    elif bc_index in [2, 3]:
        coords = coords.at[:, 0].multiply(Level["h"][0])
        coords = coords.at[:, 1].multiply(Level["h"][2])
    else:
        coords = coords.at[:, 0].multiply(Level["h"][0])
        coords = coords.at[:, 1].multiply(Level["h"][1])
    N, _, wq = computeQuad2dFemShapeFunctions_jax(coords)

    open_surfaces = Level["S1faces"][bc_index]

    def calcCR(i):
        """
        Compute the integrated heat flux vector for a single top-surface element.
        """
        _, _, _, idx = convert2XYZ(i, ne_x, ne_y, nn_x, nn_y)

        # Interpolate nodal temperatures to quadrature points
        Tq = jnp.matmul(N, temperature[idx[local_indices]])
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
        return (
            jnp.matmul(N.T, q_flux * wq.reshape(-1)) * open_surfaces[i],
            idx[local_indices],
        )

    aT, aidx = jax.vmap(calcCR)(jnp.arange(num_elems))
    NeumannBC = jnp.bincount(aidx.reshape(-1), aT.reshape(-1), length=num_nodes)

    return flux_vector + NeumannBC


@partial(jax.jit, static_argnames=["num_elems", "num_nodes", "bc_index"])
def computeConvectionBC(
    Level: dict,
    temperature: jnp.ndarray,
    num_elems: int,
    num_nodes: int,
    properties: dict,
    flux_vector: jnp.ndarray,
    local_indices: jnp.ndarray,
    bc_index: int,
    RC: float,
) -> jnp.ndarray:
    pass


@partial(jax.jit, static_argnames=["number_elems", "elements"])
def get_surface_faces(level, solid_indicator, number_elems, elements):
    def calcVal(i):
        _, _, _, idx = convert2XYZ(
            i,
            elements[0],
            elements[1],
            level["nodes"][0],
            level["nodes"][1],
        )
        nodes_S1 = solid_indicator[idx]
        full_element = 1 * (nodes_S1.sum() > 7)

        # Hexahedral (cube) element with nodes 0–7
        #
        #        7 -------- 6
        #       /|         /|
        #      / |        / |
        #     4 -------- 5  |
        #     |  |       |  |
        #     |  3 ------|--2
        #     | /        | /
        #     |/         |/
        #     0 -------- 1
        #
        # Node coordinates (unit cube, right-handed axes):
        #   0: (0,0,0)  bottom-front-left
        #   1: (1,0,0)  bottom-front-right
        #   2: (1,1,0)  bottom-back-right
        #   3: (0,1,0)  bottom-back-left
        #   4: (0,0,1)  top-front-left
        #   5: (1,0,1)  top-front-right
        #   6: (1,1,1)  top-back-right
        #   7: (0,1,1)  top-back-left
        # Faces (node connectivity, normal out):
        #   Face 0 (West):   0 - 4 - 7 - 3
        #   Face 1 (East):   1 - 2 - 6 - 5
        #   Face 2 (South):  0 - 1 - 5 - 4
        #   Face 3 (North):  3 - 7 - 6 - 2
        #   Face 4 (Bottom): 0 - 3 - 2 - 1
        #   Face 5 (Top):    4 - 5 - 6 - 7
        return full_element

    vcalcVal = jax.vmap(calcVal)
    S1ele = vcalcVal(jnp.arange(number_elems))

    # To get free faces
    S1ele = S1ele.reshape(elements[2], elements[1], elements[0])
    S1ele_west = (
        S1ele.at[:, :, 1:]
        .set(jnp.maximum(S1ele[:, :, 1:] - S1ele[:, :, :-1], 0))
        .reshape(-1)
    )
    S1ele_east = (
        S1ele.at[:, :, :-1]
        .set(jnp.maximum(S1ele[:, :, :-1] - S1ele[:, :, 1:], 0))
        .reshape(-1)
    )
    S1ele_south = (
        S1ele.at[:, 1:, :]
        .set(jnp.maximum(S1ele[:, 1:, :] - S1ele[:, :-1, :], 0))
        .reshape(-1)
    )
    S1ele_north = (
        S1ele.at[:, :-1, :]
        .set(jnp.maximum(S1ele[:, :-1, :] - S1ele[:, 1:, :], 0))
        .reshape(-1)
    )
    S1ele_bottom = (
        S1ele.at[1:, :, :]
        .set(jnp.maximum(S1ele[1:, :, :] - S1ele[:-1, :, :], 0))
        .reshape(-1)
    )
    S1ele_top = (
        S1ele.at[:-1, :, :]
        .set(jnp.maximum(S1ele[:-1, :, :] - S1ele[1:, :, :], 0))
        .reshape(-1)
    )
    S1faces = jnp.stack(
        [S1ele_west, S1ele_east, S1ele_south, S1ele_north, S1ele_bottom, S1ele_top]
    )

    return S1ele.reshape(-1), S1faces

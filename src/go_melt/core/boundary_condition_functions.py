from functools import partial

import jax
import jax.numpy as jnp
from go_melt.utils.gaussian_quadrature_functions import (
    computeQuad2dFemShapeFunctions_jax,
)
from go_melt.utils.helper_functions import convert2XYZ


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

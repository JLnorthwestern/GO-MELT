from functools import partial
import jax
import jax.numpy as jnp
from go_melt.utils.interpolation_functions import (
    interpolate_w_matrix,
)
from go_melt.utils.gaussian_quadrature_functions import (
    computeQuad3dFemShapeFunctions_jax,
)
from .computeFunctions import *
from .boundary_condition_functions import computeConvRadBC, assignBCs, assignBCsFine
from go_melt.utils.helper_functions import convert2XYZ


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

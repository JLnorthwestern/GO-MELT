from functools import partial
import jax
import jax.numpy as jnp
from go_melt.utils.interpolation_functions import (
    interpolate_w_matrix,
)
from go_melt.utils.gaussian_quadrature_functions import (
    computeQuad3dFemShapeFunctions_jax,
)
from .mesh_functions import getSampleCoords
from .boundary_condition_functions import (
    computeConvRadBC,
    assignBCs,
    assignBCsFine,
    computeConvectionBC,
)
from go_melt.utils.helper_functions import convert2XYZ, static_set_in_array
from .phase_state_functions import computeStateProperties
from go_melt.io.save_results_functions import record_first_call


# @record_first_call("stepGOMELTDwellTime")
@partial(
    jax.jit, static_argnames=["ne_nn", "tmp_ne_nn", "substrate", "boundary_conditions"]
)
def stepGOMELTDwellTime(
    Levels: list[dict],
    tmp_ne_nn: tuple[int, int],
    ne_nn: tuple,
    properties: dict,
    dt: float,
    substrate: tuple,
    boundary_conditions: tuple,
) -> list[dict]:
    """
    Update Level 1 temperature field during laser dwell time in GO-MELT.

    Applies convection, radiation, and evaporation boundary conditions,
    computes temperature-dependent properties, and enforces ambient
    temperature above the powder line.
    """
    L1 = Levels[1]
    num_nodes_L1 = ne_nn[1][1]
    num_elements_L1 = tmp_ne_nn[0]
    inactive_start_idx = tmp_ne_nn[1]

    # Dwell time does not apply a heat source
    source_term = jnp.zeros_like(L1["T0"])

    # Compute boundary conditions (includes convection, radiation, evaporation)
    for bc_index in range(6):
        if (
            boundary_conditions[1][0][bc_index] == 1
            and boundary_conditions[1][1][bc_index] == 0
        ):
            # Type: Neumann; Function: Surface
            source_term = computeConvRadBC(
                L1,
                L1["T0"],
                num_elements_L1,
                num_nodes_L1,
                properties,
                source_term,
                jnp.array(boundary_conditions[1][4][bc_index]),
                bc_index,
            )
        elif (
            boundary_conditions[1][0][bc_index] == 1
            and boundary_conditions[1][1][bc_index] == 1
        ):
            # Type: Neumann; Function: Convection
            Fc = computeConvectionBC(
                L1,
                L1["T0"],
                ne_nn[0][1],
                ne_nn[1][1],
                properties,
                Fc,
                jnp.array(boundary_conditions[1][4][bc_index]),
                bc_index,
                value=boundary_conditions[1][2][bc_index],
            )
        elif (
            boundary_conditions[1][0][bc_index] == 1
            and boundary_conditions[1][1][bc_index] == 2
        ):
            # Type: Neumann; Function: Adiabatic
            pass

    # Compute temperature-dependent properties
    _, _, k, rhocp = computeStateProperties(
        L1["T0"], L1["S1"], properties, substrate[1]
    )

    # Solve temperature field using matrix-free FEM
    T_new = solveMatrixFreeFE(
        L1, num_nodes_L1, num_elements_L1, k, rhocp, dt, L1["T0"], source_term, 0
    )

    # Enforce ambient temperature in inactive region above surface
    T_new = static_set_in_array(T_new, inactive_start_idx, properties["T_amb"])

    # Apply boundary conditions and update Level 1 temperature
    for bc_index in range(6):
        if (
            boundary_conditions[1][0][bc_index] == 0
            and boundary_conditions[1][1][bc_index] == 0
        ):
            # Type: Dirichlet, Function: Constant
            T_new = assignBCs(
                T_new,
                global_indices=jnp.array(boundary_conditions[1][3][bc_index]),
                value=boundary_conditions[1][2][bc_index],
            )

    # Assign final temperature to Level 1's temperature field
    Levels[1]["T0"] = T_new

    return Levels


# @record_first_call("computeL1Temperature")
@partial(jax.jit, static_argnames=["ne_nn", "tmp_ne_nn", "boundary_conditions"])
def computeL1Temperature(
    Levels: list[dict],
    ne_nn: tuple,
    tmp_ne_nn: tuple,
    Level1_source: jnp.ndarray,
    Level1_Tprime_source: jnp.ndarray,
    L1k: jnp.ndarray,
    L1rhocp: jnp.ndarray,
    dt: float,
    properties: dict,
    boundary_conditions: tuple,
) -> jnp.ndarray:
    """
    Compute the updated temperature field for Level 1.

    This function performs the temperature update for Level 1 using a matrix-free
    finite element solver. It includes source terms, subgrid corrections, and
    enforces boundary and ambient conditions.
    """
    num_nodes_L1 = ne_nn[1][1]
    num_elements_L1 = tmp_ne_nn[0]
    inactive_start_idx = tmp_ne_nn[1]

    # Solve temperature field using matrix-free FEM
    L1T = solveMatrixFreeFE(
        Levels[1],
        num_nodes_L1,
        num_elements_L1,
        L1k,
        L1rhocp,
        dt,
        Levels[1]["T0"],
        Level1_source,
        Level1_Tprime_source,
    )

    # Apply ambient temperature to inactive region above surface
    L1T = static_set_in_array(L1T, inactive_start_idx, properties["T_amb"])

    # Enforce Dirichlet boundary conditions
    for bc_index in range(6):
        if (
            boundary_conditions[1][0][bc_index] == 0
            and boundary_conditions[1][1][bc_index] == 0
        ):
            # Type: Dirichlet, Function: Constant
            L1T = assignBCs(
                L1T,
                global_indices=jnp.array(boundary_conditions[1][3][bc_index]),
                value=boundary_conditions[1][2][bc_index],
            )
    return L1T


@partial(jax.jit, static_argnames=["ne_nn"])
def computeL2Temperature(
    new_Level1_temperature: jnp.ndarray,
    interpolate_Level1_to_Level2: list[jnp.ndarray],
    Levels: list[dict],
    ne_nn: tuple,
    Level2_temperature: jnp.ndarray,
    Level2_source: jnp.ndarray,
    Level2_Tprime_source: jnp.ndarray,
    L2k: jnp.ndarray,
    L2rhocp: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """
    Compute the updated temperature field for Level 2.

    This function solves the meso-scale (Level 2) temperature field using
    matrix-free FEM, incorporating subgrid corrections and boundary conditions
    interpolated from the coarse-scale (Level 1) solution.
    """
    num_elems_L2 = ne_nn[0][2]
    num_nodes_L2 = ne_nn[1][2]

    # Interpolate Level 1 temperature to Level 2 boundary
    TfAll = interpolate_w_matrix(interpolate_Level1_to_Level2, new_Level1_temperature)

    # Solve Level 2 temperature using matrix-free FEM
    L2T = solveMatrixFreeFE(
        Levels[2],
        num_nodes_L2,
        num_elems_L2,
        L2k,
        L2rhocp,
        dt,
        Level2_temperature,
        Level2_source,
        Level2_Tprime_source,
    )

    # Apply Dirichlet boundary conditions from interpolated Level 1 solution
    new_Level2_temperature = assignBCsFine(L2T, TfAll, Levels[2])

    return new_Level2_temperature


@partial(jax.jit, static_argnames=["ne_nn"])
def computeSolutions_L3(
    new_Level2_temperature: jnp.ndarray,
    interpolate_Level2_to_Level3: list[jnp.ndarray],
    Levels: list[dict],
    ne_nn: tuple,
    Level3_temperature: jnp.ndarray,
    Level3_source: jnp.ndarray,
    L3k: jnp.ndarray,
    L3rhocp: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """
    Compute the updated temperature field for Level 3.

    This function solves the fine-scale (Level 3) temperature field using
    matrix-free FEM, applying Dirichlet boundary conditions interpolated
    from the meso-scale (Level 2) solution.
    """
    num_elems_L3 = ne_nn[0][3]
    num_nodes_L3 = ne_nn[1][3]

    # Interpolate Level 2 temperature to Level 3 boundary
    TfAll = interpolate_w_matrix(interpolate_Level2_to_Level3, new_Level2_temperature)

    # Solve Level 3 temperature using matrix-free FEM
    new_Level3_temperature = solveMatrixFreeFE(
        Levels[3],
        num_nodes_L3,
        num_elems_L3,
        L3k,
        L3rhocp,
        dt,
        Level3_temperature,
        Level3_source,
        subgrid_source_term=0,
    )

    # Apply Dirichlet boundary conditions from interpolated Level 2 solution
    new_Level3_temperature = assignBCsFine(new_Level3_temperature, TfAll, Levels[3])

    return new_Level3_temperature


# @record_first_call("computeSolutions")
@partial(jax.jit, static_argnames=["ne_nn", "tmp_ne_nn", "boundary_conditions"])
def computeSolutions(
    Levels: list[dict],
    ne_nn: tuple,
    tmp_ne_nn: list,
    Levels_sources: list[jnp.ndarray],
    Level1_Tprime_source: jnp.ndarray,
    LInterp: list,
    Lk: list[jnp.ndarray],
    Lrhocp: list[jnp.ndarray],
    Level2_Tprime_source: jnp.ndarray,
    dt: float,
    properties: dict,
    boundary_conditions: tuple,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute temperature solutions across three levels using a matrix-free FEM solver.

    This function performs a multiscale temperature update by solving the heat equation
    at coarse, meso, and fine levels. It uses matrix-free finite element methods and
    interpolates between levels to apply boundary conditions and source terms.
    """
    num_elems_L2 = ne_nn[0][2]
    num_elems_L3 = ne_nn[0][3]
    num_nodes_L1 = ne_nn[1][1]
    num_nodes_L2 = ne_nn[1][2]
    num_nodes_L3 = ne_nn[1][3]
    num_elements_L1 = tmp_ne_nn[0]
    inactive_start_idx = tmp_ne_nn[1]
    interpolate_Level1_to_Level2 = LInterp[0]
    interpolate_Level2_to_Level3 = LInterp[1]

    # Solve coarse-level problem (Level 1)
    L1T = solveMatrixFreeFE(
        Levels[1],
        num_nodes_L1,
        num_elements_L1,
        Lk[1],
        Lrhocp[1],
        dt,
        Levels[1]["T0"],
        Levels_sources[1],
        Level1_Tprime_source,
    )
    L1T = static_set_in_array(L1T, inactive_start_idx, properties["T_amb"])

    for bc_index in range(6):
        if (
            boundary_conditions[1][0][bc_index] == 0
            and boundary_conditions[1][1][bc_index] == 0
        ):
            # Type: Dirichlet, Function: Constant
            L1T = assignBCs(
                L1T,
                global_indices=jnp.array(boundary_conditions[1][3][bc_index]),
                value=boundary_conditions[1][2][bc_index],
            )
    new_Level1_temperature = L1T

    # Interpolate Level 1 solution to Level 2 for source term
    TfAll = interpolate_w_matrix(interpolate_Level1_to_Level2, new_Level1_temperature)

    # Solve meso-level problem (Level 2)
    L2T = solveMatrixFreeFE(
        Levels[2],
        num_nodes_L2,
        num_elems_L2,
        Lk[2],
        Lrhocp[2],
        dt,
        Levels[2]["T0"],
        Levels_sources[2],
        Level2_Tprime_source,
    )
    new_Level2_temperature = assignBCsFine(L2T, TfAll, Levels[2])

    # Interpolate Level 2 solution to Level 3 for boundary conditions
    TfAll = interpolate_w_matrix(interpolate_Level2_to_Level3, new_Level2_temperature)

    # Solve fine-level problem (Level 3)
    new_Level3_temperature = solveMatrixFreeFE(
        Levels[3],
        num_nodes_L3,
        num_elems_L3,
        Lk[3],
        Lrhocp[3],
        dt,
        Levels[3]["T0"],
        Levels_sources[3],
        subgrid_source_term=0,
    )
    new_Level3_temperature = assignBCsFine(new_Level3_temperature, TfAll, Levels[3])

    return new_Level1_temperature, new_Level2_temperature, new_Level3_temperature


# @record_first_call("solveMatrixFreeFE")
@partial(jax.jit, static_argnames=["num_elems", "num_nodes"])
def solveMatrixFreeFE(
    Level: dict,
    num_nodes: int,
    num_elems: int,
    k: jnp.ndarray,
    rhocp: jnp.ndarray,
    dt: float,
    temperature: jnp.ndarray,
    source_term: jnp.ndarray,
    subgrid_source_term: jnp.ndarray,
) -> jnp.ndarray:
    """
    Perform an explicit matrix-free finite element thermal solve.

    This function computes the temperature update for a single timestep
    using a matrix-free approach, avoiding global matrix assembly.
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

        return jnp.matmul(LHSe, temperature[idx]), Me, idx

    # Vectorized element-wise computation
    vcalcVal = jax.vmap(calcVal)
    aT, aMe, aidx = vcalcVal(jnp.arange(num_elems))

    # Assemble global temperature and mass contributions
    newT = jnp.bincount(aidx.reshape(-1), aT.reshape(-1), length=num_nodes)
    newM = jnp.bincount(aidx.reshape(-1), aMe.reshape(-1), length=num_nodes)

    return (newT + source_term + subgrid_source_term) / newM

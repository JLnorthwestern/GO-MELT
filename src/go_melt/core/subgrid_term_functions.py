from jax.numpy import multiply
import jax.numpy as jnp
import jax
from functools import partial
from .mesh_functions import getSampleCoords
from go_melt.utils.gaussian_quadrature_functions import (
    computeQuad3dFemShapeFunctions_jax,
)
from go_melt.utils.helper_functions import (
    convert2XYZ,
    set_in_array,
    getOverlapRegion,
)
from go_melt.utils.interpolation_functions import (
    interpolate_w_matrix,
    interpolatePoints,
)


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
    CoarseT = set_in_array(CoarseT, _idx, _val)

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

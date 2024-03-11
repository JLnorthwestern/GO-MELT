from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.config import config
from jax.experimental import sparse
from pyevtk.hl import gridToVTK

# True is for convergence (double precision), False is single precision
config.update("jax_enable_x64", False)


def calcNumNodes(x):
    """calcNumNodes finds the number of nodes from number of elements.
    :param x: list of elements (3D)
    :return x: list of nodes (3D)
    """
    return [x[0] + 1, x[1] + 1, x[2] + 1]


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
    nx = jnp.linspace(x[0], x[1], x[2])
    ny = jnp.linspace(y[0], y[1], y[2])
    nz = jnp.linspace(z[0], z[1], z[2])

    # Connectivity in x, y, and z
    nconn_x = jnp.concatenate(
        [
            jnp.arange(0, x[2] - 1).reshape(-1, 1),
            jnp.arange(1, x[2]).reshape(-1, 1),
            jnp.arange(1, x[2]).reshape(-1, 1),
            jnp.arange(0, x[2] - 1).reshape(-1, 1),
            jnp.arange(0, x[2] - 1).reshape(-1, 1),
            jnp.arange(1, x[2]).reshape(-1, 1),
            jnp.arange(1, x[2]).reshape(-1, 1),
            jnp.arange(0, x[2] - 1).reshape(-1, 1),
        ],
        axis=1,
    )
    nconn_y = jnp.concatenate(
        [
            jnp.arange(0, y[2] - 1).reshape(-1, 1),
            jnp.arange(0, y[2] - 1).reshape(-1, 1),
            jnp.arange(1, y[2]).reshape(-1, 1),
            jnp.arange(1, y[2]).reshape(-1, 1),
            jnp.arange(0, y[2] - 1).reshape(-1, 1),
            jnp.arange(0, y[2] - 1).reshape(-1, 1),
            jnp.arange(1, y[2]).reshape(-1, 1),
            jnp.arange(1, y[2]).reshape(-1, 1),
        ],
        axis=1,
    )
    nconn_z = jnp.concatenate(
        [
            jnp.arange(0, z[2] - 1).reshape(-1, 1),
            jnp.arange(0, z[2] - 1).reshape(-1, 1),
            jnp.arange(0, z[2] - 1).reshape(-1, 1),
            jnp.arange(0, z[2] - 1).reshape(-1, 1),
            jnp.arange(1, z[2]).reshape(-1, 1),
            jnp.arange(1, z[2]).reshape(-1, 1),
            jnp.arange(1, z[2]).reshape(-1, 1),
            jnp.arange(1, z[2]).reshape(-1, 1),
        ],
        axis=1,
    )
    return [nx, ny, nz], [nconn_x, nconn_y, nconn_z]


@partial(jax.jit, static_argnames=["ne", "nn"])
def meshlessKM(node_coords, conn, nn, ne, k, rho, cp, dt, T, Fc, Corr):
    """meshlessKM computes the thermal solve for the explicit timestep.
    :param node_coords: nodal coordinates along x, y, or z
    :param conn[0], conn[1], conn[2]: connectivities along x, y, or z
    :param nn, ne: number of nodes, number of elements
    :param k, rho, cp: material properties for conductivity, density, heat capacity
    :param dt: timestep
    :param T: previous temperature for mesh
    :param Fc: integrated RHS values (including heat source)
    :param Corr: integrated correction terms
    :return (newT + Fc + Corr) / newM
    :return newT: Temperture for next timestep
    """

    ne_x = jnp.size(conn[0], 0)
    ne_y = jnp.size(conn[1], 0)

    nn_x = ne_x + 1
    nn_y = ne_y + 1

    nen = jnp.size(conn[0], 1)
    ndim = 3

    coords_x = node_coords[0][conn[0][0, :]].reshape(-1, 1)
    coords_y = node_coords[1][conn[1][0, :]].reshape(-1, 1)
    coords_z = node_coords[2][conn[2][0, :]].reshape(-1, 1)

    coords = jnp.concatenate([coords_x, coords_y, coords_z], axis=1)
    N, dNdx, wq = computeQuad3dFemShapeFunctions_jax(coords)

    def calcVal(i):
        ix, iy, iz, idx = convert2XYZ(i, ne_x, ne_y, nn_x, nn_y)
        iT = T[idx]
        kvec = k * jnp.ones([8, 1])
        mvec = ((rho * jnp.ones([8, 1])) * (cp * jnp.ones([8, 1]))) / dt

        Me = jnp.sum(jnp.matmul(N.T, jnp.multiply(N, wq * mvec)), 0)
        Ke = jnp.zeros((nen, nen))

        for idim in range(ndim):
            Ke += jnp.matmul(dNdx[:, :, idim].T, kvec * dNdx[:, :, idim]) * wq
        LHSe = jnp.diag(Me) - Ke

        return jnp.matmul(LHSe, iT), Me, idx

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
    N = (
        (1 / 8)
        * (1 + ksi_q[:, jnp.newaxis] @ ksi_i[jnp.newaxis, :])
        * (1 + eta_q[:, jnp.newaxis] @ eta_i[jnp.newaxis, :])
        * (1 + zeta_q[:, jnp.newaxis] @ zeta_i[jnp.newaxis, :])
    )
    dNdksi = (
        (1 / 8)
        * ksi_i[jnp.newaxis, :]
        * (1 + eta_q[:, jnp.newaxis] @ eta_i[jnp.newaxis, :])
        * (1 + zeta_q[:, jnp.newaxis] @ zeta_i[jnp.newaxis, :])
    )
    dNdeta = (
        (1 / 8)
        * eta_i[jnp.newaxis, :]
        * (1 + ksi_q[:, jnp.newaxis] @ ksi_i[jnp.newaxis, :])
        * (1 + zeta_q[:, jnp.newaxis] @ zeta_i[jnp.newaxis, :])
    )
    dNdzeta = (
        (1 / 8)
        * zeta_i[jnp.newaxis, :]
        * (1 + ksi_q[:, jnp.newaxis] @ ksi_i[jnp.newaxis, :])
        * (1 + eta_q[:, jnp.newaxis] @ eta_i[jnp.newaxis, :])
    )

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
                [
                    dNdksi[q, :, jnp.newaxis],
                    dNdeta[q, :, jnp.newaxis],
                    dNdzeta[q, :, jnp.newaxis],
                ],
                axis=1,
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

    tmp_wq = jnp.ones(ngp)

    N = (
        (1 / 4)
        * (1 + ksi_q[:, jnp.newaxis] @ ksi_i[jnp.newaxis, :])
        * (1 + eta_q[:, jnp.newaxis] @ eta_i[jnp.newaxis, :])
    )
    dNdksi = (
        (1 / 4)
        * ksi_i[jnp.newaxis, :]
        * (1 + eta_q[:, jnp.newaxis] @ eta_i[jnp.newaxis, :])
    )
    dNdeta = (
        (1 / 4)
        * eta_i[jnp.newaxis, :]
        * (1 + ksi_q[:, jnp.newaxis] @ ksi_i[jnp.newaxis, :])
    )

    dxdksi = jnp.matmul(dNdksi, coords[4:, 0])
    dydeta = jnp.matmul(dNdeta, coords[4:, 1])

    J = jnp.array([[dxdksi[0], 0], [0, dydeta[0]]])
    Jinv = jnp.array([[1 / dxdksi[0], 0], [0, 1 / dydeta[0]]])
    dNdx = jnp.zeros([ngp, ngp, ndim])
    wq = jnp.zeros([ngp, 1])
    for q in range(ngp):
        dNdx = dNdx.at[q, :, :].set(
            jnp.concatenate(
                [dNdksi[q, :, jnp.newaxis], dNdeta[q, :, jnp.newaxis]], axis=1
            )
            @ Jinv
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


@partial(jax.jit, static_argnames=["nn1", "nn2", "nn3"])
def computeSources(
    nodal_coords,
    conn,
    t,
    v,
    Nc1,
    Nc2,
    nodes1,
    nodes2,
    nn1,
    nn2,
    nn3,
    laserr,
    laserd,
    laserP,
    lasereta,
):
    """computeSources computes the integrated source term for all three levels using
    the mesh from Level 3.
    :param nodal_coords[0], nodal_coords[1], nodal_coords[2]: nodal coordinates
    :param conn[0], conn[1], conn[2]: connectivity matrix
    :param _t: current time
    :param v: current position of laser (from reading file)
    :param Nc1: Level3NcLevel1, shape functions between Level 3 and Level 1 (symmetric)
    :param nc2: Level3NcLevel2, shape functions between Level 3 and Level 2 (symmetric)
    :param nodes1: Level3nodesLevel1, node indices in Level3 for shape functions Level1
    :param nodes2: Level3nodesLevel2, node indices in Level3 for shape functions Level2
    :param nn1, nn2, nn3: total number of nodes (active and deactive)
    :param laserr: laser radius
    :param laserd: laser depth of penetration
    :param laserP: laser Power
    :param lasereta: laser absorptivity
    :return Fc, Fm, Ff
    :return Fc: integrated source term for Level1
    :return Fm: integrated source term for Level2
    :return Ff: integrated source term for Level3
    """
    nef_x = conn[0].shape[0]
    nef_y = conn[1].shape[0]
    nef_z = conn[2].shape[0]
    nef = nef_x * nef_y * nef_z
    nnf_x, nnf_y = nef_x + 1, nef_y + 1

    # Get shape functions and weights
    coords_x = nodal_coords[0][conn[0][0, :]].reshape(-1, 1)
    coords_y = nodal_coords[1][conn[1][0, :]].reshape(-1, 1)
    coords_z = nodal_coords[2][conn[2][0, :]].reshape(-1, 1)
    coords = jnp.concatenate([coords_x, coords_y, coords_z], axis=1)
    Nf, dNdxf, wqf = computeQuad3dFemShapeFunctions_jax(coords)

    def stepcomputeCoarseSource(ieltf):
        # Get the nodal indices for that element
        ix, iy, iz, idx = convert2XYZ(ieltf, nef_x, nef_y, nnf_x, nnf_y)
        # Get nodal coordinates for the fine element
        coords_x = nodal_coords[0][conn[0][ix, :]].reshape(-1, 1)
        coords_y = nodal_coords[1][conn[1][iy, :]].reshape(-1, 1)
        coords_z = nodal_coords[2][conn[2][iz, :]].reshape(-1, 1)
        # Do all of the quadrature points simultaneously
        x = Nf @ coords_x
        y = Nf @ coords_y
        z = Nf @ coords_z
        w = wqf
        # Compute the source at the quadrature point location
        Q = computeSourceFunction_jax(x, y, z, t, v, laserr, laserd, laserP, lasereta)
        return Q * w, Nf @ Q * w, idx

    vstepcomputeCoarseSource = jax.vmap(stepcomputeCoarseSource)

    # Returns data for Nc1/Nc2, data premultiplied by Nf, and nodes for Level 3
    _data, _data3, nodes3 = vstepcomputeCoarseSource(jnp.arange(nef))

    # This will be equivalent to a transverse matrix operation for each fine element
    _data1 = jnp.multiply(Nc1, _data).sum(axis=1)
    _data2 = jnp.multiply(Nc2, _data).sum(axis=1)
    Fc = nodes1 @ _data1.reshape(-1)
    Fm = nodes2 @ _data2.reshape(-1)
    Ff = bincount(nodes3.reshape(-1), _data3.reshape(-1), nn3)
    return Fc, Fm, Ff


@jax.jit
def computeSourceFunction_jax(x, y, z, t, v, r, d, P, eta):
    """computeSourceFunction_jax computes a 3D Gaussian term.
    :param x, y, z: nodal coordinates
    :param t: time
    :param v: laser center
    :param r: laser radius
    :param d: laser penetration depth
    :param P: laser power
    :param eta: laser emissivity
    :return F: output of source equation
    """
    # Source term params
    xm = v[0]
    ym = v[1]
    zm = v[2]

    # Assume each source is independent, multiply afterwards
    Qx = 1 / (r * jnp.sqrt(jnp.pi)) * jnp.exp(-3 * (x - xm) ** 2 / (r**2))
    Qy = 1 / (r * jnp.sqrt(jnp.pi)) * jnp.exp(-3 * (y - ym) ** 2 / (r**2))
    Qz = 1 / (d * jnp.sqrt(jnp.pi)) * jnp.exp(-3 * (z - zm) ** 2 / (d**2))

    return 6 * jnp.sqrt(3) * P * eta * Qx * Qy * Qz


@jax.jit
def interpolatePointsMatrix(node_coords, conn, node_coords_new):
    """interpolatePointsMatrix computes shape functions and node indices
    to interpolate solutions located on node_coords and connected with
    conn to new coordinates node_coords_new. This function outputs
    shape functions for interpolation that is between levels
    :param node_coords: source nodal coordinates
    :param conn: connectivity matrix
    :param node_coords_new: output nodal coordinates
    :return _Nc, _node
    :return _Nc: shape function connecting source to output
    :return _node: coarse node indices
    """
    ne_x = conn[0].shape[0]
    ne_y = conn[1].shape[0]
    ne_z = conn[2].shape[0]
    nn_x, nn_y, nn_z = ne_x + 1, ne_y + 1, ne_z + 1

    nn_xn = len(node_coords_new[0])
    nn_yn = len(node_coords_new[1])
    nn_zn = len(node_coords_new[2])
    nn2 = nn_xn * nn_yn * nn_zn
    h_x = node_coords[0][1] - node_coords[0][0]
    h_y = node_coords[1][1] - node_coords[1][0]
    h_z = node_coords[2][1] - node_coords[2][0]

    def stepInterpolatePoints(ielt):
        # Get nodal indices
        izn, _ = jnp.divmod(ielt, (nn_xn) * (nn_yn))
        iyn, _ = jnp.divmod(ielt, nn_xn)
        iyn -= izn * nn_yn
        ixn = jnp.mod(ielt, nn_xn)

        _x = node_coords_new[0][ixn, jnp.newaxis]
        _y = node_coords_new[1][iyn, jnp.newaxis]
        _z = node_coords_new[2][izn, jnp.newaxis]

        x_comp = (ne_x - 1) * jnp.ones_like(_x)
        y_comp = (ne_y - 1) * jnp.ones_like(_y)
        z_comp = (ne_z - 1) * jnp.ones_like(_z)

        x_comp2 = jnp.zeros_like(_x)
        y_comp2 = jnp.zeros_like(_y)
        z_comp2 = jnp.zeros_like(_z)

        # Figure out which coarse element we are in
        _floorx = jnp.floor((_x - node_coords[0][0]) / h_x)
        _conx = jnp.concatenate((_floorx, x_comp))
        _ielt_x = jnp.min(_conx)
        _conx = jnp.concatenate((_ielt_x[jnp.newaxis], x_comp2))
        ielt_x = jnp.max(_conx).T.astype(int)

        _floory = jnp.floor((_y - node_coords[1][0]) / h_y)
        _cony = jnp.concatenate((_floory, y_comp))
        _ielt_y = jnp.min(_cony)
        _cony = jnp.concatenate((_ielt_y[jnp.newaxis], y_comp2))
        ielt_y = jnp.max(_cony).T.astype(int)

        _floorz = jnp.floor((_z - node_coords[2][0]) / h_z)
        _conz = jnp.concatenate((_floorz, z_comp))
        _ielt_z = jnp.min(_conz).T.astype(int)
        _conz = jnp.concatenate((_ielt_z[jnp.newaxis], z_comp2))
        ielt_z = jnp.max(_conz).T.astype(int)

        nodex = conn[0][ielt_x, :]
        nodey = conn[1][ielt_y, :]
        nodez = conn[2][ielt_z, :]
        node = nodex + nodey * nn_x + nodez * (nn_x * nn_y)

        xx = node_coords[0][nodex]
        yy = node_coords[1][nodey]
        zz = node_coords[2][nodez]

        xc0 = xx[0]
        xc1 = xx[1]
        yc0 = yy[0]
        yc3 = yy[3]
        zc0 = zz[0]
        zc5 = zz[5]

        # Evaluate shape functions associated with coarse nodes
        Nc = jnp.concatenate(
            [
                ((xc1 - _x) / h_x * (yc3 - _y) / h_y * (zc5 - _z) / h_z),
                ((_x - xc0) / h_x * (yc3 - _y) / h_y * (zc5 - _z) / h_z),
                ((_x - xc0) / h_x * (_y - yc0) / h_y * (zc5 - _z) / h_z),
                ((xc1 - _x) / h_x * (_y - yc0) / h_y * (zc5 - _z) / h_z),
                ((xc1 - _x) / h_x * (yc3 - _y) / h_y * (_z - zc0) / h_z),
                ((_x - xc0) / h_x * (yc3 - _y) / h_y * (_z - zc0) / h_z),
                ((_x - xc0) / h_x * (_y - yc0) / h_y * (_z - zc0) / h_z),
                ((xc1 - _x) / h_x * (_y - yc0) / h_y * (_z - zc0) / h_z),
            ]
        )
        Nc = (
            Nc
            * (Nc >= -1e-2).all().astype(float)
            * (Nc <= 1 + 1e-2).all().astype(float)
        )
        return Nc, node

    vstepInterpolatePoints = jax.vmap(stepInterpolatePoints)
    _Nc, _node = vstepInterpolatePoints(jnp.arange(nn2))
    return _Nc, _node


@jax.jit
def interpolate_w_matrix(intmat, node, T):
    """interpolate_w_matrix uses shape functions from interpolatePointsMatrix
    to interpolate the solution to the new nodal coordinates
    :param intmat: shape functions for interpolation
    :param node: nodal indices of source solution
    :param T: source solution
    :return T_new: interpolated solution at new nodal coordinates
    """
    return jnp.multiply(intmat, T[node]).sum(axis=1)


def interpolatePoints_jax(node_coords, conn, u, node_coords_new):
    """interpolatePoints_jax interpolate solutions located on node_coords
    and connected with conn to new coordinates node_coords_new. Values
    that are later bin counted are the output
    :param node_coords: source nodal coordinates
    :param conn: connectivity matrix
    :param node_coords_new: output nodal coordinates
    :return _val: nodal values that need to be bincounted
    """
    ne_x = conn[0].shape[0]
    ne_y = conn[1].shape[0]
    ne_z = conn[2].shape[0]
    nn_x, nn_y, nn_z = ne_x + 1, ne_y + 1, ne_z + 1

    nn_xn = len(node_coords_new[0])
    nn_yn = len(node_coords_new[1])
    nn_zn = len(node_coords_new[2])
    nn2 = nn_xn * nn_yn * nn_zn
    h_x = node_coords[0][1] - node_coords[0][0]
    h_y = node_coords[1][1] - node_coords[1][0]
    h_z = node_coords[2][1] - node_coords[2][0]

    def stepInterpolatePoints(ielt):
        # Get nodal indices
        izn, _ = jnp.divmod(ielt, (nn_xn) * (nn_yn))
        iyn, _ = jnp.divmod(ielt, nn_xn)
        iyn -= izn * nn_yn
        ixn = jnp.mod(ielt, nn_xn)

        _x = node_coords_new[0][ixn, jnp.newaxis]
        _y = node_coords_new[1][iyn, jnp.newaxis]
        _z = node_coords_new[2][izn, jnp.newaxis]

        x_comp = (ne_x - 1) * jnp.ones_like(_x)
        y_comp = (ne_y - 1) * jnp.ones_like(_y)
        z_comp = (ne_z - 1) * jnp.ones_like(_z)

        x_comp2 = jnp.zeros_like(_x)
        y_comp2 = jnp.zeros_like(_y)
        z_comp2 = jnp.zeros_like(_z)

        # Figure out which coarse element we are in
        _floorx = jnp.floor((_x - node_coords[0][0]) / h_x)
        _conx = jnp.concatenate((_floorx, x_comp))
        _ielt_x = jnp.min(_conx)
        _conx = jnp.concatenate((_ielt_x[jnp.newaxis], x_comp2))
        ielt_x = jnp.max(_conx).T.astype(int)

        _floory = jnp.floor((_y - node_coords[1][0]) / h_y)
        _cony = jnp.concatenate((_floory, y_comp))
        _ielt_y = jnp.min(_cony)
        _cony = jnp.concatenate((_ielt_y[jnp.newaxis], y_comp2))
        ielt_y = jnp.max(_cony).T.astype(int)

        _floorz = jnp.floor((_z - node_coords[2][0]) / h_z)
        _conz = jnp.concatenate((_floorz, z_comp))
        _ielt_z = jnp.min(_conz).T.astype(int)
        _conz = jnp.concatenate((_ielt_z[jnp.newaxis], z_comp2))
        ielt_z = jnp.max(_conz).T.astype(int)

        nodex = conn[0][ielt_x, :]
        nodey = conn[1][ielt_y, :]
        nodez = conn[2][ielt_z, :]
        node = nodex + nodey * nn_x + nodez * (nn_x * nn_y)

        xx = node_coords[0][nodex]
        yy = node_coords[1][nodey]
        zz = node_coords[2][nodez]

        xc0 = xx[0]
        xc1 = xx[1]
        yc0 = yy[0]
        yc3 = yy[3]
        zc0 = zz[0]
        zc5 = zz[5]

        # Evaluate shape functions associated with coarse nodes
        Nc = jnp.concatenate(
            [
                ((xc1 - _x) / h_x * (yc3 - _y) / h_y * (zc5 - _z) / h_z),
                ((_x - xc0) / h_x * (yc3 - _y) / h_y * (zc5 - _z) / h_z),
                ((_x - xc0) / h_x * (_y - yc0) / h_y * (zc5 - _z) / h_z),
                ((xc1 - _x) / h_x * (_y - yc0) / h_y * (zc5 - _z) / h_z),
                ((xc1 - _x) / h_x * (yc3 - _y) / h_y * (_z - zc0) / h_z),
                ((_x - xc0) / h_x * (yc3 - _y) / h_y * (_z - zc0) / h_z),
                ((_x - xc0) / h_x * (_y - yc0) / h_y * (_z - zc0) / h_z),
                ((xc1 - _x) / h_x * (_y - yc0) / h_y * (_z - zc0) / h_z),
            ]
        )
        Nc = (
            Nc
            * (Nc >= -1e-2).all().astype(float)
            * (Nc <= 1 + 1e-2).all().astype(float)
        )
        return Nc @ u[node]

    vstepInterpolatePoints = jax.vmap(stepInterpolatePoints)
    return vstepInterpolatePoints(jnp.arange(nn2))


@jax.jit
def computeCoarseFineShapeFunctions(
    node_coords_coarse, conn_coarse, node_coords_fine, conn_fine
):
    """computeCoarseFineShapeFunctions finds the shape functions of
    the fine scale quadrature points for the coarse element
    :param node_coords_coarse: nodal coordinates of global coarse
    :param conn_coarse: indices to get coordinates of nodes of coarse element
    :param node_coords_fine: nodal coordinates of global fine
    :param conn_fine: indices to get x coordinates of nodes of fine element
    :return Nc, dNcdx, dNcdy, dNcdz, _nodes.reshape(-1)
    :return Nc: (Num fine elements, 8 quadrature, 8), coarse shape function for fine element
    :return dNcdx: (Num fine elements, 8 quadrature, 8), coarse x-derivate shape function for fine element
    :return dNcdy: (Num fine elements, 8 quadrature, 8), coarse y-derivate shape function for fine element
    :return dNcdz: (Num fine elements, 8 quadrature, 8), coarse z-derivate shape function for fine element
    :return _nodes: (Num fine elements * 8 * 8), coarse nodal indices
    """
    # Get number of elements and nodes for both coarse and fine
    nec_x = conn_coarse[0].shape[0]
    nec_y = conn_coarse[1].shape[0]
    nec_z = conn_coarse[2].shape[0]
    nnc_x = node_coords_coarse[0].shape[0]
    nnc_y = node_coords_coarse[1].shape[0]
    nnc_z = node_coords_coarse[2].shape[0]
    nnc = nnc_x * nnc_y * nnc_z
    nef_x = conn_fine[0].shape[0]
    nef_y = conn_fine[1].shape[0]
    nef_z = conn_fine[2].shape[0]
    nef = nef_x * nef_y * nef_z
    nnf_x = node_coords_fine[0].shape[0]
    nnf_y = node_coords_fine[1].shape[0]

    # Assume constant mesh sizes
    hc_x = node_coords_coarse[0][1] - node_coords_coarse[0][0]
    hc_y = node_coords_coarse[1][1] - node_coords_coarse[1][0]
    hc_z = node_coords_coarse[2][1] - node_coords_coarse[2][0]

    # Get lower bounds of meshes
    xminc_x = node_coords_coarse[0][0]
    xminc_y = node_coords_coarse[1][0]
    xminc_z = node_coords_coarse[2][0]

    # Get shape functions and weights
    coords_x = node_coords_fine[0][conn_fine[0][0, :]].reshape(-1, 1)
    coords_y = node_coords_fine[1][conn_fine[1][0, :]].reshape(-1, 1)
    coords_z = node_coords_fine[2][conn_fine[2][0, :]].reshape(-1, 1)
    coords = jnp.concatenate([coords_x, coords_y, coords_z], axis=1)
    Nf, dNdxf, wqf = computeQuad3dFemShapeFunctions_jax(coords)

    def stepComputeCoarseFineTerm(ieltf):
        ix, iy, iz, idx = convert2XYZ(ieltf, nef_x, nef_y, nnf_x, nnf_y)
        coords_x = node_coords_fine[0][conn_fine[0][ix, :]].reshape(-1, 1)
        coords_y = node_coords_fine[1][conn_fine[1][iy, :]].reshape(-1, 1)
        coords_z = node_coords_fine[2][conn_fine[2][iz, :]].reshape(-1, 1)

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
            nodec_x = conn_coarse[0][ieltc_x[iq], :].astype(int)
            nodec_y = conn_coarse[1][ieltc_y[iq], :].astype(int)
            nodec_z = conn_coarse[2][ieltc_z[iq], :].astype(int)
            nodes = nodec_x + nodec_y * nnc_x + nodec_z * nnc_x * nnc_y

            _x = x[iq]
            _y = y[iq]
            _z = z[iq]

            xc0 = node_coords_coarse[0][conn_coarse[0][ieltc_x[iq], 0]]
            xc1 = node_coords_coarse[0][conn_coarse[0][ieltc_x[iq], 1]]
            yc0 = node_coords_coarse[1][conn_coarse[1][ieltc_y[iq], 0]]
            yc3 = node_coords_coarse[1][conn_coarse[1][ieltc_y[iq], 3]]
            zc0 = node_coords_coarse[2][conn_coarse[2][ieltc_z[iq], 0]]
            zc5 = node_coords_coarse[2][conn_coarse[2][ieltc_z[iq], 5]]

            # Evaluate shape functions associated with coarse nodes
            Nc = jnp.array(
                [
                    ((xc1 - _x) / hc_x * (yc3 - _y) / hc_y * (zc5 - _z) / hc_z),
                    ((_x - xc0) / hc_x * (yc3 - _y) / hc_y * (zc5 - _z) / hc_z),
                    ((_x - xc0) / hc_x * (_y - yc0) / hc_y * (zc5 - _z) / hc_z),
                    ((xc1 - _x) / hc_x * (_y - yc0) / hc_y * (zc5 - _z) / hc_z),
                    ((xc1 - _x) / hc_x * (yc3 - _y) / hc_y * (_z - zc0) / hc_z),
                    ((_x - xc0) / hc_x * (yc3 - _y) / hc_y * (_z - zc0) / hc_z),
                    ((_x - xc0) / hc_x * (_y - yc0) / hc_y * (_z - zc0) / hc_z),
                    ((xc1 - _x) / hc_x * (_y - yc0) / hc_y * (_z - zc0) / hc_z),
                ]
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
    return Nc, [dNcdx, dNcdy, dNcdz], test


@partial(jax.jit, static_argnames=["nn1", "nn2"])
def computeCoarseTprimeMassTerm_jax(
    xnf,  # Level3
    xnm,  # Level2
    nconnf,  # Level3
    nconnm,  # Level2
    Tprimef,
    Tprimef0,
    Tprimem,
    Tprimem0,
    rho,
    cp,
    dt,
    Nc31,
    Nc21,
    Nc32,
    nodes31,
    nodes21,
    nodes32,
    Vcu,
    Vmu,
    nn1,
    nn2,
):

    Tprimef_new = Tprimef - Tprimef0
    Tprimem_new = Tprimem - Tprimem0

    # Level 3
    nef_x = nconnf[0].shape[0]
    nef_y = nconnf[1].shape[0]
    nef_z = nconnf[2].shape[0]
    nef = nef_x * nef_y * nef_z
    nnf_x = xnf[0].shape[0]
    nnf_y = xnf[1].shape[0]

    # Level 3 Get shape functions and weights
    coordsf_x = xnf[0][nconnf[0][0, :]].reshape(-1, 1)
    coordsf_y = xnf[1][nconnf[1][0, :]].reshape(-1, 1)
    coordsf_z = xnf[2][nconnf[2][0, :]].reshape(-1, 1)
    coordsf = jnp.concatenate([coordsf_x, coordsf_y, coordsf_z], axis=1)
    Nf, dNdxf, wqf = computeQuad3dFemShapeFunctions_jax(coordsf)

    # Level 3
    _, _, _, idxf = convert2XYZ(jnp.arange(nef), nef_x, nef_y, nnf_x, nnf_y)
    _Tprimef = Nf @ Tprimef_new[idxf]
    _data1 = jnp.multiply(
        jnp.multiply(-Nc31, _Tprimef.T[:, :, jnp.newaxis]),
        (rho * cp / dt) * wqf[jnp.newaxis, jnp.newaxis, :],
    ).sum(axis=2)
    _data2 = jnp.multiply(
        jnp.multiply(-Nc32, _Tprimef.T[:, :, jnp.newaxis]),
        (rho * cp / dt) * wqf[jnp.newaxis, jnp.newaxis, :],
    ).sum(axis=2)

    # Level 2
    nem_x = nconnm[0].shape[0]
    nem_y = nconnm[1].shape[0]
    nem_z = nconnm[2].shape[0]
    nem = nem_x * nem_y * nem_z
    nnm_x = xnm[0].shape[0]
    nnm_y = xnm[1].shape[0]

    # Level 2 Get shape functions and weights
    coordsm_x = xnm[0][nconnm[0][0, :]].reshape(-1, 1)
    coordsm_y = xnm[1][nconnm[1][0, :]].reshape(-1, 1)
    coordsm_z = xnm[2][nconnm[2][0, :]].reshape(-1, 1)
    coordsm = jnp.concatenate([coordsm_x, coordsm_y, coordsm_z], axis=1)
    Nm, dNdxm, wqm = computeQuad3dFemShapeFunctions_jax(coordsm)

    # Level 2
    _, _, _, idxm = convert2XYZ(jnp.arange(nem), nem_x, nem_y, nnm_x, nnm_y)
    _Tprimem = Nm @ Tprimem_new[idxm]
    _data3 = jnp.multiply(
        jnp.multiply(-Nc21, _Tprimem.T[:, :, jnp.newaxis]),
        (rho * cp / dt) * wqm[jnp.newaxis, jnp.newaxis, :],
    ).sum(axis=2)

    Vcu += nodes31 @ _data1.reshape(-1) + nodes21 @ _data3.reshape(-1)
    Vmu += nodes32 @ _data2.reshape(-1)

    return Vcu, Vmu


@partial(jax.jit, static_argnames=["nn1", "nn2"])
def computeCoarseTprimeTerm_jax(
    xnf,  # Level3
    xnm,  # Level2
    nconnf,  # Level3
    nconnm,  # Level2
    Tprimef,
    Tprimem,  # Level3, Level2
    k,
    dNc31d,
    nodes31,  # Level3 Level1
    dNc21d,
    nodes21,  # Level2 Level1
    dNc32d,
    nodes32,  # Level3 Level2
    nn1,
    nn2,
):
    # Level 3
    nef_x = nconnf[0].shape[0]
    nef_y = nconnf[1].shape[0]
    nef_z = nconnf[2].shape[0]
    nef = nef_x * nef_y * nef_z
    nnf_x = xnf[0].shape[0]
    nnf_y = xnf[1].shape[0]

    # Level 3 Get shape functions and weights
    coordsf_x = xnf[0][nconnf[0][0, :]].reshape(-1, 1)
    coordsf_y = xnf[1][nconnf[1][0, :]].reshape(-1, 1)
    coordsf_z = xnf[2][nconnf[2][0, :]].reshape(-1, 1)
    coordsf = jnp.concatenate([coordsf_x, coordsf_y, coordsf_z], axis=1)
    Nf, dNdxf, wqf = computeQuad3dFemShapeFunctions_jax(coordsf)

    # Level 3
    # idxf: (8, nef), indexing in Tprimef for later shape function use
    _, _, _, idxf = convert2XYZ(jnp.arange(nef), nef_x, nef_y, nnf_x, nnf_y)
    _Tprimef = Tprimef[idxf]
    dTprimefdx = dNdxf[:, :, 0] @ _Tprimef
    dTprimefdy = dNdxf[:, :, 1] @ _Tprimef
    dTprimefdz = dNdxf[:, :, 2] @ _Tprimef

    _data1 = jnp.multiply(
        jnp.multiply(-dNc31d[0], dTprimefdx.T[:, :, jnp.newaxis]),
        k * wqf[jnp.newaxis, jnp.newaxis, :],
    ).sum(axis=2)
    _data1 += jnp.multiply(
        jnp.multiply(-dNc31d[1], dTprimefdy.T[:, :, jnp.newaxis]),
        k * wqf[jnp.newaxis, jnp.newaxis, :],
    ).sum(axis=2)
    _data1 += jnp.multiply(
        jnp.multiply(-dNc31d[2], dTprimefdz.T[:, :, jnp.newaxis]),
        k * wqf[jnp.newaxis, jnp.newaxis, :],
    ).sum(axis=2)

    _data2 = jnp.multiply(
        jnp.multiply(-dNc32d[0], dTprimefdx.T[:, :, jnp.newaxis]),
        k * wqf[jnp.newaxis, jnp.newaxis, :],
    ).sum(axis=2)
    _data2 += jnp.multiply(
        jnp.multiply(-dNc32d[1], dTprimefdy.T[:, :, jnp.newaxis]),
        k * wqf[jnp.newaxis, jnp.newaxis, :],
    ).sum(axis=2)
    _data2 += jnp.multiply(
        jnp.multiply(-dNc32d[2], dTprimefdz.T[:, :, jnp.newaxis]),
        k * wqf[jnp.newaxis, jnp.newaxis, :],
    ).sum(axis=2)

    # Level 2
    nem_x = nconnm[0].shape[0]
    nem_y = nconnm[1].shape[0]
    nem_z = nconnm[2].shape[0]
    nem = nem_x * nem_y * nem_z
    nnm_x = xnm[0].shape[0]
    nnm_y = xnm[1].shape[0]

    # Level 2 Get shape functions and weights
    coordsm_x = xnm[0][nconnm[0][0, :]].reshape(-1, 1)
    coordsm_y = xnm[1][nconnm[1][0, :]].reshape(-1, 1)
    coordsm_z = xnm[2][nconnm[2][0, :]].reshape(-1, 1)
    coordsm = jnp.concatenate([coordsm_x, coordsm_y, coordsm_z], axis=1)
    Nm, dNdxm, wqm = computeQuad3dFemShapeFunctions_jax(coordsm)

    # Level 2
    _, _, _, idxm = convert2XYZ(jnp.arange(nem), nem_x, nem_y, nnm_x, nnm_y)
    _Tprimem = Tprimem[idxm]
    dTprimemdx = dNdxm[:, :, 0] @ _Tprimem
    dTprimemdy = dNdxm[:, :, 1] @ _Tprimem
    dTprimemdz = dNdxm[:, :, 2] @ _Tprimem

    _data3 = jnp.multiply(
        jnp.multiply(-dNc21d[0], dTprimemdx.T[:, :, jnp.newaxis]),
        k * wqm[jnp.newaxis, jnp.newaxis, :],
    ).sum(axis=2)
    _data3 += jnp.multiply(
        jnp.multiply(-dNc21d[1], dTprimemdy.T[:, :, jnp.newaxis]),
        k * wqm[jnp.newaxis, jnp.newaxis, :],
    ).sum(axis=2)
    _data3 += jnp.multiply(
        jnp.multiply(-dNc21d[2], dTprimemdz.T[:, :, jnp.newaxis]),
        k * wqm[jnp.newaxis, jnp.newaxis, :],
    ).sum(axis=2)

    Vcu = nodes31 @ _data1.reshape(-1) + nodes21 @ _data3.reshape(-1)
    Vmu = nodes32 @ _data2.reshape(-1)

    return Vcu, Vmu


def getBCindices(x):
    nx, ny, nz = x.nodes[0], x.nodes[1], x.nodes[2]
    nn = x.nn

    bidx = jnp.arange(0, nx * ny)
    tidx = jnp.arange(nx * ny * (nz - 1), nn)
    widx = jnp.arange(0, nn, nx)
    eidx = jnp.arange(nx - 1, nn, nx)
    sidx = (
        jnp.arange(0, nx)[:, jnp.newaxis]
        + (nx * ny * jnp.arange(0, nz))[jnp.newaxis, :]
    )
    sidx = sidx.reshape(-1)
    nidx = (
        jnp.arange(nx * (ny - 1), nx * ny)[:, jnp.newaxis]
        + (nx * ny * jnp.arange(0, nz))[jnp.newaxis, :]
    )
    nidx = nidx.reshape(-1)
    return [widx, eidx, sidx, nidx, bidx, tidx]


@jax.jit
def assignBCs(RHS, TS, TN, TW, TE, TB, TT, BC):
    _RHS = RHS
    _RHS = _RHS.at[BC[2]].set(TS)
    _RHS = _RHS.at[BC[3]].set(TN)
    _RHS = _RHS.at[BC[0]].set(TW)
    _RHS = _RHS.at[BC[1]].set(TE)
    _RHS = _RHS.at[BC[4]].set(TB)
    # _RHS = _RHS.at[BC[5]].set(TT)
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
def jit_constrain_v(vx, vy, vz, iE, iN, iT, iW, iS, iB):
    vx = jnp.minimum(vx, iE)
    vy = jnp.minimum(vy, iN)
    vz = jnp.minimum(vz, iT)
    vx = jnp.maximum(vx, iW)
    vy = jnp.maximum(vy, iS)
    vz = jnp.maximum(vz, iB)
    return vx, vy, vz


@jax.jit
def move_fine_mesh(x, y, z, hx, hy, hz, vx, vy, vz):
    vx_ = jnp.round(vx / hx)
    vy_ = jnp.round(vy / hy)
    vz_ = jnp.round(vz / hz)
    xnf_x = x + hx * vx_
    xnf_y = y + hy * vy_
    xnf_z = z + hz * vz_
    return [xnf_x, xnf_y, xnf_z], vx_.astype(int), vy_.astype(int), vz_.astype(int)


@jax.jit
def update_overlap_nodes_coords(
    overlapNodes_x_orig,
    overlapNodes_y_orig,
    overlapNodes_z_orig,
    overlapCoords_x_orig,
    overlapCoords_y_orig,
    overlapCoords_z_orig,
    vx_tot_con,
    vy_tot_con,
    vz_tot_con,
    hc_x,
    hc_y,
    hc_z,
):
    overlapNodes_x = overlapNodes_x_orig + jnp.round(vx_tot_con / hc_x).astype(int)
    overlapNodes_y = overlapNodes_y_orig + jnp.round(vy_tot_con / hc_y).astype(int)
    overlapNodes_z = overlapNodes_z_orig + jnp.round(vz_tot_con / hc_z).astype(int)
    overlapCoords_x = overlapCoords_x_orig + hc_x * jnp.round(vx_tot_con / hc_x)
    overlapCoords_y = overlapCoords_y_orig + hc_y * jnp.round(vy_tot_con / hc_y)
    overlapCoords_z = overlapCoords_z_orig + hc_z * jnp.round(vz_tot_con / hc_z)
    return (
        overlapNodes_x,
        overlapNodes_y,
        overlapNodes_z,
        overlapCoords_x,
        overlapCoords_y,
        overlapCoords_z,
    )


def add_vectors(a, b):
    return a + b


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
    iB = (
        CoarseLevel.bounds.z[0] - FinerLevel.bounds.z[0]
    )  # Number of elements to bottom
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


def save_result(Level, save_str, record_lab, save_path, zoffset):
    """save_result saves a vtk for the current level's temperature field
    :param Level: structure of Level
    :param save_str: prefix of save string
    :param record_lab: recording label that is incremented after each save
    :param save_path: folder where file is saved
    :param zoffset: used for rendering purposes, no effect on model itself
    """
    # List coordinates in each direction for structured save
    vtkcx = np.array(Level.node_coords[0])
    vtkcy = np.array(Level.node_coords[1])
    vtkcz = np.array(Level.node_coords[2] - zoffset)
    # Reshape the temperature field for correct rendering later
    vtkT = np.array(
        Level.T0.reshape(Level.nodes[2], Level.nodes[1], Level.nodes[0])
    ).transpose((2, 1, 0))
    # Save a vtr
    gridToVTK(
        save_path + save_str + str(record_lab).zfill(8),
        vtkcx,
        vtkcy,
        vtkcz,
        pointData={"Temperature (K)": vtkT},
    )


@jax.jit
def getNewTprime(
    lnc,
    lcon,
    lT0,
    lov,
    lovn,
    uT,
    un0,
    un1,
    ulmat,
    ulnode,
    lumat,
    lunode,
):
    # l: lower (e.g. Level3, Level3)
    # u: upper (e.g. Level2, Level2)
    # nc: node_coords
    # con: connect
    # ov: overlapCoords
    # ovn: overlapNodes

    # Go from fine to coarse
    _ = interpolate_w_matrix(lumat, lunode, lT0)
    # Go from coarse to fine
    _ = interpolate_w_matrix(ulmat, ulnode, _)
    # Directly substitute into T0 to save deepcopy
    Tprime = lT0 - _
    # Find new T
    _val = interpolatePoints_jax(lnc, lcon, lT0, lov)
    _idx = getOverlapRegion(lovn, un0, un1)
    # Directly substitute into T0 to save deepcopy
    uT = substitute_Tbar2(uT, _idx, _val)
    return Tprime, uT


@jax.jit
def getBothNewTprimes(
    lnc,
    lcon,
    lT0,
    lov,
    lovn,
    mT,
    mn,
    mlmat,
    mlnode,
    lmmat,
    lmnode,
    mnc,
    mcon,
    mov,
    movn,
    uT,
    un,
    ummat,
    umnode,
    mumat,
    munode,
):
    lTprime, mT0 = getNewTprime(
        lnc,
        lcon,
        lT0,
        lov,
        lovn,
        mT,
        mn[0],
        mn[1],
        mlmat,
        mlnode,
        lmmat,
        lmnode,
    )
    mTprime, uT0 = getNewTprime(
        mnc,
        mcon,
        mT0,
        mov,
        movn,
        uT,
        un[0],
        un[1],
        ummat,
        umnode,
        mumat,
        munode,
    )
    return lTprime, mTprime, mT0, uT0


@partial(
    jax.jit,
    static_argnames=[
        "Level1nn",
        "Level1tmpne",
        "Level1tmpnn",
        "Level2nn",
        "Level2ne",
        "Level3nn",
        "Level3ne",
    ],
)
def computeSolutions(
    Level1nc,
    Level1con,
    Level1nn,
    Level1tmpne,
    Level1tmpnn,
    Level1T0,
    Level1F,
    Level1V,
    Level1x0,
    Level1x1,
    Level1y0,
    Level1y1,
    Level1z0,
    Level1z1,
    Level1BC,
    Level1Level2_intmat,
    Level1Level2_node,
    Level2nc,
    Level2con,
    Level2nn,
    Level2ne,
    Level2T0,
    Level2F,
    Level2V,
    Level2BC,
    Level2Level3_intmat,
    Level2Level3_node,
    Level3nc,
    Level3con,
    Level3nn,
    Level3ne,
    Level3T0,
    Level3F,
    Level3BC,
    k,
    rho,
    cp,
    dt,
    T_amb,
):
    Level1T = meshlessKM(
        Level1nc,
        Level1con,
        Level1nn,
        Level1tmpne,
        k,
        rho,
        cp,
        dt,
        Level1T0,
        Level1F,
        Level1V,
    )
    Level1T = substitute_Tbar(Level1T, Level1tmpnn, T_amb)
    FinalLevel1 = assignBCs(
        Level1T,
        Level1y0,
        Level1y1,
        Level1x0,
        Level1x1,
        Level1z0,
        Level1z1,
        Level1BC,
    )

    # Compute source term for medium scale problem using fine mesh
    TfAll = interpolate_w_matrix(Level1Level2_intmat, Level1Level2_node, FinalLevel1)
    # Avoids assembling LHS matrix
    Level2T = meshlessKM(
        Level2nc,
        Level2con,
        Level2nn,
        Level2ne,
        k,
        rho,
        cp,
        dt,
        Level2T0,
        Level2F,
        Level2V,
    )
    FinalLevel2 = assignBCsFine(Level2T, TfAll, Level2BC)

    # Use Level2.T to get Dirichlet BCs for fine-scale solution
    TfAll = interpolate_w_matrix(Level2Level3_intmat, Level2Level3_node, FinalLevel2)
    FinalLevel3 = meshlessKM(
        Level3nc,
        Level3con,
        Level3nn,
        Level3ne,
        k,
        rho,
        cp,
        dt,
        Level3T0,
        Level3F,
        0,
    )
    FinalLevel3 = assignBCsFine(FinalLevel3, TfAll, Level3BC)
    return FinalLevel1, FinalLevel2, FinalLevel3


@partial(jax.jit, static_argnames=["ne", "nn"])
def computeConvectionRadiation(
    xyz, c, T, ne, nn, T_amb, h_conv, sigma_sb, emissivity, F
):
    # Stefan-Boltzmann constant: 5.67e-8 [W/m^2/K^4]
    # Emissivity and Heat transfer coefficient are unknown and not readily measurable, require calibration
    # Heat transfer cofficient by convection: [W/m^2 k]
    # Emissivity of the powder bed: unitless
    # Equation is k grad(T) = h_conv * (T - T_amb) + sigma_sb * emissivity * (T**4 - T_amb**4)
    # This would be applied to "finest" levels
    h_conv = h_conv / 1e6  # convert to W/mm^2K
    sigma_sb = sigma_sb / 1e6  # convert to W/mm^2/K^4

    x, y, z = xyz[0], xyz[1], xyz[2]
    cx, cy, cz = c[0], c[1], c[2]

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
        _iT = jnp.matmul(N, T[idx[4:]])
        iT = h_conv * (T_amb - _iT) + sigma_sb * emissivity * (T_amb**4 - _iT**4)
        return jnp.matmul(N.T, jnp.multiply(iT, wq.reshape(-1))), idx[4:]

    vcalcCR = jax.vmap(calcCR)
    aT, aidx = vcalcCR(jnp.arange(top_ne, ne))
    NeumannBC = jnp.bincount(aidx.reshape(-1), aT.reshape(-1), length=nn)

    # Returns k grad(T) integral, which is Neumann BC (expect ambient < body)
    return F + NeumannBC


@partial(
    jax.jit,
    static_argnames=["nn1", "nn2", "nn3", "tmp_ne", "tmp_nn", "Level2ne", "Level3ne"],
)
def doExplicitTimestep(
    Level3nc,
    Level3con,
    Level3ne,
    Level3oC,
    Level3oN,
    Level2nc,
    Level2con,
    Level2ne,
    Level2no,
    Level2oC,
    Level2oN,
    Level1nc,
    Level1con,
    Level1no,
    Level1condx,
    Level1condy,
    Level1condz,
    Level3BC,
    Level2BC,
    Level1BC,
    tmp_ne,
    tmp_nn,
    nn1,
    nn2,
    nn3,
    Level3Tp0,
    Level2Tp0,
    Level3T0,
    Level2T0,
    Level1T0,
    Level3NcLevel1,
    Level3NcLevel2,
    Level2NcLevel1,
    Level3dNcdLevel1,
    Level3nodesLevel1,
    Level3dNcdLevel2,
    Level3nodesLevel2,
    Level2dNcdLevel1,
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
    laserr,
    laserd,
    laserP,
    lasereta,
    T_amb,
    h_conv,
    vareps,
):
    """doExplicitTimestep computes a Level1 explicit timestep starting by
    computing the source terms for all three levels, computing the convection/radiation
    terms for all three levels, computing the previous step volumetric correction terms,
    computing solutions for all three levels (predictor step), calculating new Tprime terms
    and their volumetric correction terms, computing solutions for all three levels (corrector step),
    and finally updating the Tprime terms for the next explicit time step
    :param nc: nodal coordinates
    :param con: connectivity matrix
    :param ne: number of elements
    :param oC: overlapping coordinates between level and lower level
    :param oN: overlapping nodes between level and lower level
    :param no: number of nodes (in one direction)
    :param condx, condy, condz: Dirichlet boundary conditions for Level1 mesh
    :param Level#BC: indices for boundary surface nodes
    :param tmp_ne, tmp_nn: number of elements and nodes active on Level1 mesh (based on layer)
    :param nn1, nn2, nn3: total number of nodes (active and deactive)
    :param Tp0: previous correction term for Tprime
    :param T0: previous temperature
    :param Level3NcLevel1: shape functions from Level3 to Level1
    :param Level3dNcdLevel1: derivative of shape functions from Level3 to Level1
    :param Level3nodesLevel1: node indices for Level3 to Level 1 Tprime calculations
    :param Level2Level3_intmat: interpolation matrix from Level2 to Level3
    :param Level2Level3_node: nodes indexing into Level2 for interpolation
    :param _t: current time
    :param v: current position of laser (from reading file)
    :param k: thermal conductivity
    :param rho: material density
    :param cp: specific heat
    :param dt: timestep
    :param laserr: laser radius
    :param laserd: laser depth of penetration
    :param laserP: laser Power
    :param lasereta: laser absorptivity
    :param T_amb: ambient temperature
    :param h_conv: convection coefficient
    :param vareps: emissivity coefficient
    :return Level3T0, Level2T0, Level1T0, Level3Tp0, Level2Tp0
    :return Level3T0: Temperature for Level 3 (assigned back to previous temperature)
    :return Level2T0: Temperature for Level 2 (assigned back to previous temperature)
    :return Level1T0: Temperature for Level 1 (assigned back to previous temperature)
    :return Level3Tp0: Temperature correction from Level 3 (assigned to previous)
    :return Level2Tp0: Temperature correction from Level 2 (assigned to previous)
    """
    Fc, Fm, Ff = computeSources(
        Level3nc,
        Level3con,
        _t,
        v,
        Level3NcLevel1,
        Level3NcLevel2,
        Level3nodesLevel1,
        Level3nodesLevel2,
        nn1,
        nn2,
        nn3,
        laserr,
        laserd,
        laserP,
        lasereta,
    )
    Fc = computeConvectionRadiation(
        Level1nc, Level1con, Level1T0, tmp_ne, nn1, T_amb, h_conv, 5.67e-8, vareps, Fc
    )
    Fm = computeConvectionRadiation(
        Level2nc, Level2con, Level2T0, Level2ne, nn2, T_amb, h_conv, 5.67e-8, vareps, Fm
    )
    Ff = computeConvectionRadiation(
        Level3nc, Level3con, Level3T0, Level3ne, nn3, T_amb, h_conv, 5.67e-8, vareps, Ff
    )
    Vcu, Vmu = computeCoarseTprimeTerm_jax(
        Level3nc,
        Level2nc,
        Level3con,
        Level2con,
        Level3Tp0,
        Level2Tp0,
        k,
        Level3dNcdLevel1,
        Level3nodesLevel1,
        Level2dNcdLevel1,
        Level2nodesLevel1,
        Level3dNcdLevel2,
        Level3nodesLevel2,
        nn1,
        nn2,
    )
    Level1T, Level2T, Level3T = computeSolutions(
        Level1nc,
        Level1con,
        nn1,
        tmp_ne,
        tmp_nn,
        Level1T0,
        Fc,
        Vcu,
        Level1condx[0],
        Level1condx[1],
        Level1condy[0],
        Level1condy[1],
        Level1condz[0],
        Level1condz[1],
        Level1BC,
        Level1Level2_intmat,
        Level1Level2_node,
        Level2nc,
        Level2con,
        nn2,
        Level2ne,
        Level2T0,
        Fm,
        Vmu,
        Level2BC,
        Level2Level3_intmat,
        Level2Level3_node,
        Level3nc,
        Level3con,
        nn3,
        Level3ne,
        Level3T0,
        Ff,
        Level3BC,
        k,
        rho,
        cp,
        dt,
        T_amb,
    )
    Level3Tp, Level2Tp, Level2T, Level1T = getBothNewTprimes(
        Level3nc,
        Level3con,
        Level3T,
        Level3oC,
        Level3oN,
        Level2T,
        Level2no,
        Level2Level3_intmat,
        Level2Level3_node,
        Level3Level2_intmat,
        Level3Level2_node,
        Level2nc,
        Level2con,
        Level2oC,
        Level2oN,
        Level1T,
        Level1no,
        Level1Level2_intmat,
        Level1Level2_node,
        Level2Level1_intmat,
        Level2Level1_node,
    )
    Vcu, Vmu = computeCoarseTprimeMassTerm_jax(
        Level3nc,
        Level2nc,
        Level3con,
        Level2con,
        Level3Tp,
        Level3Tp0,
        Level2Tp,
        Level2Tp0,
        rho,
        cp,
        dt,
        Level3NcLevel1,
        Level2NcLevel1,
        Level3NcLevel2,
        Level3nodesLevel1,
        Level2nodesLevel1,
        Level3nodesLevel2,
        Vcu,
        Vmu,
        nn1,
        nn2,
    )
    Level1T, Level2T, Level3T0 = computeSolutions(
        Level1nc,
        Level1con,
        nn1,
        tmp_ne,
        tmp_nn,
        Level1T0,
        Fc,
        Vcu,
        Level1condx[0],
        Level1condx[1],
        Level1condy[0],
        Level1condy[1],
        Level1condz[0],
        Level1condz[1],
        Level1BC,
        Level1Level2_intmat,
        Level1Level2_node,
        Level2nc,
        Level2con,
        nn2,
        Level2ne,
        Level2T0,
        Fm,
        Vmu,
        Level2BC,
        Level2Level3_intmat,
        Level2Level3_node,
        Level3nc,
        Level3con,
        nn3,
        Level3ne,
        Level3T0,
        Ff,
        Level3BC,
        k,
        rho,
        cp,
        dt,
        T_amb,
    )
    Level3Tp0, Level2Tp0, Level2T0, Level1T0 = getBothNewTprimes(
        Level3nc,
        Level3con,
        Level3T0,
        Level3oC,
        Level3oN,
        Level2T,
        Level2no,
        Level2Level3_intmat,
        Level2Level3_node,
        Level3Level2_intmat,
        Level3Level2_node,
        Level2nc,
        Level2con,
        Level2oC,
        Level2oN,
        Level1T,
        Level1no,
        Level1Level2_intmat,
        Level1Level2_node,
        Level2Level1_intmat,
        Level2Level1_node,
    )
    return Level3T0, Level2T0, Level1T0, Level3Tp0, Level2Tp0


@jax.jit
def moveLevel3Mesh(
    v,
    vstart,
    Level3pnc,
    Level3con,
    Level3inc,
    Level3ooN,
    Level3ooC,
    Level3bx,
    Level3by,
    Level3bz,
    Level3Tp0,
    Level2nc,
    Level2con,
    Level2h,
    Level2Tp0,
    Level1nc,
    Level1con,
    Level1T0,
):
    # Level3nc: node_coords
    # Level3con: connect
    # Level3inc: init_node_coords
    # Level3pnc: prev_node_coords
    # Level3oN: overlapNodes
    # Level3oC: overlapCoords
    # Level3ooN: orig_overlap_nodes
    # Level3ooC: orig_overlap_coors
    # Level3bx,by,bz: bounds.ix,iy,iz

    vtot = v - vstart
    _Level3vx_tot_con, _Level3vy_tot_con, _Level3vz_tot_con = jit_constrain_v(
        vtot[0],
        vtot[1],
        vtot[2],
        Level3bx[1],
        Level3by[1],
        Level3bz[1],
        Level3bx[0],
        Level3by[0],
        Level3bz[0],
    )

    ### Correction step (fine) ###
    Level3nc, _a, _b, _c = move_fine_mesh(
        Level3inc[0],
        Level3inc[1],
        Level3inc[2],
        Level2h[0],
        Level2h[1],
        Level2h[2],
        _Level3vx_tot_con,
        _Level3vy_tot_con,
        _Level3vz_tot_con,
    )

    Level3oNx, Level3oNy, Level3oNz, Level3oCx, Level3oCy, Level3oCz = (
        update_overlap_nodes_coords(
            Level3ooN[0],
            Level3ooN[1],
            Level3ooN[2],
            Level3ooC[0],
            Level3ooC[1],
            Level3ooC[2],
            _Level3vx_tot_con,
            _Level3vy_tot_con,
            _Level3vz_tot_con,
            Level2h[0],
            Level2h[1],
            Level2h[2],
        )
    )

    Level3T0 = interpolatePoints_jax(Level1nc, Level1con, Level1T0, Level3nc)
    _ = interpolatePoints_jax(Level2nc, Level2con, Level2Tp0, Level3nc)
    Level3T0 = add_vectors(Level3T0, _)

    Level3Tp0 = interpolatePoints_jax(Level3pnc, Level3con, Level3Tp0, Level3nc)
    Level3T0 = add_vectors(Level3T0, Level3Tp0)
    return (
        Level3nc,
        [Level3oNx, Level3oNy, Level3oNz],
        [Level3oCx, Level3oCy, Level3oCz],
        Level3T0,
        Level3Tp0,
        vtot,
    )


@jax.jit
def updateLevel3AfterMove(
    Level3nc,
    Level3con,
    Level3oN,
    Level2nc,
    Level2con,
    Level1nc,
    Level1con,
    vx,
    vy,
    vz,
):
    Level3oN[0], Level3oN[1], Level3oN[2] = (
        Level3oN[0] - vx,
        Level3oN[1] - vy,
        Level3oN[2] - vz,
    )

    # If mesh moves, recalculate shape functions
    (
        Level3NcLevel1,
        Level3dNcdLevel1,
        Level3nodesLevel1,
    ) = computeCoarseFineShapeFunctions(
        Level1nc,
        Level1con,
        Level3nc,
        Level3con,
    )

    # Move Level3 with respect to Level2
    (
        Level3NcLevel2,
        Level3dNcdLevel2,
        Level3nodesLevel2,
    ) = computeCoarseFineShapeFunctions(
        Level2nc,
        Level2con,
        Level3nc,
        Level3con,
    )

    Level2Level3_intmat, Level2Level3_node = interpolatePointsMatrix(
        Level2nc,
        Level2con,
        Level3nc,
    )
    Level3Level2_intmat, Level3Level2_node = interpolatePointsMatrix(
        Level3nc,
        Level3con,
        Level2nc,
    )
    return (
        Level3oN,
        Level3NcLevel1,
        Level3dNcdLevel1,
        Level3nodesLevel1,
        Level3NcLevel2,
        Level3dNcdLevel2,
        Level3nodesLevel2,
        Level2Level3_intmat,
        Level2Level3_node,
        Level3Level2_intmat,
        Level3Level2_node,
    )


@jax.jit
def prepLevel2Move(
    Level2inc,
    Level2h,
    Level2ooN,
    Level2ooC,
    Level2bx,
    Level2by,
    Level2bz,
    Level1h,
    vtot,
    _vx,
    _vy,
    _vz,
):
    _vx_tot_con, _vy_tot_con, _vz_tot_con = jit_constrain_v(
        vtot[0],
        vtot[1],
        vtot[2],
        Level2bx[1],
        Level2by[1],
        Level2bz[1],
        Level2bx[0],
        Level2by[0],
        Level2bz[0],
    )
    # Need to round due to numerical round off error and truncation
    _tmp_x = (jnp.round(Level1h[0] / Level2h[0])).astype(int)
    _tmp_y = (jnp.round(Level1h[1] / Level2h[1])).astype(int)
    _tmp_z = (jnp.round(Level1h[2] / Level2h[2])).astype(int)
    _vvx, _vvy, _vvz = _vx, _vy, _vz

    Level2nc, _vx, _vy, _vz = move_fine_mesh(
        Level2inc[0],
        Level2inc[1],
        Level2inc[2],
        Level1h[0],
        Level1h[1],
        Level1h[2],
        _vx_tot_con,
        _vy_tot_con,
        _vz_tot_con,
    )

    _vx, _vy, _vz = _vx * _tmp_x, _vy * _tmp_y, _vz * _tmp_z
    moveLevel2 = (_vvx != _vx) | (_vvy != _vy) | (_vvz != _vz)

    Level2oNx, Level2oNy, Level2oNz, Level2oCx, Level2oCy, Level2oCz = (
        update_overlap_nodes_coords(
            Level2ooN[0],
            Level2ooN[1],
            Level2ooN[2],
            Level2ooC[0],
            Level2ooC[1],
            Level2ooC[2],
            _vx_tot_con,
            _vy_tot_con,
            _vz_tot_con,
            Level1h[0],
            Level1h[1],
            Level1h[2],
        )
    )
    return (
        Level2nc,
        [Level2oNx, Level2oNy, Level2oNz],
        [Level2oCx, Level2oCy, Level2oCz],
        _vx,
        _vy,
        _vz,
        moveLevel2,
    )


def moveLevel2Mesh(
    Level2nc, Level2con, Level2pnc, Level2Tp0, Level1nc, Level1con, Level1T0
):
    Level2T0 = interpolatePoints_jax(
        Level1nc,
        Level1con,
        Level1T0,
        Level2nc,
    )
    Level2Tp0 = interpolatePoints_jax(
        Level2pnc,
        Level2con,
        Level2Tp0,
        Level2nc,
    )
    Level2T0 = add_vectors(Level2T0, Level2Tp0)
    return Level2T0, Level2Tp0


@jax.jit
def updateLevel2objects(
    Level2nc, Level2con, Level2pnc, Level2Tp0, Level1nc, Level1con, Level1T0
):

    Level2T0, Level2Tp0 = moveLevel2Mesh(
        Level2nc, Level2con, Level2pnc, Level2Tp0, Level1nc, Level1con, Level1T0
    )
    # If mesh moves, recalculate shape functions
    Level2NcLevel1, Level2dNcdLevel1, Level2nodesLevel1 = (
        computeCoarseFineShapeFunctions(Level1nc, Level1con, Level2nc, Level2con)
    )
    Level1Level2_intmat, Level1Level2_node = interpolatePointsMatrix(
        Level1nc, Level1con, Level2nc
    )
    Level2Level1_intmat, Level2Level1_node = interpolatePointsMatrix(
        Level2nc, Level2con, Level1nc
    )
    return (
        Level2T0,
        Level2Tp0,
        Level2NcLevel1,
        Level2dNcdLevel1,
        Level2nodesLevel1,
        Level1Level2_intmat,
        Level1Level2_node,
        Level2Level1_intmat,
        Level2Level1_node,
    )


@jax.jit
def moveEverything(
    v,
    vstart,
    Level3pnc,
    Level3con,
    Level3inc,
    Level3ooN,
    Level3ooC,
    Level3bx,
    Level3by,
    Level3bz,
    Level3Tp0,
    Level2pnc,
    Level2con,
    Level2h,
    Level2Tp0,
    Level1nc,
    Level1con,
    Level1T0,
    Level2inc,
    Level2ooN,
    Level2ooC,
    Level2bx,
    Level2by,
    Level2bz,
    Level1h,
    _vx,
    _vy,
    _vz,
):
    Level3nc, Level3oN, Level3oC, Level3T0, Level3Tp0, vtot = moveLevel3Mesh(
        v,
        vstart,
        Level3pnc,
        Level3con,
        Level3inc,
        Level3ooN,
        Level3ooC,
        Level3bx,
        Level3by,
        Level3bz,
        Level3Tp0,
        Level2pnc,
        Level2con,
        Level2h,
        Level2Tp0,
        Level1nc,
        Level1con,
        Level1T0,
    )
    Level2nc, Level2oN, Level2oC, _vx, _vy, _vz, moveLevel2 = prepLevel2Move(
        Level2inc,
        Level2h,
        Level2ooN,
        Level2ooC,
        Level2bx,
        Level2by,
        Level2bz,
        Level1h,
        vtot,
        _vx,
        _vy,
        _vz,
    )
    (
        Level2T0,
        Level2Tp0,
        Level2NcLevel1,
        Level2dNcdLevel1,
        Level2nodesLevel1,
        Level1Level2_intmat,
        Level1Level2_node,
        Level2Level1_intmat,
        Level2Level1_node,
    ) = updateLevel2objects(
        Level2nc, Level2con, Level2pnc, Level2Tp0, Level1nc, Level1con, Level1T0
    )
    (
        Level3oN,
        Level3NcLevel1,
        Level3dNcdLevel1,
        Level3nodesLevel1,
        Level3NcLevel2,
        Level3dNcdLevel2,
        Level3nodesLevel2,
        Level2Level3_intmat,
        Level2Level3_node,
        Level3Level2_intmat,
        Level3Level2_node,
    ) = updateLevel3AfterMove(
        Level3nc,
        Level3con,
        Level3oN,
        Level2nc,
        Level2con,
        Level1nc,
        Level1con,
        _vx,
        _vy,
        _vz,
    )
    return (
        Level3nc,
        Level3oN,
        Level3oC,
        Level3T0,
        Level3Tp0,
        vtot,
        Level2nc,
        Level2oN,
        Level2oC,
        Level2T0,
        Level2Tp0,
        Level2NcLevel1,
        Level2dNcdLevel1,
        Level2nodesLevel1,
        Level1Level2_intmat,
        Level1Level2_node,
        Level2Level1_intmat,
        Level2Level1_node,
        Level3NcLevel1,
        Level3dNcdLevel1,
        Level3nodesLevel1,
        Level3NcLevel2,
        Level3dNcdLevel2,
        Level3nodesLevel2,
        Level2Level3_intmat,
        Level2Level3_node,
        Level3Level2_intmat,
        Level3Level2_node,
        _vx,
        _vy,
        _vz,
    )

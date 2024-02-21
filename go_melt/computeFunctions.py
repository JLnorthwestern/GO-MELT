import numpy as np
import scipy
import jax.numpy as jnp
import jax
from functools import partial
from jax.config import config
from jax.experimental import sparse
from pyevtk.hl import gridToVTK

# True is for convergence (double precision), False is single precision
config.update("jax_enable_x64", False)

def calcNumNodes(x):
    """ calcNumNodes finds the number of nodes from number of elements.
        :param x: list of elements (3D)
        :return x: list of nodes (3D)
    """
    return [x[0] + 1, x[1] + 1, x[2] + 1]

def createMesh3D(x, y, z):
    """ createMesh3D finds nodal coordinates and mesh
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
        [jnp.arange(0, x[2]-1).reshape(-1, 1),
         jnp.arange(1, x[2]).reshape(-1, 1),
         jnp.arange(1, x[2]).reshape(-1, 1),
         jnp.arange(0, x[2]-1).reshape(-1, 1),
         jnp.arange(0, x[2]-1).reshape(-1, 1),
         jnp.arange(1, x[2]).reshape(-1, 1),
         jnp.arange(1, x[2]).reshape(-1, 1),
         jnp.arange(0, x[2]-1).reshape(-1, 1)],
        axis=1)
    nconn_y = jnp.concatenate(
        [jnp.arange(0, y[2]-1).reshape(-1, 1),
         jnp.arange(0, y[2]-1).reshape(-1, 1),
         jnp.arange(1, y[2]).reshape(-1, 1),
         jnp.arange(1, y[2]).reshape(-1, 1),
         jnp.arange(0, y[2]-1).reshape(-1, 1),
         jnp.arange(0, y[2]-1).reshape(-1, 1),
         jnp.arange(1, y[2]).reshape(-1, 1),
         jnp.arange(1, y[2]).reshape(-1, 1)],
        axis=1)
    nconn_z = jnp.concatenate(
        [jnp.arange(0, z[2]-1).reshape(-1, 1),
         jnp.arange(0, z[2]-1).reshape(-1, 1),
         jnp.arange(0, z[2]-1).reshape(-1, 1),
         jnp.arange(0, z[2]-1).reshape(-1, 1),
         jnp.arange(1, z[2]).reshape(-1, 1),
         jnp.arange(1, z[2]).reshape(-1, 1),
         jnp.arange(1, z[2]).reshape(-1, 1),
         jnp.arange(1, z[2]).reshape(-1, 1)],
        axis=1)
    return [nx, ny, nz], [nconn_x, nconn_y, nconn_z]

@partial(jax.jit, static_argnames=['ne','nn'])
def meshlessKM(x, y, z, cx, cy, cz, nn, ne, k, rho, cp, dt, T, Fc, Corr):
    """ meshlessKM computes the thermal solve for the explicit timestep.
        :param x, y, z: nodal coordinates along x, y, or z
        :param cx, cy, cz: connectivities along x, y, or z
        :param nn, ne: number of nodes, number of elements
        :param k, rho, cp: material properties for conductivity, density, heat capacity
        :param dt: timestep
        :param T: previous temperature for mesh
        :param Fc: integrated RHS values (including heat source)
        :param Corr: integrated correction terms
        :return (newT + Fc + Corr) / newM
        :return newT: Temperture for next timestep
    """

    ne_x = jnp.size(cx, 0)
    ne_y = jnp.size(cy, 0)

    nn_x = ne_x + 1
    nn_y = ne_y + 1

    nen = jnp.size(cx, 1)
    ndim = 3

    coords_x = x[cx[0, :]].reshape(-1, 1)
    coords_y = y[cy[0, :]].reshape(-1, 1)
    coords_z = z[cz[0, :]].reshape(-1, 1)

    coords = jnp.concatenate([coords_x, coords_y, coords_z], axis=1)
    N, dNdx, wq = computeQuad3dFemShapeFunctions_jax(coords)

    def calcVal(i):
        ix, iy, iz, idx = convert2XYZ(i, ne_x, ne_y, nn_x, nn_y)
        iT = T[idx]
        kvec = k * jnp.ones([8,1])
        mvec = ((rho * jnp.ones([8,1])) * (cp * jnp.ones([8,1]))) / dt

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
    """ convert2XYZ computes the indices for each node w.r.t. each axis. 
        It also computes the connectivity matrix in terms of global indices.
        :param i: element id
        :param ne_x, ne_y: number of elements in x and y directions
        :param nn_x, nn_y: number of nodes in x and y directions
        :return ix, iy, iz, idx
        :return ix, iy, iz: Node id in either x, y, or z axis
        :return idx: connectivity vector of global node ids
    """

    iz, _ = jnp.divmod(i, (ne_x)*(ne_y))
    iy, _ = jnp.divmod(i, ne_x)
    iy -= iz * ne_y
    ix = jnp.mod(i, ne_x)

    idx = jnp.array([ix + iy * (nn_x) + iz * (nn_x * nn_y),
                    (ix + 1) + iy * (nn_x) + iz * (nn_x * nn_y),
                    (ix + 1) + (iy + 1) * (nn_x) + iz * (nn_x * nn_y),
                    ix + (iy + 1) * (nn_x) + iz * (nn_x * nn_y),
                    ix + iy * (nn_x) + (iz + 1) * (nn_x * nn_y),
                    (ix + 1) + iy * (nn_x) + (iz + 1) * (nn_x * nn_y),
                    (ix + 1) + (iy + 1) * (nn_x) + (iz + 1) * (nn_x * nn_y),
                    ix + (iy + 1) * (nn_x) + (iz + 1) * (nn_x * nn_y)])

    return ix, iy, iz, idx

def getBC(LHS, nx, ny, nz):
    """ getBC updates the implicit LHS to have Dirichlet boundary conditions.
        In its current form, the top and bottom are not included (since testing 2D).
        :param LHS: current left-hand side matrix (implicit)
        :param nx, ny, nz: number of nodes in x, y, and z directions
        :return LHS: left-hand side matrix updated with Dirichlet BCs
    """

    # Top surface is arange(0, nnx * nny)
    # Bottom surface is arange(nnx * nny * (nnz - 1), nnx * nny * nnz)
    # West is arange(0, nnx * nny * nnz, nnx) (spaced by nnx)
    # East is arange(nnx - 1, nnx * nny * nnz, nnx)
    # South is arange(0, nnx) + nnx * nny * arange(0, nn_z-1)
    # North is arange(nnx * nny - nnx - 1, nnx * nny) + nnx * nny * arange(0, nn_z)
    nn = nx * ny * nz
    bidx = jnp.arange(0, nx * ny)
    tidx = jnp.arange(nx * ny * (nz - 1), nn)
    widx = jnp.arange(0, nn, nx)
    eidx = jnp.arange(nx - 1, nn, nx)
    sidx = jnp.arange(0, nx)[:, jnp.newaxis] + \
        (nx * ny * jnp.arange(0, nz))[jnp.newaxis, :]
    sidx = sidx.reshape(-1)
    nidx = jnp.arange(
        nx * (ny - 1), nx * ny)[:, jnp.newaxis] + (nx * ny * jnp.arange(0, nz))[jnp.newaxis, :]
    nidx = nidx.reshape(-1)

    # Ignore top and bottom
    # idx = jnp.concatenate([tidx, bidx, widx, eidx, sidx, nidx])
    idx = jnp.concatenate([widx, eidx, sidx, nidx])
    idx = jnp.unique(idx)
    data = jnp.ones_like(idx)
    identity = scipy.sparse.csr_matrix((data, (idx, idx)), shape=(nn, nn))

    BCLHS = identity @ LHS
    LHS = LHS - BCLHS
    LHS = LHS + identity
    return LHS


@jax.jit
def computeQuad3dFemShapeFunctions_jax(coords):
    """ def computeQuad3dFemShapeFunctions_jax calculates the 3D shape functions
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
    N = (1 / 8) * (1 + ksi_q[:, jnp.newaxis] @ ksi_i[jnp.newaxis, :]) * \
        (1 + eta_q[:, jnp.newaxis] @ eta_i[jnp.newaxis, :]) * \
        (1 + zeta_q[:, jnp.newaxis] @ zeta_i[jnp.newaxis, :])
    dNdksi = (1 / 8) * ksi_i[jnp.newaxis, :] * \
        (1 + eta_q[:, jnp.newaxis] @ eta_i[jnp.newaxis, :]) * \
        (1 + zeta_q[:, jnp.newaxis] @ zeta_i[jnp.newaxis, :])
    dNdeta = (1 / 8) * eta_i[jnp.newaxis, :] * \
        (1 + ksi_q[:, jnp.newaxis] @ ksi_i[jnp.newaxis, :]) * \
        (1 + zeta_q[:, jnp.newaxis] @ zeta_i[jnp.newaxis, :])
    dNdzeta = (1 / 8) * zeta_i[jnp.newaxis, :] * \
        (1 + ksi_q[:, jnp.newaxis] @ ksi_i[jnp.newaxis, :]) * \
        (1 + eta_q[:, jnp.newaxis] @ eta_i[jnp.newaxis, :])

    # Find derivative of parent coordinates w.r.t. isoparametric space
    dxdksi = jnp.matmul(dNdksi, coords[:, 0])
    dydeta = jnp.matmul(dNdeta, coords[:, 1])
    dzdzeta = jnp.matmul(dNdzeta, coords[:, 2])

    # Find Jacobian matrices and calculate quadrature weights and dNdx
    J = jnp.array([[dxdksi[0], 0, 0],
                   [0, dydeta[0], 0],
                   [0, 0, dzdzeta[0]]])
    Jinv = jnp.array([[1/dxdksi[0], 0, 0],
                      [0, 1/dydeta[0], 0],
                      [0, 0, 1/dzdzeta[0]]])
    dNdx = jnp.zeros([ngp, ngp, ndim])
    wq = jnp.zeros([ngp, 1])
    for q in range(ngp):
        dNdx = dNdx.at[q, :, :].set(jnp.concatenate(
            [dNdksi[q, :, jnp.newaxis],
             dNdeta[q, :, jnp.newaxis],
             dNdzeta[q, :, jnp.newaxis]], axis=1) @ Jinv)
        wq = wq.at[q].set(jnp.linalg.det(J) * tmp_wq[q])
    return jnp.array(N), jnp.array(dNdx), jnp.array(wq)

@jax.jit
def computeQuad2dFemShapeFunctions_jax(coords):
    """ def computeQuad2dFemShapeFunctions_jax calculates the 2D shape functions
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

    N = (1 / 4) * (1 + ksi_q[:, jnp.newaxis] @ ksi_i[jnp.newaxis, :]) * \
        (1 + eta_q[:, jnp.newaxis] @ eta_i[jnp.newaxis, :])
    dNdksi = (1 / 4) * ksi_i[jnp.newaxis, :] * \
        (1 + eta_q[:, jnp.newaxis] @ eta_i[jnp.newaxis, :])
    dNdeta = (1 / 4) * eta_i[jnp.newaxis, :] * \
        (1 + ksi_q[:, jnp.newaxis] @ ksi_i[jnp.newaxis, :])

    dxdksi = jnp.matmul(dNdksi, coords[4:, 0])
    dydeta = jnp.matmul(dNdeta, coords[4:, 1])

    J = jnp.array([[dxdksi[0], 0],
                   [0, dydeta[0]]])
    Jinv = jnp.array([[1/dxdksi[0], 0],
                      [0, 1/dydeta[0]]])
    dNdx = jnp.zeros([ngp, ngp, ndim])
    wq = jnp.zeros([ngp, 1])
    for q in range(ngp):
        dNdx = dNdx.at[q, :, :].set(jnp.concatenate(
            [dNdksi[q, :, jnp.newaxis],
             dNdeta[q, :, jnp.newaxis]], axis=1) @ Jinv)
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

@partial(jax.jit, static_argnames=['nn1', 'nn2', 'nn3'])
def computeSources(xf, yf, zf, cfx, cfy, cfz,
                   t, v, Nc1, Nc2, nodes1, nodes2, nn1, nn2, nn3,
                   laserr, laserd, laserP, lasereta):
    """ computeSources computes the integrated source term for all three levels using
        the mesh from Level 3.
        :param xf, yf, zf: nodal coordinates
        :param cfx, cfy, cfz: connectivity matrix
        :param _t: current time
        :param v: current position of laser (from reading file)
        :param Nc1: sub2NcLevel1, shape functions between Level 3 and Level 1 (symmetric)
        :param nc2: sub2NcLevel2, shape functions between Level 3 and Level 2 (symmetric)
        :param nodes1: sub2nodesLevel1, node indices in Level3 for shape functions Level1
        :param nodes2: sub2nodesLevel2, node indices in Level3 for shape functions Level2
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
    nef_x = cfx.shape[0]
    nef_y = cfy.shape[0]
    nef_z = cfz.shape[0]
    nef = nef_x * nef_y * nef_z
    nnf_x, nnf_y = nef_x + 1, nef_y + 1

    # Get shape functions and weights
    coords_x = xf[cfx[0, :]].reshape(-1, 1)
    coords_y = yf[cfy[0, :]].reshape(-1, 1)
    coords_z = zf[cfz[0, :]].reshape(-1, 1)
    coords = jnp.concatenate([coords_x, coords_y, coords_z], axis=1)
    Nf, dNdxf, wqf = computeQuad3dFemShapeFunctions_jax(coords)

    def stepcomputeCoarseSource(ieltf):
        # Get the nodal indices for that element
        ix, iy, iz, idx = convert2XYZ(ieltf, nef_x, nef_y, nnf_x, nnf_y)
        # Get nodal coordinates for the fine element
        coords_x = xf[cfx[ix, :]].reshape(-1, 1)
        coords_y = yf[cfy[iy, :]].reshape(-1, 1)
        coords_z = zf[cfz[iz, :]].reshape(-1, 1)
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
    _data1 = jnp.multiply(Nc1,_data).sum(axis=1)
    _data2 = jnp.multiply(Nc2,_data).sum(axis=1)
    Fc = nodes1 @ _data1.reshape(-1)
    Fm = nodes2 @ _data2.reshape(-1)
    Ff = bincount(nodes3.reshape(-1), _data3.reshape(-1), nn3)
    return Fc, Fm, Ff

@jax.jit
def computeSourceFunction_jax(x, y, z, t, v, r, d, P, eta):
    # Source term params
    xm = v[0]
    ym = v[1]
    zm = v[2]

    # Assume each source is independent, multiply afterwards
    Qx = 1 / (r * jnp.sqrt(jnp.pi)) * \
        jnp.exp(-3*(x - xm) ** 2 / (r ** 2))
    Qy = 1 / (r * jnp.sqrt(jnp.pi)) * \
        jnp.exp(-3*(y - ym) ** 2 / (r ** 2))
    Qz = 1 / (d * jnp.sqrt(jnp.pi)) * \
        jnp.exp(-3*(z - zm) ** 2 / (d ** 2))

    return 6 * jnp.sqrt(3) * P * eta * Qx * Qy * Qz


@jax.jit
def interpolatePointsMatrix(x, y, z, cx, cy, cz, xn, yn, zn):
    ne_x = cx.shape[0]
    ne_y = cy.shape[0]
    ne_z = cz.shape[0]
    nn_x, nn_y, nn_z = ne_x + 1, ne_y + 1, ne_z + 1

    nn_xn = len(xn)
    nn_yn = len(yn)
    nn_zn = len(zn)
    nn2 = nn_xn * nn_yn * nn_zn
    h_x = x[1] - x[0]
    h_y = y[1] - y[0]
    h_z = z[1] - z[0]

    def stepInterpolatePoints(ielt):
        # Get nodal indices
        izn, _ = jnp.divmod(ielt, (nn_xn)*(nn_yn))
        iyn, _ = jnp.divmod(ielt, nn_xn)
        iyn -= izn * nn_yn
        ixn = jnp.mod(ielt, nn_xn)

        _x = xn[ixn, jnp.newaxis]
        _y = yn[iyn, jnp.newaxis]
        _z = zn[izn, jnp.newaxis]

        x_comp = (ne_x - 1) * jnp.ones_like(_x)
        y_comp = (ne_y - 1) * jnp.ones_like(_y)
        z_comp = (ne_z - 1) * jnp.ones_like(_z)

        x_comp2 = jnp.zeros_like(_x)
        y_comp2 = jnp.zeros_like(_y)
        z_comp2 = jnp.zeros_like(_z)

        # Figure out which coarse element we are in
        _floorx = jnp.floor((_x - x[0]) / h_x)
        _conx = jnp.concatenate((_floorx, x_comp))
        _ielt_x = jnp.min(_conx)
        _conx = jnp.concatenate((_ielt_x[jnp.newaxis], x_comp2))
        ielt_x = jnp.max(_conx).T.astype(int)

        _floory = jnp.floor((_y - y[0]) / h_y)
        _cony = jnp.concatenate((_floory, y_comp))
        _ielt_y = jnp.min(_cony)
        _cony = jnp.concatenate((_ielt_y[jnp.newaxis], y_comp2))
        ielt_y = jnp.max(_cony).T.astype(int)

        _floorz = jnp.floor((_z - z[0]) / h_z)
        _conz = jnp.concatenate((_floorz, z_comp))
        _ielt_z = jnp.min(_conz).T.astype(int)
        _conz = jnp.concatenate((_ielt_z[jnp.newaxis], z_comp2))
        ielt_z = jnp.max(_conz).T.astype(int)

        nodex = cx[ielt_x, :]
        nodey = cy[ielt_y, :]
        nodez = cz[ielt_z, :]
        node = nodex + nodey * nn_x + nodez * (nn_x * nn_y)

        xx = x[nodex]
        yy = y[nodey]
        zz = z[nodez]

        xc0 = xx[0]
        xc1 = xx[1]
        yc0 = yy[0]
        yc3 = yy[3]
        zc0 = zz[0]
        zc5 = zz[5]

        # Evaluate shape functions associated with coarse nodes
        Nc = jnp.concatenate([((xc1 - _x) / h_x * (yc3 - _y) / h_y * (zc5 - _z) / h_z),
                              ((_x - xc0) / h_x * (yc3 - _y) / h_y * (zc5 - _z) / h_z),
                              ((_x - xc0) / h_x * (_y - yc0) / h_y * (zc5 - _z) / h_z),
                              ((xc1 - _x) / h_x * (_y - yc0) / h_y * (zc5 - _z) / h_z),
                              ((xc1 - _x) / h_x * (yc3 - _y) / h_y * (_z - zc0) / h_z),
                              ((_x - xc0) / h_x * (yc3 - _y) / h_y * (_z - zc0) / h_z),
                              ((_x - xc0) / h_x * (_y - yc0) / h_y * (_z - zc0) / h_z),
                              ((xc1 - _x) / h_x * (_y - yc0) / h_y * (_z - zc0) / h_z)])
        Nc = Nc * (Nc >= -1e-2).all().astype(float) * (Nc <= 1+1e-2).all().astype(float)
        return Nc, node
    vstepInterpolatePoints = jax.vmap(stepInterpolatePoints)
    _Nc, _node = vstepInterpolatePoints(jnp.arange(nn2))
    return _Nc, _node

@jax.jit 
def interpolate_w_matrix(intmat, node, T):
    return jnp.multiply(intmat, T[node]).sum(axis=1)


def interpolatePoints_jax(x, y, z, cx, cy, cz,
                          u, xn, yn, zn):
    ne_x = cx.shape[0]
    ne_y = cy.shape[0]
    ne_z = cz.shape[0]
    nn_x, nn_y, nn_z = ne_x + 1, ne_y + 1, ne_z + 1

    nn_xn = len(xn)
    nn_yn = len(yn)
    nn_zn = len(zn)
    nn2 = nn_xn * nn_yn * nn_zn
    h_x = x[1] - x[0]
    h_y = y[1] - y[0]
    h_z = z[1] - z[0]

    def stepInterpolatePoints(ielt):
        # Get nodal indices
        izn, _ = jnp.divmod(ielt, (nn_xn)*(nn_yn))
        iyn, _ = jnp.divmod(ielt, nn_xn)
        iyn -= izn * nn_yn
        ixn = jnp.mod(ielt, nn_xn)

        _x = xn[ixn, jnp.newaxis]
        _y = yn[iyn, jnp.newaxis]
        _z = zn[izn, jnp.newaxis]

        x_comp = (ne_x - 1) * jnp.ones_like(_x)
        y_comp = (ne_y - 1) * jnp.ones_like(_y)
        z_comp = (ne_z - 1) * jnp.ones_like(_z)

        x_comp2 = jnp.zeros_like(_x)
        y_comp2 = jnp.zeros_like(_y)
        z_comp2 = jnp.zeros_like(_z)

        # Figure out which coarse element we are in
        _floorx = jnp.floor((_x - x[0]) / h_x)
        _conx = jnp.concatenate((_floorx, x_comp))
        _ielt_x = jnp.min(_conx)
        _conx = jnp.concatenate((_ielt_x[jnp.newaxis], x_comp2))
        ielt_x = jnp.max(_conx).T.astype(int)

        _floory = jnp.floor((_y - y[0]) / h_y)
        _cony = jnp.concatenate((_floory, y_comp))
        _ielt_y = jnp.min(_cony)
        _cony = jnp.concatenate((_ielt_y[jnp.newaxis], y_comp2))
        ielt_y = jnp.max(_cony).T.astype(int)

        _floorz = jnp.floor((_z - z[0]) / h_z)
        _conz = jnp.concatenate((_floorz, z_comp))
        _ielt_z = jnp.min(_conz).T.astype(int)
        _conz = jnp.concatenate((_ielt_z[jnp.newaxis], z_comp2))
        ielt_z = jnp.max(_conz).T.astype(int)

        nodex = cx[ielt_x, :]
        nodey = cy[ielt_y, :]
        nodez = cz[ielt_z, :]
        node = nodex + nodey * nn_x + nodez * (nn_x * nn_y)

        xx = x[nodex]
        yy = y[nodey]
        zz = z[nodez]

        xc0 = xx[0]
        xc1 = xx[1]
        yc0 = yy[0]
        yc3 = yy[3]
        zc0 = zz[0]
        zc5 = zz[5]

        # Evaluate shape functions associated with coarse nodes
        Nc = jnp.concatenate([((xc1 - _x) / h_x * (yc3 - _y) / h_y * (zc5 - _z) / h_z),
                              ((_x - xc0) / h_x * (yc3 - _y) / h_y * (zc5 - _z) / h_z),
                              ((_x - xc0) / h_x * (_y - yc0) / h_y * (zc5 - _z) / h_z),
                              ((xc1 - _x) / h_x * (_y - yc0) / h_y * (zc5 - _z) / h_z),
                              ((xc1 - _x) / h_x * (yc3 - _y) / h_y * (_z - zc0) / h_z),
                              ((_x - xc0) / h_x * (yc3 - _y) / h_y * (_z - zc0) / h_z),
                              ((_x - xc0) / h_x * (_y - yc0) / h_y * (_z - zc0) / h_z),
                              ((xc1 - _x) / h_x * (_y - yc0) / h_y * (_z - zc0) / h_z)])
        Nc = Nc * (Nc >= -1e-2).all().astype(float) * (Nc <= 1+1e-2).all().astype(float)
        return Nc @ u[node]
    vstepInterpolatePoints = jax.vmap(stepInterpolatePoints)
    return vstepInterpolatePoints(jnp.arange(nn2))


@jax.jit
def computeCoarseFineShapeFunctions(xnc_x, xnc_y, xnc_z,
                                    nconnc_x, nconnc_y, nconnc_z,
                                    xnf_x, xnf_y, xnf_z,
                                    nconnf_x, nconnf_y, nconnf_z):
    """ computeCoarseFineShapeFunctions finds the shape functions of
        the fine scale quadrature points for the coarse element
        :param xnc_x: nodal coordinates of global coarse in x 
        :param xnc_y: nodal coordinates of global coarse in y 
        :param xnc_z: nodal coordinates of global coarse in z 
        :param nconnc_x: indices to get x coordinates of nodes of coarse element 
        :param nconnc_y: indices to get x coordinates of nodes of coarse element 
        :param nconnc_z: indices to get x coordinates of nodes of coarse element 
        :param xnf_x: nodal coordinates of global fine in x 
        :param xnf_y: nodal coordinates of global fine in y 
        :param xnf_z: nodal coordinates of global fine in z 
        :param nconnf_x: indices to get x coordinates of nodes of fine element 
        :param nconnf_y: indices to get x coordinates of nodes of fine element 
        :param nconnf_z: indices to get x coordinates of nodes of fine element 
        :return Nc, dNcdx, dNcdy, dNcdz, _nodes.reshape(-1)
        :return Nc: (Num fine elements, 8 quadrature, 8), coarse shape function for fine element
        :return dNcdx: (Num fine elements, 8 quadrature, 8), coarse x-derivate shape function for fine element
        :return dNcdy: (Num fine elements, 8 quadrature, 8), coarse y-derivate shape function for fine element
        :return dNcdz: (Num fine elements, 8 quadrature, 8), coarse z-derivate shape function for fine element
        :return _nodes: (Num fine elements * 8 * 8), coarse nodal indices
    """
    # Get number of elements and nodes for both coarse and fine
    nec_x = nconnc_x.shape[0]
    nec_y = nconnc_y.shape[0]
    nec_z = nconnc_z.shape[0]
    nnc_x = xnc_x.shape[0]
    nnc_y = xnc_y.shape[0]
    nnc_z = xnc_z.shape[0]
    nnc = nnc_x * nnc_y * nnc_z
    nef_x = nconnf_x.shape[0]
    nef_y = nconnf_y.shape[0]
    nef_z = nconnf_z.shape[0]
    nef = nef_x * nef_y * nef_z
    nnf_x = xnf_x.shape[0]
    nnf_y = xnf_y.shape[0]

    # Assume constant mesh sizes
    hc_x = xnc_x[1] - xnc_x[0]
    hc_y = xnc_y[1] - xnc_y[0]
    hc_z = xnc_z[1] - xnc_z[0]

    # Get lower bounds of meshes
    xminc_x = xnc_x[0]
    xminc_y = xnc_y[0]
    xminc_z = xnc_z[0]

    # Get shape functions and weights
    coords_x = xnf_x[nconnf_x[0, :]].reshape(-1, 1)
    coords_y = xnf_y[nconnf_y[0, :]].reshape(-1, 1)
    coords_z = xnf_z[nconnf_z[0, :]].reshape(-1, 1)
    coords = jnp.concatenate([coords_x, coords_y, coords_z], axis=1)
    Nf, dNdxf, wqf = computeQuad3dFemShapeFunctions_jax(coords)

    def stepComputeCoarseFineTerm(ieltf):
        ix, iy, iz, idx = convert2XYZ(ieltf, nef_x, nef_y, nnf_x, nnf_y)
        coords_x = xnf_x[nconnf_x[ix, :]].reshape(-1, 1)
        coords_y = xnf_y[nconnf_y[iy, :]].reshape(-1, 1)
        coords_z = xnf_z[nconnf_z[iz, :]].reshape(-1, 1)

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
            nodec_x = nconnc_x[ieltc_x[iq], :].astype(int)
            nodec_y = nconnc_y[ieltc_y[iq], :].astype(int)
            nodec_z = nconnc_z[ieltc_z[iq], :].astype(int)
            nodes = nodec_x + nodec_y * nnc_x + nodec_z * nnc_x * nnc_y

            _x = x[iq]
            _y = y[iq]
            _z = z[iq]

            xc0 = xnc_x[nconnc_x[ieltc_x[iq], 0]]
            xc1 = xnc_x[nconnc_x[ieltc_x[iq], 1]]
            yc0 = xnc_y[nconnc_y[ieltc_y[iq], 0]]
            yc3 = xnc_y[nconnc_y[ieltc_y[iq], 3]]
            zc0 = xnc_z[nconnc_z[ieltc_z[iq], 0]]
            zc5 = xnc_z[nconnc_z[ieltc_z[iq], 5]]

            # Evaluate shape functions associated with coarse nodes
            Nc = jnp.array([((xc1 - _x) / hc_x * (yc3 - _y) / hc_y * (zc5 - _z) / hc_z),
                            ((_x - xc0) / hc_x * (yc3 - _y) / hc_y * (zc5 - _z) / hc_z),
                            ((_x - xc0) / hc_x * (_y - yc0) / hc_y * (zc5 - _z) / hc_z),
                            ((xc1 - _x) / hc_x * (_y - yc0) / hc_y * (zc5 - _z) / hc_z),
                            ((xc1 - _x) / hc_x * (yc3 - _y) / hc_y * (_z - zc0) / hc_z),
                            ((_x - xc0) / hc_x * (yc3 - _y) / hc_y * (_z - zc0) / hc_z),
                            ((_x - xc0) / hc_x * (_y - yc0) / hc_y * (_z - zc0) / hc_z),
                            ((xc1 - _x) / hc_x * (_y - yc0) / hc_y * (_z - zc0) / hc_z)])

            # Evaluate shape functions associated with coarse nodes
            dNcdx = jnp.array([((-1) / hc_x * (yc3 - _y) / hc_y * (zc5 - _z) / hc_z),
                               ((1) / hc_x * (yc3 - _y) / hc_y * (zc5 - _z) / hc_z),
                               ((1) / hc_x * (_y - yc0) / hc_y * (zc5 - _z) / hc_z),
                               ((-1) / hc_x * (_y - yc0) / hc_y * (zc5 - _z) / hc_z),
                               ((-1) / hc_x * (yc3 - _y) / hc_y * (_z - zc0) / hc_z),
                               ((1) / hc_x * (yc3 - _y) / hc_y * (_z - zc0) / hc_z),
                               ((1) / hc_x * (_y - yc0) / hc_y * (_z - zc0) / hc_z),
                               ((-1) / hc_x * (_y - yc0) / hc_y * (_z - zc0) / hc_z)])
            # Evaluate shape functions associated with coarse nodes
            dNcdy = jnp.array([((xc1 - _x) / hc_x * (-1) / hc_y * (zc5 - _z) / hc_z),
                               ((_x - xc0) / hc_x * (-1) / hc_y * (zc5 - _z) / hc_z),
                               ((_x - xc0) / hc_x * (1) / hc_y * (zc5 - _z) / hc_z),
                               ((xc1 - _x) / hc_x * (1) / hc_y * (zc5 - _z) / hc_z),
                               ((xc1 - _x) / hc_x * (-1) / hc_y * (_z - zc0) / hc_z),
                               ((_x - xc0) / hc_x * (-1) / hc_y * (_z - zc0) / hc_z),
                               ((_x - xc0) / hc_x * (1) / hc_y * (_z - zc0) / hc_z),
                               ((xc1 - _x) / hc_x * (1) / hc_y * (_z - zc0) / hc_z)])
            # Evaluate shape functions associated with coarse nodes
            dNcdz = jnp.array([((xc1 - _x) / hc_x * (yc3 - _y) / hc_y * (-1) / hc_z),
                               ((_x - xc0) / hc_x * (yc3 - _y) / hc_y * (-1) / hc_z),
                               ((_x - xc0) / hc_x * (_y - yc0) / hc_y * (-1) / hc_z),
                               ((xc1 - _x) / hc_x * (_y - yc0) / hc_y * (-1) / hc_z),
                               ((xc1 - _x) / hc_x * (yc3 - _y) / hc_y * (1) / hc_z),
                               ((_x - xc0) / hc_x * (yc3 - _y) / hc_y * (1) / hc_z),
                               ((_x - xc0) / hc_x * (_y - yc0) / hc_y * (1) / hc_z),
                               ((xc1 - _x) / hc_x * (_y - yc0) / hc_y * (1) / hc_z)])
            return Nc, dNcdx, dNcdy, dNcdz, nodes
        viqLoopMass = jax.vmap(iqLoopMass)
        return viqLoopMass(jnp.arange(8))
    vstepComputeCoarseFineTerm = jax.vmap(
        stepComputeCoarseFineTerm)
    Nc, dNcdx, dNcdy, dNcdz, _nodes = vstepComputeCoarseFineTerm(jnp.arange(nef))
    _nodes = _nodes[:,0,:]
    indices = jnp.concatenate([_nodes.reshape(-1,1),jnp.arange(_nodes.size).reshape(-1,1)], axis = 1)
    test = jax.experimental.sparse.BCOO([jnp.ones(_nodes.size),indices],shape=(nnc,_nodes.size))
    return Nc, dNcdx, dNcdy, dNcdz, test
    
@partial(jax.jit, static_argnames=['nn1','nn2'])
def computeCoarseTprimeMassTerm_jax(xnf_x, xnf_y, xnf_z, # sub2
                                xnm_x, xnm_y, xnm_z, # sub1
                                nconnf_x, nconnf_y, nconnf_z, # sub2
                                nconnm_x, nconnm_y, nconnm_z, # sub1
                                Tprimef, Tprimef0,
                                Tprimem, Tprimem0,
                                rho, cp, dt,
                                Nc31, Nc21, Nc32,
                                nodes31, nodes21, nodes32,
                                Vcu, Vmu, nn1, nn2):

    Tprimef_new = Tprimef - Tprimef0
    Tprimem_new = Tprimem - Tprimem0

    # Level 3
    nef_x = nconnf_x.shape[0]
    nef_y = nconnf_y.shape[0]
    nef_z = nconnf_z.shape[0]
    nef = nef_x * nef_y * nef_z
    nnf_x = xnf_x.shape[0]
    nnf_y = xnf_y.shape[0]

    # Level 3 Get shape functions and weights
    coordsf_x = xnf_x[nconnf_x[0, :]].reshape(-1, 1)
    coordsf_y = xnf_y[nconnf_y[0, :]].reshape(-1, 1)
    coordsf_z = xnf_z[nconnf_z[0, :]].reshape(-1, 1)
    coordsf = jnp.concatenate([coordsf_x, coordsf_y, coordsf_z], axis=1)
    Nf, dNdxf, wqf = computeQuad3dFemShapeFunctions_jax(coordsf)

    # Level 3
    _, _, _, idxf = convert2XYZ(jnp.arange(nef), nef_x, nef_y, nnf_x, nnf_y)
    _Tprimef = Nf @ Tprimef_new[idxf]
    _data1 = jnp.multiply(jnp.multiply(-Nc31,_Tprimef.T[:,:,jnp.newaxis]),
                         (rho * cp / dt) * wqf[jnp.newaxis, jnp.newaxis, :]).sum(axis=2)
    _data2 = jnp.multiply(jnp.multiply(-Nc32,_Tprimef.T[:,:,jnp.newaxis]),
                         (rho * cp / dt) * wqf[jnp.newaxis, jnp.newaxis, :]).sum(axis=2)
    
    # Level 2
    nem_x = nconnm_x.shape[0]
    nem_y = nconnm_y.shape[0]
    nem_z = nconnm_z.shape[0]
    nem = nem_x * nem_y * nem_z
    nnm_x = xnm_x.shape[0]
    nnm_y = xnm_y.shape[0]

    # Level 2 Get shape functions and weights
    coordsm_x = xnm_x[nconnm_x[0, :]].reshape(-1, 1)
    coordsm_y = xnm_y[nconnm_y[0, :]].reshape(-1, 1)
    coordsm_z = xnm_z[nconnm_z[0, :]].reshape(-1, 1)
    coordsm = jnp.concatenate([coordsm_x, coordsm_y, coordsm_z], axis=1)
    Nm, dNdxm, wqm = computeQuad3dFemShapeFunctions_jax(coordsm)

    # Level 2
    _, _, _, idxm = convert2XYZ(jnp.arange(nem), nem_x, nem_y, nnm_x, nnm_y)
    _Tprimem = Nm @ Tprimem_new[idxm]
    _data3 = jnp.multiply(jnp.multiply(-Nc21,_Tprimem.T[:,:,jnp.newaxis]),
                         (rho * cp / dt) * wqm[jnp.newaxis, jnp.newaxis, :]).sum(axis=2)
    
    Vcu += nodes31 @ _data1.reshape(-1) + nodes21 @ _data3.reshape(-1)
    Vmu += nodes32 @ _data2.reshape(-1)

    return Vcu, Vmu

@partial(jax.jit, static_argnames=['nn1','nn2'])
def computeCoarseTprimeTerm_jax(xnf_x, xnf_y, xnf_z, # sub2
                                xnm_x, xnm_y, xnm_z, # sub1
                                nconnf_x, nconnf_y, nconnf_z, # sub2
                                nconnm_x, nconnm_y, nconnm_z, # sub1
                                Tprimef, Tprimem, # sub2, sub1
                                k,
                                dNc31dx, dNc31dy, dNc31dz, nodes31, # sub2 Level1
                                dNc21dx, dNc21dy, dNc21dz, nodes21, # sub1 Level1
                                dNc32dx, dNc32dy, dNc32dz, nodes32, # sub2 Level2
                                nn1, nn2):
    # Level 3
    nef_x = nconnf_x.shape[0]
    nef_y = nconnf_y.shape[0]
    nef_z = nconnf_z.shape[0]
    nef = nef_x * nef_y * nef_z
    nnf_x = xnf_x.shape[0]
    nnf_y = xnf_y.shape[0]

    # Level 3 Get shape functions and weights
    coordsf_x = xnf_x[nconnf_x[0, :]].reshape(-1, 1)
    coordsf_y = xnf_y[nconnf_y[0, :]].reshape(-1, 1)
    coordsf_z = xnf_z[nconnf_z[0, :]].reshape(-1, 1)
    coordsf = jnp.concatenate([coordsf_x, coordsf_y, coordsf_z], axis=1)
    Nf, dNdxf, wqf = computeQuad3dFemShapeFunctions_jax(coordsf)

    # Level 3
    # idxf: (8, nef), indexing in Tprimef for later shape function use
    _, _, _, idxf = convert2XYZ(jnp.arange(nef), nef_x, nef_y, nnf_x, nnf_y)
    _Tprimef = Tprimef[idxf]
    dTprimefdx = dNdxf[:, :, 0] @ _Tprimef
    dTprimefdy = dNdxf[:, :, 1] @ _Tprimef
    dTprimefdz = dNdxf[:, :, 2] @ _Tprimef

    _data1 = jnp.multiply(jnp.multiply(-dNc31dx,dTprimefdx.T[:,:,jnp.newaxis]), k * wqf[jnp.newaxis, jnp.newaxis, :]).sum(axis=2)
    _data1 += jnp.multiply(jnp.multiply(-dNc31dy,dTprimefdy.T[:,:,jnp.newaxis]), k * wqf[jnp.newaxis, jnp.newaxis, :]).sum(axis=2)
    _data1 += jnp.multiply(jnp.multiply(-dNc31dz,dTprimefdz.T[:,:,jnp.newaxis]), k * wqf[jnp.newaxis, jnp.newaxis, :]).sum(axis=2)

    _data2 = jnp.multiply(jnp.multiply(-dNc32dx,dTprimefdx.T[:,:,jnp.newaxis]), k * wqf[jnp.newaxis, jnp.newaxis, :]).sum(axis=2)
    _data2 += jnp.multiply(jnp.multiply(-dNc32dy,dTprimefdy.T[:,:,jnp.newaxis]), k * wqf[jnp.newaxis, jnp.newaxis, :]).sum(axis=2)
    _data2 += jnp.multiply(jnp.multiply(-dNc32dz,dTprimefdz.T[:,:,jnp.newaxis]), k * wqf[jnp.newaxis, jnp.newaxis, :]).sum(axis=2)

    # Level 2
    nem_x = nconnm_x.shape[0]
    nem_y = nconnm_y.shape[0]
    nem_z = nconnm_z.shape[0]
    nem = nem_x * nem_y * nem_z
    nnm_x = xnm_x.shape[0]
    nnm_y = xnm_y.shape[0]

    # Level 2 Get shape functions and weights
    coordsm_x = xnm_x[nconnm_x[0, :]].reshape(-1, 1)
    coordsm_y = xnm_y[nconnm_y[0, :]].reshape(-1, 1)
    coordsm_z = xnm_z[nconnm_z[0, :]].reshape(-1, 1)
    coordsm = jnp.concatenate([coordsm_x, coordsm_y, coordsm_z], axis=1)
    Nm, dNdxm, wqm = computeQuad3dFemShapeFunctions_jax(coordsm)

    # Level 2
    _, _, _, idxm = convert2XYZ(jnp.arange(nem), nem_x, nem_y, nnm_x, nnm_y)
    _Tprimem = Tprimem[idxm]
    dTprimemdx = dNdxm[:, :, 0] @ _Tprimem
    dTprimemdy = dNdxm[:, :, 1] @ _Tprimem
    dTprimemdz = dNdxm[:, :, 2] @ _Tprimem

    _data3 = jnp.multiply(jnp.multiply(-dNc21dx,dTprimemdx.T[:,:,jnp.newaxis]), k * wqm[jnp.newaxis, jnp.newaxis, :]).sum(axis=2)
    _data3 += jnp.multiply(jnp.multiply(-dNc21dy,dTprimemdy.T[:,:,jnp.newaxis]), k * wqm[jnp.newaxis, jnp.newaxis, :]).sum(axis=2)
    _data3 += jnp.multiply(jnp.multiply(-dNc21dz,dTprimemdz.T[:,:,jnp.newaxis]), k * wqm[jnp.newaxis, jnp.newaxis, :]).sum(axis=2)
    
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
    sidx = jnp.arange(0, nx)[:, jnp.newaxis] + \
        (nx * ny * jnp.arange(0, nz))[jnp.newaxis, :]
    sidx = sidx.reshape(-1)
    nidx = jnp.arange(
        nx * (ny - 1), nx * ny)[:, jnp.newaxis] + (nx * ny * jnp.arange(0, nz))[jnp.newaxis, :]
    nidx = nidx.reshape(-1)
    return widx, eidx, sidx, nidx, bidx, tidx


@jax.jit
def assignBCs(RHS, TS, TN, TW, TE, TB, TT, widx, eidx, sidx, nidx, bidx, tidx):
    _RHS = RHS
    _RHS = _RHS.at[sidx].set(TS)
    _RHS = _RHS.at[nidx].set(TN)
    _RHS = _RHS.at[widx].set(TW)
    _RHS = _RHS.at[eidx].set(TE)
    _RHS = _RHS.at[bidx].set(TB)
    _RHS = _RHS.at[tidx].set(TT)
    return _RHS


@jax.jit
def assignBCsFine(RHS, TfAll, widx, eidx, sidx, nidx, bidx, tidx):
    _RHS = RHS
    _RHS = _RHS.at[sidx].set(TfAll[sidx])
    _RHS = _RHS.at[nidx].set(TfAll[nidx])
    _RHS = _RHS.at[widx].set(TfAll[widx])
    _RHS = _RHS.at[eidx].set(TfAll[eidx])
    _RHS = _RHS.at[bidx].set(TfAll[bidx])
    _RHS = _RHS.at[tidx].set(TfAll[tidx])
    return _RHS

@partial(jax.jit, static_argnames=['nn'])
def bincount(N, D, nn):
    return jnp.bincount(N, D, length=nn)

@jax.jit
def getOverlapRegion(x, y, z, nx, ny):
    _x = jnp.tile(x, y.shape[0] * z.shape[0]).reshape(-1)
    _y = jnp.repeat(jnp.tile(y, z.shape[0]), x.shape[0]).reshape(-1)
    _z = jnp.repeat(z, x.shape[0] * y.shape[0])
    return _x + _y * nx + _z * nx * ny

@jax.jit
def jit_constrain_v(vx, vy, vz, iE, iN, iT, iW, iS, iB):
    vx = jnp.minimum(vx,iE)
    vy = jnp.minimum(vy,iN)
    vz = jnp.minimum(vz,iT)
    vx = jnp.maximum(vx,iW)
    vy = jnp.maximum(vy,iS)
    vz = jnp.maximum(vz,iB)
    return vx, vy, vz

@jax.jit
def move_fine_mesh(x,y,z,hx,hy,hz,vx,vy,vz):
    vx_ = jnp.round(vx / hx)
    vy_ = jnp.round(vy / hy)
    vz_ = jnp.round(vz / hz)
    xnf_x = x + hx * vx_
    xnf_y = y + hy * vy_
    xnf_z = z + hz * vz_
    return xnf_x, xnf_y, xnf_z, vx_.astype(int), vy_.astype(int), vz_.astype(int)

@jax.jit
def update_overlap_nodes_coords(overlapNodes_x_orig, overlapNodes_y_orig, overlapNodes_z_orig,
    overlapCoords_x_orig, overlapCoords_y_orig, overlapCoords_z_orig,
    vx_tot_con, vy_tot_con, vz_tot_con, hc_x, hc_y, hc_z):
    overlapNodes_x = overlapNodes_x_orig + jnp.round(vx_tot_con / hc_x).astype(int)
    overlapNodes_y = overlapNodes_y_orig + jnp.round(vy_tot_con / hc_y).astype(int)
    overlapNodes_z = overlapNodes_z_orig + jnp.round(vz_tot_con / hc_z).astype(int)
    overlapCoords_x = overlapCoords_x_orig + hc_x * jnp.round(vx_tot_con / hc_x)
    overlapCoords_y = overlapCoords_y_orig + hc_y * jnp.round(vy_tot_con / hc_y)
    overlapCoords_z = overlapCoords_z_orig + hc_z * jnp.round(vz_tot_con / hc_z)
    return overlapNodes_x, overlapNodes_y, overlapNodes_z, overlapCoords_x, overlapCoords_y, overlapCoords_z

def add_vectors(a, b):
    return a + b

@partial(jax.jit, static_argnames=['_idx', '_val'])
def substitute_Tbar(Tbar, _idx, _val):
    return Tbar.at[_idx:].set(_val)

@jax.jit
def substitute_Tbar2(Tbar, _idx, _val):
    return Tbar.at[_idx].set(_val)

def find_max_const(full, sub):
    # Used to find the maximum number of elements the subdomain can move
    iE = full.bounds.x[1] - sub.bounds.x[1] # Number of elements to east
    iN = full.bounds.y[1] - sub.bounds.y[1] # Number of elements to north
    iT = full.bounds.z[1] - sub.bounds.z[1] # Number of elements to top
    iW = full.bounds.x[0] - sub.bounds.x[0] # Number of elements to west
    iS = full.bounds.y[0] - sub.bounds.y[0] # Number of elements to south
    iB = full.bounds.z[0] - sub.bounds.z[0] # Number of elements to bottom
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

def save_result(full, save_str, record_lab, save_path):
    vtkcx = np.array(full.node_coords[0])
    vtkcy = np.array(full.node_coords[1])
    vtkcz = np.array(full.node_coords[2])
    vtkT = np.array(full.T0.reshape(full.nodes[2],
                                    full.nodes[1],
                                    full.nodes[0])).transpose((2,1,0))
    gridToVTK(save_path+save_str+str(record_lab).zfill(8),
            vtkcx, vtkcy, vtkcz, pointData = {"Temperature (K)" : vtkT})

@jax.jit
def getNewTprime(lnc0, lnc1, lnc2, lcon0, lcon1, lcon2, lT0, 
                 lov0, lov1, lov2, lovn0, lovn1, lovn2, uT,
                 un0, un1, ulmat, ulnode, lumat, lunode):
    # l: lower (e.g. Level3, sub2)
    # u: upper (e.g. Level2, sub1)
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
    _val = interpolatePoints_jax(lnc0,lnc1,lnc2,lcon0,lcon1,lcon2,
                                lT0,lov0,lov1,lov2)
    _idx = getOverlapRegion(lovn0,lovn1,lovn2,
                            un0,un1)
    # Directly substitute into T0 to save deepcopy
    uT = substitute_Tbar2(uT, _idx, _val)
    return Tprime, uT

@jax.jit
def getBothNewTprimes(lnc0, lnc1, lnc2, lcon0, lcon1, lcon2, lT0, 
                 lov0, lov1, lov2, lovn0, lovn1, lovn2, mT,
                 mn0, mn1, mlmat, mlnode, lmmat, lmnode,
                 mnc0, mnc1, mnc2, mcon0, mcon1, mcon2,
                 mov0, mov1, mov2, movn0, movn1, movn2, uT,
                 un0, un1, ummat, umnode, mumat, munode):
    lTprime, mT0 = getNewTprime(lnc0, lnc1, lnc2, lcon0, lcon1, lcon2, lT0, 
                 lov0, lov1, lov2, lovn0, lovn1, lovn2, mT,
                 mn0, mn1, mlmat, mlnode, lmmat, lmnode)
    mTprime, uT0 = getNewTprime(mnc0, mnc1, mnc2, mcon0, mcon1, mcon2, mT0, 
                 mov0, mov1, mov2, movn0, movn1, movn2, uT,
                 un0, un1, ummat, umnode, mumat, munode)
    return lTprime, mTprime, mT0, uT0

@partial(jax.jit, static_argnames=['fullnn','fulltmpne','fulltmpnn',
                                   'sub1nn','sub1ne', 'sub2nn', 'sub2ne'])
def computeSolutions(fullnc0, fullnc1, fullnc2,
                     fullcon0, fullcon1, fullcon2,
                     fullnn, fulltmpne, fulltmpnn, fullT0, fullF, fullV,
                     fullx0, fullx1, fully0, fully1, fullz0, fullz1,
                     fullwi, fullei, fullsi, fullni, fullbi, fullti,
                     fullsub1_intmat, fullsub1_node,
                     sub1nc0, sub1nc1, sub1nc2,
                     sub1con0, sub1con1, sub1con2,
                     sub1nn, sub1ne, sub1T0, sub1F, sub1V,
                     sub1wi, sub1ei, sub1si, sub1ni, sub1bi, sub1ti,
                     sub1sub2_intmat, sub1sub2_node,
                     sub2nc0, sub2nc1, sub2nc2,
                     sub2con0, sub2con1, sub2con2,
                     sub2nn, sub2ne, sub2T0, sub2F,
                     sub2wi, sub2ei, sub2si, sub2ni, sub2bi, sub2ti,
                     k, rho, cp, dt, T_amb):
    fullT = meshlessKM(fullnc0, fullnc1, fullnc2,
                    fullcon0, fullcon1, fullcon2,
                    fullnn, fulltmpne,
                    k, rho, cp, dt, fullT0, fullF, fullV)
    fullT = substitute_Tbar(fullT, fulltmpnn, T_amb)
    FinalFull = assignBCs(fullT, fully0, fully1,
                       fullx0, fullx1,
                       fullz0, fullz1,
                       fullwi, fullei,
                       fullsi, fullni,
                       fullbi, fullti)

    # Compute source term for medium scale problem using fine mesh
    TfAll = interpolate_w_matrix(fullsub1_intmat, fullsub1_node, FinalFull)
    # Avoids assembling LHS matrix
    sub1T = meshlessKM(sub1nc0, sub1nc1, sub1nc2,
                    sub1con0, sub1con1, sub1con2,
                    sub1nn, sub1ne,
                    k, rho, cp, dt, sub1T0, sub1F, sub1V)
    FinalSub1 = assignBCsFine(sub1T, TfAll,
                        sub1wi, sub1ei,
                        sub1si, sub1ni,
                        sub1bi, sub1ti)

    # Use sub1.T to get Dirichlet BCs for fine-scale solution
    TfAll = interpolate_w_matrix(sub1sub2_intmat, sub1sub2_node, FinalSub1)
    FinalSub2 = meshlessKM(sub2nc0, sub2nc1, sub2nc2,
                    sub2con0, sub2con1, sub2con2,
                    sub2nn, sub2ne,
                    k, rho, cp, dt, sub2T0, sub2F, 0)
    FinalSub2 = assignBCsFine(FinalSub2,
                            TfAll,
                            sub2wi, sub2ei,
                            sub2si, sub2ni,
                            sub2bi, sub2ti)
    return FinalFull, FinalSub1, FinalSub2

@partial(jax.jit, static_argnames=['ne','nn'])
def computeConvectionRadiation(xyz, c, T, ne, nn, T_amb, h_conv, sigma_sb, emissivity, F):
    # Stefan-Boltzmann constant: 5.67e-8 [W/m^2/K^4]
    # Emissivity and Heat transfer coefficient are unknown and not readily measurable, require calibration
    # Heat transfer cofficient by convection: [W/m^2 k]
    # Emissivity of the powder bed: unitless
    # Equation is k grad(T) = h_conv * (T - T_amb) + sigma_sb * emissivity * (T**4 - T_amb**4)
    # This would be applied to "finest" levels
    h_conv = h_conv / 1e6 # convert to W/mm^2K
    sigma_sb = sigma_sb / 1e6 # convert to W/mm^2/K^4

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
    aT, aidx = vcalcCR(jnp.arange(top_ne,ne))
    NeumannBC = jnp.bincount(aidx.reshape(-1), aT.reshape(-1), length=nn)

    # Returns k grad(T) integral, which is Neumann BC (expect ambient < body)
    return F + NeumannBC

@partial(jax.jit, static_argnames=['nn1','nn2','nn3','tmp_ne','tmp_nn','sub1ne','sub2ne'])
def doExplicitTimestep(sub2nc, sub2con, sub2ne, sub2oC, sub2oN,
                       sub1nc, sub1con, sub1ne, sub1no, sub1oC, sub1oN,
                       fullnc, fullcon, fullno, fullcondx, fullcondy, fullcondz,
                       sub2widx, sub2eidx, sub2sidx, sub2nidx, sub2bidx, sub2tidx,
                       sub1widx, sub1eidx, sub1sidx, sub1nidx, sub1bidx, sub1tidx,
                       fullwidx, fulleidx, fullsidx, fullnidx, fullbidx, fulltidx,
                       tmp_ne, tmp_nn,
                       nn1, nn2, nn3,
                       sub2Tp0, sub1Tp0,
                       sub2T0, sub1T0, fullT0,
                       sub2NcLevel1, sub2NcLevel2, sub1NcLevel1,
                       sub2dNcdxLevel1, sub2dNcdyLevel1, sub2dNcdzLevel1, sub2nodesLevel1,
                       sub2dNcdxLevel2, sub2dNcdyLevel2, sub2dNcdzLevel2, sub2nodesLevel2,
                       sub1dNcdxLevel1, sub1dNcdyLevel1, sub1dNcdzLevel1, sub1nodesLevel1,
                       sub1sub2_intmat, sub1sub2_node, sub2sub1_intmat, sub2sub1_node,
                       fullsub1_intmat, fullsub1_node, sub1full_intmat, sub1full_node,
                       _t, v, k, rho, cp, dt, laserr, laserd, laserP, lasereta, T_amb, h_conv, vareps):
    """ doExplicitTimestep computes a full explicit timestep starting by
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
        :param condx, condy, condz: Dirichlet boundary conditions for full mesh
        :param widx, eidx, sidx...: indices for boundary surface nodes
        :param tmp_ne, tmp_nn: number of elements and nodes active on full mesh (based on layer)
        :param nn1, nn2, nn3: total number of nodes (active and deactive)
        :param Tp0: previous correction term for Tprime
        :param T0: previous temperature
        :param sub2NcLevel1: shape functions from sub2 to Level1
        :param sub2dNcdxLevel1: derivative of shape functions from sub2 to Level1
        :param sub2nodesLevel1: node indices for sub2 to Level 1 Tprime calculations
        :param sub1sub2_intmat: interpolation matrix from sub1 to sub2
        :param sub1sub2_node: nodes indexing into sub1 for interpolation
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
        :return sub2T0, sub1T0, fullT0, sub2Tp0, sub1Tp0
        :return sub2T0: Temperature for Level 3 (assigned back to previous temperature)
        :return sub1T0: Temperature for Level 2 (assigned back to previous temperature)
        :return fullT0: Temperature for Level 1 (assigned back to previous temperature)
        :return sub2Tp0: Temperature correction from Level 3 (assigned to previous)
        :return sub1Tp0: Temperature correction from Level 2 (assigned to previous)
    """
    Fc, Fm, Ff = computeSources(sub2nc[0],
                                sub2nc[1],
                                sub2nc[2],
                                sub2con[0],
                                sub2con[1],
                                sub2con[2],
                                _t, v,
                                sub2NcLevel1,
                                sub2NcLevel2,
                                sub2nodesLevel1,
                                sub2nodesLevel2,
                                nn1,
                                nn2,
                                nn3,
                                laserr,
                                laserd,
                                laserP,
                                lasereta)
    Fc = computeConvectionRadiation(fullnc, fullcon, fullT0, tmp_ne, nn1, T_amb, h_conv, 5.67e-8, vareps, Fc)
    Fm = computeConvectionRadiation(sub1nc, sub1con, sub1T0, sub1ne, nn2, T_amb, h_conv, 5.67e-8, vareps, Fm)
    Ff = computeConvectionRadiation(sub2nc, sub2con, sub2T0, sub2ne, nn3, T_amb, h_conv, 5.67e-8, vareps, Ff)
    Vcu, Vmu = computeCoarseTprimeTerm_jax(
                        sub2nc[0],
                        sub2nc[1],
                        sub2nc[2],
                        sub1nc[0],
                        sub1nc[1],
                        sub1nc[2],
                        sub2con[0],
                        sub2con[1],
                        sub2con[2],
                        sub1con[0],
                        sub1con[1],
                        sub1con[2],
                        sub2Tp0,
                        sub1Tp0,
                        k,
                        sub2dNcdxLevel1, 
                        sub2dNcdyLevel1,
                        sub2dNcdzLevel1,
                        sub2nodesLevel1,
                        sub1dNcdxLevel1, 
                        sub1dNcdyLevel1,
                        sub1dNcdzLevel1,
                        sub1nodesLevel1,
                        sub2dNcdxLevel2, 
                        sub2dNcdyLevel2,
                        sub2dNcdzLevel2,
                        sub2nodesLevel2,
                        nn1,
                        nn2)
    fullT, sub1T, sub2T = computeSolutions(
                            fullnc[0],
                            fullnc[1],
                            fullnc[2],
                            fullcon[0],
                            fullcon[1],
                            fullcon[2],
                            nn1,
                            tmp_ne,
                            tmp_nn,
                            fullT0,
                            Fc,
                            Vcu,
                            fullcondx[0],
                            fullcondx[1],
                            fullcondy[0],
                            fullcondy[1],
                            fullcondz[0],
                            fullcondz[1],
                            fullwidx,
                            fulleidx,
                            fullsidx,
                            fullnidx,
                            fullbidx,
                            fulltidx,
                            fullsub1_intmat,
                            fullsub1_node,
                            sub1nc[0],
                            sub1nc[1],
                            sub1nc[2],
                            sub1con[0],
                            sub1con[1],
                            sub1con[2],
                            nn2,
                            sub1ne,
                            sub1T0,
                            Fm,
                            Vmu,
                            sub1widx,
                            sub1eidx,
                            sub1sidx,
                            sub1nidx,
                            sub1bidx,
                            sub1tidx,
                            sub1sub2_intmat,
                            sub1sub2_node,
                            sub2nc[0],
                            sub2nc[1],
                            sub2nc[2],
                            sub2con[0],
                            sub2con[1],
                            sub2con[2],
                            nn3,
                            sub2ne,
                            sub2T0,
                            Ff,
                            sub2widx,
                            sub2eidx,
                            sub2sidx,
                            sub2nidx,
                            sub2bidx,
                            sub2tidx,
                            k, rho, cp, dt, T_amb)
    sub2Tp, sub1Tp, sub1T, fullT = getBothNewTprimes(
                            sub2nc[0],
                            sub2nc[1],
                            sub2nc[2],
                            sub2con[0],
                            sub2con[1],
                            sub2con[2],
                            sub2T, 
                            sub2oC[0],
                            sub2oC[1],
                            sub2oC[2],
                            sub2oN[0],
                            sub2oN[1],
                            sub2oN[2],
                            sub1T,
                            sub1no[0],
                            sub1no[1],
                            sub1sub2_intmat,
                            sub1sub2_node,
                            sub2sub1_intmat,
                            sub2sub1_node,
                            sub1nc[0],
                            sub1nc[1],
                            sub1nc[2],
                            sub1con[0],
                            sub1con[1],
                            sub1con[2],
                            sub1oC[0],
                            sub1oC[1],
                            sub1oC[2],
                            sub1oN[0],
                            sub1oN[1],
                            sub1oN[2],
                            fullT,
                            fullno[0],
                            fullno[1],
                            fullsub1_intmat,
                            fullsub1_node,
                            sub1full_intmat,
                            sub1full_node)
    Vcu, Vmu = computeCoarseTprimeMassTerm_jax(
        sub2nc[0],
        sub2nc[1],
        sub2nc[2],
        sub1nc[0],
        sub1nc[1],
        sub1nc[2],
        sub2con[0],
        sub2con[1],
        sub2con[2],
        sub1con[0],
        sub1con[1],
        sub1con[2],
        sub2Tp,
        sub2Tp0,
        sub1Tp,
        sub1Tp0,
        rho, cp, dt,
        sub2NcLevel1,
        sub1NcLevel1,
        sub2NcLevel2,
        sub2nodesLevel1,
        sub1nodesLevel1,
        sub2nodesLevel2,
        Vcu,
        Vmu,
        nn1,
        nn2)
    fullT, sub1T, sub2T0 = computeSolutions(
        fullnc[0],
        fullnc[1],
        fullnc[2],
        fullcon[0],
        fullcon[1],
        fullcon[2],
        nn1,
        tmp_ne,
        tmp_nn,
        fullT0,
        Fc,
        Vcu,
        fullcondx[0],
        fullcondx[1],
        fullcondy[0],
        fullcondy[1],
        fullcondz[0],
        fullcondz[1],
        fullwidx,
        fulleidx,
        fullsidx,
        fullnidx,
        fullbidx,
        fulltidx,
        fullsub1_intmat,
        fullsub1_node,
        sub1nc[0],
        sub1nc[1],
        sub1nc[2],
        sub1con[0],
        sub1con[1],
        sub1con[2],
        nn2,
        sub1ne,
        sub1T0,
        Fm,
        Vmu,
        sub1widx,
        sub1eidx,
        sub1sidx,
        sub1nidx,
        sub1bidx,
        sub1tidx,
        sub1sub2_intmat,
        sub1sub2_node,
        sub2nc[0],
        sub2nc[1],
        sub2nc[2],
        sub2con[0],
        sub2con[1],
        sub2con[2],
        nn3,
        sub2ne,
        sub2T0,
        Ff,
        sub2widx,
        sub2eidx,
        sub2sidx,
        sub2nidx,
        sub2bidx,
        sub2tidx,
        k, rho, cp, dt, T_amb)
    sub2Tp0, sub1Tp0, sub1T0, fullT0 = getBothNewTprimes(
            sub2nc[0],
            sub2nc[1],
            sub2nc[2],
            sub2con[0],
            sub2con[1],
            sub2con[2],
            sub2T0, 
            sub2oC[0],
            sub2oC[1],
            sub2oC[2],
            sub2oN[0],
            sub2oN[1],
            sub2oN[2],
            sub1T,
            sub1no[0],
            sub1no[1],
            sub1sub2_intmat,
            sub1sub2_node,
            sub2sub1_intmat,
            sub2sub1_node,
            sub1nc[0],
            sub1nc[1],
            sub1nc[2],
            sub1con[0],
            sub1con[1],
            sub1con[2],
            sub1oC[0],
            sub1oC[1],
            sub1oC[2],
            sub1oN[0],
            sub1oN[1],
            sub1oN[2],
            fullT,
            fullno[0],
            fullno[1],
            fullsub1_intmat,
            fullsub1_node,
            sub1full_intmat,
            sub1full_node)
    return sub2T0, sub1T0, fullT0, sub2Tp0, sub1Tp0

@jax.jit
def moveSub2Mesh(v, vstart,
             sub2pnc, sub2con, sub2inc, sub2ooN, sub2ooC,
             sub2bx, sub2by, sub2bz, sub2Tp0,
             sub1nc, sub1con, sub1h, sub1Tp0,
             fullnc, fullcon, fullT0):
    # sub2nc: node_coords
    # sub2con: connect
    # sub2inc: init_node_coords
    # sub2pnc: prev_node_coords
    # sub2oN: overlapNodes
    # sub2oC: overlapCoords
    # sub2ooN: orig_overlap_nodes
    # sub2ooC: orig_overlap_coors
    # sub2bx,by,bz: bounds.ix,iy,iz

    vtot = v - vstart
    _sub2vx_tot_con, _sub2vy_tot_con, _sub2vz_tot_con = jit_constrain_v(vtot[0],
                                                        vtot[1],
                                                        vtot[2],
                                                        sub2bx[1],
                                                        sub2by[1],
                                                        sub2bz[1],
                                                        sub2bx[0],
                                                        sub2by[0],
                                                        sub2bz[0])

    ### Correction step (fine) ###
    sub2ncx, sub2ncy, sub2ncz, _a, _b, _c = move_fine_mesh(sub2inc[0],
                        sub2inc[1],
                        sub2inc[2],
                        sub1h[0],
                        sub1h[1],
                        sub1h[2],
                        _sub2vx_tot_con,
                        _sub2vy_tot_con,
                        _sub2vz_tot_con)
    sub2oNx, sub2oNy, sub2oNz, sub2oCx, sub2oCy, sub2oCz\
        = update_overlap_nodes_coords(sub2ooN[0],
                                    sub2ooN[1],
                                    sub2ooN[2],
                                    sub2ooC[0],
                                    sub2ooC[1],
                                    sub2ooC[2],
                                    _sub2vx_tot_con,
                                    _sub2vy_tot_con,
                                    _sub2vz_tot_con,
                                    sub1h[0],
                                    sub1h[1],
                                    sub1h[2])
    
    sub2T0 = interpolatePoints_jax(fullnc[0],
                                    fullnc[1],
                                    fullnc[2],
                                    fullcon[0],
                                    fullcon[1],
                                    fullcon[2],
                                    fullT0,
                                    sub2ncx,
                                    sub2ncy,
                                    sub2ncz)
    _ = interpolatePoints_jax(sub1nc[0],
                                sub1nc[1],
                                sub1nc[2],
                                sub1con[0],
                                sub1con[1],
                                sub1con[2],
                                sub1Tp0,
                                sub2ncx,
                                sub2ncy,
                                sub2ncz)
    sub2T0 = add_vectors(sub2T0, _)

    sub2Tp0 = interpolatePoints_jax(sub2pnc[0],
                                    sub2pnc[1],
                                    sub2pnc[2],
                                    sub2con[0],
                                    sub2con[1],
                                    sub2con[2],
                                    sub2Tp0,
                                    sub2ncx,
                                    sub2ncy,
                                    sub2ncz)
    sub2T0 = add_vectors(sub2T0, sub2Tp0)
    return [sub2ncx, sub2ncy, sub2ncz], [sub2oNx, sub2oNy, sub2oNz],\
           [sub2oCx, sub2oCy, sub2oCz], sub2T0, sub2Tp0, vtot

@jax.jit
def updateSub2AfterMove(sub2nc, sub2con, sub2oN,
                        sub1nc, sub1con,
                        fullnc, fullcon,
                        vx, vy, vz):
    sub2oN[0], sub2oN[1], sub2oN[2] = sub2oN[0] - vx, sub2oN[1] - vy, sub2oN[2] - vz

    # If mesh moves, recalculate shape functions
    sub2NcLevel1, sub2dNcdxLevel1, sub2dNcdyLevel1,\
            sub2dNcdzLevel1, sub2nodesLevel1 =\
            computeCoarseFineShapeFunctions(fullnc[0],
                                        fullnc[1],
                                        fullnc[2],
                                        fullcon[0],
                                        fullcon[1],
                                        fullcon[2],
                                        sub2nc[0],
                                        sub2nc[1],
                                        sub2nc[2],
                                        sub2con[0],
                                        sub2con[1],
                                        sub2con[2])

    # Move sub2 with respect to sub1
    sub2NcLevel2, sub2dNcdxLevel2, sub2dNcdyLevel2,\
            sub2dNcdzLevel2, sub2nodesLevel2 =\
            computeCoarseFineShapeFunctions(sub1nc[0],
                                        sub1nc[1],
                                        sub1nc[2],
                                        sub1con[0],
                                        sub1con[1],
                                        sub1con[2],
                                        sub2nc[0],
                                        sub2nc[1],
                                        sub2nc[2],
                                        sub2con[0],
                                        sub2con[1],
                                        sub2con[2])
    
    sub1sub2_intmat, sub1sub2_node = interpolatePointsMatrix(
                                sub1nc[0],
                                sub1nc[1],
                                sub1nc[2],
                                sub1con[0],
                                sub1con[1],
                                sub1con[2],
                                sub2nc[0],
                                sub2nc[1],
                                sub2nc[2])
    sub2sub1_intmat, sub2sub1_node = interpolatePointsMatrix(
                                sub2nc[0],
                                sub2nc[1],
                                sub2nc[2],
                                sub2con[0],
                                sub2con[1],
                                sub2con[2],
                                sub1nc[0],
                                sub1nc[1],
                                sub1nc[2])
    return sub2oN, sub2NcLevel1, sub2dNcdxLevel1, sub2dNcdyLevel1,\
            sub2dNcdzLevel1, sub2nodesLevel1,\
            sub2NcLevel2, sub2dNcdxLevel2, sub2dNcdyLevel2,\
            sub2dNcdzLevel2, sub2nodesLevel2,\
            sub1sub2_intmat, sub1sub2_node,\
            sub2sub1_intmat, sub2sub1_node

@jax.jit
def prepSub1Move(sub1inc, sub1h, sub1ooN, sub1ooC,
                 sub1bx, sub1by, sub1bz,
                 fullh,
                 vtot, _vx, _vy, _vz):
    _vx_tot_con, _vy_tot_con, _vz_tot_con = jit_constrain_v(vtot[0],
                                                            vtot[1],
                                                            vtot[2],
                                                            sub1bx[1],
                                                            sub1by[1],
                                                            sub1bz[1],
                                                            sub1bx[0],
                                                            sub1by[0],
                                                            sub1bz[0])
    # Need to round due to numerical round off error and truncation
    _tmp_x = (jnp.round(fullh[0] / sub1h[0])).astype(int)
    _tmp_y = (jnp.round(fullh[1] / sub1h[1])).astype(int)
    _tmp_z = (jnp.round(fullh[2] / sub1h[2])).astype(int)
    _vvx, _vvy, _vvz = _vx, _vy, _vz

    sub1ncx, sub1ncy, sub1ncz, _vx, _vy, _vz = move_fine_mesh(sub1inc[0],
                        sub1inc[1],
                        sub1inc[2],
                        fullh[0],
                        fullh[1],
                        fullh[2],
                        _vx_tot_con,
                        _vy_tot_con,
                        _vz_tot_con)

    _vx, _vy, _vz = _vx * _tmp_x, _vy * _tmp_y, _vz * _tmp_z
    moveSub1 = (_vvx != _vx) | (_vvy != _vy) | (_vvz != _vz)

    sub1oNx, sub1oNy, sub1oNz, sub1oCx, sub1oCy, sub1oCz\
                    = update_overlap_nodes_coords(sub1ooN[0],
                                                sub1ooN[1],
                                                sub1ooN[2],
                                                sub1ooC[0],
                                                sub1ooC[1],
                                                sub1ooC[2],
                                                _vx_tot_con,
                                                _vy_tot_con,
                                                _vz_tot_con,
                                                fullh[0],
                                                fullh[1],
                                                fullh[2])
    return [sub1ncx, sub1ncy, sub1ncz], [sub1oNx, sub1oNy, sub1oNz],\
           [sub1oCx, sub1oCy, sub1oCz], _vx, _vy, _vz, moveSub1

def moveSub1Mesh(sub1nc, sub1con, sub1pnc, sub1Tp0,
                 fullnc, fullcon, fullT0):
    sub1T0 = interpolatePoints_jax(fullnc[0],
                                    fullnc[1],
                                    fullnc[2],
                                    fullcon[0],
                                    fullcon[1],
                                    fullcon[2],
                                    fullT0,
                                    sub1nc[0],
                                    sub1nc[1],
                                    sub1nc[2])
    sub1Tp0 = interpolatePoints_jax(sub1pnc[0],
                                    sub1pnc[1],
                                    sub1pnc[2],
                                    sub1con[0],
                                    sub1con[1],
                                    sub1con[2],
                                    sub1Tp0,
                                    sub1nc[0],
                                    sub1nc[1],
                                    sub1nc[2])
    sub1T0 = add_vectors(sub1T0, sub1Tp0)
    return sub1T0, sub1Tp0

@jax.jit
def updateSub1objects(sub1nc, sub1con, sub1pnc, sub1Tp0,
                 fullnc, fullcon, fullT0):

    sub1T0, sub1Tp0 = moveSub1Mesh(sub1nc,
                                   sub1con,
                                   sub1pnc,
                                   sub1Tp0,
                                   fullnc,
                                   fullcon,
                                   fullT0)
    # If mesh moves, recalculate shape functions
    sub1NcLevel1, sub1dNcdxLevel1, sub1dNcdyLevel1,\
        sub1dNcdzLevel1, sub1nodesLevel1 =\
        computeCoarseFineShapeFunctions(fullnc[0],
                                        fullnc[1],
                                        fullnc[2],
                                        fullcon[0],
                                        fullcon[1],
                                        fullcon[2],
                                        sub1nc[0],
                                        sub1nc[1],
                                        sub1nc[2],
                                        sub1con[0],
                                        sub1con[1],
                                        sub1con[2])
    fullsub1_intmat, fullsub1_node = interpolatePointsMatrix(
                                        fullnc[0],
                                        fullnc[1],
                                        fullnc[2],
                                        fullcon[0],
                                        fullcon[1],
                                        fullcon[2],
                                        sub1nc[0],
                                        sub1nc[1],
                                        sub1nc[2])
    sub1full_intmat, sub1full_node = interpolatePointsMatrix(
                            sub1nc[0],
                            sub1nc[1],
                            sub1nc[2],
                            sub1con[0],
                            sub1con[1],
                            sub1con[2],
                            fullnc[0],
                            fullnc[1],
                            fullnc[2])
    return sub1T0, sub1Tp0, sub1NcLevel1, sub1dNcdxLevel1, sub1dNcdyLevel1,\
        sub1dNcdzLevel1, sub1nodesLevel1, fullsub1_intmat, fullsub1_node,\
        sub1full_intmat, sub1full_node

@jax.jit
def moveEverything(v, vstart, sub2pnc, sub2con, sub2inc, sub2ooN, sub2ooC,
             sub2bx, sub2by, sub2bz, sub2Tp0,
             sub1pnc, sub1con, sub1h, sub1Tp0,
             fullnc, fullcon, fullT0,
             sub1inc, sub1ooN, sub1ooC, sub1bx, sub1by, sub1bz, fullh, _vx, _vy, _vz):
    sub2nc ,sub2oN, sub2oC, sub2T0, sub2Tp0, vtot = moveSub2Mesh(v, vstart,
             sub2pnc, sub2con, sub2inc, sub2ooN, sub2ooC,
             sub2bx, sub2by, sub2bz, sub2Tp0,
             sub1pnc, sub1con, sub1h, sub1Tp0,
             fullnc, fullcon, fullT0)
    sub1nc, sub1oN, sub1oC, _vx, _vy, _vz, moveSub1 = prepSub1Move(sub1inc,
                                                                   sub1h,
                                                                   sub1ooN,
                                                                   sub1ooC,
                                                                   sub1bx,
                                                                   sub1by,
                                                                   sub1bz,
                                                                   fullh,
                                                                   vtot,
                                                                   _vx,
                                                                   _vy,
                                                                   _vz)
    sub1T0, sub1Tp0, sub1NcLevel1, sub1dNcdxLevel1, sub1dNcdyLevel1,\
    sub1dNcdzLevel1, sub1nodesLevel1, fullsub1_intmat, fullsub1_node,\
    sub1full_intmat, sub1full_node = updateSub1objects(sub1nc, 
                                                       sub1con,
                                                       sub1pnc,
                                                       sub1Tp0,
                                                       fullnc,
                                                       fullcon,
                                                       fullT0)
    sub2oN, sub2NcLevel1, sub2dNcdxLevel1, sub2dNcdyLevel1,\
    sub2dNcdzLevel1, sub2nodesLevel1,\
    sub2NcLevel2, sub2dNcdxLevel2, sub2dNcdyLevel2,\
    sub2dNcdzLevel2, sub2nodesLevel2,\
    sub1sub2_intmat, sub1sub2_node,\
    sub2sub1_intmat, sub2sub1_node = updateSub2AfterMove(sub2nc, sub2con, sub2oN,
                        sub1nc, sub1con,
                        fullnc, fullcon,
                        _vx, _vy, _vz)
    return sub2nc, sub2oN, sub2oC, sub2T0, sub2Tp0, vtot, sub1nc, sub1oN, sub1oC,\
        sub1T0, sub1Tp0, sub1NcLevel1, sub1dNcdxLevel1, sub1dNcdyLevel1,\
        sub1dNcdzLevel1, sub1nodesLevel1, fullsub1_intmat, fullsub1_node,\
        sub1full_intmat, sub1full_node, sub2NcLevel1, sub2dNcdxLevel1, sub2dNcdyLevel1,\
        sub2dNcdzLevel1, sub2nodesLevel1,\
        sub2NcLevel2, sub2dNcdxLevel2, sub2dNcdyLevel2,\
        sub2dNcdzLevel2, sub2nodesLevel2,\
        sub1sub2_intmat, sub1sub2_node,\
        sub2sub1_intmat, sub2sub1_node, _vx, _vy, _vz

import jax
import jax.numpy as jnp


@jax.jit
def computeQuad2dFemShapeFunctions_jax(coords):
    """
    Compute shape functions, their derivatives, and quadrature weights
    for a 4-node quadrilateral element using 2D Gaussian quadrature.

    This function evaluates the shape function matrix (N), its spatial
    derivatives (dNdx), and the quadrature weights (wq) at 4 integration
    points in the reference element.

    Parameters:
    coords (array): Nodal coordinates of the quadrilateral element, shape (8, 2). Only
                    the last 4 coordinates (top surface of element) are used.

    Returns:
    N (array): Shape function values at each quadrature point, shape (4, 4).
    dNdx (array): Derivatives of shape functions w.r.t. global coordinates,
                  shape (4, 4, 2).
    wq (array): Quadrature weights for each integration point, shape (4, 1).
    """
    ngp = 4  # Number of Gauss points
    ndim = 2  # Number of spatial dimensions

    # Isoparametric coordinates for 4-node quadrilateral element
    ksi_i = jnp.array([-1, 1, 1, -1])
    eta_i = jnp.array([-1, -1, 1, 1])

    # Gauss points in reference space (±1/√3)
    ksi_q = (1 / jnp.sqrt(3)) * ksi_i
    eta_q = (1 / jnp.sqrt(3)) * eta_i

    # Uniform quadrature weights for 2-point Gauss rule
    tmp_wq = jnp.ones(ngp)

    # Evaluate shape functions at Gauss points
    _ksi = 1 + ksi_q[:, None] @ ksi_i[None, :]
    _eta = 1 + eta_q[:, None] @ eta_i[None, :]
    N = (1 / 4) * _ksi * _eta

    # Derivatives w.r.t. reference coordinates
    dNdksi = (1 / 4) * ksi_i[None, :] * _eta
    dNdeta = (1 / 4) * eta_i[None, :] * _ksi

    # Compute Jacobian components (diagonal for structured mesh)
    dxdksi = jnp.matmul(dNdksi, coords[:, 0])
    dydeta = jnp.matmul(dNdeta, coords[:, 1])

    # Manually construct Jacobian and its inverse
    J = jnp.array(
        [
            [dxdksi[0], 0.0],
            [0.0, dydeta[0]],
        ]
    )
    Jinv = jnp.array(
        [
            [1.0 / dxdksi[0], 0.0],
            [0.0, 1.0 / dydeta[0]],
        ]
    )

    # Allocate arrays for shape function derivatives and weights
    dNdx = jnp.zeros((ngp, ngp, ndim))
    wq = jnp.zeros((ngp, 1))

    # Loop over Gauss points to compute global derivatives and weights
    for q in range(ngp):
        dN_dxi = jnp.concatenate(
            [
                dNdksi[q, :, None],
                dNdeta[q, :, None],
            ],
            axis=1,
        )
        dNdx = dNdx.at[q, :, :].set(dN_dxi @ Jinv)
        wq = wq.at[q].set(jnp.linalg.det(J) * tmp_wq[q])

    return jnp.array(N), jnp.array(dNdx), jnp.array(wq)


@jax.jit
def computeQuad3dFemShapeFunctions_jax(coords):
    """
    Compute shape functions, their derivatives, and quadrature weights
    for an 8-node hexahedral element using 3D Gaussian quadrature.

    This function evaluates the shape function matrix (N), its spatial
    derivatives (dNdx), and the quadrature weights (wq) at 8 integration
    points in the reference element.

    Parameters:
    coords (array): Nodal coordinates of the hexahedral element, shape (8, 3).

    Returns:
    N (array): Shape function values at each quadrature point, shape (8, 8).
    dNdx (array): Derivatives of shape functions w.r.t. global coordinates,
                  shape (8, 8, 3).
    wq (array): Quadrature weights for each integration point, shape (8, 1).
    """
    ngp = 8  # Number of Gauss points
    ndim = 3  # Number of spatial dimensions

    # Isoparametric coordinates for 8-node hexahedral element
    ksi_i = jnp.array([-1, 1, 1, -1, -1, 1, 1, -1])
    eta_i = jnp.array([-1, -1, 1, 1, -1, -1, 1, 1])
    zeta_i = jnp.array([-1, -1, -1, -1, 1, 1, 1, 1])

    # Gauss points in reference space (±1/√3)
    ksi_q = (1 / jnp.sqrt(3)) * ksi_i
    eta_q = (1 / jnp.sqrt(3)) * eta_i
    zeta_q = (1 / jnp.sqrt(3)) * zeta_i

    # Uniform quadrature weights for 2-point Gauss rule
    tmp_wq = jnp.ones(ngp)

    # Evaluate shape functions at Gauss points
    _ksi = 1 + ksi_q[:, None] @ ksi_i[None, :]
    _eta = 1 + eta_q[:, None] @ eta_i[None, :]
    _zeta = 1 + zeta_q[:, None] @ zeta_i[None, :]
    N = (1 / 8) * _ksi * _eta * _zeta
    # Derivatives w.r.t. reference coordinates
    dNdksi = (1 / 8) * ksi_i[None, :] * _eta * _zeta
    dNdeta = (1 / 8) * eta_i[None, :] * _ksi * _zeta
    dNdzeta = (1 / 8) * zeta_i[None, :] * _ksi * _eta

    # Compute Jacobian components (diagonal for structured mesh)
    dxdksi = jnp.matmul(dNdksi, coords[:, 0])
    dydeta = jnp.matmul(dNdeta, coords[:, 1])
    dzdzeta = jnp.matmul(dNdzeta, coords[:, 2])

    # Manually construct Jacobian and its inverse
    J = jnp.array(
        [
            [dxdksi[0], 0.0, 0.0],
            [0.0, dydeta[0], 0.0],
            [0.0, 0.0, dzdzeta[0]],
        ]
    )
    Jinv = jnp.array(
        [
            [1.0 / dxdksi[0], 0.0, 0.0],
            [0.0, 1.0 / dydeta[0], 0.0],
            [0.0, 0.0, 1.0 / dzdzeta[0]],
        ]
    )

    # Allocate arrays for shape function derivatives and weights
    dNdx = jnp.zeros((ngp, ngp, ndim))
    wq = jnp.zeros((ngp, 1))

    # Loop over Gauss points to compute global derivatives and weights
    for q in range(ngp):
        dN_dxi = jnp.concatenate(
            [
                dNdksi[q, :, None],
                dNdeta[q, :, None],
                dNdzeta[q, :, None],
            ],
            axis=1,
        )
        dNdx = dNdx.at[q, :, :].set(dN_dxi @ Jinv)
        wq = wq.at[q].set(jnp.linalg.det(J) * tmp_wq[q])

    return jnp.array(N), jnp.array(dNdx), jnp.array(wq)


def getQuadratureCoords(Level, ix, iy, iz, Nf):
    """
    Computes the quadrature point coordinates for a given element.

    Parameters:
    Level (dict): Mesh data for a given level.
    ix, iy, iz (int): Element indices in x, y, z directions.
    Nf (array): Shape function matrix.

    Returns:
    tuple: x, y, z coordinates at quadrature points.
    """
    coords_x = Level["node_coords"][0][Level["connect"][0][ix, :]].reshape(-1, 1)
    coords_y = Level["node_coords"][1][Level["connect"][1][iy, :]].reshape(-1, 1)
    coords_z = Level["node_coords"][2][Level["connect"][2][iz, :]].reshape(-1, 1)
    return Nf @ coords_x, Nf @ coords_y, Nf @ coords_z

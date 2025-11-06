import jax.numpy as jnp
import jax
from jax.experimental import sparse
from .gaussian_quadrature_functions import computeQuad3dFemShapeFunctions_jax
from .helper_functions import convert2XYZ


def compute3DN(q, x, y, z, h):
    """
    Compute trilinear shape functions for a hexahedral element at a given point.

    Parameters:
    q (list or array): Evaluation point [xq, yq, zq].
    x (list): x-coordinates of the element's bounding box [x0, x1].
    y (list): y-coordinates of the element's bounding box [y0, y1].
    z (list): z-coordinates of the element's bounding box [z0, z1].
    h (list): Element sizes in x, y, z directions [hx, hy, hz].

    Returns:
    array: Shape function values at point q, shape (8,).
    """
    inv_vol = 1.0 / (h[0] * h[1] * h[2])

    N = (
        jnp.array(
            [
                (x[1] - q[0]) * (y[1] - q[1]) * (z[1] - q[2]),
                (q[0] - x[0]) * (y[1] - q[1]) * (z[1] - q[2]),
                (q[0] - x[0]) * (q[1] - y[0]) * (z[1] - q[2]),
                (x[1] - q[0]) * (q[1] - y[0]) * (z[1] - q[2]),
                (x[1] - q[0]) * (y[1] - q[1]) * (q[2] - z[0]),
                (q[0] - x[0]) * (y[1] - q[1]) * (q[2] - z[0]),
                (q[0] - x[0]) * (q[1] - y[0]) * (q[2] - z[0]),
                (x[1] - q[0]) * (q[1] - y[0]) * (q[2] - z[0]),
            ]
        )
        * inv_vol
    )

    return N


@jax.jit
def computeCoarseFineShapeFunctions(Coarse, Fine):
    """
    Compute coarse shape functions and their derivatives at fine-scale
    quadrature points, and return a sparse projection matrix.

    Parameters:
    Coarse (dict): Coarse mesh data.
                   - "node_coords": list of 1D arrays for x, y, z coordinates.
                   - "connect": list of connectivity arrays in x, y, z.
    Fine (dict): Fine mesh data.
                 - "node_coords": list of 1D arrays for x, y, z coordinates.
                 - "connect": list of connectivity arrays in x, y, z.

    Returns:
    Nc (array): Coarse shape functions at fine quadrature points,
                shape (n_fine_elem, 8, 8).
    dNcdx, dNcdy, dNcdz (arrays): Derivatives of coarse shape functions,
                                  each of shape (n_fine_elem, 8, 8).
    test (BCOO): Sparse projection matrix from coarse nodes to fine quadrature.
    """
    # Mesh sizes
    nec_x, nec_y, nec_z = [Coarse["connect"][i].shape[0] for i in range(3)]
    nnc_x, nnc_y, nnc_z = [Coarse["node_coords"][i].shape[0] for i in range(3)]
    nnc = nnc_x * nnc_y * nnc_z
    nef_x, nef_y, nef_z = [Fine["connect"][i].shape[0] for i in range(3)]
    nef = nef_x * nef_y * nef_z
    nnf_x, nnf_y = [Fine["node_coords"][i].shape[0] for i in range(2)]

    # Coarse mesh spacing
    hc_x = Coarse["node_coords"][0][1] - Coarse["node_coords"][0][0]
    hc_y = Coarse["node_coords"][1][1] - Coarse["node_coords"][1][0]
    hc_z = Coarse["node_coords"][2][1] - Coarse["node_coords"][2][0]
    hc_xyz = hc_x * hc_y * hc_z
    xminc_x, xminc_y, xminc_z = [Coarse["node_coords"][i][0] for i in range(3)]

    # Reference shape functions for fine elements
    coords = jnp.stack(
        [
            Fine["node_coords"][0][Fine["connect"][0][0, :]],
            Fine["node_coords"][1][Fine["connect"][1][0, :]],
            Fine["node_coords"][2][Fine["connect"][2][0, :]],
        ],
        axis=1,
    )
    Nf, _, _ = computeQuad3dFemShapeFunctions_jax(coords)

    def stepComputeCoarseFineTerm(ieltf):
        ix, iy, iz, _ = convert2XYZ(ieltf, nef_x, nef_y, nnf_x, nnf_y)
        coords_x = Fine["node_coords"][0][Fine["connect"][0][ix, :]].reshape(-1, 1)
        coords_y = Fine["node_coords"][1][Fine["connect"][1][iy, :]].reshape(-1, 1)
        coords_z = Fine["node_coords"][2][Fine["connect"][2][iz, :]].reshape(-1, 1)

        x = (Nf @ coords_x).reshape(-1)
        y = (Nf @ coords_y).reshape(-1)
        z = (Nf @ coords_z).reshape(-1)

        # Determine coarse element indices
        ieltc_x = jnp.clip(jnp.floor((x - xminc_x) / hc_x).astype(int), 0, nec_x - 1)
        ieltc_y = jnp.clip(jnp.floor((y - xminc_y) / hc_y).astype(int), 0, nec_y - 1)
        ieltc_z = jnp.clip(jnp.floor((z - xminc_z) / hc_z).astype(int), 0, nec_z - 1)

        def iqLoop(iq):
            # Coarse element node indices
            nodec_x = Coarse["connect"][0][ieltc_x[iq], :]
            nodec_y = Coarse["connect"][1][ieltc_y[iq], :]
            nodec_z = Coarse["connect"][2][ieltc_z[iq], :]
            nodes = nodec_x + nodec_y * nnc_x + nodec_z * nnc_x * nnc_y

            # Bounding box corners
            xc0 = Coarse["node_coords"][0][nodec_x[0]]
            xc1 = Coarse["node_coords"][0][nodec_x[1]]
            yc0 = Coarse["node_coords"][1][nodec_y[0]]
            yc3 = Coarse["node_coords"][1][nodec_y[3]]
            zc0 = Coarse["node_coords"][2][nodec_z[0]]
            zc5 = Coarse["node_coords"][2][nodec_z[5]]

            _x, _y, _z = x[iq], y[iq], z[iq]

            Nc = compute3DN(
                [_x, _y, _z], [xc0, xc1], [yc0, yc3], [zc0, zc5], [hc_x, hc_y, hc_z]
            )

            dNcdx = (
                jnp.array(
                    [
                        (-1 * (yc3 - _y) * (zc5 - _z)),
                        (1 * (yc3 - _y) * (zc5 - _z)),
                        (1 * (_y - yc0) * (zc5 - _z)),
                        (-1 * (_y - yc0) * (zc5 - _z)),
                        (-1 * (yc3 - _y) * (_z - zc0)),
                        (1 * (yc3 - _y) * (_z - zc0)),
                        (1 * (_y - yc0) * (_z - zc0)),
                        (-1 * (_y - yc0) * (_z - zc0)),
                    ]
                )
                / hc_xyz
            )

            dNcdy = (
                jnp.array(
                    [
                        ((xc1 - _x) * -1 * (zc5 - _z)),
                        ((_x - xc0) * -1 * (zc5 - _z)),
                        ((_x - xc0) * 1 * (zc5 - _z)),
                        ((xc1 - _x) * 1 * (zc5 - _z)),
                        ((xc1 - _x) * -1 * (_z - zc0)),
                        ((_x - xc0) * -1 * (_z - zc0)),
                        ((_x - xc0) * 1 * (_z - zc0)),
                        ((xc1 - _x) * 1 * (_z - zc0)),
                    ]
                )
                / hc_xyz
            )

            dNcdz = (
                jnp.array(
                    [
                        ((xc1 - _x) * (yc3 - _y) * -1),
                        ((_x - xc0) * (yc3 - _y) * -1),
                        ((_x - xc0) * (_y - yc0) * -1),
                        ((xc1 - _x) * (_y - yc0) * -1),
                        ((xc1 - _x) * (yc3 - _y) * 1),
                        ((_x - xc0) * (yc3 - _y) * 1),
                        ((_x - xc0) * (_y - yc0) * 1),
                        ((xc1 - _x) * (_y - yc0) * 1),
                    ]
                )
                / hc_xyz
            )

            return Nc, dNcdx, dNcdy, dNcdz, nodes

        return jax.vmap(iqLoop)(jnp.arange(8))

    Nc, dNcdx, dNcdy, dNcdz, _nodes = jax.vmap(stepComputeCoarseFineTerm)(
        jnp.arange(nef)
    )
    _nodes = _nodes[:, 0, :]  # Only need one set of node indices per element

    # Construct sparse projection matrix
    indices = jnp.stack([_nodes.reshape(-1), jnp.arange(_nodes.size)], axis=1)
    test = jax.experimental.sparse.BCOO(
        (jnp.ones(_nodes.size), indices), shape=(nnc, _nodes.size)
    )
    return [Nc, [dNcdx, dNcdy, dNcdz], test]

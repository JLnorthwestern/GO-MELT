import jax.numpy as jnp
import jax
from jax.experimental import sparse
from jax.experimental.sparse import BCOO
from typing import List, Dict, Tuple, Union
from .gaussian_quadrature_functions import computeQuad3dFemShapeFunctions_jax
from .helper_functions import convert2XYZ


def compute3DN(
    test_pt: Union[List[float], jnp.ndarray],
    x_lims: Union[List[float], jnp.ndarray],
    y_lims: Union[List[float], jnp.ndarray],
    z_lims: Union[List[float], jnp.ndarray],
    elem_sizes: Union[List[float], jnp.ndarray],
) -> jnp.ndarray:
    """
    Compute trilinear shape functions for a hexahedral element at a given point.
    """
    inv_vol = 1.0 / (elem_sizes[0] * elem_sizes[1] * elem_sizes[2])

    x, y, z = test_pt
    N = (
        jnp.array(
            [
                (x_lims[1] - x) * (y_lims[1] - y) * (z_lims[1] - z),
                (x - x_lims[0]) * (y_lims[1] - y) * (z_lims[1] - z),
                (x - x_lims[0]) * (y - y_lims[0]) * (z_lims[1] - z),
                (x_lims[1] - x) * (y - y_lims[0]) * (z_lims[1] - z),
                (x_lims[1] - x) * (y_lims[1] - y) * (z - z_lims[0]),
                (x - x_lims[0]) * (y_lims[1] - y) * (z - z_lims[0]),
                (x - x_lims[0]) * (y - y_lims[0]) * (z - z_lims[0]),
                (x_lims[1] - x) * (y - y_lims[0]) * (z - z_lims[0]),
            ]
        )
        * inv_vol
    )

    return N


@jax.jit
def computeCoarseFineShapeFunctions(
    Coarse: Dict[str, List[jnp.ndarray]],
    Fine: Dict[str, List[jnp.ndarray]],
) -> Tuple[jnp.ndarray, List[jnp.ndarray], BCOO]:
    """
    Compute coarse shape functions and their derivatives at fine-scale
    quadrature points, and return a sparse projection matrix.
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

    def stepComputeCoarseFineTerm(
        ieltf: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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

        def iqLoop(
            iq: int,
        ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
    # First column represents coarse node indices, second is fine node indices
    indices = jnp.stack([_nodes.reshape(-1), jnp.arange(_nodes.size)], axis=1)
    summation_operator = jax.experimental.sparse.BCOO(
        (jnp.ones(_nodes.size), indices), shape=(nnc, _nodes.size)
    )
    return [Nc, [dNcdx, dNcdy, dNcdz], summation_operator]

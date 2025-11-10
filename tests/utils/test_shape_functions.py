import numpy as np
import jax.numpy as jnp
import pytest

from go_melt.utils.shape_functions import (
    compute3DN,
    computeCoarseFineShapeFunctions,
)


def test_compute3DN_unit_box_center_and_corners():
    # Define element bounding box [0,1] in each direction and h = 1
    x = [0.0, 1.0]
    y = [0.0, 1.0]
    z = [0.0, 1.0]
    h = [1.0, 1.0, 1.0]

    # Center point q = (0.5,0.5,0.5) -> each trilinear shape function should be 1/8
    q_center = [0.5, 0.5, 0.5]
    N_center = compute3DN(q_center, x, y, z, h)
    Nc = np.asarray(N_center)
    assert Nc.shape == (8,)
    assert np.allclose(Nc, np.ones(8) * (1.0 / 8.0), atol=1e-12)

    # Corner at (0,0,0) should give N[0] = 1 and others 0
    q_corner = [0.0, 0.0, 0.0]
    N_corner = np.asarray(compute3DN(q_corner, x, y, z, h))
    assert np.isclose(N_corner[0], 1.0)
    assert np.allclose(N_corner[1:], np.zeros(7), atol=1e-12)

    # Opposite corner at (1,1,1) should give N[6] = 1 (ordering follows provided compute3DN)
    q_opposite = [1.0, 1.0, 1.0]
    N_opp = np.asarray(compute3DN(q_opposite, x, y, z, h))
    # The function ordering places the (1,1,1) node at index 6 according to the code
    assert np.isclose(N_opp[6], 1.0)
    zeros_except = np.delete(N_opp, 6)
    assert np.allclose(zeros_except, np.zeros(7), atol=1e-12)


def test_computeCoarseFineShapeFunctions_single_element_basic():
    coarse_node_coords_x = jnp.linspace(0, 4, 5)
    coarse_node_coords_y = jnp.linspace(0, 4, 5)
    coarse_node_coords_z = jnp.linspace(0, 4, 5)
    fine_node_coords_x = jnp.linspace(1, 3, 5)
    fine_node_coords_y = jnp.linspace(1, 3, 5)
    fine_node_coords_z = jnp.linspace(1, 3, 5)

    connect_x = jnp.array(
        [
            [0, 1, 1, 0, 0, 1, 1, 0],
            [1, 2, 2, 1, 1, 2, 2, 1],
            [2, 3, 3, 2, 2, 3, 3, 2],
            [3, 4, 4, 3, 3, 4, 4, 3],
        ]
    )
    connect_y = jnp.array(
        [
            [0, 0, 1, 1, 0, 0, 1, 1],
            [1, 1, 2, 2, 1, 1, 2, 2],
            [2, 2, 3, 3, 2, 2, 3, 3],
            [3, 3, 4, 4, 3, 3, 4, 4],
        ]
    )
    connect_z = jnp.array(
        [
            [0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 2, 2, 2, 2],
            [2, 2, 2, 2, 3, 3, 3, 3],
            [3, 3, 3, 3, 4, 4, 4, 4],
        ]
    )

    Coarse = {
        "node_coords": [
            coarse_node_coords_x,
            coarse_node_coords_y,
            coarse_node_coords_z,
        ],
        "connect": [connect_x, connect_y, connect_z],
    }

    Fine = {
        "node_coords": [fine_node_coords_x, fine_node_coords_y, fine_node_coords_z],
        "connect": [connect_x, connect_y, connect_z],
    }

    # Call the function
    Nc, dN_list, sparse_proj = computeCoarseFineShapeFunctions(Coarse, Fine)

    # Nc should be an array with shape (n_fine_elems, 8, 8). For our setup nef = 1 element => shape[0] == 1
    Nc_np = np.asarray(Nc)
    assert Nc_np.ndim == 3
    assert Nc_np.shape[1:] == (8, 8)

    # For each fine quadrature point, coarse shape functions should sum to 1
    sums = Nc_np.sum(axis=2)  # sum over coarse nodes for each of 8 quadrature points
    assert np.allclose(sums, np.ones_like(sums), atol=1e-6)

    # Derivative arrays shapes
    dNcdx, dNcdy, dNcdz = dN_list
    dNcdx_np = np.asarray(dNcdx)
    dNcdy_np = np.asarray(dNcdy)
    dNcdz_np = np.asarray(dNcdz)
    assert dNcdx_np.shape == Nc_np.shape
    assert dNcdy_np.shape == Nc_np.shape
    assert dNcdz_np.shape == Nc_np.shape

    # Sparse projection matrix should be a BCOO with shape (nnc, nfq) where nfq = nef * 8
    nnc = int(
        Coarse["node_coords"][0].size
        * Coarse["node_coords"][1].size
        * Coarse["node_coords"][2].size
    )
    nef = int(
        Fine["connect"][0].shape[0]
        * Fine["connect"][1].shape[0]
        * Fine["connect"][2].shape[0]
    )
    nfq = nef * 8
    assert tuple(sparse_proj.shape) == (nnc, nfq)

    # Non-zero values in sparse_proj should be ones
    data = np.asarray(sparse_proj.data)
    assert np.allclose(data, np.ones_like(data))

    # Basic sanity: values in Nc are between 0 and 1
    assert np.all((Nc_np >= -1e-12) & (Nc_np <= 1.0 + 1e-12))


if __name__ == "__main__":
    pytest.main([__file__])

import numpy as np
import jax.numpy as jnp
import pytest

from go_melt.utils.interpolation_functions import (
    interpolatePoints,
    interpolatePointsMatrix,
    interpolate_w_matrix,
)


def test_interpolatePoints_identity_on_node_grid():
    node_coords_x = jnp.linspace(0, 4, 5)
    node_coords_y = jnp.linspace(0, 4, 5)
    node_coords_z = jnp.linspace(0, 4, 5)

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

    Level = {
        "node_coords": [node_coords_x, node_coords_y, node_coords_z],
        "connect": [connect_x, connect_y, connect_z],
    }

    # Build u as values at every coarse node arranged consistent with indexing used in code:
    # total coarse nodes = sizes of node_coords product
    nnx = Level["node_coords"][0].size
    nny = Level["node_coords"][1].size
    nnz = Level["node_coords"][2].size
    total_nodes = int(nnx * nny * nnz)

    # u is a simple sequence so we can identify exact matches
    u = jnp.arange(total_nodes, dtype=jnp.float32)

    # Use node_coords_new equal to the coarse node coordinates (so interpolation should reproduce u)
    node_coords_new = [
        jnp.asarray(Level["node_coords"][0]),
        jnp.asarray(Level["node_coords"][1]),
        jnp.asarray(Level["node_coords"][2]),
    ]

    # interpolatePoints returns an array of length nnx_new * nny_new * nnz_new which equals total_nodes
    interpolated = interpolatePoints(Level, u, node_coords_new)
    interp_np = np.asarray(interpolated).ravel()

    # Because the indexing scheme in the function maps the new points in the same flattened order,
    # the interpolated result should match u for points that coincide with coarse nodes.
    assert interp_np.shape[0] == total_nodes
    assert np.allclose(interp_np, np.asarray(u), atol=1e-6)

    # Single point test
    interpolated = interpolatePoints(
        Level, u, [jnp.asarray([2.5]), jnp.asarray([2.5]), jnp.asarray([2.5])]
    )
    interp_np = np.asarray(interpolated).ravel()
    assert interp_np.shape[0] == 1
    assert np.allclose(interp_np, 77.5, atol=1e-6)


def test_interpolatePointsMatrix_and_interpolate_w_matrix_consistency():
    node_coords_x = jnp.linspace(0, 4, 5)
    node_coords_y = jnp.linspace(0, 4, 5)
    node_coords_z = jnp.linspace(0, 4, 5)

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

    Level = {
        "node_coords": [node_coords_x, node_coords_y, node_coords_z],
        "connect": [connect_x, connect_y, connect_z],
    }

    nnx = Level["node_coords"][0].size
    nny = Level["node_coords"][1].size
    nnz = Level["node_coords"][2].size
    total_nodes = int(nnx * nny * nnz)

    # field values
    T = 0.1 * jnp.arange(total_nodes, dtype=jnp.float32)

    node_coords_new = [
        np.asarray(Level["node_coords"][0]),
        np.asarray(Level["node_coords"][1]),
        np.asarray(Level["node_coords"][2]),
    ]

    # Get interpolation shape functions and node indices
    C2F = interpolatePointsMatrix(Level, node_coords_new)
    C_weights = C2F[0]  # shape (n_new, 8)
    C_nodes = C2F[1]  # shape (n_new, 8)

    C_weights_np = np.asarray(C_weights)
    C_nodes_np = np.asarray(C_nodes)

    # Use interpolate_w_matrix to reconstruct interpolated values
    reconstructed = interpolate_w_matrix((C_weights, C_nodes), T)
    reconstructed_np = np.asarray(reconstructed).ravel()

    # Direct interpolation via interpolatePoints should match reconstructed values
    direct = np.asarray(interpolatePoints(Level, T, node_coords_new)).ravel()

    assert reconstructed_np.shape == direct.shape
    assert np.allclose(reconstructed_np, direct, atol=1e-6)

    # Sanity checks on shapes and ranges
    assert C_weights_np.shape[0] == total_nodes
    assert C_weights_np.shape[1] == 8
    assert C_nodes_np.shape == C_weights_np.shape
    # weights should be between 0 and 1
    assert np.all(C_weights_np >= -1e-8)
    assert np.all(C_weights_np <= 1.0 + 1e-8)


if __name__ == "__main__":
    pytest.main([__file__])

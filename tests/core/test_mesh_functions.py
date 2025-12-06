import pytest
import jax.numpy as jnp

from go_melt.core.mesh_functions import (
    calc_length_h,
    createMesh3D,
    calcNumNodes,
    getSampleCoords,
    getSubstrateNodes,
)


class DummyBounds:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class DummyObj:
    def __init__(self, bounds, elements):
        self.bounds = bounds
        self.elements = elements


def test_calc_length_h_simple_cube():
    bounds = DummyBounds(x=(0.0, 2.0), y=(0.0, 4.0), z=(0.0, 6.0))
    mesh = DummyObj(bounds=bounds, elements=(2, 4, 6))
    lengths, spacings = calc_length_h(mesh)

    assert lengths == [2.0, 4.0, 6.0]
    assert spacings == [1.0, 1.0, 1.0]  # each element size should be 1


def test_createMesh3D_coordinates_and_connectivity():
    x = [0.0, 1.0, 3]  # 3 nodes in x
    y = [0.0, 2.0, 2]  # 2 nodes in y
    z = [0.0, 1.0, 2]  # 2 nodes in z

    node_coords, connectivity = createMesh3D(x, y, z)

    # Check node coordinates
    assert jnp.allclose(node_coords[0], jnp.linspace(0.0, 1.0, 3))
    assert jnp.allclose(node_coords[1], jnp.linspace(0.0, 2.0, 2))
    assert jnp.allclose(node_coords[2], jnp.linspace(0.0, 1.0, 2))

    # Connectivity matrices should have 8 columns
    for conn in connectivity:
        assert conn.shape[1] == 8


def test_calcNumNodes_increments_elements():
    elements = [2, 3, 4]
    result = calcNumNodes(elements)
    assert result == [3, 4, 5]


def test_getSampleCoords_extracts_first_element():
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

    coords = getSampleCoords(Level)
    # Should return 2 nodes with 3 coordinates each
    assert coords.shape == (8, 3)
    assert jnp.allclose(
        coords[:, 0], jnp.array([0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
    )
    assert jnp.allclose(
        coords[:, 1], jnp.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    )
    assert jnp.allclose(
        coords[:, 2], jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    )


def test_getSubstrateNodes_counts_nodes_below_zero():
    node_coords_x0 = jnp.linspace(0, 4, 5)
    node_coords_y0 = jnp.linspace(0, 4, 5)
    node_coords_z0 = jnp.linspace(1, 2, 5)

    node_coords_x1 = jnp.linspace(0, 4, 5)
    node_coords_y1 = jnp.linspace(0, 4, 5)
    node_coords_z1 = jnp.linspace(-2, 2, 5)

    node_coords_x2 = jnp.linspace(0, 4, 5)
    node_coords_y2 = jnp.linspace(0, 4, 5)
    node_coords_z2 = jnp.linspace(0, 2, 5)

    node_coords_x3 = jnp.linspace(0, 4, 5)
    node_coords_y3 = jnp.linspace(0, 4, 5)
    node_coords_z3 = jnp.linspace(1, 2, 5)

    Levels = [
        {
            "node_coords": [node_coords_x0, node_coords_y0, node_coords_z0],
            "nodes": [5, 5, 5],
        },
        {
            "node_coords": [node_coords_x1, node_coords_y1, node_coords_z1],
            "nodes": [5, 5, 5],
        },
        {
            "node_coords": [node_coords_x2, node_coords_y2, node_coords_z2],
            "nodes": [5, 5, 5],
        },
        {
            "node_coords": [node_coords_x3, node_coords_y3, node_coords_z3],
            "nodes": [5, 5, 5],
        },
    ]

    result = getSubstrateNodes(Levels)
    assert result == (0, 75, 25, 0)


if __name__ == "__main__":
    pytest.main([__file__])

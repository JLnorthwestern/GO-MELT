import numpy as np
import jax.numpy as jnp
import pytest

from go_melt.utils.gaussian_quadrature_functions import (
    computeQuad2dFemShapeFunctions_jax,
    computeQuad3dFemShapeFunctions_jax,
    getQuadratureCoords,
)


def test_computeQuad2dFemShapeFunctions_basic():
    # Construct coords with last 4 nodes forming a unit square in XY:
    # ordering of last 4 nodes: we'll set them as (0,0),(2,0),(2,2),(0,2)
    # coords shape must be (8,2); first 4 rows can be dummy
    coords = jnp.vstack(
        [
            jnp.zeros((4, 2)),
            jnp.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]),
        ]
    )

    N, dNdx, wq = computeQuad2dFemShapeFunctions_jax(coords)

    # Shapes: N (4 gauss points x 4 shape funcs), dNdx (4,4,2), wq (4,1)
    assert N.shape == (4, 4)
    assert dNdx.shape == (4, 4, 2)
    assert wq.shape == (4, 1)

    # For a unit square with isotropic mapping, quadrature weights should be equal and
    # positive
    wq_np = np.asarray(wq).ravel()
    assert np.all(wq_np > 0)
    assert np.allclose(wq_np, wq_np[0], rtol=1e-6)

    # Check partition of unity: sum of shape functions at each gauss point == 1
    sums = np.asarray(N).sum(axis=1)
    assert np.allclose(sums, np.ones_like(sums), atol=1e-6)

    # Check that dNdx values are finite numbers
    assert np.isfinite(np.asarray(dNdx)).all()


def test_computeQuad3dFemShapeFunctions_basic():
    # Construct coords for an 8-node unit cube centered at origin corner (0 or 1)
    # Node ordering follows ksi_i, eta_i, zeta_i sign patterns used in function
    # We'll create a unit cube [0,1]^3 with nodes in the same sign order:
    coords = jnp.array(
        [
            [0.0, 0.0, 0.0],  # - - -
            [1.0, 0.0, 0.0],  # + - -
            [1.0, 1.0, 0.0],  # + + -
            [0.0, 1.0, 0.0],  # - + -
            [0.0, 0.0, 1.0],  # - - +
            [1.0, 0.0, 1.0],  # + - +
            [1.0, 1.0, 1.0],  # + + +
            [0.0, 1.0, 1.0],  # - + +
        ]
    )

    N, dNdx, wq = computeQuad3dFemShapeFunctions_jax(coords)

    # Shapes: N (8x8), dNdx (8,8,3), wq (8,1)
    assert N.shape == (8, 8)
    assert dNdx.shape == (8, 8, 3)
    assert wq.shape == (8, 1)

    # Partition of unity: sum of shape functions at each gauss point == 1
    sums = np.asarray(N).sum(axis=1)
    assert np.allclose(sums, np.ones_like(sums), atol=1e-6)

    # All weights equal and positive for unit cube mapping
    wq_np = np.asarray(wq).ravel()
    assert np.all(wq_np > 0)
    assert np.allclose(wq_np, wq_np[0], rtol=1e-6)

    # Derivatives finite
    assert np.isfinite(np.asarray(dNdx)).all()


def test_getQuadratureCoords():
    # Test 2D case
    coords2D = jnp.vstack(
        [
            jnp.zeros((4, 2)),
            jnp.array([[0.0, 0.0], [2.0, 0.0], [2.0, 3.0], [0.0, 3.0]]),
        ]
    )
    Nf, _, _ = computeQuad2dFemShapeFunctions_jax(coords2D)

    node_coords_x = jnp.linspace(0, 4, 3)
    node_coords_y = jnp.linspace(0, 3, 2)
    node_coords_z = jnp.linspace(0, 4, 5)

    connect_x = jnp.array([[0, 1, 1, 0]])
    connect_y = jnp.array([[0, 0, 1, 1]])
    connect_z = jnp.array([[0, 0, 0, 0]])

    Level = {
        "node_coords": [node_coords_x, node_coords_y, node_coords_z],
        "connect": [connect_x, connect_y, connect_z],
    }

    # Element indices ix=0,iy=0,iz=0
    xq, yq, _ = getQuadratureCoords(Level, 0, 0, 0, Nf)

    # Each returned array should equal Nf @ corresponding coords
    expected_x = np.asarray(Nf @ coords2D[4:, 0]).reshape(-1, 1)
    expected_y = np.asarray(Nf @ coords2D[4:, 1]).reshape(-1, 1)

    assert np.allclose(np.asarray(xq), expected_x)
    assert np.allclose(np.asarray(yq), expected_y)

    # Test 3D case
    coords3D = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 3.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
            [2.0, 3.0, 1.0],
            [0.0, 3.0, 1.0],
        ]
    )

    Nf, _, _ = computeQuad3dFemShapeFunctions_jax(coords3D)

    node_coords_x = jnp.linspace(0, 4, 3)
    node_coords_y = jnp.linspace(0, 3, 2)
    node_coords_z = jnp.linspace(0, 4, 5)

    connect_x = jnp.array([[0, 1, 1, 0, 0, 1, 1, 0]])
    connect_y = jnp.array([[0, 0, 1, 1, 0, 0, 1, 1]])
    connect_z = jnp.array([[0, 0, 0, 0, 1, 1, 1, 1]])

    Level = {
        "node_coords": [node_coords_x, node_coords_y, node_coords_z],
        "connect": [connect_x, connect_y, connect_z],
    }

    # Element indices ix=0,iy=0,iz=0
    xq, yq, zq = getQuadratureCoords(Level, 0, 0, 0, Nf)

    # Each returned array should equal Nf @ corresponding coords
    expected_x = np.asarray(Nf @ coords3D[:, 0]).reshape(-1, 1)
    expected_y = np.asarray(Nf @ coords3D[:, 1]).reshape(-1, 1)
    expected_z = np.asarray(Nf @ coords3D[:, 2]).reshape(-1, 1)

    assert np.allclose(np.asarray(xq), expected_x)
    assert np.allclose(np.asarray(yq), expected_y)
    assert np.allclose(np.asarray(zq), expected_z)


if __name__ == "__main__":
    pytest.main([__file__])

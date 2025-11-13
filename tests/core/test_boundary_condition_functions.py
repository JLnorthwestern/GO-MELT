import pytest
import jax.numpy as jnp

from go_melt.core.boundary_condition_functions import (
    getBCindices,
    assignBCs,
    assignBCsFine,
    computeConvRadBC,
)


class DummyObj:
    def __init__(self, nodes):
        self.nodes = nodes
        self.nn = nodes[0] * nodes[1] * nodes[2]


def test_getBCindices_simple_cube():
    x = DummyObj([2, 2, 2])  # 2x2x2 cube
    indices = getBCindices(x)

    widx, eidx, sidx, nidx, bidx, tidx = indices

    # Bottom face should be first nx*ny nodes
    assert jnp.all(bidx == jnp.array([0, 1, 2, 3]))
    # Top face should be last nx*ny nodes
    assert jnp.all(tidx == jnp.array([4, 5, 6, 7]))
    # West face should be nodes at x=0
    assert jnp.all(widx == jnp.array([0, 2, 4, 6]))
    # East face should be nodes at x=nx-1
    assert jnp.all(eidx == jnp.array([1, 3, 5, 7]))
    # South face should include y=0 plane
    assert sidx.shape[0] == 2 * 2  # nx * nz
    # North face should include y=ny-1 plane
    assert nidx.shape[0] == 2 * 2


def test_assignBCs_applies_conditions():
    RHS = jnp.zeros(8)
    Levels = [
        {},  # Level 0 unused
        {
            "BC": [
                jnp.array([0]),  # x-min
                jnp.array([1]),  # x-max
                jnp.array([2]),  # y-min
                jnp.array([3]),  # y-max
                jnp.array([4]),  # z-min
            ],
            "conditions": {"x": [10.0, 20.0], "y": [30.0, 40.0], "z": [50.0]},
        },
    ]

    result = assignBCs(RHS, Levels)
    assert result[0] == 10.0
    assert result[1] == 20.0
    assert result[2] == 30.0
    assert result[3] == 40.0
    assert result[4] == 50.0


def test_assignBCsFine_copies_values():
    RHS = jnp.zeros(6)
    TfAll = jnp.arange(6) * 10.0
    BC = [
        jnp.array([0]),  # x-min
        jnp.array([1]),  # x-max
        jnp.array([2]),  # y-min
        jnp.array([3]),  # y-max
        jnp.array([4]),  # z-min
    ]

    result = assignBCsFine(RHS, TfAll, BC)
    assert result[0] == TfAll[0]
    assert result[1] == TfAll[1]
    assert result[2] == TfAll[2]
    assert result[3] == TfAll[3]
    assert result[4] == TfAll[4]


def test_computeConvRadBC_runs(monkeypatch):
    # Minimal mesh: 1 element in x, 1 in y, 2 in z
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
    temperature = jnp.array([300.0, 300.0, 300.0, 300.0, 300.0])
    num_elems = 2
    num_nodes = 5
    properties = {
        "T_amb": 300.0,
        "T_boiling": 373.0,
        "vareps": 0.9,
        "evc": 1.0,
        "Lev": 2256000.0,
        "cp_fluid": 4180.0,
        "h_conv": 10.0,
        "sigma_sb": 5.67e-8,
        "CM_coeff": 1.0,
        "CT_coeff": 1.0,
        "CP_coeff": 1.0,
    }
    flux_vector = jnp.zeros(num_nodes)

    result = computeConvRadBC(
        Level, temperature, num_elems, num_nodes, properties, flux_vector
    )
    assert result.shape == (num_nodes,)
    # Should produce nonzero flux contributions
    assert jnp.any(result != 0.0)


if __name__ == "__main__":
    pytest.main([__file__])

import pytest
import jax.numpy as jnp

from go_melt.core.boundary_condition_functions import (
    getBCindices,
    assignBCs,
    assignBCsFine,
    computeConvRadBC,
    get_surface_faces,
    computeConvectionBC,
)


class Conditions:
    def __init__(self):
        self.west = type("c", (), {})()
        self.east = type("c", (), {})()
        self.south = type("c", (), {})()
        self.north = type("c", (), {})()
        self.bottom = type("c", (), {})()
        self.top = type("c", (), {})()


class DummyObj:
    def __init__(self, nodes):
        self.nodes = nodes
        self.nn = nodes[0] * nodes[1] * nodes[2]
        self.conditions = Conditions()


def test_getBCindices():
    x = DummyObj([2, 2, 2])  # 2x2x2 cube
    x = getBCindices(x)

    # Bottom face (z=0)
    assert jnp.all(x.conditions.bottom.indices == jnp.array([0, 1, 2, 3]))
    # Top face (z=1)
    assert jnp.all(x.conditions.top.indices == jnp.array([4, 5, 6, 7]))
    # West face (x=0)
    assert jnp.all(x.conditions.west.indices == jnp.array([0, 2, 4, 6]))
    # East face (x=1)
    assert jnp.all(x.conditions.east.indices == jnp.array([1, 3, 5, 7]))
    # South face (y=0)
    assert jnp.all(x.conditions.south.indices == jnp.array([0, 4, 1, 5]))
    # North face (y=1)
    assert jnp.all(x.conditions.north.indices == jnp.array([2, 6, 3, 7]))


def test_assignBCs():
    RHS = jnp.zeros(8)
    conditions = {
        "west": {"indices": jnp.array([0]), "value": 10.0},
        "east": {"indices": jnp.array([1]), "value": 20.0},
        "south": {"indices": jnp.array([2]), "value": 30.0},
        "north": {"indices": jnp.array([3]), "value": 40.0},
        "bottom": {"indices": jnp.array([4]), "value": 50.0},
        "top": {"indices": jnp.array([5]), "value": 60.0},
    }

    # Apply each boundary condition
    for bc in conditions.values():
        RHS = assignBCs(RHS, bc["indices"], bc["value"])

    assert RHS[0] == 10.0  # west
    assert RHS[1] == 20.0  # east
    assert RHS[2] == 30.0  # south
    assert RHS[3] == 40.0  # north
    assert RHS[4] == 50.0  # bottom
    assert RHS[5] == 60.0  # top


def test_assignBCsFine():
    RHS = jnp.zeros(6)
    TfAll = jnp.arange(6) * 10.0
    level = {
        "conditions": {
            "west": {"indices": jnp.array([0]), "value": 10.0},
            "east": {"indices": jnp.array([1]), "value": 20.0},
            "south": {"indices": jnp.array([2]), "value": 30.0},
            "north": {"indices": jnp.array([3]), "value": 40.0},
            "bottom": {"indices": jnp.array([4]), "value": 50.0},
            "top": {"indices": jnp.array([5]), "value": 60.0},  # optional
        }
    }

    # Apply each boundary condition using its indices
    for cond in level["conditions"].values():
        RHS = assignBCsFine(RHS, TfAll, cond["indices"])

    # Now check that RHS matches TfAll at those indices
    assert RHS[0] == TfAll[0]  # west
    assert RHS[1] == TfAll[1]  # east
    assert RHS[2] == TfAll[2]  # south
    assert RHS[3] == TfAll[3]  # north
    assert RHS[4] == TfAll[4]  # bottom
    assert RHS[5] == TfAll[5]  # top


def test_get_surface_faces():
    # 4x4x4 element cube
    level = {
        "nodes": [5, 5],  # dummy values for convert2XYZ
    }
    number_elems = 4 * 4 * 4  # 64 elements
    elements = (4, 4, 4)

    # Each element has 8 nodes, but we only need a solid_indicator of length >= number_elems*8
    # For simplicity, mark all nodes active
    solid_indicator = jnp.ones(number_elems * 8, dtype=bool)

    S1ele, S1faces = get_surface_faces(level, solid_indicator, number_elems, elements)

    # All 64 elements are solid
    assert S1ele.shape == (64,)
    assert jnp.all(S1ele == 1)

    # S1faces has 6 rows (faces), each flattened to length 64
    assert S1faces.shape == (6, 64)

    # Check that interior faces cancel out (0) and boundary faces remain free (1)
    west_faces = S1faces[0].reshape(elements[2], elements[1], elements[0])
    east_faces = S1faces[1].reshape(elements[2], elements[1], elements[0])
    south_faces = S1faces[2].reshape(elements[2], elements[1], elements[0])
    north_faces = S1faces[3].reshape(elements[2], elements[1], elements[0])
    bottom_faces = S1faces[4].reshape(elements[2], elements[1], elements[0])
    top_faces = S1faces[5].reshape(elements[2], elements[1], elements[0])

    # Outer boundaries should be 1
    assert jnp.all(west_faces[:, :, 0] == 1)
    assert jnp.all(east_faces[:, :, -1] == 1)
    assert jnp.all(south_faces[:, 0, :] == 1)
    assert jnp.all(north_faces[:, -1, :] == 1)
    assert jnp.all(bottom_faces[0, :, :] == 1)
    assert jnp.all(top_faces[-1, :, :] == 1)

    # Interior faces should be 0
    assert jnp.all(west_faces[:, :, 1:] == 0)
    assert jnp.all(east_faces[:, :, :-1] == 0)
    assert jnp.all(south_faces[:, 1:, :] == 0)
    assert jnp.all(north_faces[:, :-1, :] == 0)
    assert jnp.all(bottom_faces[1:, :, :] == 0)
    assert jnp.all(top_faces[:-1, :, :] == 0)


def test_computeConvRadBC(monkeypatch):
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

    num_elems = 4 * 4 * 4  # 64 elements
    num_nodes = 5 * 5 * 5  # 125 nodes

    # Build Level dict with S1faces marking outer boundary faces as 1
    # Shape (6, num_elems)
    S1faces = jnp.zeros((6, num_elems), dtype=int)
    # Mark all outer faces as open (simplified assumption for test)
    S1faces = S1faces.at[:, :].set(1)

    Level = {
        "node_coords": [node_coords_x, node_coords_y, node_coords_z],
        "connect": [connect_x, connect_y, connect_z],
        "h": [1.0, 1.0, 1.0],  # element dimensions
        "S1faces": S1faces,
    }

    temperature = jnp.full(num_nodes, 400.0)
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

    # local_indices: which nodes of each element belong to the boundary face
    local_indices = jnp.array([0, 1, 2, 3])
    bc_index = 0  # pick one boundary face

    result = computeConvRadBC(
        Level,
        temperature,
        num_elems,
        num_nodes,
        properties,
        flux_vector,
        local_indices,
        bc_index,
    )

    assert result.shape == (num_nodes,)
    # Should produce nonzero flux contributions
    assert jnp.any(result != 0.0)


def test_computeConvectionBC():
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

    num_elems = 4 * 4 * 4  # 64 elements
    num_nodes = 5 * 5 * 5  # 125 nodes

    # Build Level dict with S1faces marking outer boundary faces as 1
    # Shape (6, num_elems)
    S1faces = jnp.zeros((6, num_elems), dtype=int)
    # Mark all outer faces as open (simplified assumption for test)
    S1faces = S1faces.at[:, :].set(1)

    Level = {
        "node_coords": [node_coords_x, node_coords_y, node_coords_z],
        "connect": [connect_x, connect_y, connect_z],
        "h": [1.0, 1.0, 1.0],  # element dimensions
        "S1faces": S1faces,
        "active": jnp.ones(num_elems, dtype=bool),  # all elements active
    }

    temperature = jnp.full(num_nodes, 400.0)
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

    # local_indices: which nodes of each element belong to the boundary face
    local_indices = jnp.array([0, 1, 2, 3])
    bc_index = 0  # pick one boundary face
    tau = 1.0
    rhocp = jnp.ones(num_nodes) * 1000.0  # uniform density*cp

    result = computeConvectionBC(
        Level,
        temperature,
        num_elems,
        num_nodes,
        properties,
        flux_vector,
        local_indices,
        bc_index,
        tau,
        rhocp,
    )

    # Check output shape
    assert result.shape == (num_nodes,)
    # Should produce nonzero flux contributions
    assert jnp.any(result != 0.0)


if __name__ == "__main__":
    pytest.main([__file__])

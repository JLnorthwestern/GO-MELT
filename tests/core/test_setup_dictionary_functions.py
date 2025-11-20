import pytest
import json
import jax.numpy as jnp
from go_melt.core.setup_dictionary_functions import (
    obj,
    SetupProperties,
    SetupLevels,
    SetupNonmesh,
    SetupStaticNodesAndElements,
    SetupStaticSubcycle,
    dict2obj,
    structure_to_dict,
    calcStaticTmpNodesAndElements,
    find_max_const,
)


def test_SetupProperties_SetupLevels_SetupNonmesh(monkeypatch):
    input_file = "examples/example_unit.json"
    with open(input_file, "r") as read_file:
        solver_input = json.load(read_file)

    Properties = SetupProperties(solver_input.get("properties", {}))
    assert isinstance(Properties, dict)

    Levels = SetupLevels(solver_input, Properties)
    assert isinstance(Levels, list)

    # Patch os.path.exists to always return False, so the code will try to call os.makedirs
    called = {}
    monkeypatch.setattr("os.path.exists", lambda path: False)
    monkeypatch.setattr("os.makedirs", lambda path: called.setdefault("path", path))

    Nonmesh = SetupNonmesh(solver_input.get("nonmesh", {}))
    assert isinstance(Nonmesh, dict)

    # Verify that os.makedirs was "called" with the expected save_path
    assert "path" in called
    assert called["path"] == Nonmesh["save_path"]


def test_SetupProperties_SetupLevels_SetupNonmesh_edge():
    input_file = "examples/example_unit.json"
    with open(input_file, "r") as read_file:
        solver_input = json.load(read_file)

    # --- Modify solver_input in memory ---
    solver_input["Level2"]["bounds"]["z"] = [-2.5, 0.5]
    solver_input["Level3"]["bounds"]["z"] = [-1.0, 0.5]
    solver_input["Level1"]["conditions"]["bottom"]["type"] = "Neumann"
    solver_input["Level1"]["conditions"]["bottom"]["function"] = "Convection"

    Properties = SetupProperties(solver_input.get("properties", {}))
    assert isinstance(Properties, dict)
    Levels = SetupLevels(solver_input, Properties)
    assert isinstance(Levels, list)
    Nonmesh = SetupNonmesh(solver_input.get("nonmesh", {}))
    assert isinstance(Nonmesh, dict)


def test_SetupStaticNodesAndElements():
    Levels = [
        {},  # index 0 unused
        {"nn": jnp.array(10)},  # Level 1
        {"ne": jnp.array(20), "nn": jnp.array(30)},  # Level 2
        {"ne": jnp.array(40), "nn": jnp.array(50)},  # Level 3
    ]
    result = SetupStaticNodesAndElements(Levels)
    assert result == (20, 40, 10, 30, 50)


def test_SetupStaticSubcycle():
    nonmesh = {"subcycle_num_L2": 2, "subcycle_num_L3": 3, "loop_GOMELT": 1}
    result = SetupStaticSubcycle(nonmesh)
    # Expect (2, 3, 6, 2.0, 3.0, 6.0)
    assert result == (2, 3, 6, 2.0, 3.0, 6.0, 1)


def test_dict2obj_and_structure_to_dict_roundtrip():
    d = {"a": 1, "b": {"c": 2}}
    o = dict2obj(d)
    assert isinstance(o, obj)
    # Access like attribute
    assert o.b.c == 2

    # Convert back to dict
    d2 = structure_to_dict(o)
    assert d2 == d


def test_structure_to_dict_with_array():
    class NestedObj:
        def __init__(self):
            self.arr = jnp.array([1, 2, 3])
            self.value = 42

    nested = NestedObj()
    d = structure_to_dict(nested)
    # arr should remain as numpy array (has tolist method)
    assert isinstance(d["arr"], jnp.ndarray)
    assert d["value"] == 42


def test_calcStaticTmpNodesAndElements():
    Levels = [
        {},  # index 0 unused
        {
            "node_coords": [
                jnp.array([-0.5, 0.0, 0.5, 1.0]),
                jnp.array([-0.5, 0.0, 0.5, 1.0]),
                jnp.array([-0.5, 0.0, 0.5, 1.0]),
            ],  # pretend z coords
            "elements": [jnp.array(3), jnp.array(3), jnp.array(3)],
            "nodes": [jnp.array(4), jnp.array(4), jnp.array(4)],
        },
    ]
    toolpath_input = [0.0, 0.0, 0.5]  # z threshold = 0.5
    result = calcStaticTmpNodesAndElements(Levels, toolpath_input)
    # 3 * 3 * 2 = 18 active elements
    # 4 * 4 * 3 = 48 active nodes
    assert result == (18, 48)


class DummyBounds:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class DummyLevel:
    def __init__(self, bounds):
        self.bounds = bounds


def test_find_max_const_finer_inside_coarse():
    coarse = DummyLevel(DummyBounds(x=(0, 10), y=(0, 10), z=(0, 10)))
    finer = DummyLevel(DummyBounds(x=(2, 8), y=(3, 7), z=(1, 9)))

    shifts = find_max_const(coarse, finer)
    # West shift: coarse.x0 - finer.x0 = 0 - 2 = -2
    # East shift: coarse.x1 - finer.x1 = 10 - 8 = 2
    assert shifts[0] == [-2, 2]
    # South shift: 0 - 3 = -3, North shift: 10 - 7 = 3
    assert shifts[1] == [-3, 3]
    # Bottom shift: 0 - 1 = -1, Top shift: 10 - 9 = 1
    assert shifts[2] == [-1, 1]


def test_find_max_const_finer_equal_to_coarse():
    coarse = DummyLevel(DummyBounds(x=(0, 5), y=(0, 5), z=(0, 5)))
    finer = DummyLevel(DummyBounds(x=(0, 5), y=(0, 5), z=(0, 5)))

    shifts = find_max_const(coarse, finer)
    # All shifts should be zero
    assert shifts == ([0, 0], [0, 0], [0, 0])


def test_find_max_const_finer_extends_beyond_coarse():
    coarse = DummyLevel(DummyBounds(x=(0, 5), y=(0, 5), z=(0, 5)))
    finer = DummyLevel(DummyBounds(x=(-1, 6), y=(-2, 7), z=(-3, 8)))

    shifts = find_max_const(coarse, finer)
    # West shift: 0 - (-1) = 1, East shift: 5 - 6 = -1
    assert shifts[0] == [1, -1]
    # South shift: 0 - (-2) = 2, North shift: 5 - 7 = -2
    assert shifts[1] == [2, -2]
    # Bottom shift: 0 - (-3) = 3, Top shift: 5 - 8 = -3
    assert shifts[2] == [3, -3]


def test_find_max_const_finer_completely_inside_coarse():
    coarse = DummyLevel(DummyBounds(x=(0, 20), y=(0, 20), z=(0, 20)))
    finer = DummyLevel(DummyBounds(x=(5, 15), y=(5, 15), z=(5, 15)))

    shifts = find_max_const(coarse, finer)
    # West shift: 0 - 5 = -5, East shift: 20 - 15 = 5
    assert shifts[0] == [-5, 5]
    # South shift: 0 - 5 = -5, North shift: 20 - 15 = 5
    assert shifts[1] == [-5, 5]
    # Bottom shift: 0 - 5 = -5, Top shift: 20 - 15 = 5
    assert shifts[2] == [-5, 5]


if __name__ == "__main__":
    pytest.main([__file__])

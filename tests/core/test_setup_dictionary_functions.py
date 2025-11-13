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
)


def test_SetupProperties_SetupLevels_SetupNonmesh():
    input_file = "examples/example_unit.json"
    with open(input_file, "r") as read_file:
        solver_input = json.load(read_file)
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
    nonmesh = {"subcycle_num_L2": 2, "subcycle_num_L3": 3}
    result = SetupStaticSubcycle(nonmesh)
    # Expect (2, 3, 6, 2.0, 3.0, 6.0)
    assert result == (2, 3, 6, 2.0, 3.0, 6.0)


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


if __name__ == "__main__":
    pytest.main([__file__])

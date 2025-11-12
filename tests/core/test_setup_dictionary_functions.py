import pytest
import json
from go_melt.core.setup_dictionary_functions import (
    SetupProperties,
)


def test_SetupProperties():
    input_file = "examples/example_unit.json"
    with open(input_file, "r") as read_file:
        solver_input = json.load(read_file)
    Properties = SetupProperties(solver_input.get("properties", {}))
    assert isinstance(Properties, dict)


if __name__ == "__main__":
    pytest.main([__file__])

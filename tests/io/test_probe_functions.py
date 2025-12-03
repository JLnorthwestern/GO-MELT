import pytest
import os
import csv
import jax.numpy as jnp
import numpy as np
from unittest.mock import mock_open, patch

from go_melt.io.probe_functions import (
    update_probes,
    initialize_probe_csv,
    get_probe_regions,
)


def test_initialize_probe_csv_creates_header_without_file_io():
    save_path = "/fake/path"
    num_probes = 3
    expected_header = ["time", "Probe1", "Probe2", "Probe3"]

    # Patch builtins.open so no file is created
    m = mock_open()
    with patch("builtins.open", m):
        initialize_probe_csv(save_path, num_probes)

    # Grab the actual written content
    handle = m()
    written = "".join(call.args[0] for call in handle.write.call_args_list)

    # Use csv.reader on the string to parse what would have been written
    reader = csv.reader(written.splitlines())
    header = next(reader)

    assert header == expected_header


@patch("go_melt.io.probe_functions.interpolatePoints")
def test_update_probes_writes_correct_values(mock_interp):
    # Mock interpolatePoints to return predictable values
    mock_interp.side_effect = lambda level, field, coords: jnp.array([1.0])

    Levels = {
        1: {"T0": "dummy"},
        2: {"Tprime0": "dummy"},
        3: {"Tprime0": "dummy"},
    }
    grids = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    Nonmesh = {"save_path": "/fake/path"}
    time_inc = 0.12345678

    # Patch open so no file is created
    m = mock_open()
    with patch("builtins.open", m):
        update_probes(Levels, grids, Nonmesh, time_inc)

    # Collect what was written
    handle = m()
    written = "".join(call.args[0] for call in handle.write.call_args_list)

    # Parse the written string with csv.reader
    reader = list(csv.reader(written.splitlines()))

    # First row should contain formatted time and probe values
    row = reader[0]
    assert row[0] == f"{time_inc:.8f}"
    # Each probe value should be 3.0000 (since 1+1+1 from mock)
    assert all(val == "3.0000" for val in row[1:])


def test_get_probe_regions_returns_expected_arrays():
    locations = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]
    result = get_probe_regions(locations)

    assert len(result) == 2
    assert all(len(region) == 3 for region in result)

    # Check values
    assert np.allclose(result[0][0], jnp.array([1.0]))
    assert np.allclose(result[0][1], jnp.array([2.0]))
    assert np.allclose(result[0][2], jnp.array([3.0]))
    assert np.allclose(result[1][0], jnp.array([4.0]))
    assert np.allclose(result[1][1], jnp.array([5.0]))
    assert np.allclose(result[1][2], jnp.array([6.0]))


if __name__ == "__main__":
    pytest.main([__file__])

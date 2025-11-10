import numpy as np
import jax.numpy as jnp
import pytest

from go_melt.utils.helper_functions import (
    convert2XYZ,
    bincount,
    getCoarseNodesInLargeFineRegion,
    getCoarseNodesInFineRegion,
    getOverlapRegion,
    static_set_in_array,
    set_in_array,
    melting_temp,
)


def test_convert2XYZ_basic():
    # small 2x2x2 element grid (elements) -> nodes: 3x3x3
    num_elems_x = 2
    num_elems_y = 2
    num_nodes_x = 3
    num_nodes_y = 3

    # Test element 0 (corner at origin)
    ex, ey, ez, global_indices = convert2XYZ(
        0, num_elems_x, num_elems_y, num_nodes_x, num_nodes_y
    )
    assert (ex, ey, ez) == (0, 0, 0)
    expected0 = jnp.array([0, 1, 4, 3, 9, 10, 13, 12])
    assert jnp.array_equal(global_indices, expected0)

    # Test middle element index 3 (last in first z-layer)
    # element indexing order: z major, then y, then x (element index linear)
    idx = 3
    ex, ey, ez, global_indices = convert2XYZ(
        idx, num_elems_x, num_elems_y, num_nodes_x, num_nodes_y
    )
    assert (ex, ey, ez) == (1, 1, 0)
    # compute expected base: x + y*num_nodes_x + z*num_nodes_xy
    base = 1 + 1 * num_nodes_x + 0 * (num_nodes_x * num_nodes_y)
    expected3 = jnp.array(
        [
            base,
            base + 1,
            base + 1 + num_nodes_x,
            base + num_nodes_x,
            base + num_nodes_x * num_nodes_y,
            base + 1 + num_nodes_x * num_nodes_y,
            base + 1 + num_nodes_x + num_nodes_x * num_nodes_y,
            base + num_nodes_x + num_nodes_x * num_nodes_y,
        ]
    )
    assert jnp.array_equal(global_indices, expected3)


def test_bincount_basic():
    # node indices with repeats and values
    node_indices = jnp.array([0, 1, 1, 2, 4, 4], dtype=jnp.int32)
    values = jnp.array([1.0, 2.0, 3.0, 1.0, 5.0, 2.0], dtype=jnp.float32)
    total_node_num = 6
    bc = bincount(node_indices, values, total_node_num)
    # expected sums per index: 0->1, 1->5, 2->1, 3->0, 4->7, 5->0
    expected = jnp.array([1.0, 5.0, 1.0, 0.0, 7.0, 0.0], dtype=jnp.float32)
    assert jnp.allclose(bc, expected)


def test_getCoarseNodesInLargeFineRegion_simple():
    # fine grid larger than coarse: fine from 0 to 4 with 5 nodes (elem size 1)
    # coarse from 1 to 3 with 3 nodes (elem size 1)
    fine_coords = jnp.linspace(1.0, 5.0, 9)
    coarse_coords = jnp.linspace(2.0, 4.0, 3)
    overlap = getCoarseNodesInLargeFineRegion(coarse_coords, fine_coords)
    # coarse nodes at fine indices [1,2,3] -> expect [1,2,3]
    assert np.array_equal(np.asarray(overlap), np.array([2, 4, 6]))

    # If coarse has larger element size (coarse every 2 fine nodes)
    coarse_coords2 = jnp.linspace(2.0, 4.0, 2)
    overlap2 = getCoarseNodesInLargeFineRegion(coarse_coords2, fine_coords)
    # coarse nodes at fine indices 1 and 3
    assert np.array_equal(np.asarray(overlap2), np.array([2, 6]))


def test_getCoarseNodesInFineRegion_simple():
    # coarse from 0 to 4 (5 nodes), fine from 1 to 3 (5 nodes)
    coarse_coords = jnp.linspace(0.0, 4.0, 5)
    fine_coords = jnp.linspace(1.0, 3.0, 5)
    overlap = getCoarseNodesInFineRegion(fine_coords, coarse_coords)
    # coarse indices that fall inside fine extent are 1,2,3
    assert np.array_equal(np.asarray(overlap), np.array([1, 2, 3]))

    # coarse from 0 to 4 (5 nodes), fine from 1 to 3 (9 nodes)
    fine_coords2 = jnp.linspace(1.0, 3.0, 9)
    overlap2 = getCoarseNodesInFineRegion(fine_coords2, coarse_coords)
    assert np.array_equal(np.asarray(overlap2), np.array([1, 2, 3]))


def test_getOverlapRegion_3d_indices():
    # Construct a small 3D index set
    x_idx = jnp.array([0, 2])
    y_idx = jnp.array([0, 1])
    z_idx = jnp.array([0, 1])
    node_indices = [x_idx, y_idx, z_idx]
    num_nodes_x = 4
    num_nodes_y = 5

    ov = getOverlapRegion(node_indices, num_nodes_x, num_nodes_y)
    # build expected using explicit loops
    expected_list = []
    for z in z_idx:
        for y in y_idx:
            for x in x_idx:
                expected_list.append(
                    int(x + y * num_nodes_x + z * num_nodes_x * num_nodes_y)
                )
    expected = jnp.array(expected_list)
    assert jnp.array_equal(ov, expected)


def test_static_set_in_array_and_set_in_array():
    arr = jnp.arange(8)
    new_vals = jnp.array([100, 101, 102, 103, 104])
    res_static = static_set_in_array(arr, 3, new_vals)
    expected_static = arr.at[3:].set(new_vals)
    assert jnp.array_equal(res_static, expected_static)

    # set_in_array with indices
    arr2 = jnp.zeros(6, dtype=jnp.int32)
    indices = jnp.array([0, 2, 5])
    vals = jnp.array([7, 8, 9])
    res_set = set_in_array(arr2, indices, vals)
    expected_set = arr2.at[indices].set(vals)
    assert jnp.array_equal(res_set, expected_set)


def test_melting_temp_updates():
    # temps: numpy array, some above T_melt
    temps = np.array([300.0, 350.0, 250.0, 400.0])
    delt_T = 2.5
    T_melt = 300.0
    # accum_time is a jnp array of zeros
    accum_time = jnp.zeros(4)
    # indices we want to update: all indices
    idx = jnp.arange(4)
    updated = melting_temp(temps, delt_T, T_melt, accum_time, idx)
    # temps > T_melt: positions 1 and 3 (350,400); temps == T_melt should be False -> only > counts
    expected = jnp.array([0.0, delt_T, 0.0, delt_T])
    assert jnp.allclose(updated, expected)

    # calling again with same temps should accumulate
    updated2 = melting_temp(temps, delt_T, T_melt, updated, idx)
    expected2 = expected + expected
    assert jnp.allclose(updated2, expected2)


if __name__ == "__main__":
    pytest.main([__file__])

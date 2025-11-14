import jax
import jax.numpy as jnp
from functools import partial
import numpy as np
from typing import Tuple, List


@jax.jit
def convert2XYZ(
    elem_index: int,
    num_elems_x: int,
    num_elems_y: int,
    num_nodes_x: int,
    num_nodes_y: int,
) -> Tuple[int, int, int, jnp.ndarray]:
    """
    Compute element indices (used to determine which connectivity row) and global node
    indices for an 8-node hexahedron.
    """

    num_elems_xy = num_elems_x * num_elems_y
    num_nodes_xy = num_nodes_x * num_nodes_y

    elem_index_z = elem_index // num_elems_xy
    elem_index_y = (elem_index // num_elems_x) - elem_index_z * num_elems_y
    elem_index_x = elem_index % num_elems_x

    # base: global node index for the element's (0,0,0) corner
    base = elem_index_x + elem_index_y * num_nodes_x + elem_index_z * num_nodes_xy
    dx = 1
    dy = num_nodes_x
    dz = num_nodes_xy

    # global_indices: global indices for the 8 nodes of the hexahedral element
    global_indices = jnp.array(
        [
            base,
            base + dx,
            base + dx + dy,
            base + dy,
            base + dz,
            base + dx + dz,
            base + dx + dy + dz,
            base + dy + dz,
        ]
    )

    return elem_index_x, elem_index_y, elem_index_z, global_indices


@partial(jax.jit, static_argnames=["total_node_num"])
def bincount(
    node_indices: jnp.ndarray, values: jnp.ndarray, total_node_num: int
) -> jnp.ndarray:
    """Static bincount function call"""
    return jnp.bincount(node_indices, values, length=total_node_num)


def getCoarseNodesInLargeFineRegion(
    coarse_coords: jnp.ndarray, fine_coords: jnp.ndarray
) -> jnp.ndarray:
    """
    Identify fine grid indices (1D) that correspond to coarse grid nodes
    when the fine grid spans a larger domain than the coarse grid.

    Both grids are assumed uniformly spaced.
    """
    fine_min = fine_coords.min()
    fine_max = fine_coords.max()
    coarse_min = coarse_coords.min()
    coarse_max = coarse_coords.max()

    num_fine_nodes = fine_coords.size
    num_fine_elems = num_fine_nodes - 1
    fine_elem_size = (fine_max - fine_min) / num_fine_elems

    num_coarse_nodes = coarse_coords.size
    num_coarse_elems = num_coarse_nodes - 1
    coarse_elem_size = (coarse_max - coarse_min) / num_coarse_elems

    overlapMin = jnp.round((coarse_min - fine_min) / fine_elem_size)
    overlapMax = jnp.round((coarse_max - fine_min) / fine_elem_size) + 1

    step = int(jnp.round(coarse_elem_size / fine_elem_size))
    overlap = jnp.arange(overlapMin, overlapMax, step).astype(int)

    return overlap


def getCoarseNodesInFineRegion(
    fine_coords: jnp.ndarray, coarse_coords: jnp.ndarray
) -> jnp.ndarray:
    """
    Identify coarse-grid node indices (1D) that fall inside the fine-grid extent.

    Both grids are assumed uniformly spaced.
    """
    fine_min = fine_coords.min()
    fine_max = fine_coords.max()
    coarse_min = coarse_coords.min()
    coarse_max = coarse_coords.max()

    num_coarse_nodes = coarse_coords.size
    num_coarse_elems = num_coarse_nodes - 1
    coarse_elem_size = (coarse_max - coarse_min) / num_coarse_elems

    overlapMin = jnp.round((fine_min - coarse_min) / coarse_elem_size)
    overlapMax = jnp.round((fine_max - coarse_min) / coarse_elem_size) + 1

    overlap = jnp.arange(overlapMin, overlapMax).astype(int)

    return overlap


@jax.jit
def getOverlapRegion(
    node_indices: List[jnp.ndarray], num_nodes_x: int, num_nodes_y: int
) -> jnp.ndarray:
    """
    Compute flattened global node indices for a structured 3D grid.

    This function generates a 1D array of global node indices based on
    the Cartesian product of x, y, and z coordinate arrays. It assumes
    a structured grid with dimensions (num_nodes_x, num_nodes_y, num_nodes_z).
    """
    x_index = jnp.tile(
        node_indices[0], node_indices[1].shape[0] * node_indices[2].shape[0]
    ).reshape(-1)

    y_index = jnp.repeat(
        jnp.tile(node_indices[1], node_indices[2].shape[0]), node_indices[0].shape[0]
    ).reshape(-1)

    z_index = jnp.repeat(
        node_indices[2], node_indices[0].shape[0] * node_indices[1].shape[0]
    )

    return x_index + y_index * num_nodes_x + z_index * num_nodes_x * num_nodes_y


@partial(jax.jit, static_argnames=["index"])
def static_set_in_array(
    input_array: jnp.ndarray, index: int, values: jnp.ndarray
) -> jnp.ndarray:
    """
    Replace a slice of the input_array array starting at a given index with new values.

    The index is treated as static arguments for JAX compilation efficiency.
    """
    return input_array.at[index:].set(values)


@jax.jit
def set_in_array(input_array: jnp.ndarray, indices: jnp.ndarray, values: jnp.ndarray):
    """
    Replace a slice of the input_array array with new values.
    """
    return input_array.at[indices].set(values)

import jax
import jax.numpy as jnp
from functools import partial
import numpy as np


@jax.jit
def convert2XYZ(i, ne_x, ne_y, nn_x, nn_y):
    """
    Compute local element indices and global node connectivity in 3D.

    Parameters:
    i (int): Element index (flattened).
    ne_x (int): Number of elements in the x-direction.
    ne_y (int): Number of elements in the y-direction.
    nn_x (int): Number of nodes in the x-direction.
    nn_y (int): Number of nodes in the y-direction.

    Returns:
    tuple:
        ix (int): Element index in x-direction.
        iy (int): Element index in y-direction.
        iz (int): Element index in z-direction.
        idx (jnp.ndarray): Global node indices for the 8-node hexahedral element.
    """
    ne_xy = ne_x * ne_y
    nn_xy = nn_x * nn_y

    iz = i // ne_xy
    iy = (i // ne_x) - iz * ne_y
    ix = i % ne_x

    base = ix + iy * nn_x + iz * nn_xy
    dx = 1
    dy = nn_x
    dz = nn_xy

    idx = jnp.array(
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

    return ix, iy, iz, idx


@partial(jax.jit, static_argnames=["nn"])
def bincount(N, D, nn):
    """
    Perform a weighted bin count operation.

    This function accumulates values from `D` into bins specified by `N`,
    producing a 1D array of length `nn`.

    Parameters:
    N (array): Bin indices (integer array).
    D (array): Values to accumulate (same shape as N).
    nn (int): Total number of bins (output length).

    Returns:
    array: Binned sum of values, shape (nn,).
    """
    return jnp.bincount(N, D, length=nn)


def getCoarseNodesInLargeFineRegion(xnc, xnf):
    """
    Identify fine grid indices that correspond to coarse grid nodes
    when the fine grid spans a larger domain than the coarse grid.

    This function computes the indices of fine grid nodes that align
    with the coarse grid node positions, assuming both grids are
    uniformly spaced.

    Parameters:
    xnc (array): Coordinates of the coarse grid nodes.
    xnf (array): Coordinates of the fine grid nodes.

    Returns:
    array: Indices of fine grid nodes that align with coarse grid nodes.
    """
    xfmin = xnf.min()
    xfmax = xnf.max()
    xcmin = xnc.min()
    xcmax = xnc.max()

    nnf = xnf.size
    nef = nnf - 1
    hf = (xfmax - xfmin) / nef

    nnc = xnc.size
    nec = nnc - 1
    hc = (xcmax - xcmin) / nec

    overlapMin = jnp.round((xcmin - xfmin) / hf)
    overlapMax = jnp.round((xcmax - xfmin) / hf) + 1

    step = int(jnp.round(hc / hf))
    overlap = jnp.arange(overlapMin, overlapMax, step).astype(int)

    return overlap


def getCoarseNodesInFineRegion(xnf, xnc):
    """
    Identify coarse grid nodes that overlap with the fine grid region.

    This function determines which coarse grid nodes fall within the
    spatial extent of a given fine grid. It assumes uniform spacing
    in the coarse grid.

    Parameters:
    xnf (array): Coordinates of the fine grid nodes.
    xnc (array): Coordinates of the coarse grid nodes.

    Returns:
    array: Indices of coarse grid nodes that overlap with the fine grid.
    """
    xfmin = xnf.min()
    xfmax = xnf.max()
    xcmin = xnc.min()
    xcmax = xnc.max()

    nnc = xnc.size
    nec = nnc - 1
    hc = (xcmax - xcmin) / nec

    overlapMin = jnp.round((xfmin - xcmin) / hc)
    overlapMax = jnp.round((xfmax - xcmin) / hc) + 1

    overlap = jnp.arange(overlapMin, overlapMax).astype(int)

    return overlap


@jax.jit
def getOverlapRegion(node_coords, nx, ny):
    """
    Compute flattened global node indices for a structured 3D grid.

    This function generates a 1D array of global node indices based on
    the Cartesian product of x, y, and z coordinate arrays. It assumes
    a structured grid with dimensions (nx, ny, nz).

    Parameters:
    node_coords (list): List of 1D arrays [x, y, z] representing node coordinates.
    nx (int): Number of nodes in the x-direction.
    ny (int): Number of nodes in the y-direction.

    Returns:
    array: Flattened global node indices corresponding to the overlap region.
    """
    _x = jnp.tile(
        node_coords[0], node_coords[1].shape[0] * node_coords[2].shape[0]
    ).reshape(-1)

    _y = jnp.repeat(
        jnp.tile(node_coords[1], node_coords[2].shape[0]), node_coords[0].shape[0]
    ).reshape(-1)

    _z = jnp.repeat(node_coords[2], node_coords[0].shape[0] * node_coords[1].shape[0])

    return _x + _y * nx + _z * nx * ny


@partial(jax.jit, static_argnames=["_idx"])
def substitute_Tbar(Tbar, _idx, _val):
    """
    Replace a slice of the Tbar array starting at a given index with a new value.

    This function sets all elements from index `_idx` to the end of the array
    to the value `_val`. The index and value are treated as static arguments
    for JAX compilation efficiency.

    Parameters:
    Tbar (array): Input array to be modified.
    _idx (int): Starting index for substitution.
    _val (float or array): Value(s) to assign from _idx onward.

    Returns:
    array: Modified Tbar array with values substituted from _idx onward.
    """
    return Tbar.at[_idx:].set(_val)


@jax.jit
def substitute_Tbar2(Tbar, _idx, _val):
    """
    Replace a single element in the Tbar array at a given index.

    This function sets the element at index `_idx` to `_val`.

    Parameters:
    Tbar (array): Input array to be modified.
    _idx (int): Index of the element to be replaced.
    _val (float): New value to assign at the specified index.

    Returns:
    array: Modified Tbar array with the specified element updated.
    """
    return Tbar.at[_idx].set(_val)


def melting_temp(temps, delt_T, T_melt, accum_time, idx):
    """
    Update accumulated melt time for nodes above melting temperature.

    Parameters:
    temps (array): Current temperature field.
    delt_T (float): Time step duration.
    T_melt (float): Melting temperature threshold.
    accum_time (array): Accumulated melt time array.
    idx (array): Indices of nodes to update.

    Returns:
    array: Updated accumulated melt time.
    """
    T_above_threshold = np.array(temps > T_melt)
    accum_time = accum_time.at[idx].add(T_above_threshold * delt_T)
    return accum_time

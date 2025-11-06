import jax
import jax.numpy as jnp


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

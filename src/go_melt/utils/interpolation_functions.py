import jax
import jax.numpy as jnp
from jax.numpy import multiply
from .shape_functions import compute3DN


@jax.jit
def interpolatePoints(
    Level: dict[str, list[jnp.ndarray]],
    u: jnp.ndarray,
    node_coords_new: list[jnp.ndarray],
) -> jnp.ndarray:
    """
    Interpolate a scalar field from a structured mesh to new coordinates.

    This function evaluates shape functions at new nodal coordinates and
    uses them to interpolate values from the source field `u` defined on
    Level["node_coords"] and Level["connect"].

    Parameters:
    Level (dict): Contains mesh information:
                  - "node_coords": list of 1D arrays for x, y, z coordinates.
                  - "connect": list of connectivity arrays in x, y, z.
    u (array): Field values at the source mesh nodes.
    node_coords_new (list): New nodal coordinates [x_new, y_new, z_new].

    Returns:
    array: Interpolated values at the new nodal coordinates.
    """
    ne_x = Level["connect"][0].shape[0]
    ne_y = Level["connect"][1].shape[0]
    ne_z = Level["connect"][2].shape[0]
    nn_x, nn_y = ne_x + 1, ne_y + 1

    nn_xn = len(node_coords_new[0])
    nn_yn = len(node_coords_new[1])
    nn_zn = len(node_coords_new[2])
    total_points = nn_xn * nn_yn * nn_zn

    h_x = Level["node_coords"][0][1] - Level["node_coords"][0][0]
    h_y = Level["node_coords"][1][1] - Level["node_coords"][1][0]
    h_z = Level["node_coords"][2][1] - Level["node_coords"][2][0]

    def stepInterpolatePoints(ielt):
        izn, rem = jnp.divmod(ielt, nn_xn * nn_yn)
        iyn, ixn = jnp.divmod(rem, nn_xn)

        _x = node_coords_new[0][ixn]
        _y = node_coords_new[1][iyn]
        _z = node_coords_new[2][izn]

        # Determine coarse element indices
        ielt_x = jnp.clip(
            jnp.floor((_x - Level["node_coords"][0][0]) / h_x).astype(int), 0, ne_x - 1
        )
        ielt_y = jnp.clip(
            jnp.floor((_y - Level["node_coords"][1][0]) / h_y).astype(int), 0, ne_y - 1
        )
        ielt_z = jnp.clip(
            jnp.floor((_z - Level["node_coords"][2][0]) / h_z).astype(int), 0, ne_z - 1
        )

        # Get node indices
        nodex = Level["connect"][0][ielt_x, :]
        nodey = Level["connect"][1][ielt_y, :]
        nodez = Level["connect"][2][ielt_z, :]
        node = nodex + nodey * nn_x + nodez * (nn_x * nn_y)

        # Get coordinates of the coarse element nodes
        xx = Level["node_coords"][0][nodex]
        yy = Level["node_coords"][1][nodey]
        zz = Level["node_coords"][2][nodez]

        # Bounding box corners
        xc0, xc1 = xx[0], xx[1]
        yc0, yc3 = yy[0], yy[3]
        zc0, zc5 = zz[0], zz[5]

        # Compute shape functions
        Nc = compute3DN(
            [_x, _y, _z], [xc0, xc1], [yc0, yc3], [zc0, zc5], [h_x, h_y, h_z]
        )

        # Clip and mask invalid values
        valid = jnp.logical_and((Nc >= -1e-2).all(), (Nc <= 1 + 1e-2).all())
        Nc = jax.lax.select(valid, jnp.clip(Nc, 0.0, 1.0), jnp.zeros_like(Nc))

        return Nc @ u[node]

    return jax.vmap(stepInterpolatePoints)(jnp.arange(total_points))


@jax.jit
def interpolatePointsMatrix(Level, node_coords_new):
    """
    Compute interpolation shape functions and node indices for mapping
    values from a coarse mesh to a new set of coordinates.

    Parameters:
    Level (dict): Contains mesh information:
                  - "node_coords": list of 1D arrays for x, y, z coordinates.
                  - "connect": list of connectivity arrays in x, y, z.
    node_coords_new (list): New nodal coordinates [x_new, y_new, z_new],
                            each as a 2D array.

    Returns:
    list: [_Nc, _node]
          _Nc (array): Shape function values for interpolation.
          _node (array): Indices of coarse nodes contributing to each point.
    """
    ne_x = Level["connect"][0].shape[0]
    ne_y = Level["connect"][1].shape[0]
    ne_z = Level["connect"][2].shape[0]
    nn_x, nn_y = ne_x + 1, ne_y + 1

    nn_xn = len(node_coords_new[0])
    nn_yn = len(node_coords_new[1])
    nn_zn = len(node_coords_new[2])
    total_points = nn_xn * nn_yn * nn_zn

    h_x = Level["node_coords"][0][1] - Level["node_coords"][0][0]
    h_y = Level["node_coords"][1][1] - Level["node_coords"][1][0]
    h_z = Level["node_coords"][2][1] - Level["node_coords"][2][0]

    def stepInterpolatePoints(ielt):
        izn, rem = jnp.divmod(ielt, nn_xn * nn_yn)
        iyn, ixn = jnp.divmod(rem, nn_xn)

        _x = node_coords_new[0][ixn]
        _y = node_coords_new[1][iyn]
        _z = node_coords_new[2][izn]

        # Determine coarse element indices
        ielt_x = jnp.clip(
            jnp.floor((_x - Level["node_coords"][0][0]) / h_x).astype(int), 0, ne_x - 1
        )
        ielt_y = jnp.clip(
            jnp.floor((_y - Level["node_coords"][1][0]) / h_y).astype(int), 0, ne_y - 1
        )
        ielt_z = jnp.clip(
            jnp.floor((_z - Level["node_coords"][2][0]) / h_z).astype(int), 0, ne_z - 1
        )

        # Get node indices
        nodex = Level["connect"][0][ielt_x, :]
        nodey = Level["connect"][1][ielt_y, :]
        nodez = Level["connect"][2][ielt_z, :]
        node = nodex + nodey * nn_x + nodez * (nn_x * nn_y)

        # Get coordinates of the coarse element nodes
        xx = Level["node_coords"][0][nodex]
        yy = Level["node_coords"][1][nodey]
        zz = Level["node_coords"][2][nodez]

        # Bounding box corners
        xc0, xc1 = xx[0], xx[1]
        yc0, yc3 = yy[0], yy[3]
        zc0, zc5 = zz[0], zz[5]

        # Compute shape functions
        Nc = compute3DN(
            [_x, _y, _z], [xc0, xc1], [yc0, yc3], [zc0, zc5], [h_x, h_y, h_z]
        )

        # Clip and mask invalid values
        valid = jnp.logical_and((Nc >= -1e-2).all(), (Nc <= 1 + 1e-2).all())
        Nc = jax.lax.select(valid, jnp.clip(Nc, 0.0, 1.0), jnp.zeros_like(Nc))

        return Nc, node

    _Nc, _node = jax.vmap(stepInterpolatePoints)(jnp.arange(total_points))
    return [_Nc, _node]


@jax.jit
def interpolate_w_matrix(C2F, T):
    """
    Interpolate a solution field to new nodal coordinates using shape functions.

    This function applies precomputed shape functions and node indices
    (from `interpolatePointsMatrix`) to interpolate the solution `T`
    from a source mesh to a new set of points.

    Parameters:
    C2F (tuple): Interpolation data.
                 - C2F[0]: Shape function weights (array of shape [n_new, n_basis]).
                 - C2F[1]: Indices of source nodes (array of shape [n_new, n_basis]).
    T (array): Source solution values at coarse nodes.

    Returns:
    array: Interpolated solution at new nodal coordinates.
    """
    return multiply(C2F[0], T[C2F[1]]).sum(axis=1)

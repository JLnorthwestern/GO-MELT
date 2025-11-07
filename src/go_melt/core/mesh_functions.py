import jax.numpy as jnp


def calc_length_h(A):
    """
    Compute domain lengths and element sizes in each spatial direction.

    This function calculates the physical length of the domain and the
    corresponding element size (grid spacing) in the x, y, and z directions.

    Parameters:
    A (object): Mesh object with attributes:
                - bounds.x, bounds.y, bounds.z: tuples (min, max)
                - elements: tuple (nx, ny, nz) representing number of elements

    Returns:
    tuple:
        - [Lx, Ly, Lz]: Physical domain lengths in x, y, z directions.
        - [hx, hy, hz]: Element sizes in x, y, z directions.
    """
    # Domain length
    Lx = A.bounds.x[1] - A.bounds.x[0]
    Ly = A.bounds.y[1] - A.bounds.y[0]
    Lz = A.bounds.z[1] - A.bounds.z[0]
    # Element length
    hx = Lx / A.elements[0]
    hy = Ly / A.elements[1]
    hz = Lz / A.elements[2]

    return [Lx, Ly, Lz], [hx, hy, hz]


def createMesh3D(x, y, z):
    """
    Generate nodal coordinates and connectivity matrices for a 3D mesh.

    Parameters:
    x (list): [x_min, x_max, num_nodes_x] — bounds and number of nodes in x.
    y (list): [y_min, y_max, num_nodes_y] — bounds and number of nodes in y.
    z (list): [z_min, z_max, num_nodes_z] — bounds and number of nodes in z.

    Returns:
    tuple:
        node_coords (list of jnp.ndarray): Coordinates along x, y, and z axes.
        connect (list of jnp.ndarray): Connectivity matrices for x, y, and z.
    """
    # Node positions in x, y, z
    nx, ny, nz = [jnp.linspace(*axis) for axis in (x, y, z)]

    # Connectivity in x-direction
    cx0 = jnp.arange(0, x[2] - 1).reshape(-1, 1)
    cx1 = jnp.arange(1, x[2]).reshape(-1, 1)
    nconn_x = jnp.concatenate([cx0, cx1, cx1, cx0, cx0, cx1, cx1, cx0], axis=1)

    # Connectivity in y-direction
    cy0 = jnp.arange(0, y[2] - 1).reshape(-1, 1)
    cy1 = jnp.arange(1, y[2]).reshape(-1, 1)
    nconn_y = jnp.concatenate([cy0, cy0, cy1, cy1, cy0, cy0, cy1, cy1], axis=1)

    # Connectivity in z-direction
    cz0 = jnp.arange(0, z[2] - 1).reshape(-1, 1)
    cz1 = jnp.arange(1, z[2]).reshape(-1, 1)
    nconn_z = jnp.concatenate([cz0, cz0, cz0, cz0, cz1, cz1, cz1, cz1], axis=1)

    return [nx, ny, nz], [nconn_x, nconn_y, nconn_z]


def calcNumNodes(elements):
    """
    Increment each of the three input elements by 1.

    Parameters:
    elements (list or tuple of int): A sequence of three integers, typically
    representing node indices or positions.

    Returns:
    list of int: A list where each element is incremented by 1.
    """
    return [elements[0] + 1, elements[1] + 1, elements[2] + 1]


def getSampleCoords(Level):
    """
    Extracts the nodal coordinates of the first element in the mesh.

    Parameters:
    Level (dict): Mesh data for a given level.

    Returns:
    array: (n_nodes, 3) array of x, y, z coordinates.
    """
    x = Level["node_coords"][0][Level["connect"][0][0, :]].reshape(-1, 1)
    y = Level["node_coords"][1][Level["connect"][1][0, :]].reshape(-1, 1)
    z = Level["node_coords"][2][Level["connect"][2][0, :]].reshape(-1, 1)
    return jnp.concatenate([x, y, z], axis=1)


def getSubstrateNodes(Levels):
    """
    Calculate the number of substrate nodes (z ≤ 0) for Levels 1 to 3.

    Parameters:
    Levels (list): A list of level dictionaries, each containing:
        - "node_coords": list of arrays for x, y, z coordinates.
        - "nodes": list of node counts in x, y, z directions.

    Returns:
    tuple: A tuple of substrate node counts for Levels 0 to 3,
           where Level 0 is always 0.
    """
    substrate = [
        ((L["node_coords"][2] < 1e-5).sum() * L["nodes"][0] * L["nodes"][1]).tolist()
        for L in Levels[:4]
    ]
    return tuple(substrate)

import jax.numpy as jnp
from .data_structures import obj


def calc_length_h(Level: obj) -> tuple[list[float], list[float]]:
    """
    Compute domain lengths and element sizes in each spatial direction.

    This function calculates the physical length of the domain and the
    corresponding element size (grid spacing) in the x, y, and z directions.
    """
    # Domain length
    bounds = [Level.bounds.x, Level.bounds.y, Level.bounds.z]
    lengths = [b[1] - b[0] for b in bounds]
    element_length = [L / n for L, n in zip(lengths, Level.elements)]

    return lengths, element_length


def createMesh3D(
    x: list[float, float, int], y: list[float, float, int], z: list[float, float, int]
) -> tuple[list[jnp.ndarray], list[jnp.ndarray]]:
    """
    Generate nodal coordinates and connectivity matrices for a 3D mesh.
    """
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

    # Node positions in x, y, z
    node_coordinates = [jnp.linspace(*axis) for axis in (x, y, z)]

    # Connectivity
    connectivity_matrices = [nconn_x, nconn_y, nconn_z]

    return node_coordinates, connectivity_matrices


def calcNumNodes(elements: list[int]) -> list[int]:
    """
    Increment each of the three input elements by 1 to get number of nodes along each
    axis.
    """
    return [elements[0] + 1, elements[1] + 1, elements[2] + 1]


def getSampleCoords(Level: dict) -> jnp.ndarray:
    """
    Extracts the nodal coordinates of the first element in the mesh.
    """
    x = Level["node_coords"][0][Level["connect"][0][0, :]].reshape(-1, 1)
    y = Level["node_coords"][1][Level["connect"][1][0, :]].reshape(-1, 1)
    z = Level["node_coords"][2][Level["connect"][2][0, :]].reshape(-1, 1)
    return jnp.concatenate([x, y, z], axis=1)


def getSubstrateNodes(Levels: list[dict]) -> tuple[int]:
    """
    Calculate the number of substrate nodes (z ≤ 0) for Levels 1 to 3.
    """
    substrate = [
        ((L["node_coords"][2] < 1e-5).sum() * L["nodes"][0] * L["nodes"][1]).tolist()
        for L in Levels[:4]
    ]
    return tuple(substrate)

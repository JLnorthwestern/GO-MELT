from functools import partial
import jax
import jax.numpy as jnp
from jax.numpy import multiply
from go_melt.utils.gaussian_quadrature_functions import (
    getQuadratureCoords,
    computeQuad3dFemShapeFunctions_jax,
)
from go_melt.utils.helper_functions import convert2XYZ, bincount
from .mesh_functions import getSampleCoords


@partial(jax.jit, static_argnames=["ne_nn"])
def computeLevelSource(
    Levels: list[dict],
    ne_nn: tuple,
    laser_position: jnp.ndarray,
    LevelShape: tuple,
    properties: dict,
    laser_powers: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute the volumetric heat source for Level 1 or Level 2 using Level 3 mesh.

    This function evaluates the laser heat source on the Level 3 mesh using
    quadrature integration, then projects the result onto the target level
    (Level 1 or Level 2) using the provided shape function matrices.
    """
    Level3_num_elems = ne_nn[0][3]
    Level3_to_Target_N = LevelShape[0]
    Level3_to_Target_sum_operator = LevelShape[2]

    # Get quadrature shape functions and weights for Level 3
    coords = getSampleCoords(Levels[3])
    Nf, _, wqf = computeQuad3dFemShapeFunctions_jax(coords)

    def stepLaserPosition(ilaser):
        def stepcomputeCoarseSource(fine_element_index):
            # Convert element index to 3D indices and global node index
            ix, iy, iz, idx = convert2XYZ(
                fine_element_index,
                Levels[3]["elements"][0],
                Levels[3]["elements"][1],
                Levels[3]["nodes"][0],
                Levels[3]["nodes"][1],
            )
            # Get quadrature point coordinates for this element
            x, y, z = getQuadratureCoords(Levels[3], ix, iy, iz, Nf)
            w = wqf

            # Evaluate laser source at quadrature points
            Q = computeSourceFunction_jax(
                x, y, z, laser_position[ilaser], properties, laser_powers[ilaser]
            )
            return Q * w  # Weighted source term

        # Vectorize over all Level 3 elements
        vstepcomputeCoarseSource = jax.vmap(stepcomputeCoarseSource)
        _data = vstepcomputeCoarseSource(jnp.arange(Level3_num_elems))

        # Integrate over elements using shape functions
        _data1tmp = multiply(Level3_to_Target_N, _data).sum(axis=1)
        return _data1tmp

    # Vectorize over all laser positions
    vstepLaserPosition = jax.vmap(stepLaserPosition)

    # Average source over all laser positions
    lshape = laser_position.shape[0]
    _data1 = vstepLaserPosition(jnp.arange(lshape)).sum(axis=0) / lshape

    # Project integrated source to global vector of target level
    return Level3_to_Target_sum_operator @ _data1.reshape(-1)


@partial(jax.jit, static_argnames=["ne_nn"])
def computeSourcesL3(
    Level: dict,
    laser_center: jnp.ndarray,
    ne_nn: tuple,
    properties: dict,
    laser_power: float,
) -> jnp.ndarray:
    """
    Compute the integrated volumetric heat source for Level 3.

    This function evaluates the laser heat source at quadrature points
    using the Level 3 mesh and assembles the global source vector.
    """
    Level3_num_elems = ne_nn[0][3]
    Level3_num_nodes = ne_nn[1][3]

    # Get shape functions and quadrature weights
    coords = getSampleCoords(Level)
    Nf, _, wqf = computeQuad3dFemShapeFunctions_jax(coords)

    def stepcomputeCoarseSource(fine_element_index):
        # Convert element index to 3D indices and global node indices
        ix, iy, iz, idx = convert2XYZ(
            fine_element_index,
            Level["elements"][0],
            Level["elements"][1],
            Level["nodes"][0],
            Level["nodes"][1],
        )
        # Get quadrature point coordinates for this element
        x, y, z = getQuadratureCoords(Level, ix, iy, iz, Nf)
        w = wqf

        # Evaluate laser source at quadrature points
        Q = computeSourceFunction_jax(x, y, z, laser_center, properties, laser_power)

        # Integrate source over element using shape functions
        return Nf @ Q * w, idx

    # Vectorize over all Level 3 elements
    vstepcomputeCoarseSource = jax.vmap(stepcomputeCoarseSource)
    _data3, nodes3 = vstepcomputeCoarseSource(jnp.arange(Level3_num_elems))

    # Assemble global source vector using node indices
    Level3_source = bincount(nodes3.reshape(-1), _data3.reshape(-1), Level3_num_nodes)

    return Level3_source


@partial(jax.jit, static_argnames=["ne_nn"])
def computeSources(
    Level: dict,
    laser_center: jnp.ndarray,
    Shapes: list,
    ne_nn: tuple,
    properties: dict,
    laser_power: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute integrated source terms for all three levels using Level 3 mesh.

    This function evaluates the laser-induced heat source at quadrature
    points of the fine mesh (Level 3), integrates it using shape functions,
    and projects the result to coarser levels using precomputed transfer
    operators.
    """
    Level3_to_Level1_N = Shapes[1][0]
    Level3_to_Level2_N = Shapes[2][0]
    Level3_to_Level1_sum_operator = Shapes[1][2]
    Level3_to_Level2_sum_operator = Shapes[2][2]
    Level3_num_elems = ne_nn[0][3]
    Level3_num_nodes = ne_nn[1][3]

    # Get shape functions and quadrature weights for Level 3 elements
    coords = getSampleCoords(Level)
    Nf, _, wqf = computeQuad3dFemShapeFunctions_jax(coords)

    def stepcomputeCoarseSource(fine_element_index):
        # Get nodal indices for the fine element
        ix, iy, iz, idx = convert2XYZ(
            fine_element_index,
            Level["elements"][0],
            Level["elements"][1],
            Level["nodes"][0],
            Level["nodes"][1],
        )
        # Evaluate coordinates at quadrature points
        x, y, z = getQuadratureCoords(Level, ix, iy, iz, Nf)
        # Evaluate source function at quadrature points
        Q = computeSourceFunction_jax(x, y, z, laser_center, properties, laser_power)
        # Return raw source, integrated source, and node indices
        return Q * wqf, Nf @ Q * wqf, idx

    # Vectorized computation over all Level 3 elements
    vstepcomputeCoarseSource = jax.vmap(stepcomputeCoarseSource)
    _data, _data3, nodes3 = vstepcomputeCoarseSource(jnp.arange(Level3_num_elems))

    # Project source terms to Levels 1 and 2 using transfer operators
    _data1 = multiply(Level3_to_Level1_N, _data).sum(axis=1)
    _data2 = multiply(Level3_to_Level2_N, _data).sum(axis=1)

    Level1_source = Level3_to_Level1_sum_operator @ _data1.reshape(-1)
    Level2_source = Level3_to_Level2_sum_operator @ _data2.reshape(-1)
    Level3_source = bincount(nodes3.reshape(-1), _data3.reshape(-1), Level3_num_nodes)

    return Level1_source, Level2_source, Level3_source


@jax.jit
def computeSourceFunction_jax(
    x: jnp.ndarray,
    y: jnp.ndarray,
    z: jnp.ndarray,
    laser_center_position: jnp.ndarray,
    properties: dict,
    laser_power: float,
) -> jnp.ndarray:
    """
    Compute a 3D Gaussian heat source term for laser-material interaction.

    This function evaluates a separable Gaussian distribution in x, y, and z
    directions centered at the laser.
    """
    # Precompute constants
    _pcoeff = 6 * jnp.sqrt(3) * laser_power * properties["laser_eta"]
    _rcoeff = 1 / (properties["laser_radius"] * jnp.sqrt(jnp.pi))
    _dcoeff = 1 / (properties["laser_depth"] * jnp.sqrt(jnp.pi))
    _rsq = properties["laser_radius"] ** 2
    _dsq = properties["laser_depth"] ** 2

    # Evaluate separable Gaussian in each direction
    Qx = _rcoeff * jnp.exp(-3 * (x - laser_center_position[0]) ** 2 / _rsq)
    Qy = _rcoeff * jnp.exp(-3 * (y - laser_center_position[1]) ** 2 / _rsq)
    Qz = _dcoeff * jnp.exp(-3 * (z - laser_center_position[2]) ** 2 / _dsq)

    return _pcoeff * Qx * Qy * Qz

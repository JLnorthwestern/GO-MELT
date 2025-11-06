from functools import partial
import jax
import jax.numpy as jnp
from .computeFunctions import *
from jax.numpy import multiply
from go_melt.utils.gaussian_quadrature_functions import (
    getQuadratureCoords,
    computeQuad3dFemShapeFunctions_jax,
)
from go_melt.utils.helper_functions import convert2XYZ


@partial(jax.jit, static_argnames=["ne_nn"])
def computeLevelSource(Levels, ne_nn, laser_position, LevelShape, properties, laserP):
    """
    Compute the volumetric heat source for Level 1 or Level 2 using Level 3 mesh.

    This function evaluates the laser heat source on the Level 3 mesh using
    quadrature integration, then projects the result onto the target level
    (Level 1 or Level 2) using the provided shape function matrices.

    Parameters:
    Levels (list of dict): Multilevel simulation state. Level 3 is used for integration.
    ne_nn (tuple): Mesh metadata.
        - ne_nn[1] (int): Number of elements in Level 3.
    laser_position (array): Laser center positions for each time step.
    LevelShape (tuple): FEM shape function data for the target level.
        - LevelShape[0] (array): Shape function values at quadrature points.
        - LevelShape[2] (array): Projection matrix to target level's global vector.
    properties (dict): Material and process properties.
    laserP (array): Laser power or intensity at each time step.

    Returns:
    array: Assembled global heat source vector for the target level.
    """
    # Get quadrature shape functions and weights for Level 3
    coords = getSampleCoords(Levels[3])
    Nf, _, wqf = computeQuad3dFemShapeFunctions_jax(coords)

    def stepLaserPosition(ilaser):
        def stepcomputeCoarseSource(ieltf):
            # Convert element index to 3D indices and global node index
            ix, iy, iz, idx = convert2XYZ(
                ieltf,
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
                x, y, z, laser_position[ilaser], properties, laserP[ilaser]
            )
            return Q * w  # Weighted source term

        # Vectorize over all Level 3 elements
        vstepcomputeCoarseSource = jax.vmap(stepcomputeCoarseSource)
        _data = vstepcomputeCoarseSource(jnp.arange(ne_nn[1]))

        # Integrate over elements using shape functions
        _data1tmp = multiply(LevelShape[0], _data).sum(axis=1)
        return _data1tmp

    # Vectorize over all laser positions
    vstepLaserPosition = jax.vmap(stepLaserPosition)

    # Average source over all laser positions
    lshape = laser_position.shape[0]
    _data1 = vstepLaserPosition(jnp.arange(lshape)).sum(axis=0) / lshape

    # Project integrated source to global vector of target level
    return LevelShape[2] @ _data1.reshape(-1)


@partial(jax.jit, static_argnames=["ne_nn"])
def computeSourcesL3(Level, v, ne_nn, properties, laserP):
    """
    Compute the integrated volumetric heat source for Level 3.

    This function evaluates the laser heat source at quadrature points
    using the Level 3 mesh and assembles the global source vector.

    Parameters:
    Level (dict): Level 3 mesh and field data.
        - Level["elements"]: Element dimensions (nx, ny, nz).
        - Level["nodes"]: Node dimensions (nx, ny, nz).
    v (array): Current laser position.
    ne_nn (tuple): Mesh metadata.
        - ne_nn[1]: Number of elements in Level 3.
        - ne_nn[4]: Total number of nodes in Level 3.
    properties (dict): Material and process properties.
    laserP (float): Laser power at the current position.

    Returns:
    array: Assembled global source vector for Level 3.
    """
    # Get shape functions and quadrature weights
    coords = getSampleCoords(Level)
    Nf, _, wqf = computeQuad3dFemShapeFunctions_jax(coords)

    def stepcomputeCoarseSource(ieltf):
        # Convert element index to 3D indices and global node indices
        ix, iy, iz, idx = convert2XYZ(
            ieltf,
            Level["elements"][0],
            Level["elements"][1],
            Level["nodes"][0],
            Level["nodes"][1],
        )
        # Get quadrature point coordinates for this element
        x, y, z = getQuadratureCoords(Level, ix, iy, iz, Nf)
        w = wqf

        # Evaluate laser source at quadrature points
        Q = computeSourceFunction_jax(x, y, z, v, properties, laserP)

        # Integrate source over element using shape functions
        return Nf @ Q * w, idx

    # Vectorize over all Level 3 elements
    vstepcomputeCoarseSource = jax.vmap(stepcomputeCoarseSource)
    _data3, nodes3 = vstepcomputeCoarseSource(jnp.arange(ne_nn[1]))

    # Assemble global source vector using node indices
    Ff = bincount(nodes3.reshape(-1), _data3.reshape(-1), ne_nn[4])

    return Ff


@partial(jax.jit, static_argnames=["ne_nn"])
def computeSources(Level, v, Shapes, ne_nn, properties, laserP):
    """
    Compute integrated source terms for all three levels using Level 3 mesh.

    This function evaluates the laser-induced heat source at quadrature
    points of the fine mesh (Level 3), integrates it using shape functions,
    and projects the result to coarser levels using precomputed transfer
    operators.

    Parameters:
    Level (dict): Mesh level containing connectivity and geometry for Level 3.
    v (array): Current laser position.
    Shapes (list): Shape transfer operators between Level 3 and Levels 1 & 2.
                   Shapes[1][0]: L3 → L1 interpolation matrix.
                   Shapes[2][0]: L3 → L2 interpolation matrix.
                   Shapes[1][2], Shapes[2][2]: Projection matrices.
    ne_nn (tuple): Element/node counts for each level.
                   ne_nn[1]: Number of elements in Level 3.
                   ne_nn[4]: Number of nodes in Level 3.
    properties (dict): Material and laser properties.
    laserP (float): Laser power.

    Returns:
    Fc (array): Integrated source term for Level 1.
    Fm (array): Integrated source term for Level 2.
    Ff (array): Integrated source term for Level 3.
    """
    # Get shape functions and quadrature weights for Level 3 elements
    coords = getSampleCoords(Level)
    Nf, _, wqf = computeQuad3dFemShapeFunctions_jax(coords)

    def stepcomputeCoarseSource(ieltf):
        # Get nodal indices for the fine element
        ix, iy, iz, idx = convert2XYZ(
            ieltf,
            Level["elements"][0],
            Level["elements"][1],
            Level["nodes"][0],
            Level["nodes"][1],
        )
        # Evaluate coordinates at quadrature points
        x, y, z = getQuadratureCoords(Level, ix, iy, iz, Nf)
        # Evaluate source function at quadrature points
        Q = computeSourceFunction_jax(x, y, z, v, properties, laserP)
        # Return raw source, integrated source, and node indices
        return Q * wqf, Nf @ Q * wqf, idx

    # Vectorized computation over all Level 3 elements
    vstepcomputeCoarseSource = jax.vmap(stepcomputeCoarseSource)
    _data, _data3, nodes3 = vstepcomputeCoarseSource(jnp.arange(ne_nn[1]))

    # Project source terms to Levels 1 and 2 using transfer operators
    _data1 = multiply(Shapes[1][0], _data).sum(axis=1)
    _data2 = multiply(Shapes[2][0], _data).sum(axis=1)

    Fc = Shapes[1][2] @ _data1.reshape(-1)
    Fm = Shapes[2][2] @ _data2.reshape(-1)
    Ff = bincount(nodes3.reshape(-1), _data3.reshape(-1), ne_nn[4])

    return Fc, Fm, Ff


@jax.jit
def computeSourceFunction_jax(x, y, z, v, properties, P):
    """
    Compute a 3D Gaussian heat source term for laser-material interaction.

    This function evaluates a separable Gaussian distribution in x, y, and z
    directions centered at the laser position `v`, scaled by laser power and
    material properties.

    Parameters:
    x, y, z (array): Coordinates of quadrature points.
    v (array): Laser center position [vx, vy, vz].
    properties (dict): Material and laser properties.
                       Required keys:
                       - "laser_eta": laser absorption efficiency.
                       - "laser_radius": standard deviation in x/y.
                       - "laser_depth": standard deviation in z.
    P (float): Laser power.

    Returns:
    array: Evaluated source term at each quadrature point.
    """
    # Precompute constants
    _pcoeff = 6 * jnp.sqrt(3) * P * properties["laser_eta"]
    _rcoeff = 1 / (properties["laser_radius"] * jnp.sqrt(jnp.pi))
    _dcoeff = 1 / (properties["laser_depth"] * jnp.sqrt(jnp.pi))
    _rsq = properties["laser_radius"] ** 2
    _dsq = properties["laser_depth"] ** 2

    # Evaluate separable Gaussian in each direction
    Qx = _rcoeff * jnp.exp(-3 * (x - v[0]) ** 2 / _rsq)
    Qy = _rcoeff * jnp.exp(-3 * (y - v[1]) ** 2 / _rsq)
    Qz = _dcoeff * jnp.exp(-3 * (z - v[2]) ** 2 / _dsq)

    return _pcoeff * Qx * Qy * Qz

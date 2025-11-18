import jax.numpy as jnp
import numpy as np


def compare_values(v1, v2, rtol=1e-4, atol=1e-3):
    """Recursively compare values that may be scalars, arrays, lists, or dicts."""
    # Arrays (NumPy or JAX)
    if isinstance(v1, (np.ndarray, jnp.ndarray)):
        np.testing.assert_allclose(v1, v2, rtol=rtol, atol=atol)
        return

    # Dicts
    if isinstance(v1, dict) and isinstance(v2, dict):
        assert v1.keys() == v2.keys()
        for k in v1:
            compare_values(v1[k], v2[k], rtol=rtol, atol=atol)
        return

    # Lists
    if isinstance(v1, list) and isinstance(v2, list):
        assert len(v1) == len(v2)
        for a, b in zip(v1, v2):
            compare_values(a, b, rtol=rtol, atol=atol)
        return

    # Scalars
    if isinstance(v1, (float, int)) and isinstance(v2, (float, int)):
        assert np.isclose(v1, v2, rtol=rtol, atol=atol)
    else:
        assert v1 == v2

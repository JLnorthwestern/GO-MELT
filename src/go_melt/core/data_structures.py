from typing import NamedTuple
import jax.numpy as jnp


class obj:
    """
    A simple wrapper class to convert a dictionary into an object
    with attributes accessible via dot notation.

    Attributes are dynamically created from the keys and values
    of the input dictionary.

    Example:
        data = {'a': 1, 'b': 2}
        o = obj(data)
        print(o.a)  # Outputs: 1
    """

    def __init__(self, dict1):
        self.__dict__.update(dict1)


class L2Carry_Predictor(NamedTuple):
    T0: jnp.ndarray  # Level 2 temperature
    S1: jnp.ndarray  # Level 2 phase state
    L3T0: jnp.ndarray  # Level 3 temperature
    L3Tprime0: jnp.ndarray  # Level 3 Tprime
    L3S1: jnp.ndarray  # Level 3 phase state S1


class L2Carry_Corrector(NamedTuple):
    T0: jnp.ndarray  # Level 2 temperature
    S1: jnp.ndarray  # Level 2 phase state
    L3T0: jnp.ndarray  # Level 3 temperature
    L3Tprime0: jnp.ndarray  # Level 3 Tprime
    L3S1: jnp.ndarray  # Level 3 phase state S1
    L3S2: jnp.ndarray
    max_accum: jnp.ndarray
    accum: jnp.ndarray


class L3Carry(NamedTuple):
    T0: jnp.ndarray  # Level 3 temperature
    S1: jnp.ndarray  # Level 3 phase state S1
    S2: jnp.ndarray  # Level 3 phase state S2
    max_accum: jnp.ndarray  # Max accumulated melt time
    accum: jnp.ndarray  # Current accumulated melt time

from typing import NamedTuple
import jax.numpy as jnp
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import io


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


class L3Carry_Predictor(NamedTuple):
    T0: jnp.ndarray  # Level 3 temperature
    S1: jnp.ndarray  # Level 3 phase state S1


class L3Carry_Corrector(NamedTuple):
    T0: jnp.ndarray  # Level 3 temperature
    S1: jnp.ndarray  # Level 3 phase state S1
    S2: jnp.ndarray  # Level 3 phase state S2


class SubcycleContext_Predictor(NamedTuple):
    Levels: list[dict]
    ne_nn: tuple[int]
    Shapes: list[list]
    substrate: tuple[int]
    LInterp: list[list]
    laser_position: jnp.ndarray
    laserP: jnp.ndarray
    subcycle: tuple[int, int, int, float, float, float]
    properties: dict
    L1T: jnp.ndarray


class SubcycleContext_Corrector(NamedTuple):
    Levels: list[dict]
    ne_nn: tuple[int]
    Shapes: list[list]
    substrate: tuple[int]
    LInterp: list[list]
    laser_position: jnp.ndarray
    laserP: jnp.ndarray
    subcycle: tuple[int, int, int, float, float, float]
    properties: dict
    L1T: jnp.ndarray
    L3Tp_L2: jnp.ndarray


@dataclass
class SimulationState:
    # Core dictionaries
    Levels: list[dict]
    Nonmesh: dict
    Properties: dict

    # Static numbers (tuples)
    ne_nn: tuple
    substrate: tuple
    tmp_ne_nn: tuple
    subcycle: tuple

    # Unchanging variables
    laser_start: np.ndarray
    L1L2Eratio: list[int]
    L2L3Eratio: list[int]
    total_t_inc: int
    tool_path_file: io.TextIOWrapper
    layer_check: int
    level_names: list[str]

    # Changing variables
    laser_prev_z: float
    time_inc: int
    checkpoint_load: bool
    move_hist: list[jnp.ndarray]
    dwell_time_count: float
    accum_time: jnp.ndarray
    max_accum_time: jnp.ndarray
    record_inc: int
    wait_inc: int
    LInterp: list
    t_add: int
    tstart: float
    t_output: float
    ongoing_simulation: bool

    # Paths
    checkpoint_path: Path

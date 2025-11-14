import pytest
import dill
import jax.numpy as jnp

from go_melt.core.predictor_corrector_functions import stepGOMELT, subcycleGOMELT


def test_stepGOMELT():
    with open("tests/core/inputs/inputs_stepGOMELT.pkl", "rb") as f:
        (
            Levels,
            ne_nn,
            tmp_ne_nn,
            Shapes,
            LInterp,
            laser_position,
            properties,
            dt,
            laser_power,
            substrate,
            max_accum_time,
            accum_time,
            record_accum,
        ) = dill.load(f)
    Levels, max_accum_time, accum_time = stepGOMELT(
        Levels,
        ne_nn,
        tmp_ne_nn,
        Shapes,
        LInterp,
        laser_position,
        properties,
        dt,
        laser_power,
        substrate,
        max_accum_time,
        accum_time,
        record_accum,
    )
    assert isinstance(Levels, list)


def test_subcycleGOMELT():
    with open("tests/core/inputs/inputs_subcycleGOMELT.pkl", "rb") as f:
        (
            Levels,
            ne_nn,
            substrate,
            LInterp,
            tmp_ne_nn,
            laser_all,
            Properties,
            subcycle,
            max_accum_time,
            accum_time,
            laser_start,
            move_hist,
            L1L2Eratio,
            L2L3Eratio,
            record_accum,
        ) = dill.load(f)
        (Levels, L2all, L3pall, move_hist, LInterp, max_accum_time, accum_time) = (
            subcycleGOMELT(
                Levels,
                ne_nn,
                substrate,
                LInterp,
                tmp_ne_nn,
                laser_all,
                Properties,
                subcycle,
                max_accum_time,
                accum_time,
                laser_start,
                move_hist,
                L1L2Eratio,
                L2L3Eratio,
                record_accum,
            )
        )
    assert isinstance(Levels, list)


def test_subcycleGOMELT_TAM():
    with open("tests/core/inputs/inputs_subcycleGOMELT.pkl", "rb") as f:
        (
            Levels,
            ne_nn,
            substrate,
            LInterp,
            tmp_ne_nn,
            laser_all,
            Properties,
            subcycle,
            max_accum_time,
            accum_time,
            laser_start,
            move_hist,
            L1L2Eratio,
            L2L3Eratio,
            record_accum,
        ) = dill.load(f)

        # Turn TAM recording on to access end part of code
        record_accum = 1

        (Levels, L2all, L3pall, move_hist, LInterp, max_accum_time, accum_time) = (
            subcycleGOMELT(
                Levels,
                ne_nn,
                substrate,
                LInterp,
                tmp_ne_nn,
                laser_all,
                Properties,
                subcycle,
                max_accum_time,
                accum_time,
                laser_start,
                move_hist,
                L1L2Eratio,
                L2L3Eratio,
                record_accum,
            )
        )
    assert isinstance(Levels, list)


if __name__ == "__main__":
    pytest.main([__file__])

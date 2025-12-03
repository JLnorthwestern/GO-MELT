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
            LPBF_indicator,
            boundary_conditions,
        ) = dill.load(f)[0]
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
        LPBF_indicator,
        boundary_conditions,
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
            boundary_conditions,
        ) = dill.load(f)[0]
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
                boundary_conditions,
            )
        )
    assert isinstance(Levels, list)


def test_stepGOMELT_TAM():
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
            LPBF_indicator,
            boundary_conditions,
        ) = dill.load(f)[0]

    # Turn TAM recording on to access end part of code
    record_accum = 1

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
        LPBF_indicator,
        boundary_conditions,
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
            boundary_conditions,
        ) = dill.load(f)[0]

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
            boundary_conditions,
        )
    )
    assert isinstance(Levels, list)


def test_stepGOMELT_DED_edge():
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
            LPBF_indicator,
            boundary_conditions,
        ) = dill.load(f)[0]

    # Turn TAM recording on to access end part of code
    LPBF_indicator = False

    # Convert inner tuples/lists to mutable lists
    bc0 = list(boundary_conditions[1][0])
    bc1 = list(boundary_conditions[1][1])
    bc2 = list(boundary_conditions[1][2])

    # Replace with Neumann
    bc0[0] = 1
    bc0[1] = 1
    # Replace functions with Convection (1) / Adiabatic (2)
    bc1[0] = 1
    bc1[1] = 2
    # Replace values with 1e-5 and nan
    bc2[0] = 1e-5
    bc2[1] = jnp.nan

    # Rebuild the outer structure
    boundary_conditions = (
        boundary_conditions[0],
        (tuple(bc0), tuple(bc1), tuple(bc2)) + boundary_conditions[1][3:],
    ) + boundary_conditions[2:]

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
        LPBF_indicator,
        boundary_conditions,
    )
    assert isinstance(Levels, list)


def test_subcycleGOMELT_edge():
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
            boundary_conditions,
        ) = dill.load(f)[0]

    # Convert inner tuples/lists to mutable lists
    bc0 = list(boundary_conditions[1][0])
    bc1 = list(boundary_conditions[1][1])
    bc2 = list(boundary_conditions[1][2])

    # Replace with Neumann
    bc0[0] = 1
    bc0[1] = 1
    # Replace functions with Convection (1) / Adiabatic (2)
    bc1[0] = 1
    bc1[1] = 2
    # Replace values with 1e-5 and nan
    bc2[0] = 1e-5
    bc2[1] = jnp.nan

    # Rebuild the outer structure
    boundary_conditions = (
        boundary_conditions[0],
        (tuple(bc0), tuple(bc1), tuple(bc2)) + boundary_conditions[1][3:],
    ) + boundary_conditions[2:]

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
            boundary_conditions,
        )
    )
    assert isinstance(Levels, list)


if __name__ == "__main__":
    pytest.main([__file__])

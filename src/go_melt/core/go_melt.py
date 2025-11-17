# stdlib
import copy
import gc
import json
import os
import time
from pathlib import Path


# third-party (lightweight at import time)
import dill
import jax.numpy as jnp
import numpy as np

# local package: grouped by subsystem, use module aliases when helpful
from .setup_dictionary_functions import (
    SetupLevels,
    SetupNonmesh,
    SetupProperties,
    SetupStaticNodesAndElements,
    SetupStaticSubcycle,
    calcStaticTmpNodesAndElements,
)
from .mesh_functions import getSubstrateNodes
from .move_mesh_functions import moveEverything
from .predictor_corrector_functions import stepGOMELT, subcycleGOMELT
from .solution_functions import stepGOMELTDwellTime
from go_melt.utils.interpolation_functions import (
    interpolatePoints,
    interpolatePointsMatrix,
)
from go_melt.io.toolpath_functions import count_lines, parsingGcode
from go_melt.io.print_functions import printLevelMaxMin
from go_melt.io.save_results_functions import (
    saveState,
    saveResults,
    saveResultsFinal,
    saveCustom,
    save_object,
)
from .data_structures import SimulationState
from .go_melt_helper_functions import single_step_execution


def go_melt(input_file: Path):
    """
    Main GO-MELT simulation driver. This function initializes the simulation,
    sets up all levels, properties, and toolpath data, and prepares for time stepping.
    Thermal solves using the GO-MELT algorithm are then used.
    """

    with open(input_file, "r") as read_file:
        solver_input = json.load(read_file)

    tstart = time.time()  # Start timer

    # -------------------------------
    # Setup: Properties, Mesh, Nonmesh
    # -------------------------------
    Properties = SetupProperties(solver_input.get("properties", {}))
    Levels = SetupLevels(solver_input, Properties)
    Nonmesh = SetupNonmesh(solver_input.get("nonmesh", {}))

    if Nonmesh["haste"]:
        level_names = ["L1", "L2", "L3", "HASTE"]
    else:
        level_names = ["L1", "L2", "L3"]

    # -------------------------------
    # Static Mesh Metadata
    # -------------------------------
    ne_nn = SetupStaticNodesAndElements(Levels)
    subcycle = SetupStaticSubcycle(Nonmesh)

    # -------------------------------
    # Mesh Ratios for Movement Logic
    # -------------------------------
    L1L2Eratio = [
        int(jnp.round(Levels[1]["h"][i] / Levels[2]["h"][i])) for i in range(2)
    ] + [int(jnp.round(Properties["layer_height"] / Levels[2]["h"][2]))]

    L2L3Eratio = [
        int(jnp.round(Levels[2]["h"][i] / Levels[3]["h"][i])) for i in range(3)
    ]

    # -------------------------------
    # Toolpath Parsing
    # -------------------------------
    if Nonmesh["use_txt"]:
        move_mesh = count_lines(Nonmesh["toolpath"])
    else:
        move_mesh = parsingGcode(Nonmesh, Properties, Levels[2]["h"])

    # -------------------------------
    # Initial Laser Position
    # -------------------------------
    if not Nonmesh["laser_center"]:
        with open(Nonmesh["toolpath"], "r") as tool_path_file:
            laser_start = np.array(
                [float(val) for val in tool_path_file.readline().split(",")]
            )
    else:
        laser_start = np.array(Nonmesh["laser_center"])

    # -------------------------------
    # Interpolation Matrices
    # -------------------------------
    L1L2Interp = interpolatePointsMatrix(Levels[1], Levels[2]["node_coords"])
    L2L3Interp = interpolatePointsMatrix(Levels[2], Levels[3]["node_coords"])
    LInterp = [L1L2Interp, L2L3Interp]

    # -------------------------------
    # Time & Output Initialization
    # -------------------------------
    time_inc = record_inc = wait_inc = 0
    t_output = 0.0
    savenum = int(time_inc / Nonmesh["record_step"]) + 1
    saveResults(Levels, Nonmesh, savenum)

    accum_time = jnp.zeros(Levels[0]["nn"])
    max_accum_time = jnp.zeros(Levels[0]["nn"])

    # -------------------------------
    # Simulation Flags
    # -------------------------------
    ongoing_simulation = single_step = True

    # -------------------------------
    # Toolpath File & Checkpointing
    # -------------------------------
    tool_path_file = open(Nonmesh["toolpath"], "r")
    checkpoint_path = Path(Nonmesh["save_path"] + "checkpoint").absolute()
    layer_check = Nonmesh["layer_num"] + Nonmesh["restart_layer_num"]

    # -------------------------------
    # Load Checkpoint if Requested
    # -------------------------------
    if Nonmesh["layer_num"] > 0:
        print(f"Checkpoint loading for start of Layer {Nonmesh['layer_num']}")
        FILENAME = f"Checkpoint{str(Nonmesh['layer_num']).zfill(4)}.pkl"

        with open(checkpoint_path.joinpath(FILENAME), "rb") as f:
            Levels, accum_time, max_accum_time, time_inc_loaded, record_inc = dill.load(
                f
            )

        checkpoint_load = True
        line_len = len(tool_path_file.readline())
        tool_path_file.seek(time_inc_loaded * line_len)
        time_inc += time_inc_loaded
    else:
        checkpoint_load = False

    simulation_state = SimulationState(
        Levels=Levels,
        Nonmesh=Nonmesh,
        Properties=Properties,
        checkpoint_path=checkpoint_path,
        ne_nn=ne_nn,
        laser_start=laser_start,
        L1L2Eratio=L1L2Eratio,
        L2L3Eratio=L2L3Eratio,
        substrate=len(Levels) * (0,),
        tmp_ne_nn=(0, 0),
        total_t_inc=move_mesh,
        laser_prev_z=float("inf"),
        time_inc=time_inc,
        checkpoint_load=checkpoint_load,
        move_hist=[jnp.array(0), jnp.array(0), jnp.array(0)],
        dwell_time_count=0.0,
        accum_time=accum_time,
        max_accum_time=max_accum_time,
        record_inc=record_inc,
        wait_inc=wait_inc,
        LInterp=LInterp,
    )

    # -------------------------------
    # Start Time Loop
    # -------------------------------
    while ongoing_simulation:
        t_loop = time.time()  # Start timer for this loop

        # -----------------------------------
        # Read laser path for one full subcycle (n2 × n3 lines)
        # -----------------------------------
        _pos = [
            tool_path_file.readline().split(",")
            for _2 in range(subcycle[0] * subcycle[1] * subcycle[-1])
        ]

        # Convert laser path to array if valid, else handle end-of-file
        if [""] not in _pos:
            t_add = subcycle[2] * subcycle[-1]
            laser_all = jnp.array([[float(val) for val in line] for line in _pos])
        else:
            # Set up for partial single time-stepping and end simulation afterwards
            t_add = _pos.index([""])
            laser_all = jnp.array(
                [[float(val) for val in _pos[i]] for i in range(t_add)]
            )
            ongoing_simulation = False
            if t_add == 0:
                break

        # -----------------------------------
        # Determine if single-step is needed
        # -----------------------------------
        single_step = (
            any(laser_all[:, 2] != simulation_state.laser_prev_z)
            or simulation_state.checkpoint_load
            or (
                simulation_state.wait_inc
                > max(
                    0,
                    simulation_state.Nonmesh["wait_time"]
                    - subcycle[0] * subcycle[1] * subcycle[-1] * 2,
                )
            )
            or not ongoing_simulation
            or laser_all.shape[0] == 1
            or (
                jnp.abs(jnp.diff(laser_all, axis=0)[:, :2] / laser_all[:-1, 5].max())
                > (100 * simulation_state.Nonmesh["laser_velocity"])
            ).any()
        )

        # -----------------------------------
        # Single-Step Execution (Equal time step for each Level)
        # -----------------------------------
        if single_step:
            simulation_state = single_step_execution(laser_all, simulation_state)
            # End simulation if final layer reached
            if simulation_state.Nonmesh["layer_num"] == layer_check:
                return

        else:  # Subcycling mode
            # Update wait time if laser is off
            simulation_state.wait_inc = (
                simulation_state.wait_inc + len(laser_all) - laser_all[:, 4].sum()
                if (laser_all[:, 4] == 0).any()
                else 0
            )

            (
                simulation_state.Levels,
                L2all,
                L3pall,
                simulation_state.move_hist,
                simulation_state.LInterp,
                simulation_state.max_accum_time,
                simulation_state.accum_time,
            ) = subcycleGOMELT(
                simulation_state.Levels,
                simulation_state.ne_nn,
                simulation_state.substrate,
                simulation_state.LInterp,
                simulation_state.tmp_ne_nn,
                laser_all,
                simulation_state.Properties,
                subcycle,
                simulation_state.max_accum_time,
                simulation_state.accum_time,
                simulation_state.laser_start,
                simulation_state.move_hist,
                simulation_state.L1L2Eratio,
                simulation_state.L2L3Eratio,
                Nonmesh["record_TAM"],
            )
            gc.collect()

            # Update counters and total elapsed time
            simulation_state.time_inc += t_add
            simulation_state.record_inc += t_add

        # -----------------------------------
        # Output and Monitoring
        # -----------------------------------
        t_output += laser_all[:, 5].sum()

        # Save results if record step reached
        if simulation_state.record_inc >= simulation_state.Nonmesh["record_step"]:
            simulation_state.record_inc = 0
            savenum = (
                int(simulation_state.time_inc / simulation_state.Nonmesh["record_step"])
                + 1
            )
            saveResults(simulation_state.Levels, simulation_state.Nonmesh, savenum)

        # Print temperature info if enabled
        if simulation_state.Nonmesh["info_T"]:
            printLevelMaxMin(simulation_state.Levels, level_names)

        # Timing diagnostics
        tend = time.time()
        t_duration = tend - tstart
        t_now = 1000 * (tend - t_loop)
        t_avg = 1000 * t_duration / simulation_state.time_inc
        execution_time_rem = (
            ((tend - t_loop) / subcycle[2] * subcycle[-1])
            * (simulation_state.total_t_inc - simulation_state.time_inc)
            / 3600
        )

        print(
            "%d/%d, Real: %.6f s, Wall: %.2f s, Loop: %5.2f ms, Avg: %5.2f ms/dt"
            % (
                simulation_state.time_inc,
                simulation_state.total_t_inc,
                t_output,
                t_duration,
                t_now,
                t_avg,
            )
        )
        print(
            "Laser location: X: %.2f, Y: %.2f, Z: %.2f"
            % (laser_all[-1, 0], laser_all[-1, 1], laser_all[-1, 2])
        )
        print(f"Estimated execution time remaining: {execution_time_rem:.4f} hours")

    # -----------------------------------
    # Finalization
    # -----------------------------------
    tool_path_file.close()

    # Save final Level 0 state and temperature fields
    saveState(
        simulation_state.Levels[0],
        "Level0_",
        simulation_state.Nonmesh["layer_num"],
        simulation_state.Nonmesh["save_path"],
        0,
    )
    saveResultsFinal(simulation_state.Levels, simulation_state.Nonmesh)
    if Nonmesh["record_TAM"] == 1:
        saveCustom(
            simulation_state.Levels[0],
            simulation_state.max_accum_time * 1e3,
            "Time Above Melting (ms)",
            simulation_state.Nonmesh["save_path"],
            "max_accum_time",
            0,
        )

    jnp.savez(
        f"{simulation_state.Nonmesh['save_path']}FinalTemperatureFields",
        L1T=simulation_state.Levels[1]["T0"],
        L2T=simulation_state.Levels[2]["T0"],
        L3T=simulation_state.Levels[3]["T0"],
    )

    if Nonmesh["record_TAM"]:
        simulation_state.accum_time = jnp.maximum(
            simulation_state.accum_time, simulation_state.max_accum_time
        )
        jnp.savez(
            simulation_state.Nonmesh["save_path"]
            + "accum_time"
            + str(simulation_state.Nonmesh["layer_num"]).zfill(4),
            accum_time=simulation_state.accum_time,
        )

    # Clear JAX caches
    try:
        stepGOMELT._clear_cache()
        stepGOMELTDwellTime._clear_cache()
        subcycleGOMELT._clear_cache()
        moveEverything._clear_cache()
        gc.collect()
        print("Cleared cache")
    except:
        gc.collect()
        print("Cleared some cache")

    print("End of simulation")

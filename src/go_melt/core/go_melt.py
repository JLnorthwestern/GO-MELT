import os
import sys
import time
from pathlib import Path

import dill
import jax
import jax.numpy as jnp
import numpy as np
from go_melt.core.computeFunctions import *
from go_melt.core.predictor_corrector_functions import stepGOMELT, subcycleGOMELT
from .setup_dictionary_functions import SetupProperties, SetupLevels, SetupNonmesh
from go_melt.io.createPath import parsingGcode, count_lines
import gc
import json
import copy


def go_melt(input_file: Path):
    """
    Main GO-MELT simulation driver. This function initializes the simulation,
    sets up all levels, properties, and toolpath data, and prepares for time stepping.
    Thermal solves using the GO-MELT algorithm are then used.
    """
    with open(input_file, "r") as read_file:
        solver_input = json.load(read_file)

    tstart = time.time()  # Start timer

    level_names = ["L1", "L2", "L3"]

    # -------------------------------
    # Setup: Properties, Mesh, Nonmesh
    # -------------------------------
    Properties = SetupProperties(solver_input.get("properties", {}))
    Levels = SetupLevels(solver_input, Properties)
    Nonmesh = SetupNonmesh(solver_input.get("nonmesh", {}))

    # -------------------------------
    # Static Mesh Metadata
    # -------------------------------
    ne_nn = getStaticNodesAndElements(Levels)
    subcycle = getStaticSubcycle(Nonmesh)

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

    total_t_inc = move_mesh  # Total time steps

    # -------------------------------
    # Initial Laser Position
    # -------------------------------
    if not Properties["laser_center"]:
        with open(Nonmesh["toolpath"], "r") as tool_path_file:
            laser_start = np.array(
                [float(val) for val in tool_path_file.readline().split(",")]
            )
    else:
        laser_start = np.array(Properties["laser_center"])

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

    # -------------------------------
    # Layer Tracking & Accumulation
    # -------------------------------
    laser_prev_z = float("inf")
    _dwell_time_count = 0
    record_accum = True

    if record_accum:
        accum_time = jnp.zeros(Levels[0]["nn"])
        max_accum_time = jnp.zeros(Levels[0]["nn"])

    move_hist = [jnp.array(0), jnp.array(0), jnp.array(0)]  # Laser movement history

    # -------------------------------
    # Simulation Flags
    # -------------------------------
    force_move = move_vert = new_checkpoint = False
    ongoing_simulation = single_step = True

    # -------------------------------
    # Toolpath File & Checkpointing
    # -------------------------------
    tool_path_file = open(Nonmesh["toolpath"], "r")
    np_path = Path(Nonmesh["save_path"] + "checkpoint").absolute()
    layer_check = Nonmesh["layer_num"] + Nonmesh["restart_layer_num"]

    # -------------------------------
    # Load Checkpoint if Requested
    # -------------------------------
    if Nonmesh["layer_num"] > 0:
        print(f"Checkpoint loading for start of Layer {Nonmesh['layer_num']}")
        FILENAME = f"Checkpoint{str(Nonmesh['layer_num']).zfill(4)}.pkl"

        with open(np_path.joinpath(FILENAME), "rb") as f:
            Levels, accum_time, max_accum_time, time_inc_loaded, record_inc = dill.load(
                f
            )

        load_chkpt = True
        line_len = len(tool_path_file.readline())
        tool_path_file.seek(time_inc_loaded * line_len)
        time_inc += time_inc_loaded
    else:
        load_chkpt = False

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
            for _2 in range(subcycle[0] * subcycle[1])
        ]

        # Convert laser path to array if valid, else handle end-of-file
        if [""] not in _pos:
            t_add = subcycle[2]
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
            any(laser_all[:, 2] != laser_prev_z)
            or load_chkpt
            or (wait_inc > max(0, Nonmesh["wait_time"] - subcycle[0] * subcycle[1] * 2))
            or not ongoing_simulation
            or laser_all.shape[0] == 1
            or (
                jnp.abs(jnp.diff(laser_all, axis=0)[:, :2] / laser_all[:-1, 5].max())
                > (100 * Nonmesh["laser_velocity"])
            ).any()
        )

        # -----------------------------------
        # Single-Step Execution (Equal time step for each Level)
        # -----------------------------------
        if single_step:
            for laser_pos in laser_all:
                wait_inc = wait_inc + 1 if laser_pos[4] == 0 else 0

                # Save checkpoint if layer changes and not from checkpoint
                if laser_pos[2] != laser_prev_z and time_inc > 0 and not load_chkpt:
                    new_checkpoint = True
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

                # -----------------------------------
                # Handle Layer Change
                # -----------------------------------
                if laser_pos[2] != laser_prev_z:
                    trying_flag = True
                    tmp_coords = copy.deepcopy(Levels[1]["orig_node_coords"])
                    _L1T_state_idx = 0

                    # Find matching z-layer in Level 1
                    while trying_flag:
                        if jnp.isclose(
                            tmp_coords[2] - laser_pos[2], 0, atol=1e-4
                        ).any():
                            trying_flag = False
                        else:
                            tmp_coords[2] += Properties["layer_height"]
                            _L1T_state_idx += 1

                    # Update Level 1 state if not loading from checkpoint
                    if not load_chkpt:
                        Levels[1]["T0"] = jnp.maximum(
                            interpolatePoints(Levels[1], Levels[1]["T0"], tmp_coords),
                            Properties["T_amb"],
                        )
                        Levels[1]["S1_storage"] = (
                            Levels[1]["S1_storage"]
                            .at[_L1T_state_idx - 1, :]
                            .set(Levels[1]["S1"])
                        )
                        Levels[1]["S1"] = Levels[1]["S1_storage"][_L1T_state_idx, :]
                        Levels[1]["node_coords"] = copy.deepcopy(tmp_coords)

                    # Update interpolation matrices and static node/element info
                    L1L2Interp = interpolatePointsMatrix(
                        Levels[1], Levels[2]["node_coords"]
                    )
                    L2L3Interp = interpolatePointsMatrix(
                        Levels[2], Levels[3]["node_coords"]
                    )
                    LInterp = [L1L2Interp, L2L3Interp]
                    tmp_ne_nn = calcStaticTmpNodesAndElements(Levels, laser_pos)

                    # Update layer tracking and flags
                    laser_prev_z = laser_pos[2]
                    force_move = True
                    wait_inc = 0
                    move_vert = True

                    if not load_chkpt:
                        # Save Level 0 state at the start of a new layer
                        saveState(
                            Levels[0],
                            "Level0_",
                            Nonmesh["layer_num"],
                            Nonmesh["save_path"],
                            0,
                        )

                        if record_accum:
                            # Save accumulated melt time
                            accum_time = jnp.maximum(accum_time, max_accum_time)
                            jnp.savez(
                                Nonmesh["save_path"]
                                + "accum_time"
                                + str(Nonmesh["layer_num"]).zfill(4),
                                accum_time=accum_time,
                            )

                        # Shift Level 0 data down to simulate vertical mesh movement
                        _0nn1 = (
                            Levels[0]["nodes"][0]
                            * Levels[0]["nodes"][1]
                            * Levels[0]["layer_idx_delta"]
                        )
                        _0nn2 = (
                            Levels[0]["nodes"][0]
                            * Levels[0]["nodes"][1]
                            * (Levels[0]["nodes"][2] - Levels[0]["layer_idx_delta"])
                        )

                        Levels[0]["S1"] = (
                            Levels[0]["S1"].at[:_0nn2].set(Levels[0]["S1"][_0nn1:])
                        )
                        Levels[0]["S1"] = Levels[0]["S1"].at[_0nn2:].set(0)

                        # Update z-coordinates for Level 0
                        Levels[0]["node_coords"][2] = (
                            Levels[0]["orig_node_coords"][2]
                            + laser_pos[2]
                            - Levels[0]["orig_node_coords"][2][-1]
                        )

                        if record_accum:
                            max_accum_time = jnp.zeros(Levels[0]["nn"])
                            accum_time = accum_time.at[:_0nn2].set(accum_time[_0nn1:])
                            accum_time = accum_time.at[_0nn2:].set(0)

                force_move = True

                # -----------------------------------
                # Move Meshes if Needed
                # -----------------------------------
                if force_move:
                    force_move = False
                    (Levels, Shapes, LInterp, move_hist) = moveEverything(
                        laser_pos,
                        laser_start,
                        Levels,
                        move_hist,
                        LInterp,
                        L1L2Eratio,
                        L2L3Eratio,
                        Properties["layer_height"],
                    )

                    if move_vert:
                        move_vert = False
                        substrate = getSubstrateNodes(Levels)
                        Levels[0]["S1"] = Levels[0]["S1"].at[: substrate[0]].set(1)
                        print("Start of new layer")

                # -----------------------------------
                # Solve Thermal Fields
                # -----------------------------------
                if wait_inc <= Nonmesh["wait_time"]:
                    # Full GO-MELT step (Levels 1–3)
                    Levels, all_reset = stepGOMELT(
                        Levels,
                        ne_nn,
                        tmp_ne_nn,
                        Shapes,
                        LInterp,
                        laser_pos,
                        Properties,
                        laser_pos[5],  # Time step size
                        laser_pos[6],  # Power
                        substrate,
                    )

                    if Nonmesh["info_T"]:
                        print(f"Step {time_inc + 1} / {total_t_inc}")
                        printLevelMaxMin(Levels, level_names)

                    # Update accumulated melt time
                    if record_accum:
                        _resetaccumtime = accum_time[Levels[0]["idx"]] * (all_reset > 0)
                        _max_check = jnp.maximum(
                            _resetaccumtime, max_accum_time[Levels[0]["idx"]]
                        )
                        max_accum_time = max_accum_time.at[Levels[0]["idx"]].set(
                            _max_check
                        )
                        accum_time = accum_time.at[Levels[0]["idx"]].add(
                            -_resetaccumtime
                        )

                        accum_time = melting_temp(
                            Levels[3]["T0"],
                            laser_pos[5],  # Time step size
                            Properties["T_liquidus"],
                            accum_time,
                            Levels[0]["idx"],
                        )
                else:
                    # Dwell time: only update Level 1
                    if (
                        not (Levels[2]["Tprime0"] == 0).all()
                        and not (Levels[3]["Tprime0"] == 0).all()
                    ):
                        _dwell_time_count = (
                            Nonmesh["wait_time"] * Nonmesh["timestep_L3"]
                        )
                        Levels[2]["Tprime0"] = Levels[2]["Tprime0"].at[:].set(0)
                        Levels[3]["Tprime0"] = Levels[3]["Tprime0"].at[:].set(0)

                    Levels = stepGOMELTDwellTime(
                        Levels,
                        tmp_ne_nn,
                        ne_nn,
                        Properties,
                        laser_pos[5],  # Time step size
                        substrate,
                    )
                    _dwell_time_count += laser_pos[5]
                    print(f"Dwell Time {_dwell_time_count:.6f} s")

                # -----------------------------------
                # Increment Time and Record Counters
                # -----------------------------------
                time_inc += 1
                record_inc += 1

            # -----------------------------------
            # Save Checkpoint if Needed
            # -----------------------------------
            if new_checkpoint:
                Nonmesh["layer_num"] += 1
                print(f"Saving checkpoint for Layer {Nonmesh['layer_num']}")

                FILENAME = f"Checkpoint{str(Nonmesh['layer_num']).zfill(4)}.pkl"
                if not os.path.exists(np_path):
                    os.makedirs(np_path)

                save_object(
                    [Levels, accum_time, max_accum_time, time_inc, record_inc],
                    Path(np_path).joinpath(FILENAME),
                )
                print("Saved Checkpoint")

                # End simulation if final layer reached
                if Nonmesh["layer_num"] == layer_check:
                    return

                new_checkpoint = False
                load_chkpt = True  # Use single-step for next layer
            else:
                load_chkpt = False

        else:  # Subcycling mode
            # Update wait time if laser is off
            wait_inc = (
                wait_inc + len(laser_all) - laser_all[:, 4].sum()
                if (laser_all[:, 4] == 0).any()
                else 0
            )

            # Always move mesh in subcycling
            (Levels, Shapes, LInterp, move_hist) = moveEverything(
                laser_all[0, :],
                laser_start,
                Levels,
                move_hist,
                LInterp,
                L1L2Eratio,
                L2L3Eratio,
                Properties["layer_height"],
            )

            # Extract power values for each substep
            _P = laser_all[:, 6]

            # Run full GO-MELT subcycling
            Levels, L2all, L3all, L3pall, _max_accum, _accum = subcycleGOMELT(
                Levels,
                ne_nn,
                Shapes,
                substrate,
                LInterp,
                tmp_ne_nn,
                laser_all,
                Properties,
                _P,
                subcycle,
                max_accum_time[Levels[0]["idx"]],
                accum_time[Levels[0]["idx"]],
            )
            gc.collect()

            if record_accum:
                max_accum_time = max_accum_time.at[Levels[0]["idx"]].set(_max_accum)
                accum_time = accum_time.at[Levels[0]["idx"]].set(_accum)

            # Update counters and total elapsed time
            time_inc += t_add
            record_inc += t_add

        # -----------------------------------
        # Output and Monitoring
        # -----------------------------------
        t_output += laser_all[:, 5].sum()

        # Save results if record step reached
        if record_inc >= Nonmesh["record_step"]:
            record_inc = 0
            savenum = int(time_inc / Nonmesh["record_step"]) + 1
            saveResults(Levels, Nonmesh, savenum)

        # Print temperature info if enabled
        if Nonmesh["info_T"]:
            printLevelMaxMin(Levels, level_names)

        # Timing diagnostics
        tend = time.time()
        t_duration = tend - tstart
        t_now = 1000 * (tend - t_loop)
        t_avg = 1000 * t_duration / time_inc
        execution_time_rem = (
            ((tend - t_loop) / subcycle[2]) * (total_t_inc - time_inc) / 3600
        )

        print(
            "%d/%d, Real: %.6f s, Wall: %.2f s, Loop: %5.2f ms, Avg: %5.2f ms/dt"
            % (time_inc, total_t_inc, t_output, t_duration, t_now, t_avg)
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
    saveState(Levels[0], "Level0_", Nonmesh["layer_num"], Nonmesh["save_path"], 0)
    saveResultsFinal(Levels, Nonmesh)

    jnp.savez(
        f"{Nonmesh['save_path']}FinalTemperatureFields",
        L1T=Levels[1]["T0"],
        L2T=Levels[2]["T0"],
        L3T=Levels[3]["T0"],
    )

    if record_accum:
        accum_time = jnp.maximum(accum_time, max_accum_time)
        jnp.savez(
            Nonmesh["save_path"] + "accum_time" + str(Nonmesh["layer_num"]).zfill(4),
            accum_time=accum_time,
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

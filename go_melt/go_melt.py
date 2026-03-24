import os
import sys
import time
from pathlib import Path

import dill
import jax
import jax.numpy as jnp
import numpy as np
from computeFunctions import *
from hasteFunctions import *
from createPath import parsingGcode, count_lines
import gc
import json


def go_melt(solver_input: dict):
    """
    Main GO-MELT simulation driver. This function initializes the simulation,
    sets up all levels, properties, and toolpath data, and prepares for time stepping.
    Thermal solves using the GO-MELT algorithm are then used.
    """
    tstart = time.time()  # Start timer

    # -------------------------------
    # Setup: Properties, Mesh, Nonmesh
    # -------------------------------
    Properties = SetupProperties(solver_input.get("properties", {}))
    Levels = SetupLevels(solver_input, Properties)
    Nonmesh = SetupNonmesh(solver_input.get("nonmesh", {}))

    if Nonmesh["haste"]:
        level_names = ["L1", "L2", "L3", "L4"]
    else:
        level_names = ["L1", "L2", "L3"]

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
    saveResults(Levels, Nonmesh, savenum, laser_start)

    # -------------------------------
    # Layer Tracking & Accumulation
    # -------------------------------
    laser_prev_z = float("inf")
    _dwell_time_count = 0
    record_accum = Nonmesh["record_TAM"]

    accum_time = jnp.zeros(Levels[0]["nn"])
    max_accum_time = jnp.zeros(Levels[0]["nn"])

    move_hist = [jnp.array(0), jnp.array(0), jnp.array(0)]  # Laser movement history

    # -------------------------------
    # Simulation Flags
    # -------------------------------
    force_move = move_vert = new_checkpoint = False
    ongoing_simulation = single_step = True

    # -------------------------------
    # HASTE Inputs
    # -------------------------------
    surrogate_count, track, num, train_num = 0, 0, 0, 0
    laser_last = np.zeros([1, 7])
    HASTE_run, HASTE_continue_track, HASTE_ready = False, True, False
    Levels = update_level_center_indices(Levels, ne_nn, Properties)
    T_DA_orig = handle_surrogate_mode(Levels, Nonmesh)
    Test2pos = jnp.zeros(7)
    flip_flag = None
    anti_flip_flag = None
    remaining_HASTE_iterations = 0

    # -------------------------------
    # Toolpath File & Checkpointing
    # -------------------------------
    tool_path_file = open(Nonmesh["toolpath"], "r")
    np_path = Path(Nonmesh["save_path"] + "/checkpoint").absolute()
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

    # Check if any value in x, y, or z is not equal to T_amb
    needs_initial_condition = any(
        any(val != Properties["T_amb"] for val in Levels[1]["conditions"][axis])
        for axis in ["x", "y", "z"]
    )
    if needs_initial_condition:
        (Levels, Shapes, LInterp, move_hist) = moveEverything(
            laser_start,
            laser_start,
            Levels,
            move_hist,
            LInterp,
            L1L2Eratio,
            L2L3Eratio,
            Properties["layer_height"],
        )
        substrate = getSubstrateNodes(Levels)
        Levels[1]["T0"] = computeInitialCondition(Levels[1], Properties, substrate)
        saveResults(Levels, Nonmesh, savenum, laser_start)

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
            any(laser_all[:, 2] != laser_prev_z)
            or load_chkpt
            or (
                wait_inc
                > max(
                    0,
                    Nonmesh["wait_time"] - subcycle[0] * subcycle[1] * subcycle[-1] * 2,
                )
            )
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
            # If running HASTE, reinitialize
            surrogate_count = 0

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
                        HASTE.clear_cache()
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
                                + "/accum_time"
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

                        # For HASTE
                        Test2pos = Test2pos.at[:].set(laser_all[0, :])
                        if Nonmesh["haste"]:
                            Levels[4]["node_coords"] = [
                                Levels[4]["orig_node_coords"][_] + Test2pos[_]
                                for _ in range(3)
                            ]
                        print("Start of new layer")

                # -----------------------------------
                # Solve Thermal Fields
                # -----------------------------------
                if wait_inc <= Nonmesh["wait_time"]:
                    # Full GO-MELT step (Levels 1–3)
                    Levels, max_accum_time, accum_time = stepGOMELT(
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
                        max_accum_time,
                        accum_time,
                        record_accum,
                    )

                    if Nonmesh["info_T"]:
                        print(f"Step {time_inc + 1} / {total_t_inc}")
                        printLevelMaxMin(Levels, level_names)

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

            if Nonmesh.get("haste", False):
                (
                    HASTE_run,
                    HASTE_continue_track,
                    HASTE_ready,
                    surrogate_count,
                    track,
                    num,
                    laser_last,
                    Test2pos,
                    Levels,
                    remaining_HASTE_iterations,
                ) = check_surrogate_conditions(
                    laser_all,
                    laser_last,
                    HASTE_ready,
                    Test2pos,
                    Levels,
                    surrogate_count,
                    track,
                    num,
                    HASTE_continue_track,
                    remaining_HASTE_iterations,
                )
            else:
                HASTE_run = False

            # Always move mesh in subcycling
            if HASTE_run:
                # Run full GO-MELT subcycling
                Levels, move_hist, LInterp, max_accum_time, accum_time = HASTE(
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
                    flip_flag,
                    record_accum,
                    laser_start,
                    move_hist,
                    L1L2Eratio,
                    L2L3Eratio,
                )
                print("Running HASTE")
            else:
                # Run full GO-MELT subcycling
                (
                    Levels,
                    L2all,
                    L3pall,
                    move_hist,
                    LInterp,
                    max_accum_time,
                    accum_time,
                ) = subcycleGOMELT(
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

                if (
                    surrogate_count > Levels[4]["surrogate_count_check"]
                    and HASTE_continue_track
                ):
                    # Construct probe indices from boundary conditions
                    L4probe = jnp.hstack(
                        [Levels[4]["BC"][:-1][i] for i in [1, 0, 2, 3, 4]]
                    )

                    if Nonmesh["training"]:
                        # Update counters for training samples
                        train_num += len(L2all)
                        num += len(L2all)

                        # Loop through training data in reverse
                        for _lp in reversed(range(len(L2all))):
                            train_num -= 1
                            num -= 1

                            # Update test position
                            Test2pos = Test2pos.at[:].set(
                                laser_all[(subcycle[1] * (_lp + 1)) - 1, :]
                            )
                            Levels, _ = getLevel4Pos(Levels, laser_all, Test2pos)

                            # Interpolate temperature from Level 2 and 3 to Level 4
                            L4Tn1 = interpolatePoints(
                                Levels[2], L2all[_lp, :], Levels[4]["node_coords"]
                            ) + interpolatePoints(
                                Levels[3], L3pall[_lp, :], Levels[4]["node_coords"]
                            )

                            # Interpolate surrogate matrix from Level 0 to Level 4
                            SurroLInterpTestL0 = interpolatePointsMatrix(
                                Levels[0], Levels[4]["node_coords"]
                            )
                            Levels[4]["S1"] = (
                                interpolate_w_matrix(
                                    SurroLInterpTestL0, Levels[0]["S1"]
                                )
                                > 1e-3
                            ).astype(int)

                            # On last iteration, compute probe temperature
                            if _lp == len(L2all) - 1:
                                (
                                    flip_flag,
                                    anti_flip_flag,
                                    HASTE_continue_track,
                                    Levels,
                                    _,
                                ) = getflipflag(T_DA_orig, Levels)
                                _L4probe = anti_flip_flag[L4probe]
                                Levels[4]["ProbeT"] = L4Tn1[_L4probe]

                            # Save training data if surrogate tracking is enabled
                            if HASTE_continue_track:
                                save_path = (
                                    f"{Nonmesh['haste_training_dir']}/Track{track:05}/"
                                )
                                os.makedirs(save_path, exist_ok=True)

                                jnp.savez(
                                    f"{save_path}FEA_{num:07}",
                                    s=Levels[4]["S1"][anti_flip_flag],
                                    probe=Levels[4]["ProbeT"],
                                    v=L4Tn1[anti_flip_flag],
                                    track=track,
                                    xpositions=Levels[4]["node_coords"][0],
                                    ypositions=Levels[4]["node_coords"][1],
                                    zpositions=Levels[4]["node_coords"][2],
                                    Test2pos=Test2pos,
                                )
                                HASTE_ready = False

                        # Restore counters
                        train_num += len(L2all)
                        num += len(L2all)
                    else:
                        # Use last laser position for inference
                        Test2pos = Test2pos.at[:].set(laser_all[-1, :])
                        Levels, direction = getLevel4Pos(Levels, laser_all, Test2pos)

                        # Interpolate initial temperature from Level 2 and 3 to Level 4
                        Levels[4]["T0"] = interpolatePoints(
                            Levels[2], Levels[2]["T0"], Levels[4]["node_coords"]
                        ) + interpolatePoints(
                            Levels[3], Levels[3]["Tprime0"], Levels[4]["node_coords"]
                        )

                        # Interpolate surrogate matrix from Level 0 to Level 4
                        SurroLInterpTestL0 = interpolatePointsMatrix(
                            Levels[0], Levels[4]["node_coords"]
                        )
                        Levels[4]["S1"] = (
                            interpolate_w_matrix(SurroLInterpTestL0, Levels[0]["S1"])
                            > 1e-2
                        ).astype(int)

                        # Compute flip flags and surrogate tracking
                        (
                            flip_flag,
                            anti_flip_flag,
                            HASTE_continue_track,
                            Levels,
                            _,
                        ) = getflipflag(T_DA_orig, Levels)

                        if HASTE_continue_track:
                            _L4probe = anti_flip_flag[L4probe]
                            Levels[4]["ProbeIdx"] = _L4probe
                            # Project temperature onto surrogate basis
                            Levels[4]["surrogateA"] = Levels[4]["u"].T @ (
                                Levels[4]["T0"][anti_flip_flag] - Levels[4]["meanA"]
                            )
                            HASTE_ready = True

                            axis_map = {
                                "east": {
                                    "axis": 0,
                                    "sign": 1,
                                    "bounds": Levels[1]["bounds"]["x"],
                                    "coord": Levels[4]["node_coords"][0],
                                },
                                "west": {
                                    "axis": 0,
                                    "sign": -1,
                                    "bounds": Levels[1]["bounds"]["x"],
                                    "coord": Levels[4]["node_coords"][0],
                                },
                                "north": {
                                    "axis": 1,
                                    "sign": 1,
                                    "bounds": Levels[1]["bounds"]["y"],
                                    "coord": Levels[4]["node_coords"][1],
                                },
                                "south": {
                                    "axis": 1,
                                    "sign": -1,
                                    "bounds": Levels[1]["bounds"]["y"],
                                    "coord": Levels[4]["node_coords"][1],
                                },
                            }

                            params = axis_map[direction]
                            axis = params["axis"]
                            sign = params["sign"]
                            bounds = params["bounds"]
                            coord = params["coord"]

                            remaining_length = (
                                (bounds[1] - coord[0])
                                if sign > 0
                                else (coord[0] - bounds[0])
                            )
                            movement_per = (
                                jnp.diff(laser_all, axis=0)[:, axis].mean()
                                * laser_all.shape[0]
                            )
                            number_remaining = remaining_length // jnp.abs(movement_per)

                            coord_range = coord[0] + movement_per * jnp.arange(
                                1, number_remaining + 1
                            )

                            if axis == 0:
                                test_coords = [
                                    coord_range,
                                    jnp.array(
                                        [
                                            Levels[4]["node_coords"][1][0],
                                            Levels[4]["node_coords"][1][-1],
                                        ]
                                    ),
                                    jnp.array([Levels[4]["node_coords"][2][-1]]),
                                ]
                                reshape_axis = 0
                            else:
                                test_coords = [
                                    jnp.array(
                                        [
                                            Levels[4]["node_coords"][0][0],
                                            Levels[4]["node_coords"][0][-1],
                                        ]
                                    ),
                                    coord_range,
                                    jnp.array([Levels[4]["node_coords"][2][-1]]),
                                ]
                                reshape_axis = 1

                            remaining_vector = interpolatePoints(
                                Levels[0], Levels[0]["S1"], test_coords
                            )
                            reshaped = remaining_vector.reshape(
                                test_coords[1].size, test_coords[0].size
                            ).transpose(1, 0)

                            if reshape_axis == 1:
                                reshaped = reshaped.T
                            test_rows = 1 * (reshaped > 0.5)
                            first_ones = jnp.argmax(test_rows == 1, axis=0)
                            has_ones = jnp.any(test_rows == 1, axis=0)
                            first_ones = jnp.where(
                                has_ones, first_ones, test_rows.shape[0]
                            )  # -1 means "not found"
                            first_zeros = jnp.argmax(test_rows == 0, axis=0)
                            has_zeros = jnp.any(test_rows == 0, axis=0)
                            first_zeros = jnp.where(
                                has_zeros, first_zeros, test_rows.shape[0]
                            )  # -1 means "not found"
                            remaining_HASTE_iterations = min(
                                [(first_zeros).max(), (first_ones).max()]
                            )

            gc.collect()

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
            saveResults(Levels, Nonmesh, savenum, laser_all[-1, :], HASTE_run)

            if laser_pos[4] != 0 and "orig_node_coords" in Levels[4]:
                process_levels_and_save(
                    Nonmesh,
                    Test2pos,
                    laser_all,
                    Levels,
                    track,
                    HASTE_run,
                    t_output,
                    savenum,
                    flip_flag,
                )

        # Print temperature info if enabled
        if Nonmesh["info_T"]:
            printLevelMaxMin(Levels, level_names)

        # Timing diagnostics
        tend = time.time()
        t_duration = tend - tstart
        t_now = 1000 * (tend - t_loop)
        t_avg = 1000 * t_duration / time_inc
        execution_time_rem = (
            ((tend - t_loop) / (subcycle[2] * subcycle[-1]))
            * (total_t_inc - time_inc)
            / 3600
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
    saveCustom(
        Levels[0],
        max_accum_time * 1e3,
        "Time Above Melting (ms)",
        Nonmesh["save_path"],
        "max_accum_time",
        0,
    )

    jnp.savez(
        f"{Nonmesh['save_path']}/FinalTemperatureFields",
        L1T=Levels[1]["T0"],
        L2T=Levels[2]["T0"],
        L3T=Levels[3]["T0"],
    )

    if record_accum:
        accum_time = jnp.maximum(accum_time, max_accum_time)
        jnp.savez(
            Nonmesh["save_path"] + "/accum_time" + str(Nonmesh["layer_num"]).zfill(4),
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


if __name__ == "__main__":
    # -------------------------------
    # Parse Command-Line Arguments
    # -------------------------------
    # Usage: python3 run_go_melt.py DEVICE_ID input_file
    DEVICE_ID = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 0
    input_file = sys.argv[2] if len(sys.argv) > 2 else "examples/haste_get_test.json"

    if len(sys.argv) <= 1:
        print("GPU ID not provided. Setting GPU to 0.")
    if len(sys.argv) <= 2:
        print("Input file not provided. Using default: 'examples/haste_get_test.json'.")

    # -------------------------------
    # Set Environment for JAX
    # -------------------------------
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".96"

    try:
        # Attempt to assign GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
    except:
        # Fallback to CPU if GPU assignment fails
        import jax

        jax.config.update("jax_platform_name", "cpu")
        print("No GPU found. Running on CPU.")

    # -------------------------------
    # Load Input File
    # -------------------------------
    try:
        with open(input_file, "r") as read_file:
            solver_input = json.load(read_file)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)

    # -------------------------------
    # Launch GO-MELT Simulation
    # -------------------------------
    print("Running GO-MELT")
    print(f"GPU: {DEVICE_ID}, Input File: {input_file}")
    go_melt(solver_input)

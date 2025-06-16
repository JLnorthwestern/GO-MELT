import os
import sys
import time
from pathlib import Path

import dill
import jax
import jax.numpy as jnp
import numpy as np
from computeFunctions import *
from createPath import parsingGcode, count_lines
import gc
import json


def go_melt(solver_input: dict):
    # Start timestamp for total time elapsed for simulation
    tstart = time.time()

    # Name of levels (L1: Level 1, L2: Level 2, L3: Level 3)
    level_names = ["L1", "L2", "L3"]

    # Set up dictionaries for properties, multilevel information, and nonmesh info
    Properties = SetupProperties(solver_input.get("properties", {}))
    Levels = SetupLevels(solver_input, Properties)
    Nonmesh = SetupNonmesh(solver_input.get("nonmesh", {}))

    # Get total number of elements, nodes, and subcycles for static variable execution
    ne_nn = getStaticNodesAndElements(Levels)
    subcycle = getStaticSubcycle(Nonmesh)

    # Determine ratio between Levels 1 and 2 for mesh movement calculations
    L1L2Eratio = [
        (jnp.round(Levels[1]["h"][0] / Levels[2]["h"][0])).astype(int),
        (jnp.round(Levels[1]["h"][1] / Levels[2]["h"][1])).astype(int),
        (jnp.round(Properties["layer_height"] / Levels[2]["h"][2])).astype(int),
    ]
    # Determine ratio between Levels 1 and 2 for mesh movement calculations
    L2L3Eratio = [
        (jnp.round(Levels[2]["h"][0] / Levels[3]["h"][0])).astype(int),
        (jnp.round(Levels[2]["h"][1] / Levels[3]["h"][1])).astype(int),
        (jnp.round(Levels[2]["h"][2] / Levels[3]["h"][2])).astype(int),
    ]

    # Create the tool path and pre-determine when to move the mesh
    if Nonmesh["use_txt"]:
        move_mesh = count_lines(Nonmesh["toolpath"])
    else:
        move_mesh = parsingGcode(Nonmesh, Properties, Levels[2]["h"])
    # For monitoring purposes, calculate the total number of time steps in this path
    total_t_inc = move_mesh

    # Find initial position of laser

    if len(Properties["laser_center"]) == 0:
        with open(Nonmesh["toolpath"], "r") as tool_path_file:
            _ = tool_path_file.readline()
            laser_start = np.array([float(_i) for _i in _.split(",")])
    else:
        laser_start = np.array(Properties["laser_center"])

    # Set up interpolation [matrices, nodes] for initial move
    L1L2Interp = interpolatePointsMatrix(Levels[1], Levels[2]["node_coords"])
    L2L3Interp = interpolatePointsMatrix(Levels[2], Levels[3]["node_coords"])
    LInterp = [L1L2Interp, L2L3Interp]

    # Initialize time counter, record counter, wait counter, and real-time elapsed
    time_inc, record_inc, wait_inc, t_output = 0, 0, 0, 0.0

    # Save initial condition for all three levels
    savenum = int(time_inc / Nonmesh["record_step"]) + 1
    saveResults(Levels, Nonmesh, savenum)

    # Initialize z coordinate for laser position comparison to identify layer change
    laser_prev_z = float("inf")
    troubleshoot = 0
    _dwell_time_count = 0
    record_accum = True

    if record_accum:
        accum_time = jnp.zeros(Levels[0]["nn"])
        # This is where the resets go
        max_accum_time = jnp.zeros(Levels[0]["nn"])

    # Track total distance moved from laser center (used for moving mesh)
    move_hist = [jnp.array(0), jnp.array(0), jnp.array(0)]

    # Initialize flags for moving meshes, calculating substrate, and saving checkpoint
    force_move, move_vert, new_checkpoint = False, False, False

    # Open laser tool path file
    tool_path_file = open(Nonmesh["toolpath"], "r")

    # Set up boolean flags for running simulation and using single-time stepping
    ongoing_simulation, single_step = True, True

    # Define checkpoint folder
    np_path = Path(Nonmesh["save_path"] + "checkpoint").absolute()

    layer_check = Nonmesh["layer_num"] + Nonmesh["restart_layer_num"]
    # If specified in input folder, load checkpoint based on layer number
    if Nonmesh["layer_num"] > 0:
        print(f"Checkpoint loading for start of Layer {Nonmesh['layer_num']}")
        FILENAME = f"Checkpoint{str(Nonmesh['layer_num']).zfill(4)}.pkl"

        # Load saved results into Levels, and load the current incrementers
        with open(Path(np_path).joinpath(FILENAME), "rb") as f:
            [Levels, accum_time, max_accum_time, time_inc_loaded, record_inc] = (
                dill.load(f)
            )

        # Set load_chkpt flag to True to use single-time stepping for first cycle
        load_chkpt = True

        # Find byte position, then load for fast reload of toolpath
        line_len = len(tool_path_file.readline())
        prev_end = time_inc_loaded * line_len
        tool_path_file.seek(prev_end)
        time_inc += time_inc_loaded
    else:
        # Flag to not use single time stepping due to loading
        load_chkpt = False

    # Start time loop
    while ongoing_simulation:
        # Start time of single time loop execution
        t_loop = time.time()

        # Read n2 x n3 number of lines in laser path file (n#: no. subcycles in Level #)
        _pos = [
            tool_path_file.readline().split(",")
            for _2 in range(subcycle[0] * subcycle[1])
        ]

        # Process laser path into np.array that can be used for calculations
        if [""] not in _pos:
            # Set up for full subcycle or single time-stepping for n2 x n3 steps
            t_add = subcycle[2]
            laser_all = jnp.array([[float(_) for _ in _pos[_2]] for _2 in range(t_add)])
        else:
            # Set up for partial single time-stepping and end simulation afterwards
            t_add = _pos.index([""])
            laser_all = jnp.array([[float(_) for _ in _pos[_2]] for _2 in range(t_add)])
            ongoing_simulation = False
            if t_add == 0:
                break

        # Single step if layer change, checkpoint, dwell time, or end. Else, subcycle.
        if (
            any(laser_all[:, 2] != laser_prev_z)
            or load_chkpt
            or (wait_inc > max(0, Nonmesh["wait_time"] - subcycle[0] * subcycle[1] * 2))
            or not ongoing_simulation
            or laser_all.shape[0] == 1
            or (
                jnp.abs(jnp.diff(laser_all, axis=0)[:, :2] / laser_all[:-1, 5].max())
                > (100 * Nonmesh["laser_velocity"])
            ).any()
        ):
            single_step = True
        else:
            single_step = False

        # Solve each level using the same time step (from the finest level)
        if single_step:
            # Use single step for entire batch
            for laser_pos in laser_all:
                # If dwell time indicator in laser_position file is 0, increment wait
                wait_inc = wait_inc + 1 if laser_pos[4] == 0 else 0

                # If layer changed and not beginning of simulation or after load, save
                if laser_pos[2] != laser_prev_z and time_inc > 0 and not load_chkpt:
                    # Save checkpoint
                    new_checkpoint = True
                    # Clear existing caches to prevent any issues with memory
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

                # If layer changed, calculate updates
                if laser_pos[2] != laser_prev_z:

                    trying_flag = True
                    tmp_coords = copy.deepcopy(Levels[1]["orig_node_coords"])
                    _L1T_state_idx = 0
                    while trying_flag:
                        if jnp.isclose(
                            tmp_coords[2] - laser_pos[2], 0, atol=1e-4
                        ).any():
                            trying_flag = False
                        else:
                            tmp_coords[2] = tmp_coords[2] + Properties["layer_height"]
                            _L1T_state_idx += 1
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
                    L1L2Interp = interpolatePointsMatrix(
                        Levels[1], Levels[2]["node_coords"]
                    )
                    L2L3Interp = interpolatePointsMatrix(
                        Levels[2], Levels[3]["node_coords"]
                    )
                    LInterp = [L1L2Interp, L2L3Interp]

                    # Find new static values for active elements in Level 1
                    tmp_ne_nn = calcStaticTmpNodesAndElements(Levels, laser_pos)

                    # Update layer number comparison
                    laser_prev_z = laser_pos[2]

                    # Move the meshes for Levels 2 and 3
                    force_move = True

                    # Reset the dwell time counter
                    wait_inc = 0

                    # Indicate additional calculation to calculate new substrate nodes
                    move_vert = True

                    if not load_chkpt:
                        # Save Level 0 layer
                        saveState(
                            Levels[0],
                            "Level0_",
                            Nonmesh["layer_num"],
                            Nonmesh["save_path"],
                            0,
                        )
                        if record_accum:
                            accum_time = jnp.maximum(accum_time, max_accum_time)
                            jnp.savez(
                                Nonmesh["save_path"]
                                + "accum_time"
                                + str(Nonmesh["layer_num"]).zfill(4),
                                accum_time=accum_time,
                            )
                        # Shift Level 0 down since moving vertically
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

                # If indicated, update mesh positions for Levels 2 and 3
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
                    # If indicated, calculate new substrate nodes for Levels 1, 2, and 3
                    if move_vert:
                        move_vert = False
                        substrate = getSubstrateNodes(Levels)
                        print("Start of new layer")

                # If not in dwell time, calculate all three levels. Else, just Level 1
                if wait_inc <= Nonmesh["wait_time"]:
                    # Calculate updated temperature and phase fields for all Levels
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
                        troubleshoot += 1
                        print(f"Step {time_inc + 1} / {total_t_inc}")
                        printLevelMaxMin(Levels, level_names)

                    # Update elapsed real time using Level 3 dt
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
                    # Reset subgrid terms for Levels 2 and 3 (important for later moves)
                    if (
                        not (Levels[2]["Tprime0"] == 0).all()
                        and not (Levels[3]["Tprime0"] == 0).all()
                    ):
                        _dwell_time_count = (
                            Nonmesh["wait_time"] * Nonmesh["timestep_L3"]
                        )
                        Levels[2]["Tprime0"] = Levels[2]["Tprime0"].at[:].set(0)
                        Levels[3]["Tprime0"] = Levels[3]["Tprime0"].at[:].set(0)

                    # Calculate updated temperature and phase fields for Level 1
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

                # Increment counters for total time steps and recording data
                time_inc += 1
                record_inc += 1

            # Save current simulation data in checkpoint file
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
                if Nonmesh["layer_num"] == layer_check:
                    return

                # Turn off flag to save new checkpoint
                new_checkpoint = False

                # Using single time steps for the next layer
                load_chkpt = True
            else:
                # Do not use single time-stepping due to checkpoint loading
                load_chkpt = False
        else:  # if single_step:
            # Do subcycling even when power is off. Still increment wait time
            wait_inc = (
                wait_inc + len(laser_all) - laser_all[:, 4].sum()
                if (laser_all[:, 4] == 0).any()
                else 0
            )

            # Always do mesh movement when using subcycling
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

            # Calculate power at each of the laser center positions (can be 0)
            _P = laser_all[:, 6]

            # Update temperature and phase fields for all three levels numerically
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

        t_output += laser_all[:, 5].sum()

        # If not in dwell time, record the temperature field for all three levels
        if record_inc >= Nonmesh["record_step"]:
            record_inc = 0
            savenum = int(time_inc / Nonmesh["record_step"]) + 1
            saveResults(Levels, Nonmesh, savenum)

        # If indicated, print out temperature information for each level
        if Nonmesh["info_T"]:
            printLevelMaxMin(Levels, level_names)

        # Record time needed to complete loop of GO-MELT
        tend = time.time()

        # Calculate timing outputs used for monitoring progress
        t_duration = tend - tstart
        t_now = 1000 * (tend - t_loop)
        t_avg = 1000 * t_duration / time_inc

        # Iteration, Real Time Sim, Wall Time Elapsed, Loop ms/dt, Average ms/dt
        print(
            "%d/%d, Real: %.6f s, Wall: %.2f s, Loop: %5.2f ms, Avg: %5.2f ms/dt"
            % (time_inc, total_t_inc, t_output, t_duration, t_now, t_avg)
        )

    # Close file
    tool_path_file.close()

    # Save final Level 0
    saveState(Levels[0], "Level0_", Nonmesh["layer_num"], Nonmesh["save_path"], 0)
    # Save final temperature fields
    saveResultsFinal(Levels, Nonmesh)
    jnp.savez(
        f"{Nonmesh['save_path']}FinalTemperatureFields",
        L1T=Levels[1]["T0"],
        L2T=Levels[2]["T0"],
        L3T=Levels[3]["T0"],
    )
    if record_accum:
        accum_time = jnp.maximum(accum_time, max_accum_time)
        # Save final accumulation time
        jnp.savez(
            Nonmesh["save_path"] + "accum_time" + str(Nonmesh["layer_num"]).zfill(4),
            accum_time=accum_time,
        )

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
    os.system("clear")

    # Check the number of arguments
    # Usage: python3 run_go_melt.py DEVICE_ID input_file
    # Check DEVICE_ID argument
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        DEVICE_ID = int(sys.argv[1])
    else:
        print("GPU ID not provided. Setting GPU to 0.")
        DEVICE_ID = 0

    # Check input_file argument
    if len(sys.argv) > 2:
        input_file = sys.argv[2]
    else:
        print("Input file not provided. Setting input file to 'examples/example.json'.")
        input_file = "examples/example.json"

    # Initializing GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    try:
        # Always try to run on single GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
    except:
        # Run on CPU
        import jax

        jax.config.update("jax_platform_name", "cpu")
        print("No GPU found. Running on CPU.")

    with open(input_file, "r") as read_file:
        solver_input = json.load(read_file)

    # Run GO-MELT
    print("Running GO-MELT")
    print(f"GPU: {DEVICE_ID}, Input File: {input_file}")
    go_melt(solver_input)

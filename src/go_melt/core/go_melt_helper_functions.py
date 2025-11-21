from .move_mesh_functions import moveEverything
from .predictor_corrector_functions import stepGOMELT, subcycleGOMELT
from .solution_functions import stepGOMELTDwellTime
import gc
import jax.numpy as jnp
import copy
import os
import json
import dill
import time
import numpy as np
from .data_structures import SimulationState
from go_melt.utils.interpolation_functions import (
    interpolatePoints,
    interpolatePointsMatrix,
)
from pathlib import Path
from go_melt.io.save_results_functions import (
    saveState,
    saveResults,
    saveResultsFinal,
    saveCustom,
    save_object,
    record_first_call,
)
from .setup_dictionary_functions import (
    SetupLevels,
    SetupNonmesh,
    SetupProperties,
    SetupStaticNodesAndElements,
    SetupStaticSubcycle,
    calcStaticTmpNodesAndElements,
)
from .mesh_functions import getSubstrateNodes
from go_melt.io.print_functions import printLevelMaxMin
from go_melt.io.toolpath_functions import count_lines, parsingGcode


# @record_first_call("pre_time_loop_initialization")
def pre_time_loop_initialization(input_file: Path) -> SimulationState:
    """
    This function initializes the simulation, sets up all levels, properties, and
    toolpath data, and prepares for time stepping.
    """

    # -------------------------------
    # Load solver input
    # -------------------------------
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

    # -------------------------------
    # Assemble Simulation State
    # -------------------------------
    state = SimulationState(
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
        t_add=0,
        subcycle=subcycle,
        tool_path_file=tool_path_file,
        tstart=tstart,
        t_output=t_output,
        layer_check=layer_check,
        level_names=level_names,
        ongoing_simulation=True,
        new_dwell_flag=True,
        force_move=True,
    )

    return state


# @record_first_call("time_loop_pre_execution")
def time_loop_pre_execution(
    state: SimulationState,
) -> tuple[SimulationState, jnp.ndarray, bool]:
    """Read laser path for one full subcycle and determine execution mode."""

    num_lines_expected = state.subcycle[0] * state.subcycle[1] * state.subcycle[-1]
    raw_lines = [
        state.tool_path_file.readline().strip() for _ in range(num_lines_expected)
    ]

    if "" not in raw_lines:
        state.t_add = state.subcycle[2] * state.subcycle[-1]
        laser_all = jnp.array([parse_line(line) for line in raw_lines])
    else:
        state.t_add = raw_lines.index("")
        laser_all = jnp.array([parse_line(raw_lines[i]) for i in range(state.t_add)])
        state.ongoing_simulation = False
        if state.t_add == 0:
            return state, jnp.array([[]]), False

    # --- Single-step condition checks ---
    z_mismatch = any(laser_all[:, 2] != state.laser_prev_z)
    wait_exceeded = state.wait_inc > max(
        0, state.Nonmesh["wait_time"] - num_lines_expected * 2
    )
    velocity_jump = (
        jnp.abs(jnp.diff(laser_all, axis=0)[:, :2] / laser_all[:-1, 5].max())
        > (100 * state.Nonmesh["laser_velocity"])
    ).any()

    single_step = (
        z_mismatch
        or state.checkpoint_load
        or wait_exceeded
        or not state.ongoing_simulation
        or laser_all.shape[0] == 1
        or velocity_jump
    )

    return state, laser_all, single_step


# @record_first_call("single_step_execution")
def single_step_execution(
    laser_all: jnp.ndarray, state: SimulationState
) -> SimulationState:

    new_checkpoint = False
    for laser_pos in laser_all:
        state.wait_inc = state.wait_inc + 1 if laser_pos[4] == 0 else 0

        # Save checkpoint if layer changes and not from checkpoint
        if (
            laser_pos[2] != state.laser_prev_z
            and state.time_inc > 0
            and not state.checkpoint_load
        ):
            new_checkpoint = True
            clear_jax_function_caches()

        # -----------------------------------
        # Handle Layer Change
        # -----------------------------------
        move_vert = False
        if laser_pos[2] != state.laser_prev_z:
            trying_flag = True
            tmp_coords = copy.deepcopy(state.Levels[1]["orig_node_coords"])
            _L1T_state_idx = 0

            # Find matching z-layer in Level 1
            while trying_flag:
                if jnp.isclose(tmp_coords[2] - laser_pos[2], 0, atol=1e-4).any():
                    trying_flag = False
                else:
                    tmp_coords[2] += state.Properties["layer_height"]
                    _L1T_state_idx += 1

            # Update Level 1 state if not loading from checkpoint
            if not state.checkpoint_load:
                state.Levels[1]["T0"] = jnp.maximum(
                    interpolatePoints(
                        state.Levels[1], state.Levels[1]["T0"], tmp_coords
                    ),
                    state.Properties["T_amb"],
                )
                state.Levels[1]["S1_storage"] = (
                    state.Levels[1]["S1_storage"]
                    .at[_L1T_state_idx - 1, :]
                    .set(state.Levels[1]["S1"])
                )
                state.Levels[1]["S1"] = state.Levels[1]["S1_storage"][_L1T_state_idx, :]
                state.Levels[1]["node_coords"] = copy.deepcopy(tmp_coords)

            # Update interpolation matrices and static node/element info
            L1L2Interp = interpolatePointsMatrix(
                state.Levels[1], state.Levels[2]["node_coords"]
            )
            L2L3Interp = interpolatePointsMatrix(
                state.Levels[2], state.Levels[3]["node_coords"]
            )
            state.LInterp = [L1L2Interp, L2L3Interp]
            state.tmp_ne_nn = calcStaticTmpNodesAndElements(state.Levels, laser_pos)

            # Update layer tracking and flags
            state.laser_prev_z = laser_pos[2]
            state.force_move = True
            state.wait_inc = 0
            move_vert = True

            if not state.checkpoint_load:
                # Save Level 0 state at the start of a new layer
                saveState(
                    state.Levels[0],
                    "Level0_",
                    state.Nonmesh["layer_num"],
                    state.Nonmesh["save_path"],
                    0,
                )

                if state.Nonmesh["record_TAM"]:
                    # Save accumulated melt time
                    state.max_accum_time = jnp.maximum(
                        state.accum_time, state.max_accum_time
                    )
                    jnp.savez(
                        state.Nonmesh["save_path"]
                        + "max_accum_time"
                        + str(state.Nonmesh["layer_num"]).zfill(4),
                        max_accum_time=state.max_accum_time,
                    )
                    saveCustom(
                        state.Levels[0],
                        state.max_accum_time * 1e3,
                        "Time Above Melting (ms)",
                        state.Nonmesh["save_path"],
                        "max_accum_time_" + str(state.Nonmesh["layer_num"]).zfill(4),
                        0,
                    )

                # Shift Level 0 data down to simulate vertical mesh movement
                _0nn1 = (
                    state.Levels[0]["nodes"][0]
                    * state.Levels[0]["nodes"][1]
                    * state.Levels[0]["layer_idx_delta"]
                )
                _0nn2 = (
                    state.Levels[0]["nodes"][0]
                    * state.Levels[0]["nodes"][1]
                    * (state.Levels[0]["nodes"][2] - state.Levels[0]["layer_idx_delta"])
                )

                state.Levels[0]["S1"] = (
                    state.Levels[0]["S1"].at[:_0nn2].set(state.Levels[0]["S1"][_0nn1:])
                )
                state.Levels[0]["S1"] = state.Levels[0]["S1"].at[_0nn2:].set(0)

                # Update z-coordinates for Level 0
                state.Levels[0]["node_coords"][2] = (
                    state.Levels[0]["orig_node_coords"][2]
                    + laser_pos[2]
                    - state.Levels[0]["orig_node_coords"][2][-1]
                )

                if state.Nonmesh["record_TAM"]:
                    state.accum_time = jnp.zeros(state.Levels[0]["nn"])
                    state.max_accum_time = state.max_accum_time.at[:_0nn2].set(
                        state.max_accum_time[_0nn1:]
                    )
                    state.max_accum_time = state.max_accum_time.at[_0nn2:].set(0)

        # -----------------------------------
        # Move Meshes if Needed
        # -----------------------------------
        if state.force_move:
            state.force_move = False
            (state.Levels, Shapes, state.LInterp, state.move_hist) = moveEverything(
                laser_pos,
                state.laser_start,
                state.Levels,
                state.move_hist,
                state.LInterp,
                state.L1L2Eratio,
                state.L2L3Eratio,
                state.Properties["layer_height"],
            )

            if move_vert:
                move_vert = False
                state.substrate = getSubstrateNodes(state.Levels)
                state.Levels[0]["S1"] = (
                    state.Levels[0]["S1"].at[: state.substrate[0]].set(1)
                )
                print("Start of new layer")

        # -----------------------------------
        # Solve Thermal Fields
        # -----------------------------------
        if state.wait_inc <= state.Nonmesh["wait_time"]:
            # Full GO-MELT step (state.Levels 1–3)
            state.Levels, state.max_accum_time, state.accum_time = stepGOMELT(
                state.Levels,
                state.ne_nn,
                state.tmp_ne_nn,
                Shapes,
                state.LInterp,
                laser_pos,
                state.Properties,
                laser_pos[5],  # Time step size
                laser_pos[6],  # Power
                state.substrate,
                state.max_accum_time,
                state.accum_time,
                state.Nonmesh["record_TAM"],
                state.Nonmesh["LPBF"],
            )
            state.force_move = True
            state.new_dwell_flag = True
        else:
            # Dwell time: only update Level 1
            if state.new_dwell_flag:
                state.dwell_time_count = (
                    state.Nonmesh["wait_time"] * state.Nonmesh["timestep_L3"]
                )
                state.Levels[2]["Tprime0"] = state.Levels[2]["Tprime0"].at[:].set(0)
                state.Levels[3]["Tprime0"] = state.Levels[3]["Tprime0"].at[:].set(0)
                state.new_dwell_flag = False

            state.Levels = stepGOMELTDwellTime(
                state.Levels,
                state.tmp_ne_nn,
                state.ne_nn,
                state.Properties,
                laser_pos[5],  # Time step size
                state.substrate,
            )
            state.dwell_time_count += laser_pos[5]

        # -----------------------------------
        # Increment Time and Record Counters
        # -----------------------------------
        state.time_inc += 1
        state.record_inc += 1

    # -----------------------------------
    # Save Checkpoint if Needed
    # -----------------------------------
    if new_checkpoint:
        state.Nonmesh["layer_num"] += 1
        print(f"Saving checkpoint for Layer {state.Nonmesh['layer_num']}")

        FILENAME = f"Checkpoint{str(state.Nonmesh['layer_num']).zfill(4)}.pkl"
        if not os.path.exists(state.checkpoint_path):
            os.makedirs(state.checkpoint_path)

        save_object(
            [
                state.Levels,
                state.accum_time,
                state.max_accum_time,
                state.time_inc,
                state.record_inc,
            ],
            Path(state.checkpoint_path).joinpath(FILENAME),
        )
        print("Saved Checkpoint")

        new_checkpoint = False
        state.checkpoint_load = True  # Use single-step for next layer
    else:
        state.checkpoint_load = False

    return state


# @record_first_call("multi_step_execution")
def multi_step_execution(
    laser_all: jnp.ndarray, state: SimulationState
) -> SimulationState:
    # Update wait time if laser is off
    state.wait_inc = (
        state.wait_inc + len(laser_all) - laser_all[:, 4].sum()
        if (laser_all[:, 4] == 0).any()
        else 0
    )

    (
        state.Levels,
        L2all,
        L3pall,
        state.move_hist,
        state.LInterp,
        state.max_accum_time,
        state.accum_time,
    ) = subcycleGOMELT(
        state.Levels,
        state.ne_nn,
        state.substrate,
        state.LInterp,
        state.tmp_ne_nn,
        laser_all,
        state.Properties,
        state.subcycle,
        state.max_accum_time,
        state.accum_time,
        state.laser_start,
        state.move_hist,
        state.L1L2Eratio,
        state.L2L3Eratio,
        state.Nonmesh["record_TAM"],
    )
    gc.collect()

    # Update counters and total elapsed time
    state.new_dwell_flag = True
    state.time_inc += state.t_add
    state.record_inc += state.t_add
    return state


# @record_first_call("time_loop_post_execution")
def time_loop_post_execution(
    state: SimulationState,
    laser_all: jnp.ndarray,
    t_loop: float,
) -> float:
    """
    Helper function to handle output, monitoring, and diagnostics during simulation.
    """
    print("------------------------------------------------------------")
    # -----------------------------------
    # Output accumulation
    # -----------------------------------
    state.t_output += laser_all[:, 5].sum()

    _x, _y, _z, *_, _t, _p = laser_all[-1]
    print(f"Laser pos. (mm): X: {_x:.2f}, Y: {_y:.2f}, Z: {_z:.2f}")
    print(f"Time step (s): {_t:.1e}; Power (W): {_p:.1f}")

    # Print temperature info if enabled
    if state.Nonmesh.get("info_T", False):
        if state.new_dwell_flag:
            # If not dwell time, print all levels
            printLevelMaxMin(state.Levels, state.level_names)
        else:
            # If in dwell time, only print part-scale
            printLevelMaxMin(state.Levels[:2], state.level_names[:2])

    # Save results if record step reached
    if state.record_inc >= state.Nonmesh["record_step"] and state.new_dwell_flag:
        state.record_inc = 0
        savenum = int(state.time_inc / state.Nonmesh["record_step"]) + 1
        saveResults(state.Levels, state.Nonmesh, savenum)

    # Timing diagnostics
    tend = time.time()
    t_duration = tend - state.tstart
    t_now = 1000 * (tend - t_loop)
    t_avg = 1000 * t_duration / max(state.time_inc, 1)
    execution_time_rem = (
        ((tend - t_loop) / state.subcycle[2] * state.subcycle[-1])
        * (state.total_t_inc - state.time_inc)
        / 3600
    )

    # Print diagnostics
    print("")
    print(f"{state.time_inc}/{state.total_t_inc}, Real: {state.t_output:.6f} s")
    print(f"Wall: {t_duration:.2f} s, Loop: {t_now:5.2f} ms, Avg: {t_avg:5.2f} ms/dt")
    print(f"Estimated execution time remaining: {execution_time_rem:.4f} hours")

    return state


# @record_first_call("post_time_loop_finalization")
def post_time_loop_finalization(state: SimulationState) -> None:
    """
    Perform all finalization tasks after the simulation time loop.

    This helper closes open files, saves final states and results, records
    custom metrics (e.g., Time Above Melting), stores temperature fields,
    and clears JAX caches.
    """
    # -----------------------------------
    # Finalization
    # -----------------------------------
    state.tool_path_file.close()

    # Save final Level 0 state and temperature fields
    saveState(
        state.Levels[0],
        "Level0_",
        state.Nonmesh["layer_num"],
        state.Nonmesh["save_path"],
        0,
    )

    saveResultsFinal(state.Levels, state.Nonmesh)

    jnp.savez(
        f"{state.Nonmesh['save_path']}FinalTemperatureFields",
        L1T=state.Levels[1]["T0"],
        L2T=state.Levels[2]["T0"],
        L3T=state.Levels[3]["T0"],
    )

    if state.Nonmesh.get("record_TAM", 0):
        state.max_accum_time = jnp.maximum(state.accum_time, state.max_accum_time)
        jnp.savez(
            state.Nonmesh["save_path"]
            + "max_accum_time"
            + str(state.Nonmesh["layer_num"]).zfill(4),
            max_accum_time=state.max_accum_time,
        )
        saveCustom(
            state.Levels[0],
            state.max_accum_time * 1e3,
            "Time Above Melting (ms)",
            state.Nonmesh["save_path"],
            "max_accum_time",
            0,
        )

    clear_jax_function_caches()


def parse_line(line: str) -> list[float]:
    """Convert a comma-separated line into a list of floats."""
    return [float(val) for val in line.split(",") if val]


def clear_jax_function_caches(funcs=None):
    """Clear JAX compilation caches using _clear_cache only."""
    if funcs is None:
        funcs = [
            stepGOMELT,
            stepGOMELTDwellTime,
            subcycleGOMELT,
            moveEverything,
        ]
    cleared = []
    for fn in funcs:
        try:
            fn._clear_cache()
            cleared.append(fn.__name__)
        except Exception:
            pass
    gc.collect()
    if cleared:
        print(f"Cleared caches: {', '.join(cleared)}")
    else:
        print("No caches cleared, ran garbage collection")

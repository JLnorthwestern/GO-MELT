from .move_mesh_functions import moveEverything
from .predictor_corrector_functions import stepGOMELT, subcycleGOMELT
from .solution_functions import stepGOMELTDwellTime
import gc
import jax.numpy as jnp
import copy
import os
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


def single_step_execution(
    laser_all: jnp.ndarray, state: SimulationState
) -> SimulationState:
    for laser_pos in laser_all:
        state.wait_inc = state.wait_inc + 1 if laser_pos[4] == 0 else 0

        # Save checkpoint if layer changes and not from checkpoint
        new_checkpoint = False
        if (
            laser_pos[2] != state.laser_prev_z
            and state.time_inc > 0
            and not state.checkpoint_load
        ):
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
            force_move = True
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
                    state.accum_time = jnp.maximum(
                        state.accum_time, state.max_accum_time
                    )
                    jnp.savez(
                        state.Nonmesh["save_path"]
                        + "accum_time"
                        + str(state.Nonmesh["layer_num"]).zfill(4),
                        accum_time=state.accum_time,
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
                    state.max_accum_time = jnp.zeros(state.Levels[0]["nn"])
                    state.accum_time = state.accum_time.at[:_0nn2].set(
                        state.accum_time[_0nn1:]
                    )
                    state.accum_time = state.accum_time.at[_0nn2:].set(0)

        force_move = True

        # -----------------------------------
        # Move Meshes if Needed
        # -----------------------------------
        if force_move:
            force_move = False
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
            )

        else:
            # Dwell time: only update Level 1
            if (
                not (state.Levels[2]["Tprime0"] == 0).all()
                and not (state.Levels[3]["Tprime0"] == 0).all()
            ):
                state.dwell_time_count = (
                    state.Nonmesh["wait_time"] * state.Nonmesh["timestep_L3"]
                )
                state.Levels[2]["Tprime0"] = state.Levels[2]["Tprime0"].at[:].set(0)
                state.Levels[3]["Tprime0"] = state.Levels[3]["Tprime0"].at[:].set(0)

            state.Levels = stepGOMELTDwellTime(
                state.Levels,
                state.tmp_ne_nn,
                state.ne_nn,
                state.Properties,
                laser_pos[5],  # Time step size
                state.substrate,
            )
            state.dwell_time_count += laser_pos[5]
            print(f"Dwell Time {state.dwell_time_count:.6f} s")

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


def clear_jax_function_caches():
    """Clear JAX compilation caches using _clear_cache only."""
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

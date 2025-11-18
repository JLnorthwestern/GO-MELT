# stdlib
import time
from pathlib import Path


# third-party (lightweight at import time)
import jax.numpy as jnp
import numpy as np

# local package: grouped by subsystem, use module aliases when helpful
from go_melt.io.save_results_functions import (
    saveState,
    saveResults,
    saveResultsFinal,
    saveCustom,
)
from .go_melt_helper_functions import (
    pre_time_loop_initialization,
    time_loop_pre_execution,
    single_step_execution,
    multi_step_execution,
    time_loop_post_execution,
    clear_jax_function_caches,
)


def go_melt(input_file: Path):
    """
    Main GO-MELT simulation driver. This function initializes the simulation,
    sets up all levels, properties, and toolpath data, and prepares for time stepping.
    Thermal solves using the GO-MELT algorithm are then used.
    """
    simulation_state = pre_time_loop_initialization(input_file)

    # -------------------------------
    # Start Time Loop
    # -------------------------------
    while simulation_state.ongoing_simulation:
        t_loop = time.time()  # Start timer for this loop

        simulation_state, laser_all, single_step = time_loop_pre_execution(
            simulation_state
        )
        if simulation_state.t_add == 0:
            break

        # -----------------------------------
        # Single-Step Execution (Equal time step for each Level)
        # -----------------------------------
        if single_step:
            simulation_state = single_step_execution(laser_all, simulation_state)
            # End simulation if final layer reached
            if simulation_state.Nonmesh["layer_num"] == simulation_state.layer_check:
                return

        else:  # Subcycling mode
            simulation_state = multi_step_execution(laser_all, simulation_state)

        simulation_state = time_loop_post_execution(simulation_state, laser_all, t_loop)

    # -----------------------------------
    # Finalization
    # -----------------------------------
    simulation_state.tool_path_file.close()

    # Save final Level 0 state and temperature fields
    saveState(
        simulation_state.Levels[0],
        "Level0_",
        simulation_state.Nonmesh["layer_num"],
        simulation_state.Nonmesh["save_path"],
        0,
    )
    saveResultsFinal(simulation_state.Levels, simulation_state.Nonmesh)
    if simulation_state.Nonmesh["record_TAM"] == 1:
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

    if simulation_state.Nonmesh["record_TAM"]:
        simulation_state.accum_time = jnp.maximum(
            simulation_state.accum_time, simulation_state.max_accum_time
        )
        jnp.savez(
            simulation_state.Nonmesh["save_path"]
            + "accum_time"
            + str(simulation_state.Nonmesh["layer_num"]).zfill(4),
            accum_time=simulation_state.accum_time,
        )

    clear_jax_function_caches()

    print("End of simulation")

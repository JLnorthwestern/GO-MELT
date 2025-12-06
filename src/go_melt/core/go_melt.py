import time
from pathlib import Path
from .go_melt_helper_functions import (
    pre_time_loop_initialization,
    time_loop_pre_execution,
    single_step_execution,
    multi_step_execution,
    time_loop_post_execution,
    post_time_loop_finalization,
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

    post_time_loop_finalization(simulation_state)

    print("End of simulation")

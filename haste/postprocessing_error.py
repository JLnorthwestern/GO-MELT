import os
import glob
import jax.numpy as jnp
from jax import jit
from tqdm import tqdm
import matplotlib.pyplot as plt
import jax
import gc
from go_melt.computeFunctions import *
import json
from matplotlib.ticker import FormatStrFormatter
from brokenaxes import brokenaxes
from matplotlib.gridspec import GridSpec
import re


def postprocessing_error(DEVICE_ID, running_file, truth_file):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    try:
        # Run on single GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
        # os.system("clear")
    except:
        # Run on CPU
        import jax

        jax.config.update("jax_platform_name", "cpu")
        # os.system("clear")
        print("No GPU found.")

    # Load input file
    with open(running_file, "r") as read_file:
        solver_input = json.load(read_file)
    with open(truth_file, "r") as read_file:
        solver_input_truth = json.load(read_file)

    Properties = SetupProperties(solver_input.get("properties", {}))
    Levels = SetupLevels(solver_input, Properties)
    Nonmesh = SetupNonmesh(solver_input.get("nonmesh", {}))
    Nonmesh_truth = SetupNonmesh(solver_input_truth.get("nonmesh", {}))

    def get_value_for_mean(truth_folder, run_folder):
        # Get all files in the current track folder and sort them
        truth_files = sorted(glob.glob(os.path.join(truth_folder, "check_error_*.npz")))
        run_files = sorted(glob.glob(os.path.join(run_folder, "check_error_*.npz")))

        # Initialize an empty list to store the variable "v" from each file
        dataset_L2_diff = []
        dataset_melt_diff = []
        dataset_time = []
        dataset_track = []
        dataset_surro = []
        direct_L3_measure = []

        # Ensure both lists have the same length
        if len(truth_files) != len(run_files):
            raise ValueError(
                "The number of truth files and run files must be the same."
            )

        # Iterate over each input file and append the variable "v" to the dataset list
        for truth_file, run_file in tqdm(
            zip(truth_files, run_files), desc="Processing files"
        ):
            # Load the .npz files
            truth_data = jnp.load(truth_file)
            run_data = jnp.load(run_file)

            # Extract the L3T variables
            L3T_truth = truth_data["L3T"]
            L3T_run = run_data["L3T"]
            time = run_data["time"]
            Lsurro = run_data["run_surrogate"]

            _diff = L3T_truth - L3T_run
            diff = jnp.linalg.norm(_diff) / jnp.linalg.norm(L3T_truth)
            direct_L3_measure.append(diff)

            # Reshape the temperature field for correct rendering later
            L3T_truth_3D = np.array(
                L3T_truth.reshape(
                    Levels[3]["nodes"][2], Levels[3]["nodes"][1], Levels[3]["nodes"][0]
                )
            ).transpose((2, 1, 0))
            L3T_run_3D = np.array(
                L3T_run.reshape(
                    Levels[3]["nodes"][2], Levels[3]["nodes"][1], Levels[3]["nodes"][0]
                )
            ).transpose((2, 1, 0))

            indices = find_3d_box_indices(
                L3T_truth_3D, L3T_run_3D, Properties["T_liquidus"]
            )
            # print("The 6 indices that create the 3D box are:", indices)

            if indices is not None:
                L3T_run_melt = copy.deepcopy(
                    L3T_run_3D[
                        indices[0] : indices[1] + 1,
                        indices[2] : indices[3] + 1,
                        indices[4] : indices[5] + 1,
                    ]
                )
                L3T_run_3D[
                    indices[0] : indices[1] + 1,
                    indices[2] : indices[3] + 1,
                    indices[4] : indices[5] + 1,
                ] = 0
                L3T_truth_melt = copy.deepcopy(
                    L3T_truth_3D[
                        indices[0] : indices[1] + 1,
                        indices[2] : indices[3] + 1,
                        indices[4] : indices[5] + 1,
                    ]
                )
                L3T_truth_3D[
                    indices[0] : indices[1] + 1,
                    indices[2] : indices[3] + 1,
                    indices[4] : indices[5] + 1,
                ] = 0
                _melt_diff = L3T_truth_melt - L3T_run_melt
                melt_diff = jnp.linalg.norm(_melt_diff) / jnp.linalg.norm(
                    L3T_truth_melt
                )
            else:
                melt_diff = 0

            # Here the element ratio is 2, will need to change if that changes
            L2T_truth_3D = L3T_truth_3D  # [::2, ::2, ::2]
            L2T_run_3D = L3T_run_3D  # [::2, ::2, ::2]

            _diff = L2T_truth_3D - L2T_run_3D
            diff = jnp.linalg.norm(_diff) / jnp.linalg.norm(L2T_truth_3D)

            dataset_L2_diff.append(diff)
            dataset_melt_diff.append(melt_diff)
            dataset_surro.append(Lsurro)
            dataset_time.append(time)
            dataset_track.append(run_data["track"])

        return (
            jnp.stack(dataset_L2_diff),
            jnp.stack(dataset_melt_diff),
            jnp.stack(direct_L3_measure),
            jnp.stack(dataset_surro),
            jnp.stack(dataset_time),
            jnp.stack(dataset_track),
        )

    # Get all case folders in the specified PASS_Big_Data directory
    # case_folders = sorted(glob.glob("results/PASS_Big_Data/GHOST_*"))
    # case_folders = sorted(
    #     glob.glob("results/PASS_Big_Data/GHOST_length_running_case2")
    #     + glob.glob("results/PASS_Big_Data/GHOST_length_truth_case2")
    # )
    # Save the output variables as a npz file in the specified folder
    output_path = os.path.join(Nonmesh["save_path"], "output_variables_error.npz")

    # # New date to replace
    new_date = "2025_05_02"  # Temp

    # # Replace the date in the file path using regex
    output_path = re.sub(r"\d{4}_\d{2}_\d{2}", new_date, output_path)

    # Check if output_path exists, if it does load it, otherwise run get_value_for_mean
    if os.path.exists(output_path):  # and False:
        data = jnp.load(output_path)
        T2_solns = data["T2_solns"]
        T3_solns = data["T3_solns"]
        direct_L3_measure = data["direct_L3_measure"]
        surro = data["surro"]
        time = data["time"]
        track = data["track"]
    else:
        T2_solns, T3_solns, direct_L3_measure, surro, time, track = get_value_for_mean(
            Nonmesh_truth["save_path"], Nonmesh["save_path"]
        )
        # Save the output variables as a npz file in the specified folder
        jnp.savez(
            output_path,
            T2_solns=T2_solns,
            T3_solns=T3_solns,
            direct_L3_measure=direct_L3_measure,
            surro=surro,
            time=time,
            track=track,
        )
    Total_soln = T3_solns + T2_solns

    def save_plot(
        time,
        T3_solns,
        T2_solns,
        direct_soln,
        Total_soln,
        surro,
        save_path,
        plot_name,
        xlim=None,
        idx=None,
        markers=False,
        legend=False,
    ):
        if idx is not None:
            idx = tuple(idx)
            time = time[idx]
            T3_solns = T3_solns[idx]
            T2_solns = T2_solns[idx]
            direct_soln = direct_soln[idx]
            Total_soln = Total_soln[idx]
            surro = surro[idx]

        fig = plt.figure(figsize=(10, 5))
        # Set larger font sizes for the entire plot
        plt.rcParams.update(
            {
                "font.size": 14,
                "axes.labelsize": 16,
                "xtick.labelsize": 14,
                "ytick.labelsize": 14,
            }
        )

        # Check for large breaks in time
        breaks = jnp.where(jnp.diff(time) > 1)[0]
        if len(breaks) > 0:
            # Create a GridSpec and use it for broken axes
            gs = GridSpec(1, 1, figure=fig)
            bax = brokenaxes(
                xlims=((time[0], time[breaks[0]]), (time[breaks[0] + 1], time[-1])),
                subplot_spec=gs[0],
            )
            ax = bax
        else:
            ax = plt.gca()

        if markers:
            # ax.plot(
            #     time,
            #     Total_soln,
            #     linestyle="-",
            #     marker="o",
            #     color="purple",
            #     label="Total",
            # )
            ax.plot(
                time,
                direct_soln,
                linestyle="-",
                marker="o",
                color="purple",
                label=r"$e$",
            )
            ax.plot(
                time,
                T3_solns,
                linestyle="-",
                marker="^",
                color="darkorange",
                label=r"$e_m$",
                alpha=0.75,
            )
            ax.plot(
                time,
                T2_solns,
                linestyle="-",
                marker="s",
                color="royalblue",
                label=r"$e_f$",
                alpha=0.5,
            )
        else:
            # ax.plot(time, Total_soln, linestyle="-", color="purple", label="Total")
            # ax.plot(
            #     time,
            #     direct_soln,
            #     linestyle="-",
            #     color="purple",
            #     label="Melt pool and field",
            #     alpha=0.5,
            # )
            ax.plot(
                time,
                T3_solns,
                linestyle="-",
                color="darkorange",
                label=r"$e_m$",
                alpha=1,
            )
            ax.plot(
                time,
                T2_solns,
                linestyle="-",
                color="royalblue",
                label=r"$e_f$",
                alpha=0.75,
            )
        ax.fill_between(
            time,
            -1 * jnp.ones_like(surro),
            jnp.ones_like(surro),
            where=surro,
            color="gray",
            alpha=0.2,
            label="HASTE",
        )
        if legend:
            # Create legend
            legend = ax.legend()
            # Set legend background to white
            legend.get_frame().set_facecolor("white")
            legend.get_frame().set_alpha(1.0)

        # ax.legend(facecolor="white", framealpha=1)
        if len(breaks) > 0:
            ax.set_ylabel("Normalized $\\rm{L_2}$ Error", fontsize=21.5, labelpad=60)
        else:
            ax.set_ylabel("Normalized $\\rm{L_2}$ Error", fontsize=21.5)
        _max = jnp.max(jnp.array([direct_soln.max(), T3_solns.max(), T2_solns.max()]))
        ax.set_ylim([0, _max * 1.1])
        fig.supxlabel("Time (s)")

        if xlim:
            ax.set_xlim(xlim)

        plt.savefig(f"{save_path}{plot_name}.png")
        plt.savefig(f"{save_path}{plot_name}.pdf")
        plt.close()

    # Example usage
    save_path = Nonmesh["save_path"]
    # Construct full path pattern
    pattern = os.path.join(save_path, "check_error*")
    # Find and delete matching files
    for filepath in glob.glob(pattern):
        try:
            os.remove(filepath)
            print(f"Deleted: {filepath}")
        except Exception as e:
            print(f"Failed to delete {filepath}: {e}")

    save_plot(
        time,
        T3_solns,
        T2_solns,
        direct_L3_measure,
        Total_soln,
        surro,
        save_path,
        "length_error_values_plot",
    )
    save_plot(
        time,
        T3_solns,
        T2_solns,
        direct_L3_measure,
        Total_soln,
        surro,
        save_path,
        "length_error_values_plot_2",
        xlim=[time.max() - 0.1, time.max()],
        idx=[time > time.max() - 0.2],
        markers=True,
    )
    save_plot(
        time,
        T3_solns,
        T2_solns,
        direct_L3_measure,
        Total_soln,
        surro,
        save_path,
        "length_error_values_plot_2p2",
        xlim=[time.max() - 0.2, time.max()],
        idx=[time > time.max() - 0.3],
        markers=True,
    )
    save_plot(
        time,
        T3_solns,
        T2_solns,
        direct_L3_measure,
        Total_soln,
        surro,
        save_path,
        "length_error_values_plot_2p3",
        xlim=[time.max() - 0.4, time.max()],
        idx=[time > time.max() - 0.5],
        markers=True,
    )
    save_plot(
        time,
        T3_solns,
        T2_solns,
        direct_L3_measure,
        Total_soln,
        surro,
        save_path,
        "length_error_values_plot_3",
        xlim=[0.02, 0.12],
        idx=[time < 0.2],
        markers=True,
    )
    save_plot(
        time,
        T3_solns,
        T2_solns,
        direct_L3_measure,
        Total_soln,
        surro,
        save_path,
        "length_error_values_plot_3p2",
        xlim=[0.12, 0.22],
        idx=[time < 0.3],
        markers=True,
    )
    save_plot(
        time,
        T3_solns,
        T2_solns,
        direct_L3_measure,
        Total_soln,
        surro,
        save_path,
        "length_error_values_plot_3p3",
        xlim=[0.12, 0.32],
        idx=[time < 0.4],
        markers=True,
    )
    save_plot(
        time,
        T3_solns,
        T2_solns,
        direct_L3_measure,
        Total_soln,
        surro,
        save_path,
        "length_error_values_plot_3p4",
        xlim=[0.12, 0.52],
        idx=[time < 0.6],
        markers=False,
    )
    save_plot(
        time[30:140],
        T3_solns[30:140],
        T2_solns[30:140],
        direct_L3_measure[30:140],
        Total_soln[30:140],
        surro[30:140],
        save_path,
        "length_error_values_plot_DMDc",
        xlim=[time[30], time[140]],
        markers=True,
    )
    save_plot(
        time[120:245],
        T3_solns[120:245],
        T2_solns[120:245],
        direct_L3_measure[120:245],
        Total_soln[120:245],
        surro[120:245],
        save_path,
        "legend",
        xlim=[time[120], time[245]],
        markers=True,
        legend=True,
    )
    save_plot(
        time,
        T3_solns,
        T2_solns,
        direct_L3_measure,
        Total_soln,
        surro,
        save_path,
        "legend_long",
        legend=True,
    )

    # print("Saved")
    print(f"Save path: {save_path}")
    print(f"e: {direct_L3_measure.max()}, e_m: {T3_solns.max()}, e_f: {T2_solns.max()}")


def find_3d_box_indices(array1, array2, threshold):
    # Get the indices where the array values are greater than the threshold
    indices1 = np.argwhere(array1 > threshold)
    indices2 = np.argwhere(array2 > threshold)

    # Check for zero-sized arrays and handle accordingly
    if indices1.size == 0 and indices2.size == 0:
        return None
    elif indices1.size == 0:
        min_indices = indices2.min(axis=0)
        max_indices = indices2.max(axis=0)
    elif indices2.size == 0:
        min_indices = indices1.min(axis=0)
        max_indices = indices1.max(axis=0)
    else:
        # Find the minimum and maximum indices along each dimension for both arrays
        min_indices1 = indices1.min(axis=0)
        max_indices1 = indices1.max(axis=0)
        min_indices2 = indices2.min(axis=0)
        max_indices2 = indices2.max(axis=0)

        # Find the overall minimum and maximum indices
        min_indices = np.minimum(min_indices1, min_indices2)
        max_indices = np.maximum(max_indices1, max_indices2)

    # Return the 6 indices that create the 3D box
    return (
        min_indices[0],
        max_indices[0],
        min_indices[1],
        max_indices[1],
        min_indices[2],
        max_indices[2],
    )


if __name__ == "__main__":
    if len(sys.argv) != 4:
        DEVICE_ID = 0

        # List of running and truth file pairs
        cases = [
            (
                "examples/haste_inward_spiral_10x10.json",
                "examples/truth_inward_spiral_10x10.json",
            ),
            (
                "examples/haste_inward_spiral_20x20.json",
                "examples/truth_inward_spiral_20x20.json",
            ),
            (
                "examples/haste_outward_spiral_10x10.json",
                "examples/truth_outward_spiral_10x10.json",
            ),
            (
                "examples/haste_outward_spiral_20x20.json",
                "examples/truth_outward_spiral_20x20.json",
            ),
            (
                "examples/haste_square_10x10.json",
                "examples/truth_square_10x10.json",
            ),
            (
                "examples/haste_square_20x20.json",
                "examples/truth_square_20x20.json",
            ),
            (
                "examples/haste_triangle_10x10.json",
                "examples/truth_triangle_10x10.json",
            ),
            (
                "examples/haste_triangle_20x10.json",
                "examples/truth_triangle_20x10.json",
            ),
            (
                "examples/haste_triangle_10x20.json",
                "examples/truth_triangle_10x20.json",
            ),
            ("examples/haste_boundary.json", "examples/truth_boundary.json"),
        ]
        # Loop through each case and run the plot_length_error function
        for _, (running_file, truth_file) in enumerate(cases):
            postprocessing_error(DEVICE_ID, running_file, truth_file)

        sys.exit(1)

    DEVICE_ID = int(sys.argv[1])
    running_file = sys.argv[2]
    truth_file = sys.argv[3]
    postprocessing_error(DEVICE_ID, running_file, truth_file)

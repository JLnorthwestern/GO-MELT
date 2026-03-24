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
from scipy.interpolate import interp1d


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
        dataset_length_haste = []
        dataset_width_haste = []
        dataset_depth_haste = []
        dataset_length_truth = []
        dataset_width_truth = []
        dataset_depth_truth = []
        dataset_time = []
        dataset_track = []
        dataset_surro = []

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

            # Example usage
            T_liquidus = Properties["T_liquidus"]
            spacing = Levels[3]["h"]  # [dx, dy, dz]

            truth_extent = compute_extent(L3T_truth_3D, T_liquidus, spacing)
            run_extent = compute_extent(L3T_run_3D, T_liquidus, spacing)

            dataset_length_haste.append(run_extent["x"])
            dataset_width_haste.append(run_extent["y"])
            dataset_depth_haste.append(run_extent["z"])
            dataset_length_truth.append(truth_extent["x"])
            dataset_width_truth.append(truth_extent["y"])
            dataset_depth_truth.append(truth_extent["z"])
            dataset_surro.append(Lsurro)
            dataset_time.append(time)
            dataset_track.append(run_data["track"])

        return (
            jnp.stack(dataset_length_haste),
            jnp.stack(dataset_width_haste),
            jnp.stack(dataset_depth_haste),
            jnp.stack(dataset_length_truth),
            jnp.stack(dataset_width_truth),
            jnp.stack(dataset_depth_truth),
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
    output_path = os.path.join(Nonmesh["save_path"], "output_geometry.npz")

    # # New date to replace
    new_date = "2025_05_02"  # Temp

    # # Replace the date in the file path using regex
    output_path = re.sub(r"\d{4}_\d{2}_\d{2}", new_date, output_path)

    # Check if output_path exists, if it does load it, otherwise run get_value_for_mean
    if os.path.exists(output_path):  # and False:
        data = jnp.load(output_path)
        haste_length = data["haste_length"]
        haste_width = data["haste_width"]
        haste_depth = data["haste_depth"]
        truth_length = data["truth_length"]
        truth_width = data["truth_width"]
        truth_depth = data["truth_depth"]
        surro = data["surro"]
        time = data["time"]
        track = data["track"]
    else:
        (
            haste_length,
            haste_width,
            haste_depth,
            truth_length,
            truth_width,
            truth_depth,
            surro,
            time,
            track,
        ) = get_value_for_mean(Nonmesh_truth["save_path"], Nonmesh["save_path"])
        # Save the output variables as a npz file in the specified folder
        jnp.savez(
            output_path,
            haste_length=haste_length,
            haste_width=haste_width,
            haste_depth=haste_depth,
            truth_length=truth_length,
            truth_width=truth_width,
            truth_depth=truth_depth,
            surro=surro,
            time=time,
            track=track,
        )

    # Example usage
    save_path = Nonmesh["save_path"]

    plot_geometry_vs_time(
        time,
        haste_length,
        haste_width,
        haste_depth,
        truth_length,
        truth_width,
        truth_depth,
        surro,
        save_path,
        plot_name="geometry_vs_time",
    )

    # print("Saved")
    print(f"Save path: {save_path}")


import matplotlib.pyplot as plt
import jax.numpy as jnp


def plot_geometry_vs_time(
    time,
    haste_length,
    haste_width,
    haste_depth,
    fem_length,
    fem_width,
    fem_depth,
    surro,
    save_path,
    plot_name="geometry_vs_time",
):
    # Update font sizes for publication-quality visuals
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.labelsize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "axes.titlesize": 20,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=False)

    comparisons = [
        ("Length", haste_length, fem_length),
        ("Width", haste_width, fem_width),
        ("Depth", haste_depth, fem_depth),
    ]

    for ax, (label, haste, fem) in zip(axes, comparisons):
        ax.plot(
            time,
            haste,
            linestyle="-",
            color="crimson",
            label="HASTE",
            alpha=0.9,
            linewidth=3,
        )
        ax.plot(
            time,
            fem,
            linestyle="--",
            color="black",
            label="FEM",
            alpha=0.9,
            linewidth=3,
        )

        ax.fill_between(
            time,
            -100 * jnp.ones_like(surro),
            100 * jnp.ones_like(surro),
            where=surro,
            color="gray",
            alpha=0.3,
            label=None,
        )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"{label} (mm)")
        ax.grid(True, linewidth=1.5)

        if label == "Length":
            ax.legend(
                loc="lower left",
                frameon=True,
                framealpha=1.0,  # fully opaque
                facecolor="white",  # background color
                edgecolor="black",  # border color
            )

        y_min = float(jnp.min(jnp.array([haste, fem])))
        y_max = float(jnp.max(jnp.array([haste, fem])))
        ax.set_ylim([y_min * 0.95, y_max * 1.05])

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(f"{save_path}{plot_name}.png", dpi=300)
    plt.savefig(f"{save_path}{plot_name}.pdf")
    plt.close()


def compute_extent(temp_field, T_liquidus, spacing):
    shape = temp_field.shape
    extent = {}

    for axis, axis_name in zip(range(3), ["x", "y", "z"]):
        above_liquidus = []

        # Collect max temperature per slice along the axis
        for idx in range(shape[axis]):
            slicer = [slice(None)] * 3
            slicer[axis] = idx
            slice_data = temp_field[tuple(slicer)]
            above_liquidus.append(np.max(slice_data) > T_liquidus)

        # Find longest continuous stretch
        max_len = 0
        start_idx = end_idx = None
        i = 0
        while i < len(above_liquidus):
            if above_liquidus[i]:
                j = i
                while j < len(above_liquidus) and above_liquidus[j]:
                    j += 1
                if j - i > max_len:
                    max_len = j - i
                    start_idx, end_idx = i, j - 1
                i = j
            else:
                i += 1

        if start_idx is None or end_idx is None:
            extent[axis_name] = 0.0
            continue

        # Interpolate start
        if start_idx > 0:
            slicer_prev = [slice(None)] * 3
            slicer_prev[axis] = start_idx - 1
            slicer_curr = [slice(None)] * 3
            slicer_curr[axis] = start_idx
            val_prev = np.max(temp_field[tuple(slicer_prev)])
            val_curr = np.max(temp_field[tuple(slicer_curr)])
            if val_curr != val_prev:
                f_start = interp1d(
                    [val_prev, val_curr],
                    [(start_idx - 1) * spacing[axis], start_idx * spacing[axis]],
                )
                start_pos = f_start(T_liquidus)
            else:
                start_pos = start_idx * spacing[axis]
        else:
            start_pos = start_idx * spacing[axis]

        # Interpolate end
        if end_idx < shape[axis] - 1:
            slicer_curr = [slice(None)] * 3
            slicer_curr[axis] = end_idx
            slicer_next = [slice(None)] * 3
            slicer_next[axis] = end_idx + 1
            val_curr = np.max(temp_field[tuple(slicer_curr)])
            val_next = np.max(temp_field[tuple(slicer_next)])
            if val_curr != val_next:
                f_end = interp1d(
                    [val_curr, val_next],
                    [end_idx * spacing[axis], (end_idx + 1) * spacing[axis]],
                )
                end_pos = f_end(T_liquidus)
            else:
                end_pos = end_idx * spacing[axis]
        else:
            end_pos = end_idx * spacing[axis]

        extent[axis_name] = float(end_pos - start_pos)

    return extent


if __name__ == "__main__":
    if len(sys.argv) != 4:
        DEVICE_ID = 0

        # List of running and truth file pairs
        cases = [
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

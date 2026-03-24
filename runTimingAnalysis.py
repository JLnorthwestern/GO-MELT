import glob
import dill
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from go_melt.hasteFunctions import *
from go_melt.computeFunctions import *

# -------------------------------
# Set Environment for JAX
# -------------------------------
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".96"

try:
    # Attempt to assign GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(4)
except:
    # Fallback to CPU if GPU assignment fails
    import jax

    jax.config.update("jax_platform_name", "cpu")
    print("No GPU found. Running on CPU.")

# Load existing timing data
timing_path = "test_data/all_times.pkl"
if os.path.exists(timing_path):
    with open(timing_path, "rb") as f:
        all_times = dill.load(f)
    print("Loaded saved timing data.")
else:
    all_times = {}

iterations = 200

# Get all .pkl files
truth_files = glob.glob("test_data/Truth*.pkl")
haste_files = glob.glob("test_data/HASTE*.pkl")

# Filter out files that have already been timed
new_truth_files = [
    f for f in truth_files if f.split("/")[-1].replace(".pkl", "") not in all_times
]
new_haste_files = [
    f for f in haste_files if f.split("/")[-1].replace(".pkl", "") not in all_times
]

for FILENAME in new_truth_files:
    with open(FILENAME, "rb") as f:
        (
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
            Nonmesh,
        ) = dill.load(f)

    for idx in range(13 + iterations):
        if idx == 13:
            start_time = time.time()
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

    # End timer
    elapsed_time = (time.time() - start_time) * 1e3 / iterations
    subcycleGOMELT._clear_cache()

    # Extract base name without extension
    tag = FILENAME.split("/")[-1].replace(".pkl", "")
    all_times[tag] = elapsed_time
    print(f"{tag}: {elapsed_time:.2f} ms")
    # Save updated timing data
    with open(timing_path, "wb") as f:
        dill.dump(all_times, f)
    print("Updated timing data saved.")


file_list = glob.glob("test_data/HASTE*.pkl")
for FILENAME in new_haste_files:
    with open(FILENAME, "rb") as f:
        (
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
            Nonmesh,
        ) = dill.load(f)

    for idx in range(13 + iterations):
        if idx == 13:
            start_time = time.time()
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

    # End timer
    elapsed_time = (time.time() - start_time) * 1e3 / iterations
    HASTE._clear_cache()

    # Extract base name without extension
    tag = FILENAME.split("/")[-1].replace(".pkl", "")
    all_times[tag] = elapsed_time
    print(f"{tag}: {elapsed_time:.2f} ms")

    # Save updated timing data
    with open(timing_path, "wb") as f:
        dill.dump(all_times, f)
    print("Updated timing data saved.")

# Save updated timing data
with open(timing_path, "wb") as f:
    dill.dump(all_times, f)
print("Updated timing data saved.")

# Print the median loop times
for base_name, median_loop_time in all_times.items():
    print(f"Median loop time for {base_name}: {median_loop_time}")


# Separate data for HASTE and TRUTH
haste_data = {k: v for k, v in all_times.items() if "HASTE" in k}
truth_data = {k: v for k, v in all_times.items() if "Truth" in k}


# Further separate data into base, Lil1, Big1, Lil2, etc.
categories = ["base", "Lil1", "Big1", "Lil2", "Big2", "Lil3", "Big3"]

haste_category_data = {
    category: {k: v for k, v in haste_data.items() if category in k}
    for category in categories
}
truth_category_data = {
    category: {k: v for k, v in truth_data.items() if category in k}
    for category in categories
}

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Define pairs for subfigures
pairs = [("Big1", "Lil1"), ("Big2", "Lil2"), ("Big3", "Lil3")]

# Define titles for subplots
titles = ["Level 1 Mesh Scaling", "Level 2 Mesh Scaling", "Level 3 Mesh Scaling"]

ax_label = ["a)", "b)", "c)"]

max_time = 0

for ax, (big, lil), title, ax_fig_label in zip(axes, pairs, titles, ax_label):
    fem_handles = []
    haste_handles = []
    for category in [big, "base", lil]:
        haste_numbers = [
            int(k.split("_")[-1]) for k in haste_category_data[category].keys()
        ]
        haste_times = list(haste_category_data[category].values())

        truth_numbers = [
            int(k.split("_")[-1]) for k in truth_category_data[category].keys()
        ]
        truth_times = list(truth_category_data[category].values())

        # Plotting
        # haste_numbers = 5 * np.array(haste_numbers)
        haste_numbers = np.array(haste_numbers)
        haste_times = (
            np.array(haste_times) / 2
        )  # Divide by 2 since doing 2 Level 1 time steps in a loop
        # truth_numbers = 5 * np.array(truth_numbers)
        truth_numbers = np.array(truth_numbers)
        truth_times = (
            np.array(truth_times) / 2
        )  # Divide by 2 since doing 2 Level 1 time steps in a loop

        filtered_haste_numbers = haste_numbers
        filtered_haste_times = haste_times
        filtered_truth_numbers = truth_numbers
        filtered_truth_times = truth_times

        h_idx = np.argsort(filtered_haste_numbers)
        t_idx = np.argsort(filtered_truth_numbers)

        if category == "Big1":
            leg_label = r"Level 1 DOF: $3.8 \times 10^5$"
            color = "green"
        elif category == "Lil1":
            leg_label = r"Level 1 DOF: $1.1 \times 10^5$"
            color = "blue"
        elif category == "Big2":
            leg_label = r"Level 2 DOF: $1.1 \times 10^6$"
            color = "green"
        elif category == "Lil2":
            leg_label = r"Level 2 DOF: $6.5 \times 10^5$"
            color = "blue"
        elif category == "Big3":
            leg_label = r"Level 3 DOF: $8.0 \times 10^5$"
            color = "green"
        elif category == "Lil3":
            leg_label = r"Level 3 DOF: $2.9 \times 10^5$"
            color = "blue"
        else:
            if big == "Big1":
                leg_label = r"Level 1 DOF: $2.0 \times 10^5$"
                color = "purple"
            elif big == "Big2":
                leg_label = r"Level 2 DOF: $8.5 \times 10^5$"
                color = "purple"
            else:
                leg_label = r"Level 3 DOF: $5.4 \times 10^5$"
                color = "purple"

            # leg_label = "Baseline"

        (haste_handle,) = ax.plot(
            filtered_haste_numbers[h_idx],
            filtered_haste_times[h_idx],
            marker="o",
            color=color,
            linestyle="--",
            label=f"HASTE {leg_label}",
        )
        (fem_handle,) = ax.plot(
            filtered_truth_numbers[t_idx],
            filtered_truth_times[t_idx],
            marker="^",
            color=color,
            linestyle="-",
            label=f"FEM {leg_label}",
        )

        # max_time = max(
        #     [max_time, filtered_haste_times.max(), filtered_truth_times.max()]
        # )

        fem_handles.append(fem_handle)
        haste_handles.append(haste_handle)

    # ax.set_xlabel(r"$\Delta t_1 / \Delta t_3 = 5 \Delta t_2 / \Delta t_3$", fontsize=14)
    ax.set_xlabel(r"m_2 = $\Delta t_2 / \Delta t_3$", fontsize=14)
    ax.set_ylabel(r"Median execution time per $\Delta t_1$ (ms)", fontsize=14)
    # if ax == axes[0]:  # Only add legend to the first subplot
    ax.legend(handles=fem_handles + haste_handles, loc="upper left", fontsize=11)
    # ax.set_ylim([0, np.ceil(max_time / 100) * 100])
    ax.grid("on")
    ax.xaxis.set_major_locator(
        MaxNLocator(integer=True)
    )  # Ensure x-axis shows only integer values
    ax.tick_params(
        axis="both", which="major", labelsize=12
    )  # Set font size for tick labels
    ax.set_title(title, fontsize=16)  # Add subplot title
    ax.text(
        -0.1,
        1.1,
        ax_fig_label,
        fontsize=20,
        ha="center",
        va="center",
        transform=ax.transAxes,
    )

plt.tight_layout()
plt.savefig(f"results/FigureTimingComparisonHASTE.pdf")
plt.savefig(f"results/FigureTimingComparisonHASTE.png")
plt.close()


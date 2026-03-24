import csv
import re
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import textwrap
from sklearn.linear_model import LinearRegression


# Function to process a single file and extract required information
def process_file(file_path, output_folder):
    with open(file_path, "r") as file:
        lines = file.readlines()

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    current_csv = None
    csv_writer = None
    # compare_csv = None
    # compare_csv_writer = None

    # Create new CSV files in the output folder
    current_csv = open(
        os.path.join(output_folder, f"{base_name}.csv"),
        "w",
        newline="",
    )
    csv_writer = csv.writer(current_csv)
    # compare_csv = open(
    #     os.path.join(output_folder, f"{base_name}_compare.csv"),
    #     "w",
    #     newline="",
    # )
    # compare_csv_writer = csv.writer(compare_csv)

    # Write the header rows
    csv_writer.writerow(["First Integer", "Loop Time (ms)", "Wall Time (s)"])
    # compare_csv_writer.writerow(["First Integer", "Loop Time (ms)", "Wall Time (s)"])

    # Convert the list of lines into an iterator
    lines_iter = iter(lines)

    for line in lines_iter:
        # Check for new simulation line
        if line.startswith("NEW SIMULATION:"):
            # Close the previous CSV files if they exist
            if current_csv:
                current_csv.close()
            # if compare_csv:
            #     compare_csv.close()

            # Extract the file name without extension
            file_name = (
                re.search(r"Cube_running_sqr_\d+x\d+_timing_\d+", line)
                .group()
                .replace(".json", "")
                .strip()
            )

            # Create new CSV files in the output folder
            current_csv = open(
                os.path.join(output_folder, f"{base_name}_{file_name}.csv"),
                "w",
                newline="",
            )
            csv_writer = csv.writer(current_csv)
            # compare_csv = open(
            #     os.path.join(output_folder, f"{base_name}_{file_name}_compare.csv"),
            #     "w",
            #     newline="",
            # )
            # compare_csv_writer = csv.writer(compare_csv)

            # Write the header rows
            csv_writer.writerow(["First Integer", "Loop Time (ms)", "Wall Time (s)"])
            # compare_csv_writer.writerow(
            #     ["First Integer", "Loop Time (ms)", "Wall Time (s)"]
            # )

        # Check for the pattern and extract required information
        if "Running HASTE" in line:
            try:
                # surrogate_line = next(lines_iter)
                # temps_line = next(lines_iter)
                temps_line = next(lines_iter)
                timing_line = next(lines_iter)

                first_integer = int(re.search(r"\d+", timing_line).group())
                loop_time = float(re.search(r"Loop: ([\d.]+) ms", timing_line).group(1))
                wall_time = float(re.search(r"Wall: ([\d.]+) s", timing_line).group(1))

                # Write the extracted information to the main CSV file
                if csv_writer:
                    csv_writer.writerow([first_integer, loop_time, wall_time])
            except StopIteration:
                break

        # Check for lines with loop time when "Running Surrogate" is no longer used
        elif "Temps:" in line and ("TRUTH" in file_path.upper()):
            try:
                timing_line = next(lines_iter)

                first_integer_match = re.search(r"\d+", timing_line)
                loop_time_match = re.search(r"Loop: ([\d.]+) ms", timing_line)
                wall_time_match = re.search(r"Wall: ([\d.]+) s", timing_line)

                if first_integer_match and loop_time_match and wall_time_match:
                    first_integer = int(first_integer_match.group())
                    loop_time = float(loop_time_match.group(1))
                    wall_time = float(wall_time_match.group(1))

                    # Write the extracted information to the compare CSV file
                    # if compare_csv_writer:
                    #     compare_csv_writer.writerow(
                    #         [first_integer, loop_time, wall_time]
                    #     )
                    if csv_writer:
                        csv_writer.writerow([first_integer, loop_time, wall_time])
            except StopIteration:
                break

    # Close the last CSV files if they exist
    if current_csv:
        current_csv.close()
    # if compare_csv:
    #     compare_csv.close()


# Function to process all .out files in a specified folder
def process_all_files_in_folder(folder_path, output_folder):
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".out"):
            process_file(os.path.join(folder_path, file_name), output_folder)


# Example usage with a specified folder
folder_path = "timing_output_files"
csv_path = "timing_output_files/csv_files"  # Replace with your folder path


if 1:
    process_all_files_in_folder(folder_path, csv_path)


# Function to read all CSV files in the output folder into arrays
def read_csv_files(output_folder):
    data = {}
    for file_name in sorted(os.listdir(output_folder)):
        if file_name.endswith(".csv"):
            file_path = os.path.join(output_folder, file_name)
            base_name = os.path.splitext(file_name)[0]
            with open(file_path, "r") as csv_file:
                csv_reader = csv.reader(csv_file)
                next(csv_reader)  # Skip header row
                rows = [row for row in csv_reader]
                data[base_name] = rows
    return data


# Function to find the median loop time for each CSV file
def find_median_loop_time(data):
    median_loop_times = {}
    for base_name, rows in data.items():
        loop_times = [float(row[1]) for row in rows]
        median_loop_time = np.median(loop_times)
        median_loop_times[base_name] = median_loop_time
    return median_loop_times


# Read all CSV files into arrays
data = read_csv_files(csv_path)

# Find the median loop time for each CSV file
median_loop_times = find_median_loop_time(data)

# Print the median loop times
for base_name, median_loop_time in median_loop_times.items():
    print(f"Median loop time for {base_name}: {median_loop_time}")


# Separate data for HASTE and TRUTH
haste_data = {k: v for k, v in median_loop_times.items() if "HASTE" in k}
truth_data = {k: v for k, v in median_loop_times.items() if "Truth" in k}


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

        def monotonic_outliers(numbers, times, tolerance=0.5):
            # Sort by numbers
            idx = np.argsort(numbers)
            sorted_numbers = numbers[idx]
            sorted_times = times[idx]

            violations = []

            for i in range(1, len(sorted_times) - 1):
                prev = sorted_times[i - 1]
                curr = sorted_times[i]
                next = sorted_times[i + 1]

                # Forward violation: time drops more than tolerance
                if curr < prev * (1 - tolerance):
                    violations.append(i)

                # Backward violation: time jumps more than tolerance
                elif curr > next * (1 + tolerance):
                    violations.append(i)

            return np.array(violations), sorted_numbers, sorted_times

        # Detect violations with 10% tolerance
        h_violations, sorted_hn, sorted_ht = monotonic_outliers(
            haste_numbers, haste_times
        )
        t_violations, sorted_tn, sorted_tt = monotonic_outliers(
            truth_numbers, truth_times
        )

        # Print violating points
        if len(h_violations) > 0:
            print("Haste violations at numbers:", sorted_hn[h_violations])
            print("Haste times:", sorted_ht[h_violations])
            filtered_haste_numbers = np.delete(sorted_hn, h_violations)
            filtered_haste_times = np.delete(sorted_ht, h_violations)
        else:
            print("No haste violations found.")
            filtered_haste_numbers = haste_numbers
            filtered_haste_times = haste_times
        if len(t_violations) > 0:
            print("Truth violations at numbers:", sorted_tn[t_violations])
            print("Truth times:", sorted_tt[t_violations])
            filtered_truth_numbers = np.delete(sorted_tn, t_violations)
            filtered_truth_times = np.delete(sorted_tt, t_violations)
        else:
            print("No truth violations found.")
            filtered_truth_numbers = truth_numbers
            filtered_truth_times = truth_times

        h_idx = np.argsort(filtered_haste_numbers)
        t_idx = np.argsort(filtered_truth_numbers)

        if category == "Big1":
            leg_label = r"Level 1 DOF: $2.1 \times 10^5$"
            color = "green"
        elif category == "Lil1":
            leg_label = r"Level 1 DOF: $6.0 \times 10^4$"
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
                leg_label = r"Level 1 DOF: $1.1 \times 10^5$"
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
            label=f"HASTE {leg_label}",
        )
        (fem_handle,) = ax.plot(
            filtered_truth_numbers[t_idx],
            filtered_truth_times[t_idx],
            marker="^",
            color=color,
            label=f"FEM {leg_label}",
        )

        max_time = max(
            [max_time, filtered_haste_times.max(), filtered_truth_times.max()]
        )

        fem_handles.append(fem_handle)
        haste_handles.append(haste_handle)

    # ax.set_xlabel(r"$\Delta t_1 / \Delta t_3 = 5 \Delta t_2 / \Delta t_3$", fontsize=14)
    ax.set_xlabel(r"$m_2 (\Delta t_2 / \Delta t_3)$", fontsize=14)
    ax.set_ylabel(r"Median execution time per $\Delta t_1$ (ms)", fontsize=14)
    # if ax == axes[0]:  # Only add legend to the first subplot
    ax.legend(handles=fem_handles + haste_handles, loc="upper left", fontsize=11)
    ax.set_ylim([0, np.ceil(max_time / 100) * 100])
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

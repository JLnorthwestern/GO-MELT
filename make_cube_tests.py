import json
import copy
from datetime import datetime
import numpy as np
import os

# Check if 'examples' directory exists, if not, create it
if not os.path.exists("examples"):
    os.makedirs("examples")

# Check if 'examples/gcodefiles' directory exists, if not, create it
if not os.path.exists("examples/gcodefiles"):
    os.makedirs("examples/gcodefiles")

# Template JSON structure
template = {
    "Level1": {
        "elements": [70, 70, 21],
        "bounds": {"x": [0, 14], "y": [0, 14], "z": [-4, 0.2]},
        "conditions": {
            "x": [298.15, 298.15],
            "y": [298.15, 298.15],
            "z": [298.15, 298.15],
        },
    },
    "Level2": {
        "elements": [200, 200, 10],
        "bounds": {"x": [0, 8], "y": [0, 8], "z": [-0.4, 0.0]},
    },
    "Level3": {
        "elements": [160, 160, 10],
        "bounds": {"x": [2.4, 5.6], "y": [2.4, 5.6], "z": [-0.2, 0.0]},
    },
    "Level4": {
        "elements": [230, 40, 10],
        "bounds": {"x": [-0.3, 2.0], "y": [-0.2, 0.2], "z": [-0.2, 0.0]},
        "surrogate_count_check": 2,
    },
    "properties": {
        "laser_center": [4.0, 4.0, 0.0, 0, 0, 0, 0],
        "thermal_conductivity": 20,
        "thermal_conductivity_bulk": 20,
        "thermal_conductivity_powder": 0.4,
        "heat_capacity": 750.0,
        "heat_capacity_solid": 750.0,
        "heat_capacity_mushy": 3235.0,
        "heat_capacity_fluid": 769.0,
        "density": 8e-06,
        "density_bulk": 8e-06,
        "density_powder": 8e-06,
        "laser_radius": 0.1000,
        "laser_depth": 0.1000,
        "laser_power": 285.0,
        "laser_absorptivity": 0.45,
        "T_amb": 298.15,
        "T_solidus": 1533,
        "T_liquidus": 1609,
        "T_boiling": 3038.0,
        "h_conv": 1.5e-05,
        "emissivity": 0.3,
        "evaporation_coefficient": 0.82,
        "boltzmann_constant": 1.38e-23,
        "atomic_mass": 9.746e-26,
        "saturation_prussure": 101000.0,
        "latent_heat_evap": 6457000.0,
        "layer_height": 0.04,
    },
    "nonmesh": {
        "timestep_L3": 1e-05,
        "dwell_time": 5.0,
        "wait_time": 600,
        "output_files": 1,
        "record_step": 16000,
        "Level1_record_step": 1e9,
        "savetime": 0,
        "laser_velocity": 960,
        "save_path": "./results/2025_03_05/10cm_cube_truth/",
        "gcode": "./examples/gcodefiles/10cm_cube.gcode",
        "toolpath": "./results/2025_03_05/10cm_cube_truth/toolpath.txt",
        "npz_folder": "./results/2025_03_05/10cm_cube_truth/npz_folder/",
        "layer_num": 0,
        "subcycle_num_L2": 10,
        "subcycle_num_L3": 16,
        "info_T": 1,
        "training": 0,
        "surrogate": 0,
        "dwell_time_multiplier": 2,
    },
}


#######################################################################################
# Define a function to generate G-code lines based on the identified pattern
def generate_square_gcode(
    start_x,
    start_y,
    layer_height,
    track_length,
    track_space,
    hatch_space,
    part_length,
    part_height=0.08,
):
    gcode_lines = []

    z = 0.0

    while z < part_height:
        z += layer_height
        x = start_x
        y = start_y  # + hatch_space / 2
        switchback = 1

        gcode_lines.append(f"G0 X{x:.4f} Y{y:.4f} Z{z:.4f}")

        i = 0
        while y <= part_length + start_y - hatch_space / 2 + 1e-5:
            if i % 4 == 0:
                x = x + switchback * track_length
                gcode_lines.append(f"G1 X{x:.4f} Y{y:.4f} Z{z:.4f}")
            elif i % 4 == 1:
                x = x + switchback * track_space
                gcode_lines.append(f"G0 X{x:.4f} Y{y:.4f} Z{z:.4f}")
            elif i % 4 == 2:
                y = y + hatch_space
                gcode_lines.append(f"G0 X{x:.4f} Y{y:.4f} Z{z:.4f}")
            elif i % 4 == 3:
                switchback *= -1
                x = x + switchback * track_space
                gcode_lines.append(f"G0 X{x:.4f} Y{y:.4f} Z{z:.4f}")
            i += 1
    return gcode_lines


def generate_triangle_gcode(
    start_x,
    start_y,
    layer_height,
    hatch_space,
    triangle_width,
    triangle_height,
    part_height=0.08,
):
    gcode_lines = []
    z = 0.0

    while z < part_height:
        z += layer_height
        x = start_x
        y = start_y  # + hatch_space / 2
        gcode_lines.append(f"G0 X{x:.4f} Y{y:.4f} Z{z:.4f}")

        triangle_points = np.array(
            [
                (x, y),
                (x + triangle_width, y + triangle_height / 2),
                (x, y + triangle_height),
            ]
        )

        # Define the lines every 0.1
        x_values = np.arange(
            x + hatch_space, x + triangle_width - hatch_space, hatch_space
        )

        # Function to find the intersection of a line with the triangle
        def find_intersections(triangle_points, x_values):
            intersections = []
            switchback = True
            for x in x_values:
                if switchback:
                    for i in range(len(triangle_points)):
                        p1 = triangle_points[i]
                        p2 = triangle_points[(i + 1) % len(triangle_points)]
                        if (p1[0] <= x <= p2[0] or p2[0] <= x <= p1[0]) and p1[0] != p2[
                            0
                        ]:
                            y = p1[1] + (x - p1[0]) * (p2[1] - p1[1]) / (p2[0] - p1[0])
                            intersections.append((x, y))
                else:
                    for i in reversed(range(len(triangle_points))):
                        p1 = triangle_points[i]
                        p2 = triangle_points[(i + 1) % len(triangle_points)]
                        if (p1[0] <= x <= p2[0] or p2[0] <= x <= p1[0]) and p1[0] != p2[
                            0
                        ]:
                            y = p1[1] + (x - p1[0]) * (p2[1] - p1[1]) / (p2[0] - p1[0])
                            intersections.append((x, y))
                switchback = not switchback
            return intersections

        # Find the intersection points
        intersection_points = find_intersections(triangle_points, x_values)

        # Print the intersection points
        switchback = True
        for point in intersection_points:
            if switchback:
                gcode_lines.append(f"G0 X{point[0]:.4f} Y{point[1]:.4f} Z{z:.4f}")
            else:
                gcode_lines.append(f"G1 X{point[0]:.4f} Y{point[1]:.4f} Z{z:.4f}")
            switchback = not switchback

    return gcode_lines


# Define the parameters for each G-code file
parameters = [
    (
        2.00,
        2.00,
        0.04,
        10.0,
        1.00,
        0.080,
        10,
        "examples/gcodefiles/square_10x10.gcode",
    ),
    (
        2.00,
        2.00,
        0.04,
        20.0,
        1.00,
        0.080,
        20,
        "examples/gcodefiles/square_20x20.gcode",
    ),
]
# Loop through the parameters and generate/write the G-code files
for param in parameters:
    all_gcode = generate_square_gcode(*param[:-1])
    with open(param[-1], "w") as file:
        for line in all_gcode:
            file.write(line + "\n")

# Define the parameters for each G-code file
parameters = [
    (2.00, 2.00, 0.04, 0.080, 20, 20, "examples/gcodefiles/triangle_20x20.gcode"),
    (2.00, 2.00, 0.04, 0.080, 10, 20, "examples/gcodefiles/triangle_10x20.gcode"),
    (2.00, 2.00, 0.04, 0.080, 20, 10, "examples/gcodefiles/triangle_20x10.gcode"),
    (2.00, 2.00, 0.04, 0.080, 10, 10, "examples/gcodefiles/triangle_10x10.gcode"),
]
# Loop through the parameters and generate/write the G-code files
for param in parameters:
    all_gcode = generate_triangle_gcode(*param[:-1])
    with open(param[-1], "w") as file:
        for line in all_gcode:
            file.write(line + "\n")


def generate_square_spiral_outin_gcode(
    start_x,
    start_y,
    layer_height,
    hatch_space,
    track_length,
    part_height=0.08,
):
    gcode_lines = []
    z = 0.0
    while z < part_height:
        z += layer_height
        x, y = start_x, start_y
        dx, dy = hatch_space, 0
        gcode_lines.append(f"G0 X{x:.4f} Y{y:.4f} Z{z:.4f}")

        step_size = 1
        steps_taken = 0
        steps_in_current_direction = (track_length + 1e-6) // hatch_space

        while steps_in_current_direction > 0:
            x, y = (
                x + dx * steps_in_current_direction,
                y + dy * steps_in_current_direction,
            )
            gcode_lines.append(f"G1 X{x:.4f} Y{y:.4f} Z{z:.4f}")
            steps_taken += steps_in_current_direction

            if steps_taken == steps_in_current_direction:
                dx, dy = -dy, dx
                steps_taken = 0
                if dy == 0:
                    steps_in_current_direction -= step_size

    return gcode_lines


# Generate points for the square spiral
# Define the parameters for each G-code file
parameters = [
    (2.00, 2.00, 0.04, 0.080, 10.0, "examples/gcodefiles/inward_spiral_10x10.gcode"),
    (2.00, 2.00, 0.04, 0.080, 20.0, "examples/gcodefiles/inward_spiral_20x20.gcode"),
]
# Loop through the parameters and generate/write the G-code files
for param in parameters:
    all_gcode = generate_square_spiral_outin_gcode(*param[:-1])
    with open(param[-1], "w") as file:
        for line in all_gcode:
            file.write(line + "\n")


def generate_square_spiral_inout_gcode(
    start_x,
    start_y,
    layer_height,
    hatch_space,
    track_length,
    part_height=0.08,
):
    gcode_lines = []
    z = 0.0
    while z < part_height:
        z += layer_height
        x, y = start_x + track_length / 2, start_y + track_length / 2
        dx, dy = hatch_space, 0
        gcode_lines.append(f"G0 X{x:.4f} Y{y:.4f} Z{z:.4f}")

        step_size = 1
        steps_taken = 0
        steps_in_current_direction = 1

        while steps_in_current_direction < (track_length + 1e-6) // hatch_space + 1:
            x, y = (
                x + dx * steps_in_current_direction,
                y + dy * steps_in_current_direction,
            )
            gcode_lines.append(f"G1 X{x:.4f} Y{y:.4f} Z{z:.4f}")
            steps_taken += steps_in_current_direction

            if steps_taken == steps_in_current_direction:
                dx, dy = -dy, dx
                steps_taken = 0
                if dy == 0:
                    steps_in_current_direction += step_size

    return gcode_lines


# Generate points for the square spiral
# Define the parameters for each G-code file
parameters = [
    (2.00, 2.00, 0.04, 0.080, 10.0, "examples/gcodefiles/outward_spiral_10x10.gcode"),
    (2.00, 2.00, 0.04, 0.080, 20.0, "examples/gcodefiles/outward_spiral_20x20.gcode"),
]
# Loop through the parameters and generate/write the G-code files
for param in parameters:
    all_gcode = generate_square_spiral_inout_gcode(*param[:-1])
    with open(param[-1], "w") as file:
        for line in all_gcode:
            file.write(line + "\n")


# Function to create and save JSON files
def create_json_file(
    template,
    parent_folder,
    size,
    surrogate=0,
    training=0,
    elements=[10, 10, 10],
    x=[0, 1],
    y=[0, 1],
    z=[0, 1],
    record_step=16000,
    subcycle_num_L2=10,
    surrogate_count_check=2,
):
    template_copy = copy.deepcopy(template)
    template_copy["nonmesh"]["surrogate"] = surrogate
    template_copy["nonmesh"]["training"] = training

    # Determine the case type based on surrogate and training
    if surrogate and training:
        case_type = "training"
    elif surrogate:
        case_type = "running"
    else:
        case_type = "truth"

    subfolder = f"Cube_{case_type}_{size}"
    save_path = f"{parent_folder}/{subfolder}/"
    gcode = f"./examples/gcodefiles/{size}.gcode"
    toolpath = f"{save_path}toolpath.txt"
    npz_folder = f"{save_path}npz_folder/"

    template_copy["nonmesh"]["save_path"] = save_path
    template_copy["nonmesh"]["gcode"] = gcode
    template_copy["nonmesh"]["toolpath"] = toolpath
    template_copy["nonmesh"]["npz_folder"] = npz_folder
    template_copy["Level1"]["elements"] = elements
    template_copy["Level1"]["bounds"]["x"] = x
    template_copy["Level1"]["bounds"]["y"] = y
    template_copy["Level1"]["bounds"]["z"] = z
    template_copy["nonmesh"]["record_step"] = record_step
    template_copy["Level4"]["surrogate_count_check"] = surrogate_count_check
    template_copy["nonmesh"]["subcycle_num_L2"] = subcycle_num_L2

    output_file = f"examples/{subfolder}.json"

    # with open(output_file, "w") as json_file:
    #     json.dump(template_copy, json_file, indent=4)


# Define the parent folder
current_date = datetime.now().strftime("%Y_%m_%d")
parent_folder = f"./results/{current_date}"

# Define the parameters for each case
cases = [
    ("tri_020x020", 1, 1, [120, 120, 21], [0, 24], [0, 24], [-4, 0.2], 16000, 10, 2),
    ("tri_010x010", 0, 0, [70, 70, 21], [0, 14], [0, 14], [-4, 0.2], 80, 5, 2),
    ("tri_010x010", 1, 0, [70, 70, 21], [0, 14], [0, 14], [-4, 0.2], 80, 5, 2),
    ("tri_020x010", 0, 0, [120, 120, 21], [0, 24], [0, 24], [-4, 0.2], 80, 5, 2),
    ("tri_020x010", 1, 0, [120, 120, 21], [0, 24], [0, 24], [-4, 0.2], 80, 5, 2),
    ("tri_010x020", 0, 0, [120, 120, 21], [0, 24], [0, 24], [-4, 0.2], 80, 5, 2),
    ("tri_010x020", 1, 0, [120, 120, 21], [0, 24], [0, 24], [-4, 0.2], 80, 5, 2),
    ("sqr_010x010", 0, 0, [70, 70, 21], [0, 14], [0, 14], [-4, 0.2], 80, 5, 2),
    ("sqr_010x010", 1, 0, [70, 70, 21], [0, 14], [0, 14], [-4, 0.2], 80, 5, 2),
    ("sqr_020x020", 0, 0, [120, 120, 21], [0, 24], [0, 24], [-4, 0.2], 80, 5, 2),
    ("sqr_020x020", 1, 0, [120, 120, 21], [0, 24], [0, 24], [-4, 0.2], 80, 5, 2),
    ("spi_oi_010x010", 0, 0, [70, 70, 21], [0, 14], [0, 14], [-4, 0.2], 80, 5, 2),
    ("spi_oi_010x010", 1, 0, [70, 70, 21], [0, 14], [0, 14], [-4, 0.2], 80, 5, 2),
    ("spi_oi_020x020", 0, 0, [120, 120, 21], [0, 24], [0, 24], [-4, 0.2], 80, 5, 2),
    ("spi_oi_020x020", 1, 0, [120, 120, 21], [0, 24], [0, 24], [-4, 0.2], 80, 5, 2),
    ("spi_io_010x010", 0, 0, [70, 70, 21], [0, 14], [0, 14], [-4, 0.2], 80, 5, 2),
    ("spi_io_010x010", 1, 0, [70, 70, 21], [0, 14], [0, 14], [-4, 0.2], 80, 5, 2),
    ("spi_io_020x020", 0, 0, [120, 120, 21], [0, 24], [0, 24], [-4, 0.2], 80, 5, 2),
    ("spi_io_020x020", 1, 0, [120, 120, 21], [0, 24], [0, 24], [-4, 0.2], 80, 5, 2),
    ("example", 0, 0, [50, 50, 50], [0, 10], [0, 10], [-4, 6.0], 1600, 10, 2),
]

# Loop through the cases and create the JSON files
for case in cases:
    create_json_file(
        template,
        parent_folder,
        case[0],
        case[1],
        case[2],
        case[3],
        case[4],
        case[5],
        case[6],
        case[7],
        case[8],
        case[9],
    )

print("Done.")

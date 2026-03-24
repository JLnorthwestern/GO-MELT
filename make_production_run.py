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
    part_height=10.0,
):
    gcode_lines = []

    z = 0.0

    while z + 1e-5 < part_height:
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


# Define the parameters for each G-code file
parameters = [
    (
        2.00,
        2.00,
        0.04,
        100.0,
        1.00,
        0.080,
        10,
        "examples/gcodefiles/production_100x10x10.gcode",
    ),
]
# Loop through the parameters and generate/write the G-code files
for param in parameters:
    all_gcode = generate_square_gcode(*param[:-1])
    with open(param[-1], "w") as file:
        for line in all_gcode:
            file.write(line + "\n")

import re
import numpy as np
from computeFunctions import *


def parsingGcode(Nonmesh, Properties, L2h):
    # Dwell timestep multiplier
    dwell_t_coef = float(
        Nonmesh["dwell_time_multiplier"]
        * Nonmesh["subcycle_num_L2"]
        * Nonmesh["subcycle_num_L3"]
    )
    # number of timesteps to wait before larger Nonmesh['timestep_L3'] in dwell time
    dwell_t = max(
        Nonmesh["dwell_time"] - Nonmesh["wait_time"] * Nonmesh["timestep_L3"],
        Nonmesh["dwell_time"],
    )

    # Open and read the Gcode file
    with open(Nonmesh["gcode"], "r") as gcode_file:
        gcode = gcode_file.read()

    # First pattern to match either X, Y, or Z coordinates
    pattern1 = (
        r"(?:G(\d+)\s*X(-?\d+\.\d+|-?\d+)\s*Y(-?\d+\.\d+|-?\d+)"
        r"(?:\s*Z(-?\d+\.\d+|-?\d+))?)"
    )
    matches = re.findall(pattern1, gcode)

    # Laser center coordinate (LCC) list (which is read by GO-MELT)
    LCC = []
    current_z = None  # Default z-coordinate is 0.0
    skip_segment = 0.0
    move_mesh = 0

    for current_match in matches:
        if current_match[0] != "1":  # command found, skip the segment
            skip_segment = 1.0
            if current_match[3]:  # Z-coordinate for the layer
                current_z = float(current_match[3])
            x, y, z = float(current_match[1]), float(current_match[2]), current_z
            LCC.append((x, y, z, skip_segment))
        else:  # X and Y coordinates
            if current_match[3]:  # Z-coordinate for the layer
                current_z = float(current_match[3])
            x, y, z = float(current_match[1]), float(current_match[2]), current_z
            LCC.append((x, y, z, skip_segment))
        skip_segment = 0.0

    dx = Nonmesh["laser_velocity"] * Nonmesh["timestep_L3"]

    # Calculating segments length and generating toolpath
    with open(Nonmesh["toolpath"], "w") as toolpath:
        z = LCC[0][2]

        for i in range(len(LCC) - 1):
            # Ljump == 0 during jump in laser path, Ldwell == 0 during dwell time
            Ljump, Ldwell = 1, 1

            # Check if z-coordinates are different
            if LCC[i][2] != LCC[i + 1][2]:
                new_x = LCC[i][0]
                new_y = LCC[i][1]
                # Number of ts for waiting before using larger the time step
                Ldwell = 0
                for k in range(int(Nonmesh["wait_time"])):
                    if ((k + 1) * Nonmesh["timestep_L3"]) > Nonmesh["dwell_time"]:
                        break
                    dt = Nonmesh["timestep_L3"]
                    P = 0
                    toolpath.write(
                        f"{format_fixed(new_x)},{format_fixed(new_y)},"
                        f"{format_fixed(z)},{Ljump:d},{Ldwell:d},{dt:.8e},{P:.8e}\n"
                    )
                    move_mesh += 1

                Ldwell = 0
                if dwell_t > 0:
                    _dwell_t = max(
                        0, dwell_t - Nonmesh["wait_time"] * Nonmesh["timestep_L3"]
                    )
                    dwell_t_steps = int(
                        _dwell_t / Nonmesh["timestep_L3"] / dwell_t_coef
                    )
                    for k in range(dwell_t_steps):
                        dt = Nonmesh["timestep_L3"] * dwell_t_coef
                        P = 0
                        toolpath.write(
                            f"{format_fixed(new_x)},{format_fixed(new_y)},"
                            f"{format_fixed(z)},{Ljump:d},{Ldwell:d},{dt:.8e},{P:.8e}\n"
                        )
                        move_mesh += 1
                    smalldt = (
                        _dwell_t - dwell_t_steps * Nonmesh["timestep_L3"] * dwell_t_coef
                    )
                    if smalldt > 0:
                        P = 0
                        toolpath.write(
                            f"{format_fixed(new_x)},{format_fixed(new_y)},"
                            f"{format_fixed(z)},{Ljump:d},{Ldwell:d},{smalldt:.8e},"
                            f"{P:.8e}\n"
                        )
                        move_mesh += 1
                continue  # Skip this segment
            # Check if skip-segments are different
            if LCC[i + 1][3] == 1:
                Ljump = 0

            distance_of_points = np.linalg.norm(
                np.array([LCC[i + 1][0] - LCC[i][0], LCC[i + 1][1] - LCC[i][1]])
            )
            num_pointsinSegments = int(distance_of_points // dx)
            shortdt = (distance_of_points % dx) / dx * Nonmesh["timestep_L3"]
            new_x = LCC[i][0]
            new_y = LCC[i][1]
            dx_segment = (
                Nonmesh["laser_velocity"]
                * (LCC[i + 1][0] - LCC[i][0])
                / distance_of_points
            )
            dy_segment = (
                Nonmesh["laser_velocity"]
                * (LCC[i + 1][1] - LCC[i][1])
                / distance_of_points
            )
            z = LCC[i][2]

            for j in range(num_pointsinSegments):
                new_x += dx_segment * Nonmesh["timestep_L3"]
                new_y += dy_segment * Nonmesh["timestep_L3"]
                _P = Ljump * Properties["laser_power"]
                toolpath.write(
                    f"{format_fixed(new_x)},{format_fixed(new_y)},{format_fixed(z)},"
                    f"{Ljump:d},{Ldwell:d},{Nonmesh['timestep_L3']:.8e},{_P:.8e}\n"
                )
                move_mesh += 1
            if shortdt > 0:
                dx_segment = (
                    Nonmesh["laser_velocity"]
                    * (LCC[i + 1][0] - LCC[i][0])
                    / distance_of_points
                )
                dy_segment = (
                    Nonmesh["laser_velocity"]
                    * (LCC[i + 1][1] - LCC[i][1])
                    / distance_of_points
                )
                z = LCC[i][2]
                new_x += dx_segment * shortdt
                new_y += dy_segment * shortdt
                _P = Ljump * Properties["laser_power"]
                toolpath.write(
                    f"{format_fixed(new_x)},{format_fixed(new_y)},{format_fixed(z)},"
                    f"{Ljump:d},{Ldwell:d},{shortdt:.8e},{_P:.8e}\n"
                )
                move_mesh += 1

        # End with dwell time
        P = 0
        Ldwell = 0
        # Number of ts for waiting before using larger the time step
        for k in range(int(int(Nonmesh["wait_time"]))):
            dt = Nonmesh["timestep_L3"]
            if ((k + 1) * dt) > Nonmesh["dwell_time"]:
                break
            toolpath.write(
                f"{format_fixed(new_x)},{format_fixed(new_y)},{format_fixed(z)},"
                f"{Ljump:d},{Ldwell:d},{dt:.8e},{P:.8e}\n"
            )
            move_mesh += 1
        Ldwell = 0
        _dwell_t = max(0, dwell_t - Nonmesh["wait_time"] * Nonmesh["timestep_L3"])
        dwell_t_steps = int(_dwell_t / Nonmesh["timestep_L3"] / dwell_t_coef)
        for k in range(dwell_t_steps):
            dt = Nonmesh["timestep_L3"] * dwell_t_coef
            toolpath.write(
                f"{format_fixed(new_x)},{format_fixed(new_y)},{format_fixed(z)},"
                f"{Ljump:d},{Ldwell:d},{dt:.8e},{P:.8e}\n"
            )
            move_mesh += 1
        smalldt = _dwell_t - dwell_t_steps * Nonmesh["timestep_L3"] * dwell_t_coef
        if smalldt > 0:
            toolpath.write(
                f"{format_fixed(new_x)},{format_fixed(new_y)},{format_fixed(z)},"
                f"{Ljump:d},{Ldwell:d},{smalldt:.8e},{P:.8e}\n"
            )
            move_mesh += 1
    return move_mesh


def count_lines(file_path):
    line_count = 0
    with open(file_path, "r") as file:
        for line in file:
            line_count += 1
    return line_count


def format_fixed(val, width=15, precision=8):
    formatted = f"{val:.{precision}e}"
    return formatted.rjust(width)

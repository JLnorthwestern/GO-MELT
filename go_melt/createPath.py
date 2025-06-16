import re
import numpy as np
from computeFunctions import *


# TODO, need to turn off wait time if no dwell time
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
    pattern1 = r"(?:G(\d+).*?X(\d+\.\d+|\d+).*?Y(\d+\.\d+|\d+)(?:.*?Z(\d+\.\d+|\d+))?)"
    matches = re.findall(pattern1, gcode)

    # Laser center coordinate (LCC) list (which is read by GO-MELT)
    LCC = []
    current_z = None  # Default z-coordinate is 0.0
    skip_segment = 0.0
    move_mesh = 0

    # To match Scheel validation, wait 0.0006 s after each track, otherwise 0.0
    wait_track = Nonmesh["wait_track"]

    for current_match, next_match in zip(matches, matches[1:] + [None]):
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
                        f"{new_x:.8e},{new_y:.8e},{z:.8e},{Ljump:d},{Ldwell:d},{dt:.8e},{P:.8e}\n"
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
                            f"{new_x:.8e},{new_y:.8e},{z:.8e},{Ljump:d},{Ldwell:d},{dt:.8e},{P:.8e}\n"
                        )
                        move_mesh += 1
                    smalldt = (
                        _dwell_t - dwell_t_steps * Nonmesh["timestep_L3"] * dwell_t_coef
                    )
                    if smalldt > 0:
                        P = 0
                        toolpath.write(
                            f"{new_x:.8e},{new_y:.8e},{z:.8e},{Ljump:d},{Ldwell:d},{smalldt:.8e},{P:.8e}\n"
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
                    f"{new_x:.8e},{new_y:.8e},{z:.8e},{Ljump:d},{Ldwell:d},{Nonmesh['timestep_L3']:.8e},{_P:.8e}\n"
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
                    f"{new_x:.8e},{new_y:.8e},{z:.8e},{Ljump:d},{Ldwell:d},{shortdt:.8e},{_P:.8e}\n"
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
                f"{new_x:.8e},{new_y:.8e},{z:.8e},{Ljump:d},{Ldwell:d},{dt:.8e},{P:.8e}\n"
            )
            move_mesh += 1
        Ldwell = 0
        _dwell_t = max(0, dwell_t - Nonmesh["wait_time"] * Nonmesh["timestep_L3"])
        dwell_t_steps = int(_dwell_t / Nonmesh["timestep_L3"] / dwell_t_coef)
        for k in range(dwell_t_steps):
            dt = Nonmesh["timestep_L3"] * dwell_t_coef
            toolpath.write(
                f"{new_x:.8e},{new_y:.8e},{z:.8e},{Ljump:d},{Ldwell:d},{dt:.8e},{P:.8e}\n"
            )
            move_mesh += 1
        smalldt = _dwell_t - dwell_t_steps * Nonmesh["timestep_L3"] * dwell_t_coef
        if smalldt > 0:
            toolpath.write(
                f"{new_x:.8e},{new_y:.8e},{z:.8e},{Ljump:d},{Ldwell:d},{smalldt:.8e},{P:.8e}\n"
            )
            move_mesh += 1
    return move_mesh
    # with open(Nonmesh["toolpath"], "w") as toolpath:
    #     for i in range(int(1e6)):
    #         toolpath.write("5.0,5.0,0.03,1,1\n")
    # return np.zeros(int(1e6))


# This implementation is specifically for the AM Bench Cantilever Bridge
def ExportToolpath(X, Y, Z, P, wa, Nonmesh):
    dwell_t_coef = float(
        Nonmesh["dwell_time_multiplier"]
        * Nonmesh["subcycle_num_L2"]
        * Nonmesh["subcycle_num_L3"]
    )
    buffer = 2.505
    data = np.concatenate([X, Y, Z * np.ones_like(X), P]).T
    tmp = (data[:, 1] < 7.5) | (data[:, 1] > 13)
    dwell_time = np.sum(tmp) * 1e-5
    data = data[~tmp]
    data[:, 1] += -7.75 + buffer
    data[:, 0] += 36.5 + buffer + 0.0026
    tmp = (data[:, 0] < 0) | (data[:, 1] < 0)
    data = data[~tmp]
    dwell_time += np.sum(tmp) * 1e-5

    with open(Nonmesh["toolpath"], wa) as toolpath:
        print("Writing main data to buffer")
        # Vectorized formatting (Write the main data)
        formatted_data = np.array(
            [
                f"{row[0]:.6e},{row[1]:.6e},{row[2]:.6e},1,1,1.0000e-5,{row[3]:.6e}\n"
                for row in data
            ]
        )

        # End with dwell time
        P = 0
        Ldwell = 0
        Ljump = 1
        t = 0
        dt_orig = 1.0000e-5

        print("Writing dwell time data to buffer")
        formatted_data2 = np.array(
            [
                f"{data[-1, 0]:.6e},{data[-1, 1]:.6e},{data[-1, 2]:.6e},{Ljump},{Ldwell},{dt_orig:.6e},{P:.6e}\n"
                for k in range(int(Nonmesh["wait_time"]))
            ]
        )
        if (len(formatted_data2) * dt_orig) > dwell_time:
            k = len(formatted_data2) - int(
                (-dwell_time + len(formatted_data2) * dt_orig) // dt_orig
            )
            formatted_data2 = formatted_data2[:k]

        t = len(formatted_data2) * dt_orig
        dt = dt_orig * dwell_t_coef
        dwell_time_rem = int((dwell_time - t) // dt)

        formatted_data = np.append(formatted_data, formatted_data2)

        if dwell_time_rem > 0:
            formatted_data3 = np.array(
                [
                    f"{data[-1, 0]:.6e},{data[-1, 1]:.6e},{data[-1, 2]:.6e},{Ljump},{Ldwell},{dt:.6e},{P:.6e}\n"
                    for k in range(dwell_time_rem)
                ]
            )

            t += dwell_time_rem * dt
            formatted_data = np.append(formatted_data, formatted_data3)

        dt_rem = dwell_time - t
        if dt_rem > 0:
            formatted_data = np.append(
                formatted_data,
                np.array(
                    [
                        f"{data[-1, 0]:.6e},{data[-1, 1]:.6e},{data[-1, 2]:.6e},{Ljump},{Ldwell},{dt_rem:.6e},{P:.6e}\n"
                    ]
                ),
            )

        print("Writing to file")
        # Write all buffered data to the file at once
        toolpath.writelines(formatted_data)

    # with open(Nonmesh["toolpath"], wa) as toolpath:
    #     for i in range(len(data[:, 3])):
    #         toolpath.write(
    #             f"{data[i, 0]:.6e},{data[i, 1]:.6e},{data[i, 2]:.6e},1,1,1.0000e-5,{data[i, 3]:.6e}\n"
    #         )

    #     # End with dwell time
    #     P = 0
    #     Ldwell = 0
    #     Ljump = 1
    #     t = 0
    #     # Number of ts for waiting before using larger the time step
    #     for k in range(int(int(Nonmesh["wait_time"]))):
    #         dt_orig = 1.0000e-5
    #         t += min(dwell_time - t, dt_orig)
    #         if ((k + 1) * dt_orig) > dwell_time:
    #             break
    #         toolpath.write(
    #             f"{data[-1, 0]:.8e},{data[-1, 1]:.8e},{data[-1, 2]:.8e},{Ljump:d},{Ldwell:d},{dt_orig:.6e},{P:.6e}\n"
    #         )

    #     while t < dwell_time:
    #         dt = min(dwell_time - t, dt_orig * dwell_t_coef)
    #         t += dt
    #         toolpath.write(
    #             f"{data[-1, 0]:.6e},{data[-1, 1]:.6e},{data[-1, 2]:.6e},{Ljump:d},{Ldwell:d},{dt:.6e},{P:.6e}\n"
    #         )


def count_lines(file_path):
    line_count = 0
    with open(file_path, "r") as file:
        for line in file:
            line_count += 1
    return line_count


# Example usage
# X, Y, Z, P = np.array([...]), np.array([...]), np.array([...]), np.array([...])
# ExportToolpath(X, Y, Z, P, 'w')

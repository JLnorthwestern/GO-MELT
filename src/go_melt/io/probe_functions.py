from go_melt.utils.interpolation_functions import interpolatePoints
import jax.numpy as jnp
import csv
import os


def update_probes(Levels, grids, Nonmesh, time_inc):
    probe_list = []

    for probe_coords in grids:
        ProbeT = interpolatePoints(Levels[1], Levels[1]["T0"], jnp.array(probe_coords))
        ProbeT += interpolatePoints(
            Levels[2], Levels[2]["Tprime0"], jnp.array(probe_coords)
        )
        ProbeT += interpolatePoints(
            Levels[3], Levels[3]["Tprime0"], jnp.array(probe_coords)
        )

        probe_list.append(ProbeT[0])

    # Define the CSV file path
    csv_path = os.path.join(Nonmesh["save_path"], "ProbeData.csv")

    # Format time_inc and probe values
    formatted_time = f"{time_inc:.8f}"
    formatted_probes = [f"{val:.4f}" for val in probe_list]

    # Write time_inc followed by probe values in one row
    with open(csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([formatted_time] + formatted_probes)


def initialize_probe_csv(save_path, num_probes):
    csv_path = os.path.join(save_path, "ProbeData.csv")
    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        header = ["time"] + [f"Probe{i+1}" for i in range(num_probes)]
        writer.writerow(header)


def get_probe_regions(list_probe_locations) -> list[list[jnp.ndarray]]:
    """
    Generate 3D meshgrids centered at given centroids with adjustable dimensions.
    """
    meshgrids = []

    for x0, y0, z0 in list_probe_locations:
        meshgrids.append([jnp.array([x0]), jnp.array([y0]), jnp.array([z0])])

    return meshgrids

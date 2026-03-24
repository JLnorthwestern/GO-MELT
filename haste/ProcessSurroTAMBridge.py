import os
import numpy as np
import vtk
from pyevtk.hl import gridToVTK
import gc
import matplotlib.pyplot as plt
from copy import deepcopy


def read_vtr_file(file_path):
    reader = vtk.vtkXMLRectilinearGridReader()
    reader.SetFileName(file_path)
    reader.Update()
    return reader.GetOutput()


def get_layer_height(vtr_data):
    z_coords = vtr_data.GetZCoordinates()
    return np.max(z_coords)


def get_value_ranges_and_spacing(vtr_data):
    x_coords = vtr_data.GetXCoordinates()
    y_coords = vtr_data.GetYCoordinates()
    z_coords = vtr_data.GetZCoordinates()

    x_range = (np.min(x_coords), np.max(x_coords))
    y_range = (np.min(y_coords), np.max(y_coords))

    x_spacing = (x_range[1] - x_range[0]) / (x_coords.GetNumberOfTuples() - 1)
    y_spacing = (y_range[1] - y_range[0]) / (y_coords.GetNumberOfTuples() - 1)
    z_spacing = (np.max(z_coords) - np.min(z_coords)) / (
        z_coords.GetNumberOfTuples() - 1
    )

    spacing = (x_spacing, y_spacing, z_spacing)

    return x_range, y_range, spacing


def get_number_of_data_points(vtr_data, layer_min_z, layer_height, layer_spacing):
    z_coords = vtr_data.GetZCoordinates()
    z_min = np.min(z_coords)

    count = 0
    for z in np.array(z_coords):
        if (
            layer_min_z
            <= round(z, 3)
            <= round(layer_min_z + layer_height - layer_spacing, 3)
        ):
            count += 1

    return count


def collect_layer_data(
    vtr_data, tam_data, layer_min_z, layer_height, layer_spacing, xy_num
):
    z_coords = vtr_data.GetZCoordinates()

    data_points = []
    tam_points = []
    layer_z_coords = []
    for i in range(z_coords.GetNumberOfTuples()):
        if round(z_coords.GetTuple1(i), 3) <= round(
            (layer_min_z + layer_height - layer_spacing), 3
        ):
            data_points.append(
                np.array(vtr_data.GetPointData().GetScalars())[
                    i * xy_num : (i + 1) * xy_num
                ].astype(np.int8)
            )
            tam_points.append(
                tam_data[i * xy_num : (i + 1) * xy_num].astype(np.float32)
            )
            layer_z_coords.append(round(z_coords.GetTuple1(i), 3))
            print(round(z_coords.GetTuple1(i), 3))

    return data_points, tam_points, layer_z_coords


def collect_full_point_data(vtr_data, tam_data):
    z_coords = vtr_data.GetZCoordinates()

    data_points = []
    tam_points = []
    layer_z_coords = []
    for i in range(z_coords.GetNumberOfTuples()):
        data_points.append(
            np.array(vtr_data.GetPointData().GetScalars())[
                i * xy_num : (i + 1) * xy_num
            ].astype(np.int8)
        )
        tam_points.append(tam_data[i * xy_num : (i + 1) * xy_num].astype(np.float32))
        layer_z_coords.append(round(z_coords.GetTuple1(i), 4))
        print(round(z_coords.GetTuple1(i), 4))
    return data_points, tam_points, layer_z_coords


# Root directory containing all folders
root_dir = "results"
# Store results for comparison
folder_results = {}

# Loop through all folders with "production" in their name
for folder_name in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder_name)
    if not os.path.isdir(folder_path) or "production" not in folder_name:
        continue
    production_path = folder_path

    print(f"Processing folder: {folder_path}")

    # List .vtr and TAM files
    vtr_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".vtr")])
    tam_files = sorted(
        [f for f in os.listdir(folder_path) if f.startswith("accum_time")]
    )

    if not vtr_files or not tam_files:
        print(f"No .vtr or TAM files found in {folder_path}")
        continue

    num_files_to_process = min(len(vtr_files), len(tam_files))
    samp = 1  # sampling factor

    # Read first file to get dimensions
    first_vtr_data = read_vtr_file(os.path.join(folder_path, vtr_files[0]))
    layer_height = get_layer_height(first_vtr_data)
    x_range, y_range, spacing = get_value_ranges_and_spacing(first_vtr_data)
    x_dim = int(x_range[-1] / spacing[0] + 1)
    y_dim = int(y_range[-1] / spacing[1] + 1)
    xy_num = x_dim * y_dim

    layer_min_z = np.min(first_vtr_data.GetZCoordinates())
    layer_spacing = round(spacing[2], 2)
    layer_max_z = num_files_to_process * layer_height
    z_dim = int(round((layer_max_z - layer_min_z) / layer_spacing, 2)) + 1
    z_idx = z_dim // samp + (1 if z_dim % samp == 1 else 0)

    result_array = np.zeros((x_dim, y_dim, z_idx), dtype=np.int8)[::samp, ::samp, :]
    tam_result_array = np.zeros((x_dim, y_dim, z_idx), dtype=np.float32)[
        ::samp, ::samp, :
    ]
    all_layer_z_coords = []
    z = 0

    for i in range(num_files_to_process):
        vtr_data = read_vtr_file(os.path.join(folder_path, vtr_files[i]))
        tam_data = np.load(os.path.join(folder_path, tam_files[i]))["accum_time"]

        layer_data_points, layer_tam_points, layer_z_coords = collect_full_point_data(
            vtr_data, tam_data
        )
        all_layer_z_coords.extend(layer_z_coords)

        for _z in range(len(layer_z_coords)):
            if (z + _z) % samp == 0:
                zidx = (z + _z) // samp
                if layer_z_coords[_z] <= 0:
                    result_array[:, :, zidx] = np.ones([x_dim, y_dim], dtype=np.int8)[
                        ::samp, ::samp
                    ]
                else:
                    result_array[:, :, zidx] = (
                        layer_data_points[_z].reshape((y_dim, x_dim)).T
                    )[::samp, ::samp]
                tam_result_array[:, :, zidx] = (
                    layer_tam_points[_z].reshape((y_dim, x_dim)).T
                )[::samp, ::samp]
        z += int(round(layer_height / layer_spacing, 2))
        gc.collect()

    _result_array = result_array[:, :, : zidx + 1].astype(np.int8)
    _tam_result_array = tam_result_array[:, :, : zidx + 1].astype(np.float32)

    vtkcx = np.linspace(x_range[0], x_range[1], _result_array.shape[0]).astype(
        np.float32
    )
    vtkcy = np.linspace(y_range[0], y_range[1], _result_array.shape[1]).astype(
        np.float32
    )
    vtkcz = np.linspace(layer_min_z, layer_z_coords[-1], zidx + 1).astype(np.float32)

    # Create VTK grid
    rectilinear_grid = vtk.vtkRectilinearGrid()
    rectilinear_grid.SetDimensions(_result_array.shape)

    x_coords = vtk.vtkFloatArray()
    x_coords.SetArray(vtkcx, len(vtkcx), 1)
    y_coords = vtk.vtkFloatArray()
    y_coords.SetArray(vtkcy, len(vtkcy), 1)
    z_coords = vtk.vtkFloatArray()
    z_coords.SetArray(vtkcz, len(vtkcz), 1)

    rectilinear_grid.SetXCoordinates(x_coords)
    rectilinear_grid.SetYCoordinates(y_coords)
    rectilinear_grid.SetZCoordinates(z_coords)

    data_array = vtk.vtkShortArray()
    data_array.SetName("Data")
    flat_data = _result_array.transpose(2, 1, 0).astype(np.int16).flatten()
    data_array.SetArray(flat_data, len(flat_data), 1)
    rectilinear_grid.GetPointData().AddArray(data_array)

    tam_data_array = vtk.vtkFloatArray()
    tam_data_array.SetName("SurroTAMData")
    flat_tam_data = _tam_result_array.transpose(2, 1, 0).astype(np.float32).flatten()
    tam_data_array.SetArray(flat_tam_data, len(flat_tam_data), 1)
    rectilinear_grid.GetPointData().AddArray(tam_data_array)

    # Save result in the same folder
    output_path = os.path.join(folder_path, "tamresult.vtr")
    writer = vtk.vtkXMLRectilinearGridWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(rectilinear_grid)
    writer.Write()
    print(f"Written to {output_path}")
    gc.collect()

    # Inside your folder loop, after computing _tam_result_array:
    cy_index_1 = np.argmin(np.abs(vtkcy - 6.96))
    cy_index_2 = np.argmin(np.abs(vtkcy - 7.00))
    cy_index_3 = np.argmin(np.abs(vtkcy - 7.04))
    cz_index = -1  # maximum vtkcz

    # Extract the 2D slices at vtkcy = 6.96, 7.00, and 7.04 mm, max vtkcz
    slice_data_1 = _tam_result_array[:, cy_index_1, cz_index]
    slice_data_2 = _tam_result_array[:, cy_index_2, cz_index]
    slice_data_3 = _tam_result_array[:, cy_index_3, cz_index]
    folder_results[folder_name] = (slice_data_1, slice_data_2, slice_data_3)

# After folder loop
if len(folder_results) == 2:
    folders = list(folder_results.keys())
    data1_1, data1_2, data1_3 = deepcopy(folder_results[folders[0]])  # HASTE
    data2_1, data2_2, data2_3 = deepcopy(folder_results[folders[1]])  # FEM

    # Convert to us
    data1_1 *= 1e6
    data1_2 *= 1e6
    data1_3 *= 1e6
    data2_1 *= 1e6
    data2_2 *= 1e6
    data2_3 *= 1e6

    zmax = str(round(vtkcz.max() + 4, 2))

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # fig, axs = plt.subplots(2, 3, figsize=(15, 8))

    # Enable minor ticks and subgrids
    for ax in axs:
        ax.minorticks_on()
        ax.grid(True, which="major", linestyle="-", linewidth=0.75)  # Major grid
        ax.grid(
            True, which="minor", linestyle=":", linewidth=0.5
        )  # Minor grid (subgrid)

    # Full range plots
    axs[0].plot(vtkcx, data1_1, "-", label="HASTE", color="red", linewidth=2)
    axs[0].plot(vtkcx, data2_1, "-", label="FEM", color="black", linewidth=2)
    axs[0].set_title(f"y = 6.96 mm, z = {zmax} mm")
    axs[0].set_ylabel("Time above melting (μs)")
    axs[0].set_xlabel("x (mm)")
    axs[0].legend()

    axs[1].plot(vtkcx, data1_2, "-", label="HASTE", color="red", linewidth=2)
    axs[1].plot(vtkcx, data2_2, "-", label="FEM", color="black", linewidth=2)
    axs[1].set_title(f"y = 7.00 mm, z = {zmax} mm")
    axs[1].set_ylabel("Time above melting (μs)")
    axs[1].set_xlabel("x (mm)")

    axs[2].plot(vtkcx, data1_3, "-", label="HASTE", color="red", linewidth=2)
    axs[2].plot(vtkcx, data2_3, "-", label="FEM", color="black", linewidth=2)
    axs[2].set_title(f"y = 7.04 mm, z = {zmax} mm")
    axs[2].set_ylabel("Time above melting (μs)")
    axs[2].set_xlabel("x (mm)")

    # # Full range plots
    # axs[0, 0].plot(vtkcx, data1_1, "-", label="HASTE", color="red", linewidth=2)
    # axs[0, 0].plot(vtkcx, data2_1, "--", label="FEM", color="black", linewidth=2)
    # axs[0, 0].set_title(f"y = 6.96 mm, z = {zmax} mm")
    # axs[0, 0].set_ylabel("Time above melting (μs)")
    # axs[0, 0].grid(True)
    # axs[0, 0].legend()

    # axs[0, 1].plot(vtkcx, data1_2, "-", label="HASTE", color="red", linewidth=2)
    # axs[0, 1].plot(vtkcx, data2_2, "--", label="FEM", color="black", linewidth=2)
    # axs[0, 1].set_title(f"y = 7.00 mm, z = {zmax} mm")
    # axs[0, 1].set_ylabel("Time above melting (μs)")
    # axs[0, 1].grid(True)
    # axs[0, 1].legend()

    # axs[0, 2].plot(vtkcx, data1_3, "-", label="HASTE", color="red", linewidth=2)
    # axs[0, 2].plot(vtkcx, data2_3, "--", label="FEM", color="black", linewidth=2)
    # axs[0, 2].set_title(f"y = 7.04 mm, z = {zmax} mm")
    # axs[0, 2].set_ylabel("Time above melting (μs)")
    # axs[0, 2].grid(True)
    # axs[0, 2].legend()

    # # Zoomed-in plots
    # axs[1, 0].plot(vtkcx, data1_1, "-", label="HASTE", color="red", linewidth=2)
    # axs[1, 0].plot(vtkcx, data2_1, "--", label="FEM", color="black", linewidth=2)
    # axs[1, 0].set_xlim(0, 20)
    # axs[1, 0].set_ylim(8e2, 11e2)
    # axs[1, 0].set_xlabel("x (mm)")
    # axs[1, 0].set_ylabel("Time above melting (μs)")
    # axs[1, 0].grid(True)

    # axs[1, 1].plot(vtkcx, data1_2, "-", label="HASTE", color="red", linewidth=2)
    # axs[1, 1].plot(vtkcx, data2_2, "--", label="FEM", color="black", linewidth=2)
    # axs[1, 1].set_xlim(0, 20)
    # axs[1, 1].set_ylim(7e2, 9.5e2)
    # axs[1, 1].set_xlabel("x (mm)")
    # axs[1, 1].set_ylabel("Time above melting (μs)")
    # axs[1, 1].grid(True)

    # axs[1, 2].plot(vtkcx, data1_3, "-", label="HASTE", color="red", linewidth=2)
    # axs[1, 2].plot(vtkcx, data2_3, "--", label="FEM", color="black", linewidth=2)
    # axs[1, 2].set_xlim(0, 20)
    # axs[1, 2].set_ylim(7e2, 9.5e2)
    # axs[1, 2].set_xlabel("x (mm)")
    # axs[1, 2].set_ylabel("Time above melting (μs)")
    # axs[1, 2].grid(True)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        save_path = os.path.join(
            production_path, f"tam_comparison_adjacent_tracks.{ext}"
        )
        plt.savefig(save_path, dpi=300)
        print(f"Saved comparison plot: {save_path}")
    plt.close()


print("finished")

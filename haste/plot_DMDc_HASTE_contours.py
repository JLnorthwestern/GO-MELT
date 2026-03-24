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
from scipy import ndimage
from matplotlib.colors import BoundaryNorm
import matplotlib.patches as patches
import matplotlib.ticker as mticker


def plot_length_error(DEVICE_ID, running_file, truth_file):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    try:
        # Run on single GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
        os.system("clear")
    except:
        # Run on CPU
        import jax

        jax.config.update("jax_platform_name", "cpu")
        os.system("clear")
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

    recording_time = Nonmesh["record_step"] * Nonmesh["timestep_L3"]

    def get_value_for_mean(truth_folder, run_folder, record_lab):
        # Get all files in the current track folder and sort them
        prev = 10
        prev2 = 20
        truth_file = os.path.join(truth_folder, f"check_error_{record_lab:07}.npz")
        run_file = os.path.join(run_folder, f"check_error_{record_lab:07}.npz")
        truth_file_prev1 = os.path.join(
            truth_folder, f"check_error_{record_lab-prev:07}.npz"
        )
        run_file_prev1 = os.path.join(
            run_folder, f"check_error_{record_lab-prev:07}.npz"
        )
        truth_file_prev2 = os.path.join(
            truth_folder, f"check_error_{record_lab-prev2:07}.npz"
        )
        run_file_prev2 = os.path.join(
            run_folder, f"check_error_{record_lab-prev2:07}.npz"
        )

        # Load the .npz files
        truth_data = np.load(truth_file)
        run_data = np.load(run_file)
        truth_data_prev1 = np.load(truth_file_prev1)
        run_data_prev1 = np.load(run_file_prev1)
        truth_data_prev2 = np.load(truth_file_prev2)
        run_data_prev2 = np.load(run_file_prev2)

        # Extract the L3T variables
        L3T_truth = truth_data["L3T"]
        L3T_run = run_data["L3T"]
        L3T_truth_prev1 = truth_data_prev1["L3T"]
        L3T_run_prev1 = run_data_prev1["L3T"]
        L3T_truth_prev2 = truth_data_prev2["L3T"]
        L3T_run_prev2 = run_data_prev2["L3T"]
        time = run_data["time"]
        Lsurro = run_data["run_surrogate"]

        save_str = "HASTE"
        vtkT = np.array(
            L3T_truth.reshape(
                Levels[3]["nodes"][2], Levels[3]["nodes"][1], Levels[3]["nodes"][0]
            )
        ).transpose((2, 1, 0))

        vtkT_run = np.array(
            L3T_run.reshape(
                Levels[3]["nodes"][2], Levels[3]["nodes"][1], Levels[3]["nodes"][0]
            )
        ).transpose((2, 1, 0))

        vtkT_prev1 = np.array(
            L3T_truth_prev1.reshape(
                Levels[3]["nodes"][2], Levels[3]["nodes"][1], Levels[3]["nodes"][0]
            )
        ).transpose((2, 1, 0))

        vtkT_run_prev1 = np.array(
            L3T_run_prev1.reshape(
                Levels[3]["nodes"][2], Levels[3]["nodes"][1], Levels[3]["nodes"][0]
            )
        ).transpose((2, 1, 0))

        vtkT_prev2 = np.array(
            L3T_truth_prev2.reshape(
                Levels[3]["nodes"][2], Levels[3]["nodes"][1], Levels[3]["nodes"][0]
            )
        ).transpose((2, 1, 0))

        vtkT_run_prev2 = np.array(
            L3T_run_prev2.reshape(
                Levels[3]["nodes"][2], Levels[3]["nodes"][1], Levels[3]["nodes"][0]
            )
        ).transpose((2, 1, 0))

        clims = [Properties["T_amb"], Properties["T_liquidus"]]

        temp_data = vtkT[24:-11, 65:96, :]
        temp_data_run = vtkT_run[24:-11, 65:96, :]
        temp_data_prev1 = vtkT_prev1[24:-11, 65:96, :]
        temp_data_run_prev1 = vtkT_run_prev1[24:-11, 65:96, :]
        temp_data_prev2 = vtkT_prev2[24:-11, 65:96, :]
        temp_data_run_prev2 = vtkT_run_prev2[24:-11, 65:96, :]

        fig, axs = plt.subplots(3, 2, figsize=(14, 7), dpi=400)
        norm = plt.Normalize(vmin=clims[0], vmax=clims[1], clip=True)

        cmap = plt.get_cmap("jet", 1000)
        cmap.set_over("gray")

        yz_cross_idx = 15

        # Extracting cross-sections
        xy_cross_section = temp_data[:, :, -1].T
        xy_cross_section_run = temp_data_run[:, :, -1].T
        xy_cross_section_prev1 = temp_data_prev1[:, :, -1].T
        xy_cross_section_run_prev1 = temp_data_run_prev1[:, :, -1].T
        xy_cross_section_prev2 = temp_data_prev2[:, :, -1].T
        xy_cross_section_run_prev2 = temp_data_run_prev2[:, :, -1].T

        # Plotting cross-sections
        cross_sections = [
            xy_cross_section_prev2,
            xy_cross_section_run_prev2,
            xy_cross_section_prev1,
            xy_cross_section_run_prev1,
            xy_cross_section,
            xy_cross_section_run,
        ]

        # Levels
        clevels = np.linspace(
            300, 1609, 1000
        )  # np.array([300, 500, 700, 900, 1100, 1300, 1500, 1609])

        for ax, cross_section, _num in zip(
            axs.flatten(), cross_sections, range(len(cross_sections))
        ):
            if cross_section is None:
                ax.axis("off")  # Turn off the axis for the empty subplot
                continue
            # Assuming cross_section is a 2D array
            height, width = cross_section.shape
            x = np.arange(width) * 20e-3  # mm
            y = np.arange(height) * 20e-3  # mm

            contour = ax.contourf(
                x,
                y,
                cross_section,
                levels=clevels,
                cmap=cmap,
                norm=norm,
                extend="max",
                vmin=clims[0],
                vmax=clims[1],
            )
            ax.set_aspect("equal")
            if _num // 2 == 2:
                ax.set_xlabel("X (mm)")
            if _num % 2 == 0:
                ax.set_ylabel("Y (mm)")
            # ax.axis("off")

        # Create a new axis for the colorbar
        cbar_ax = fig.add_axes(
            [0.92, 0.25, 0.02, 0.52]
        )  # [left, bottom, width, height]

        cbar = fig.colorbar(
            contour,
            cax=cbar_ax,
            extend="max",
            orientation="vertical",
            ticks=[300, 500, 700, 900, 1100, 1300, 1500, 1609],
            format=mticker.FixedFormatter(
                [
                    "300",
                    "500",
                    "700",
                    "900",
                    "1100",
                    "1300",
                    "1500",
                    "1609",
                ]
            ),
        )
        cbar.set_label("Temperature (K)")

        # Adjust layout to minimize space between plots
        plt.subplots_adjust(wspace=0.1, hspace=-0.3)

        # Draw the canvas to update the figure
        fig.canvas.draw()

        # Get the bounding box of the axis in display coordinates
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        bbox.x0 = 1.7
        bbox.y0 = 1.7
        bbox.x1 = 6.6
        bbox.y1 = 5.5

        plt.savefig(
            f"{Nonmesh['save_path']}/contour_plot_{save_str}.pdf",
            dpi=400,
            bbox_inches="tight",
        )
        plt.savefig(
            f"{Nonmesh['save_path']}/contour_plot_{save_str}.png",
            dpi=400,
            bbox_inches="tight",
        )
        plt.close()

        return 0

    # 142
    _ = get_value_for_mean(Nonmesh_truth["save_path"], Nonmesh["save_path"], 84)
    print("here")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        DEVICE_ID = 0

        running_file = "examples/haste_boundary.json"
        truth_file = "examples/truth_boundary.json"
        plot_length_error(DEVICE_ID, running_file, truth_file)

        sys.exit()

    DEVICE_ID = int(sys.argv[1])
    running_file = sys.argv[2]
    truth_file = sys.argv[3]
    plot_length_error(DEVICE_ID, running_file, truth_file)


# import os
# import numpy as np
# import matplotlib.pyplot as plt


# def get_value_for_mean(truth_folder, run_folder, record_lab):
#     # Get all files in the current track folder and sort them
#     truth_file = os.path.join(truth_folder, f"check_error_{record_lab:07}.npz")
#     run_file = os.path.join(run_folder, f"check_error_{record_lab:07}.npz")
#     truth_file_prev1 = os.path.join(truth_folder, f"check_error_{record_lab-1:07}.npz")
#     run_file_prev1 = os.path.join(run_folder, f"check_error_{record_lab-1:07}.npz")
#     truth_file_prev2 = os.path.join(truth_folder, f"check_error_{record_lab-2:07}.npz")
#     run_file_prev2 = os.path.join(run_folder, f"check_error_{record_lab-2:07}.npz")

#     # Load the .npz files
#     truth_data = np.load(truth_file)
#     run_data = np.load(run_file)
#     truth_data_prev1 = np.load(truth_file_prev1)
#     run_data_prev1 = np.load(run_file_prev1)
#     truth_data_prev2 = np.load(truth_file_prev2)
#     run_data_prev2 = np.load(run_file_prev2)

#     # Extract the L3T variables
#     L3T_truth = truth_data["L3T"]
#     L3T_run = run_data["L3T"]
#     L3T_truth_prev1 = truth_data_prev1["L3T"]
#     L3T_run_prev1 = run_data_prev1["L3T"]
#     L3T_truth_prev2 = truth_data_prev2["L3T"]
#     L3T_run_prev2 = run_data_prev2["L3T"]
#     time = run_data["time"]
#     Lsurro = run_data["run_surrogate"]

#     save_str = "HASTE"
#     vtkT = np.array(
#         L3T_truth.reshape(
#             Levels[3]["nodes"][2], Levels[3]["nodes"][1], Levels[3]["nodes"][0]
#         )
#     ).transpose((2, 1, 0))

#     vtkT_run = np.array(
#         L3T_run.reshape(
#             Levels[3]["nodes"][2], Levels[3]["nodes"][1], Levels[3]["nodes"][0]
#         )
#     ).transpose((2, 1, 0))

#     vtkT_prev1 = np.array(
#         L3T_truth_prev1.reshape(
#             Levels[3]["nodes"][2], Levels[3]["nodes"][1], Levels[3]["nodes"][0]
#         )
#     ).transpose((2, 1, 0))

#     vtkT_run_prev1 = np.array(
#         L3T_run_prev1.reshape(
#             Levels[3]["nodes"][2], Levels[3]["nodes"][1], Levels[3]["nodes"][0]
#         )
#     ).transpose((2, 1, 0))

#     vtkT_prev2 = np.array(
#         L3T_truth_prev2.reshape(
#             Levels[3]["nodes"][2], Levels[3]["nodes"][1], Levels[3]["nodes"][0]
#         )
#     ).transpose((2, 1, 0))

#     vtkT_run_prev2 = np.array(
#         L3T_run_prev2.reshape(
#             Levels[3]["nodes"][2], Levels[3]["nodes"][1], Levels[3]["nodes"][0]
#         )
#     ).transpose((2, 1, 0))

#     clims = [Properties["T_amb"], Properties["T_liquidus"]]

#     temp_data = vtkT[:, 65:95, :]
#     temp_data_run = vtkT_run[:, 65:95, :]
#     temp_data_prev1 = vtkT_prev1[:, 65:95, :]
#     temp_data_run_prev1 = vtkT_run_prev1[:, 65:95, :]
#     temp_data_prev2 = vtkT_prev2[:, 65:95, :]
#     temp_data_run_prev2 = vtkT_run_prev2[:, 65:95, :]

#     fig, axs = plt.subplots(3, 2, figsize=(12, 15), dpi=400)
#     norm = plt.Normalize(vmin=clims[0], vmax=clims[1], clip=True)

#     cmap = plt.get_cmap("jet", 15)
#     cmap.set_over("gray")

#     yz_cross_idx = 15

#     # Extracting cross-sections
#     xy_cross_section = temp_data[:, :, -1].T
#     xy_cross_section_run = temp_data_run[:, :, -1].T
#     xy_cross_section_prev1 = temp_data_prev1[:, :, -1].T
#     xy_cross_section_run_prev1 = temp_data_run_prev1[:, :, -1].T
#     xy_cross_section_prev2 = temp_data_prev2[:, :, -1].T
#     xy_cross_section_run_prev2 = temp_data_run_prev2[:, :, -1].T

#     # Plotting cross-sections
#     cross_sections = [
#         xy_cross_section,
#         xy_cross_section_run,
#         xy_cross_section_prev1,
#         xy_cross_section_run_prev1,
#         xy_cross_section_prev2,
#         xy_cross_section_run_prev2,
#     ]

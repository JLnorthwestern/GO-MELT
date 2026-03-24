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
        truth_file = os.path.join(truth_folder, f"check_error_{record_lab:07}.npz")
        run_file = os.path.join(run_folder, f"check_error_{record_lab:07}.npz")

        # Load the .npz files
        truth_data = jnp.load(truth_file)
        run_data = jnp.load(run_file)

        # Extract the L3T variables
        L3T_truth = truth_data["L3T"]
        L3T_run = run_data["L3T"]
        time = run_data["time"]
        Lsurro = run_data["run_surrogate"]

        # for T, save_str in zip([L3T_truth, L3T_run], ["truth", "HASTE"]):
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

        clims = [Properties["T_amb"], Properties["T_liquidus"]]

        # temp_data = np.clip(vtkT, clims[0], clims[1])
        temp_data = vtkT[33:103, 65:95, :]
        temp_data_run = vtkT_run[33:103, 65:95, :]

        fig, axs = plt.subplots(3, 2, figsize=(8, 6), dpi=400)
        norm = plt.Normalize(vmin=clims[0], vmax=clims[1], clip=True)

        cmap = plt.get_cmap("jet", 100)
        cmap.set_over("gray")

        yz_cross_idx = 15

        # Extracting cross-sections
        xy_cross_section = temp_data[:, :, -1].T
        xz_cross_section = temp_data[:, 1 + temp_data.shape[1] // 2, :].T
        yz_cross_section = temp_data[yz_cross_idx, :, :].T

        xy_cross_section_run = temp_data_run[:, :, -1].T
        xz_cross_section_run = temp_data_run[:, 1 + temp_data.shape[1] // 2, :].T
        yz_cross_section_run = temp_data_run[yz_cross_idx, :, :].T

        # Plotting cross-sections
        cross_sections = [
            yz_cross_section,
            yz_cross_section_run,
            xy_cross_section,
            xy_cross_section_run,
            xz_cross_section,
            xz_cross_section_run,
        ]

        for ax, cross_section, _num in zip(
            axs.flatten(), cross_sections, range(len(cross_sections))
        ):
            if cross_section is None:
                ax.axis("off")  # Turn off the axis for the empty subplot
                continue
            contour = ax.contourf(
                cross_section,
                levels=np.linspace(clims[0], clims[1], 100),
                cmap=cmap,
                norm=norm,
                extend="max",
                vmin=clims[0],
                vmax=clims[1],
            )
            ax.set_aspect("equal")
            ax.axis("off")
            if _num == 2 or _num == 3:
                # Plot a vertical dotted line at x = 50
                ax.axvline(x=yz_cross_idx, color="k", linestyle=":", linewidth=2.0)
                ax.axhline(
                    y=1 + temp_data.shape[1] // 2,
                    color="k",
                    linestyle=":",
                    linewidth=2.0,
                )

            # Define the scale bar length in pixels and the corresponding real-world length
            if _num == 0 or _num == 1:
                scale_bar_length = 5  # in pixels
                real_length = 100  # in micrometers, for example
            else:
                scale_bar_length = 10  # in pixels
                real_length = 200  # in micrometers, for example

            x_move = 1.0
            # Create a rectangle for the scale bar
            scale_bar = patches.Rectangle(
                (
                    x_move * cross_section.shape[1] / yz_cross_section.shape[1],
                    1 * cross_section.shape[1] / xy_cross_section.shape[1],
                ),
                scale_bar_length,
                1 / cross_section.shape[1],
                edgecolor="white",
                facecolor="white",
            )

            # Add the scale bar to the plot
            ax.add_patch(scale_bar)

            # Add text to indicate the real-world length
            ax.text(
                x_move * cross_section.shape[1] / yz_cross_section.shape[1]
                + scale_bar_length / 2,
                2 * cross_section.shape[1] / xy_cross_section.shape[1],
                f"{real_length} µm",
                color="white",
                ha="center",
            )

            # # Calculate the length of the arrows in data coordinates
            # arrow_length = 10 * (cross_section.shape[1] / xy_cross_section.shape[1])
            # print(arrow_length)

            # # Add arrows outside the plot area pointing orthogonally near the origin with equal length
            # ax.annotate(
            #     "",
            #     xy=(arrow_length, 0),
            #     xycoords="data",
            #     xytext=(0, 0),
            #     arrowprops=dict(facecolor="black", shrink=0.05, width=0.5, headwidth=5),
            # )
            # ax.annotate(
            #     "",
            #     xy=(0, arrow_length),
            #     xycoords="data",
            #     xytext=(0, 0),
            #     arrowprops=dict(facecolor="black", shrink=0.05, width=0.5, headwidth=5),
            # )

            # # Add labels to the arrows
            # ax.text(
            #     arrow_length + 0.5,
            #     0.5,
            #     "X",
            #     transform=ax.transData,
            #     ha="center",
            #     va="center",
            # )
            # ax.text(
            #     0.5,
            #     arrow_length + 0.5,
            #     "Y",
            #     transform=ax.transData,
            #     ha="center",
            #     va="center",
            # )

        # Create a new axis for the colorbar
        cbar_ax = fig.add_axes(
            [0.95, 0.25, 0.02, 0.52]
        )  # [left, bottom, width, height]

        cbar = fig.colorbar(
            contour,
            cax=cbar_ax,
            extend="max",
            orientation="vertical",
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
            f"{Nonmesh['save_path']}/cross_sections_{save_str}.pdf",
            dpi=400,
            bbox_inches="tight",
        )
        plt.savefig(
            f"{Nonmesh['save_path']}/cross_sections_{save_str}.png",
            dpi=400,
            bbox_inches="tight",
        )
        plt.close()

        return 0

    _ = get_value_for_mean(Nonmesh_truth["save_path"], Nonmesh["save_path"], 1000)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        DEVICE_ID = 0

        running_file = "examples/haste_square_10x10.json"
        truth_file = "examples/truth_square_10x10.json"
        plot_length_error(DEVICE_ID, running_file, truth_file)

        sys.exit()

    DEVICE_ID = int(sys.argv[1])
    running_file = sys.argv[2]
    truth_file = sys.argv[3]
    plot_length_error(DEVICE_ID, running_file, truth_file)

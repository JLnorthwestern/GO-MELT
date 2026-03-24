import os
import sys
import json
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from go_melt.computeFunctions import SetupProperties, SetupLevels, SetupNonmesh


def configure_device(device_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    except Exception:
        import jax

        jax.config.update("jax_platform_name", "cpu")
        print("No GPU found.")
    os.system("clear")


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def reshape_and_transpose(data, shape):
    return np.array(data.reshape(*shape)).transpose((2, 1, 0))


def extract_cross_sections(volume, window=41):
    # Get volume dimensions
    nx, ny, nz = volume.shape

    # Compute center indices
    cx, cy = nx // 2, ny // 2  # last slice in z

    # Compute half window
    hw = window // 2

    return {
        "xy": volume[:, :, -1].T,
        "xy_small": volume[cx - hw : cx + hw, cy - hw : cy + hw, -1].T,
        "xz_small": volume[cx - hw : cx + hw, cy, :].T,
        "yz_small": volume[cx, cy - hw : cy + hw, :].T,
    }


# def extract_cross_sections(volume):
#     return {
#         "xy": volume[:, :, -1].T,
#         "xy_small": volume[95:137, 95:137, -1].T,
#         "xz_small": volume[95:137, 115, :].T,
#         "yz_small": volume[115, 95:137, :].T,
#     }


def plot_cross_sections(sections, save_path, label):
    fig = plt.figure(figsize=(18, 24), dpi=400)
    gs = gridspec.GridSpec(6, 4, figure=fig, hspace=0.1, wspace=0.1)

    # max_TAM = 0
    # for section in sections:
    #     max_TAM = max([max_TAM, section.max()])
    max_TAM = 1100

    cmap = plt.get_cmap("jet", 100)
    levels = np.linspace(0, max_TAM, 100)

    # Create axes according to layout
    axs = [
        fig.add_subplot(gs[0, :]),  # section 1
        fig.add_subplot(gs[1:3, 0:2]),  # section 2
        fig.add_subplot(gs[1, 2:4]),  # section 3
        fig.add_subplot(gs[2, 2:4]),  # section 4
        fig.add_subplot(gs[3, :]),  # section 5
        fig.add_subplot(gs[4:6, 0:2]),  # section 6
        fig.add_subplot(gs[4, 2:4]),  # section 7
        fig.add_subplot(gs[5, 2:4]),  # section 8
    ]

    for ax, section in zip(axs, sections):
        contour = ax.contourf(section, levels=levels, cmap=cmap, extend="max")
        # cbar = plt.colorbar(contour, ax=ax)
        # cbar.set_label("Time above melting (μs)", fontsize=16)
        # cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
        ax.set_aspect("equal")
        ax.axis("off")

    plt.savefig(
        f"{save_path}/cross_sections_{label}_nocolorbar.png",
        dpi=400,
        bbox_inches="tight",
    )
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label("Time above melting (μs)", fontsize=16)
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    plt.savefig(
        f"{save_path}/cross_sections_{label}_withcolorbar.png",
        dpi=400,
        bbox_inches="tight",
    )
    plt.close()


def get_value_for_mean(truth_folder, run_folder, record_lab, levels, save_path):
    truth_file = os.path.join(truth_folder, f"accum_time{record_lab:04}.npz")
    run_file = os.path.join(run_folder, f"accum_time{record_lab:04}.npz")

    truth_data = jnp.load(truth_file)["accum_time"] * 1e6
    run_data = jnp.load(run_file)["accum_time"] * 1e6

    shape = levels[0]["nodes"][2], levels[0]["nodes"][1], levels[0]["nodes"][0]
    vtk_truth = reshape_and_transpose(truth_data, shape)
    vtk_run = reshape_and_transpose(run_data, shape)

    sections = [
        extract_cross_sections(vtk_truth)["xy"],
        extract_cross_sections(vtk_truth)["xy_small"],
        extract_cross_sections(vtk_truth)["xz_small"],
        extract_cross_sections(vtk_truth)["yz_small"],
        extract_cross_sections(vtk_run)["xy"],
        extract_cross_sections(vtk_run)["xy_small"],
        extract_cross_sections(vtk_run)["xz_small"],
        extract_cross_sections(vtk_run)["yz_small"],
    ]

    plot_cross_sections(sections, save_path, "HASTE")


def plot_length_error(device_id, running_file, truth_file):
    configure_device(device_id)

    solver_input = load_json(running_file)
    solver_input_truth = load_json(truth_file)

    properties = SetupProperties(solver_input.get("properties", {}))
    levels = SetupLevels(solver_input, properties)
    nonmesh = SetupNonmesh(solver_input.get("nonmesh", {}))
    nonmesh_truth = SetupNonmesh(solver_input_truth.get("nonmesh", {}))

    get_value_for_mean(
        nonmesh_truth["save_path"],
        nonmesh["save_path"],
        249,
        levels,
        nonmesh["save_path"],
    )


if __name__ == "__main__":
    if len(sys.argv) != 4:
        plot_length_error(
            device_id=0,
            running_file="examples/haste_production_100x10x10.json",
            truth_file="examples/truth_production_100x10x10.json",
        )
    else:
        plot_length_error(
            device_id=int(sys.argv[1]),
            running_file=sys.argv[2],
            truth_file=sys.argv[3],
        )

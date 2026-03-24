from pyevtk.hl import gridToVTK
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from go_melt.computeFunctions import *


def saveSVDImages(NC, save_str, record_lab, save_path, zoffset, U, datastring):
    """saveResult saves a vtk for the current level's temperature field
    :param Level: structure of Level
    :param save_str: prefix of save string
    :param record_lab: recording label that is incremented after each save
    :param save_path: folder where file is saved
    :param zoffset: used for rendering purposes, no effect on model itself
    """
    # List coordinates in each direction for structured save
    vtkcx = np.array(NC[0])
    vtkcy = np.array(NC[1])
    vtkcz = np.array(NC[2] - zoffset)
    node = [len(NC[0]), len(NC[1]), len(NC[2])]
    # Reshape the temperature field for correct rendering later
    vtkT = np.array(U.reshape(node[2], node[1], node[0])).transpose((2, 1, 0))
    # Save a vtr
    pointData = {datastring: vtkT}
    vtkSave = f"{save_path}{save_str}{record_lab:08}"
    gridToVTK(vtkSave, vtkcx, vtkcy, vtkcz, pointData=pointData)


def plot_modes(input_file):
    with open(input_file, "r") as read_file:
        solver_input = json.load(read_file)

    # Set up dictionaries for properties, multilevel information, and nonmesh info
    Properties = SetupProperties(solver_input.get("properties", {}))
    Levels = SetupLevels(solver_input, Properties)
    Nonmesh = SetupNonmesh(solver_input.get("nonmesh", {}))

    # Load basis functions
    basis_funs = jnp.load(Nonmesh["save_path"] + "PlottingModes.npz")
    u0 = basis_funs["u0"].T[:, : Levels[4]["nn"]]
    u1 = basis_funs["u1"].T
    meanfield = basis_funs["xmean"]

    # Reshape and transpose
    u0 = u0.reshape(
        -1, Levels[4]["nodes"][2], Levels[4]["nodes"][1], Levels[4]["nodes"][0]
    ).transpose([0, 3, 2, 1])
    u1 = u1.reshape(
        -1, Levels[4]["nodes"][2], Levels[4]["nodes"][1], Levels[4]["nodes"][0]
    ).transpose([0, 3, 2, 1])
    meanfield = (
        meanfield[: Levels[4]["nn"]]
        .reshape(Levels[4]["nodes"][2], Levels[4]["nodes"][1], Levels[4]["nodes"][0])
        .transpose([2, 1, 0])
    )

    # Determine number of input and output modes
    num_input_modes = u0.shape[0]
    num_output_modes = u1.shape[0]
    num_rows = max(num_input_modes, num_output_modes)

    # Create subplots with 2 columns and extra row for mean field
    fig, axs = plt.subplots(num_rows + 1, 2, figsize=(20, 2.5 * (num_rows + 1)))

    # Plot input modes in left column
    for i in range(num_input_modes):
        vmin, vmax = u0[i].min(), u0[i].max()
        im = axs[i, 0].imshow(
            np.flip(u0[i][:, :, -1].T, axis=0),
            aspect="auto",
            cmap="jet",
            vmin=vmin,
            vmax=vmax,
        )
        axs[i, 0].axis("equal")
        axs[i, 0].axis("off")
        axs[i, 0].set_title(f"Input Mode {i + 1}", fontsize=16)
        cbar = fig.colorbar(im, ax=axs[i, 0], aspect=5)
        cbar.ax.tick_params(labelsize=12)

    # Plot output modes in right column
    for i in range(num_output_modes):
        vmin, vmax = u0[i].min(), u0[i].max()
        im = axs[i, 1].imshow(
            np.flip(u1[i][:, :, -1].T, axis=0),
            aspect="auto",
            cmap="jet",
            vmin=vmin,
            vmax=vmax,
        )
        axs[i, 1].axis("equal")
        axs[i, 1].axis("off")
        axs[i, 1].set_title(f"Output Mode {i + 1}", fontsize=16)
        cbar = fig.colorbar(im, ax=axs[i, 1], aspect=5)
        cbar.ax.tick_params(labelsize=12)

    # Plot mean field centered below both columns
    for j in range(2):
        axs[-1, j].axis("off")

    im = axs[-1, 0].imshow(
        np.flip(meanfield[:, :, -1].T, axis=0), aspect="auto", cmap="jet"
    )
    axs[-1, 0].axis("equal")
    axs[-1, 0].set_title("Mean Field", fontsize=16)
    cbar = fig.colorbar(im, ax=axs[-1, 0], aspect=5)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(label="Temperature (K)", size=14)

    # Final layout and save
    plt.tight_layout()
    fig.savefig(Nonmesh["save_path"] + "ModeFigures.png", bbox_inches="tight")
    fig.savefig(Nonmesh["save_path"] + "ModeFigures.pdf", bbox_inches="tight")
    plt.close()

    print("Plotting complete and figure saved.")

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import sys
from collections import defaultdict
from go_melt.computeFunctions import *
from helpers import *


def generate_dmd_plots(json_path):
    # Load JSON and extract save path
    with open(json_path, "r") as f:
        config = json.load(f)
    save_path = config["nonmesh"]["save_path"]
    metric_npz_path = os.path.join(save_path, "MetricData.npz")
    global_npz_path = os.path.join(save_path, "GlobalPair.npz")

    # Load NPZ data
    data = np.load(metric_npz_path)
    error_per_sample_all = data["reconstruction_error"]
    pred_err_all = data["prediction_error"]
    s = data["s_input"]
    s_np1 = data["s_output"]
    s_norm = data["s_norm_input"]
    s_norm_np1 = data["s_norm_output"]
    s_comp = data["s_comp_input"]
    s_comp_np1 = data["s_comp_output"]
    mask = data["input_mask"]
    mask_np1 = data["output_mask"]
    rank_r = int(data["reduced_rank"])

    data = np.load(global_npz_path)
    global_pairs = data["global_pairs"]

    # Create 2x2 subplot
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # (1,1) Normalized singular values
    ax = axs[0, 0]
    k_in, k_out = s.shape[0], s_np1.shape[0]
    ax.semilogy(np.arange(1, k_in + 1), s_norm, "-", label="Input")
    ax.semilogy(np.arange(1, k_out + 1), s_norm_np1, "--", label="Output")
    ax.semilogy(np.arange(1, rank_r + 1), s_norm[mask], ":o", label="Reduced input")
    ax.semilogy(
        np.arange(1, rank_r + 1), s_norm_np1[mask_np1], "-.^", label="Reduced output"
    )
    ax.set_xlabel("Mode index")
    ax.set_ylabel("Normalized singular values")
    ax.grid(True, which="both", linestyle="-", linewidth=0.5, color="gray")
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

    # (1,2) Cumulative energy
    ax = axs[0, 1]
    ax.semilogy(np.arange(1, k_in + 1), s_comp, "-", label="Input")
    ax.semilogy(np.arange(1, k_out + 1), s_comp_np1, "--", label="Output")
    ax.semilogy(np.arange(1, rank_r + 1), s_comp[mask], ":o", label="Reduced input")
    ax.semilogy(
        np.arange(1, rank_r + 1), s_comp_np1[mask_np1], "-.^", label="Reduced output"
    )
    ax.set_xlim(0.5, rank_r + 2)
    ax.set_xlabel("Reduced dimension")
    ax.set_ylabel("Normalized cumulative energy")
    ax.grid(True, which="both", linestyle="-", linewidth=0.5, color="gray")
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

    # (2,1) Histogram: Reconstruction Error
    ax = axs[1, 0]
    bins = np.linspace(0.0, error_per_sample_all.max(initial=0.0), 31)
    ax.hist(error_per_sample_all, bins=bins, edgecolor="black")
    ax.set_xlabel("Relative Error (Reconstruction)")
    ax.set_ylabel("Count")
    ax.grid(True, which="major", linestyle="-", linewidth=0.75, color="gray")

    # (2,2) Histogram: Prediction Error
    ax = axs[1, 1]
    bins_pred = np.linspace(0.0, pred_err_all.max(initial=0.0), 31)
    ax.hist(pred_err_all, bins=bins_pred, edgecolor="black")
    ax.set_xlabel("Relative Error (Prediction)")
    ax.set_ylabel("Count")
    ax.grid(True, which="major", linestyle="-", linewidth=0.75, color="gray")

    plt.tight_layout()
    sv_png = os.path.join(save_path, "DMD_summary_plots.png")
    sv_pdf = os.path.join(save_path, "DMD_summary_plots.pdf")
    plt.savefig(sv_png, dpi=250)
    plt.savefig(sv_pdf)
    plt.close(fig)

    print(f"Saved combined plot to: {sv_png} and {sv_pdf}")

    # (2,1) Line plot of error_per_sample_all
    ax = axs[1, 0]
    ax.plot(
        np.arange(len(error_per_sample_all)),
        error_per_sample_all,
        "-",
        color="black",
        linewidth=1.5,
    )
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Relative Error (Reconstruction)")
    ax.grid(True, which="major", linestyle="-", linewidth=0.75, color="gray")

    # Save standalone version
    recon_line_png = os.path.join(save_path, "DMD_reconstruction_error_line.png")
    fig_recon = plt.figure(figsize=(6, 4))
    ax_recon = fig_recon.add_subplot(111)
    ax_recon.plot(
        np.arange(len(error_per_sample_all)),
        error_per_sample_all,
        "-",
        color="black",
        linewidth=1.5,
    )
    ax_recon.set_xlabel("Sample Index")
    ax_recon.set_ylabel("Relative Error (Reconstruction)")
    ax_recon.grid(True, which="major", linestyle="-", linewidth=0.75, color="gray")
    plt.tight_layout()
    plt.savefig(recon_line_png, dpi=250)
    plt.close(fig_recon)

    print(f"Saved reconstruction error line plot to: {recon_line_png}")

    plot_error_by_layer_sections(
        global_pairs, error_per_sample_all, pred_err_all, save_path
    )

    plot_modes(json_path)


def plot_error_by_layer_sections(
    global_pairs, reconstruction_error, prediction_error, save_path
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from collections import defaultdict

    # Group errors by track
    track_ids = [os.path.basename(pair[0]) for pair in global_pairs]
    recon_by_track = defaultdict(list)
    pred_by_track = defaultdict(list)
    for track, recon, pred in zip(track_ids, reconstruction_error, prediction_error):
        recon_by_track[track].append(recon)
        pred_by_track[track].append(pred)

    # Separate into layers
    layer1 = [t for t in recon_by_track if int(t.replace("Track", "")) <= 235]
    layer2 = [t for t in recon_by_track if int(t.replace("Track", "")) > 235]

    def split_sections(tracks):
        sorted_tracks = sorted(tracks, key=lambda x: int(x.replace("Track", "")))
        n = len(sorted_tracks)
        third = n // 3
        return [
            sorted_tracks[:third],  # Early
            sorted_tracks[third : 2 * third],  # Middle
            sorted_tracks[2 * third :],  # End
        ]

    # Split each layer into 3 sections
    sections = split_sections(layer1) + split_sections(layer2)
    section_labels = [
        "Layer 1 – Early",
        "Layer 1 – Middle",
        "Layer 1 – End",
        "Layer 2 – Early",
        "Layer 2 – Middle",
        "Layer 2 – End",
    ]

    # Flatten data for seaborn
    def flatten_errors(error_dict):
        values = []
        tags = []
        for label, group in zip(section_labels, sections):
            group_data = [error_dict[t] for t in group if t in error_dict]
            if group_data:
                for err in np.concatenate(group_data):
                    values.append(err)
                    tags.append(label)
        return values, tags

    recon_values, recon_tags = flatten_errors(recon_by_track)
    pred_values, pred_tags = flatten_errors(pred_by_track)

    # Plot side-by-side violin plots
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    sns.violinplot(
        x=recon_tags, y=recon_values, ax=axs[0], scale="width", inner="quartile", cut=0
    )
    axs[0].set_xticklabels(section_labels, rotation=30, ha="right")
    axs[0].set_ylabel("Relative Error")
    axs[0].set_title("Reconstruction Error")
    axs[0].grid(True, linestyle="-", linewidth=0.5, color="gray")

    sns.violinplot(
        x=pred_tags, y=pred_values, ax=axs[1], scale="width", inner="quartile", cut=0
    )
    axs[1].set_xticklabels(section_labels, rotation=30, ha="right")
    axs[1].set_title("Prediction Error")
    axs[1].grid(True, linestyle="-", linewidth=0.5, color="gray")

    plt.tight_layout()
    plot_path = os.path.join(save_path, "error_by_layer_sections_violin_subplot.png")
    plt.savefig(plot_path, dpi=250)
    plot_path = os.path.join(save_path, "error_by_layer_sections_violin_subplot.pdf")
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"Saved layered section violin subplot to: {plot_path}")


if __name__ == "__main__":
    generate_dmd_plots("examples/training_triangle_20x20.json")

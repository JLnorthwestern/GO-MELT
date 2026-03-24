import os
import glob
import numpy as np
import jax.numpy as jnp
from jax import jit
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import jax
import gc
from go_melt.computeFunctions import *
from helpers import *
import json
from sklearn.metrics.pairwise import cosine_similarity
import sys
import re


def DMD(DEVICE_ID, input_file, total_samples=3000):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    try:
        # Run on single GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
        os.system("clear")
    except:
        # Run on CPU
        jax.config.update("jax_platform_name", "cpu")
        os.system("clear")
        print("No GPU found.")

    VAREPS = 1 - 1e-2  # possibly needs to be 2e-2
    CASE = True
    TRACK = True

    # Configure global sample count and RNG seed
    seed = 0

    # Load input file
    with open(input_file, "r") as read_file:
        solver_input = json.load(read_file)
    Properties = SetupProperties(solver_input.get("properties", {}))
    Levels = SetupLevels(solver_input, Properties)
    Levels[4]["node_coords"] = Levels[4]["orig_node_coords"]
    Nonmesh = SetupNonmesh(solver_input.get("nonmesh", {}))

    @jit
    def perform_svd_on_dataset(v_combined):
        u, s, vh = jnp.linalg.svd(v_combined, full_matrices=False)
        return u, s, vh

    # Case folders
    case_folders = [sorted(glob.glob(f"{Nonmesh['save_path']}"))[-1]]

    # Seed for reproducibility
    key_indices = jax.random.PRNGKey(seed)

    # Preallocated outputs (will create after we learn shapes)
    all_data = None  # shape (nn + n_probe, N)
    all_data_np1 = None  # shape (nn, N)

    for case_folder in case_folders:
        if not CASE:
            continue

        # Discover track folders
        if TRACK:
            track_folders = sorted(
                glob.glob(os.path.join(Nonmesh["haste_training_dir"], "Track*"))
            )
        else:
            track_folders = [Nonmesh["haste_training_dir"]]

        # Enumerate all available (t, t+1) pairs globally
        track_to_files = {}
        global_pairs = (
            []
        )  # list[(track_folder, idx)] -> pair is files[idx], files[idx+1]
        for track_folder in track_folders:
            files = sorted(glob.glob(os.path.join(track_folder, "FEA_*.npz")))
            if len(files) < 2:
                continue
            track_to_files[track_folder] = files
            for idx in range(len(files) - 1):
                global_pairs.append((track_folder, idx))

        total_available = len(global_pairs)
        global_pair_npz = os.path.join(Nonmesh["save_path"], "GlobalPair.npz")
        jnp.savez(
            global_pair_npz,
            global_pairs=global_pairs,
        )
        if total_available == 0:
            raise RuntimeError(
                f"No (t, t+1) file pairs found under {Nonmesh['haste_training_dir']}"
            )

        # Global sampling without replacement
        safe_total = int(min(max(total_samples, 0), total_available))
        if safe_total == 0:
            raise ValueError(
                f"Requested total_samples={total_samples}, but no pairs are available."
            )

        print(f"Total available samples to select from: {total_available}")
        choice_idx = jax.random.choice(
            key_indices, jnp.arange(total_available), shape=(safe_total,), replace=False
        )
        choice_idx = jnp.sort(choice_idx).tolist()
        choice_idx = [int(i) for i in choice_idx]

        # We'll infer feature sizes from the first selected pair, then preallocate.
        nn = None  # length of v
        n_probe = None  # length of probe

        # Preallocated NumPy buffers (fill columns -> (features, N))
        buf_features = None  # (nn + n_probe, safe_total)
        buf_np1 = None  # (nn, safe_total)

        for col, ci in enumerate(
            tqdm(choice_idx, desc="Processing globally sampled pairs")
        ):
            track_folder, idx = global_pairs[ci]
            files = track_to_files[track_folder]
            f_t = files[idx]
            f_tp1 = files[idx + 1]

            # ---- load time t ----
            data_t = np.load(f_t)
            v_t = data_t["v"]
            # You can flatten if needed; the original code used vectors already:
            v_t = v_t.reshape(-1)
            Levels[4]["T0"] = v_t
            Levels[4]["S1"] = data_t["s"]

            match = re.search(r"FEA_(\d{7})\.npz", f_t)
            if match:
                record_lab = int(match.group(1))
                saveResult(Levels[4], "Level4_", record_lab, case_folder, 0.0)

            # ---- load time t+1 ----
            data_tp1 = np.load(f_tp1)
            v_tp1 = data_tp1["v"].reshape(-1)
            probe_tp1 = data_tp1["probe"].reshape(-1)

            # On first iteration, infer shapes and preallocate buffers
            if buf_features is None:
                nn = v_t.size
                n_probe = probe_tp1.size

                # Combined feature dtype (v + probe) may upcast; choose a safe result dtype
                feat_dtype = np.result_type(v_t.dtype, probe_tp1.dtype)
                # (features, samples)
                buf_features = np.empty((nn + n_probe, safe_total), dtype=feat_dtype)
                # (features_t+1, samples)
                buf_np1 = np.empty((nn, safe_total), dtype=v_tp1.dtype)

            # Fill column col
            buf_features[:nn, col] = v_t
            buf_features[nn:, col] = probe_tp1
            buf_np1[:, col] = v_tp1
        # Convert to JAX arrays once at the end
        all_data = jnp.asarray(buf_features)  # (nn + n_probe, N)
        all_data_np1 = jnp.asarray(buf_np1)  # (nn, N)

        all_data = all_data.at[nn:].set(all_data[nn:])

        # If you only have one case folder, you can break here. Otherwise, you may want to
        # concatenate across cases. This code mirrors your original single-case behavior.
        break

    # --- Clean DMD post-processing & saving ------------------------------------------
    # Ensure save path exists
    os.makedirs(Nonmesh["save_path"], exist_ok=True)

    # Convenience
    nn = int(Levels[4]["nn"])

    # 1) Center input/output and compute SVDs
    mean_field = all_data.mean(axis=1)  # (features_in,)
    Xc = all_data - mean_field[:, None]
    Yc = all_data_np1 - mean_field[:nn, None]

    u, s_orig, vh = perform_svd_on_dataset(Xc)  # Input SVD
    u_np1, s_orig_np1, vh_np1 = perform_svd_on_dataset(Yc)  # Output SVD

    # 2) Singular value stats (energies, normalized values, cumulative sums)
    s = s_orig**2
    s_np1 = s_orig_np1**2
    s_sum = jnp.maximum(s.sum(), jnp.finfo(s.dtype).eps)
    s_np1_sum = jnp.maximum(s_np1.sum(), jnp.finfo(s_np1.dtype).eps)

    s_norm = s / s_sum
    s_comp = jnp.cumsum(s_norm)

    s_norm_np1 = s_np1 / s_np1_sum
    s_comp_np1 = jnp.cumsum(s_norm_np1)

    # 3) Rank/mask selection by cumulative energy threshold with guarantees
    def energy_mask(cumvals: jnp.ndarray, vareps: float) -> jnp.ndarray:
        """
        Select modes up to the first index where cumulative energy >= vareps.
        Always returns at least one True. Shape: (k,)
        """
        # index of first cumulative >= vareps
        idx = int(jnp.searchsorted(cumvals, vareps, side="left"))
        idx = max(0, min(idx, int(cumvals.shape[0]) - 1))
        return jnp.arange(cumvals.shape[0]) <= idx

    mask_in = energy_mask(s_comp, VAREPS)
    mask_out = energy_mask(s_comp_np1, VAREPS)

    # Optional: include any single dominant mode if it alone exceeds VAREPS
    dom_in = s_norm >= VAREPS
    dom_out = s_norm_np1 >= VAREPS

    # Final mask: union of input/output energy masks and dominant-mode checks
    mask = jnp.logical_or(
        jnp.logical_or(mask_in, mask_out), jnp.logical_or(dom_in, dom_out)
    )
    mask_np1 = mask  # keep symmetry with original code
    rank_r = int(mask.sum())

    # 4) Reduced factors (keep original names for compatibility)
    u_m = u[:, mask]  # (features_in, r)
    s_m = s_orig[mask]  # (r,)
    v_m = vh[mask, :].T  # (N, r)
    u_m_np1 = u_np1[:, mask_np1]  # (features_out=nn, r)

    # 5) Reconstruction quality of Y using the output reduced subspace (TRAINING SET)
    reconstruction = u_m_np1 @ (u_m_np1.T @ Yc) + mean_field[:nn, None]
    difference = reconstruction - all_data_np1
    denom = jnp.maximum(jnp.linalg.norm(all_data_np1, axis=0), 1e-12)

    # --- NEW: keep training error separate
    error_per_sample_train = (
        jnp.linalg.norm(difference, axis=0) / denom
    )  # shape (N_train,)
    error_per_sample_train = np.asarray(error_per_sample_train)

    # 6) Plots (histogram for TRAINING ONLY)
    fig = plt.figure()
    plt.hist(error_per_sample_train, bins=30, edgecolor="black")
    plt.xlabel("Relative $\\rm{L_2}$ Error (training)")
    plt.ylabel("Count")
    plt.grid(True, which="major", linestyle="-", linewidth=0.75, color="gray")
    plt.tight_layout()
    hist_train_png = os.path.join(Nonmesh["save_path"], "DMD_histogram_train.png")
    plt.savefig(hist_train_png, dpi=250)
    plt.close(fig)

    # We only need v_{t+1} for the projection error; no need to load v_t/probe here.
    errs_all = []
    position_all = []
    for track_folder, idx in tqdm(global_pairs, desc="Evaluating error on ALL pairs"):
        files = track_to_files[track_folder]
        f_tp1 = files[idx + 1]

        data_tp1 = np.load(f_tp1)
        v_tp1 = data_tp1["v"].reshape(-1)
        laser_position = data_tp1["Test2pos"][:3]

        y = jnp.asarray(v_tp1)
        y_centered = y - mean_field[:nn]
        y_hat = u_m_np1 @ (u_m_np1.T @ y_centered) + mean_field[:nn]

        num = float(jnp.linalg.norm(y_hat - y))
        den = max(float(jnp.linalg.norm(y)), 1e-12)
        errs_all.append(num / den)
        position_all.append(laser_position)

    error_per_sample_all = np.asarray(errs_all)
    position_all = np.asarray(position_all)

    # Filter for z = 0.04
    z_target = 0.08
    tolerance = 1e-6
    mask_z = np.abs(position_all[:, 2] - z_target) < tolerance

    x = position_all[mask_z, 0]
    y = position_all[mask_z, 1]
    err = error_per_sample_all[mask_z]

    # Scale error for marker size (avoid tiny or huge values)
    size = 50 * (err - err.min()) / (err.ptp() + 1e-6) + 1e-6  # range ~10 to 110

    # Plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        x, y, c=err, s=size, cmap="jet", alpha=0.8, edgecolors="w", linewidths=0.5
    )
    plt.colorbar(scatter, label="Relative $\\rm{L_2}$ Error (Reconstruction)")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.grid(True)

    # Save the figure
    plt.savefig("results/scatter_error.png", format="png", dpi=300)
    plt.savefig("results/scatter_error.pdf", format="pdf")

    # --- NEW: Plot histogram for ALL SAMPLES in a separate figure
    # Optional: use common bins so the two histograms are comparable
    bins = np.linspace(
        0.0,
        max(
            error_per_sample_all.max(initial=0.0),
            error_per_sample_train.max(initial=0.0),
        ),
        31,
    )

    fig = plt.figure()
    plt.hist(error_per_sample_all, bins=bins, edgecolor="black")
    plt.xlabel("Relative $\\rm{L_2}$ Error (Reconstruction)")
    plt.ylabel("Count")
    plt.grid(True, which="major", linestyle="-", linewidth=0.75, color="gray")
    plt.tight_layout()
    hist_all_png = os.path.join(Nonmesh["save_path"], "DMD_histogram_all.png")
    plt.savefig(hist_all_png, dpi=250)
    plt.close(fig)

    print(f"Saved training histogram: {hist_train_png}")
    print(f"Saved all-samples histogram: {hist_all_png}")
    print(f"Maximum error: {error_per_sample_all.max()}")

    # Singular value and cumulative energy plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    k_in = int(s.shape[0])
    k_out = int(s_np1.shape[0])

    # Normalized singular values
    ax1.semilogy(np.arange(1, k_in + 1), np.asarray(s_norm), "-", label="Input")
    ax1.semilogy(np.arange(1, k_out + 1), np.asarray(s_norm_np1), "--", label="Output")
    ax1.semilogy(
        np.arange(1, rank_r + 1), np.asarray(s_norm[mask]), ":o", label="Reduced input"
    )
    ax1.semilogy(
        np.arange(1, rank_r + 1),
        np.asarray(s_norm_np1[mask_np1]),
        "-.^",
        label="Reduced output",
    )
    ax1.set_xlabel("Mode index")
    ax1.set_ylabel("Normalized singular values")
    ax1.grid(True, which="both", linestyle="-", linewidth=0.5, color="gray")
    ax1.legend()
    ax1.xaxis.set_major_locator(
        MaxNLocator(nbins=5, integer=True)
    )  # Integer ticks, ~5 total

    # Cumulative energy
    ax2.semilogy(np.arange(1, k_in + 1), np.asarray(s_comp), "-", label="Input")
    ax2.semilogy(np.arange(1, k_out + 1), np.asarray(s_comp_np1), "--", label="Output")
    ax2.semilogy(
        np.arange(1, rank_r + 1), np.asarray(s_comp[mask]), ":o", label="Reduced input"
    )
    ax2.semilogy(
        np.arange(1, rank_r + 1),
        np.asarray(s_comp_np1[mask_np1]),
        "-.^",
        label="Reduced output",
    )
    ax2.set_xlim(0.5, rank_r + 2)  # Zoom around rank_r
    ax2.set_xlabel("Reduced dimension")
    ax2.set_ylabel("Normalized cumulative energy")
    ax2.grid(True, which="both", linestyle="-", linewidth=0.5, color="gray")
    ax2.xaxis.set_major_locator(
        MaxNLocator(nbins=5, integer=True)
    )  # Integer ticks, ~5 total

    plt.tight_layout()
    sv_png = os.path.join(Nonmesh["save_path"], "singular_values_plot_DMD.png")
    sv_pdf = os.path.join(Nonmesh["save_path"], "singular_values_plot_DMD.pdf")
    plt.savefig(sv_png, dpi=250)
    plt.savefig(sv_pdf)
    plt.close(fig)

    print(f"Saved histogram to: {hist_all_png}")
    print(f"Saved singular value plots to: {sv_png} and {sv_pdf}")

    # 7) Persist plot data for reproducibility
    jnp.savez(
        os.path.join(Nonmesh["save_path"], "plot_data.npz"),
        s=s,
        s_np1=s_np1,
        s_comp=s_comp,
        s_comp_np1=s_comp_np1,
        mask=mask,
        mask_np1=mask_np1,
        vareps=jnp.asarray(VAREPS),
    )

    # Save right singular vectors (reduced modes in snapshot domain)
    vh_filtered = vh[mask]  # (r, N)
    output_path_npy = os.path.join(Nonmesh["save_path"], "DMD_modes.npy")
    jnp.save(output_path_npy, vh_filtered)
    print(f"Saved DMD modes (V^H filtered) to: {output_path_npy}")

    # 8) Build reduced operators Atilde (r x r) and Btilde (r x n_control)
    #    X = [state; control] => u_m partitioned as rows [:nn] (state) and [nn:] (control)
    G = (u_m_np1.T @ all_data_np1) @ v_m  # (r x r)
    G = G * (1.0 / s_m)  # column-scale by Σ^{-1}

    Atilde = (G @ u_m[:nn, :].T) @ u_m_np1  # (r x r)
    Btilde = G @ u_m[nn:, :].T  # (r x n_control)

    # # 9) Save reduced-order model and plotting modes
    # rom_npz = os.path.join(Nonmesh["save_path"], "DMD_ROM.npz")
    # jnp.savez(
    #     rom_npz,
    #     u=u_m_np1,
    #     s=s_m,
    #     v=v_m,
    #     A=Atilde,
    #     B=Btilde,
    #     xmean=mean_field,
    # )
    # print(f"Saved DMD ROM to: {rom_npz}")

    # plotting_npz = os.path.join(Nonmesh["save_path"], "PlottingModes.npz")
    # jnp.savez(
    #     plotting_npz,
    #     u1=u_m_np1,
    #     u0=u_m,
    #     xmean=mean_field,
    # )
    # print(f"Saved plotting modes to: {plotting_npz}")

    # # 10) Symlinks for convenience (best-effort, OS-dependent)
    # try:
    #     if os.path.islink("DMD_ROM.npz"):
    #         os.unlink("DMD_ROM.npz")
    #     os.symlink(rom_npz, "DMD_ROM.npz")
    # except Exception:
    #     pass

    # # 11) Save mean field and link
    # mean_field_path = os.path.join(Nonmesh["save_path"], "mean_field.npy")
    # jnp.save(mean_field_path, mean_field)
    # try:
    #     if os.path.islink("mean_field.npy"):
    #         os.unlink("mean_field.npy")
    #     os.symlink(mean_field_path, "mean_field.npy")
    # except Exception:
    #     pass

    # --- Prediction error on ALL pairs
    pred_errs_all = []
    position_all = []
    mean_y = mean_field[:nn]
    mean_probe = mean_field[nn:]
    for track_folder, idx in tqdm(global_pairs, desc="Prediction on ALL pairs"):
        files = track_to_files[track_folder]
        f_t = files[idx]
        f_tp1 = files[idx + 1]

        data_t = np.load(f_t)
        data_tp1 = np.load(f_tp1)

        v_t = data_t["v"].reshape(-1)  # (nn,)
        v_tp1 = data_tp1["v"].reshape(-1)  # (nn,)
        probe_tp1 = data_tp1["probe"].reshape(-1)  # (n_probe,)
        laser_position = data_tp1["Test2pos"][:3]

        z_t = u_m_np1.T @ (v_t - mean_y)  # (r,)
        u_tp1 = probe_tp1 - mean_probe  # (n_probe,)
        z_pred = Atilde @ z_t + Btilde @ u_tp1  # (r,)
        y_pred = u_m_np1 @ z_pred + mean_y  # (nn,)

        num = np.linalg.norm(y_pred - v_tp1)
        den = max(np.linalg.norm(v_tp1), 1e-12)
        position_all.append(laser_position)
        pred_errs_all.append(num / den)

    pred_err_all = np.asarray(pred_errs_all)
    position_all = np.asarray(position_all)

    # Filter for z = 0.04
    z_target = 0.08
    tolerance = 1e-6
    mask_z = np.abs(position_all[:, 2] - z_target) < tolerance

    x = position_all[mask_z, 0]
    y = position_all[mask_z, 1]
    err = pred_err_all[mask_z]

    # Scale error for marker size (avoid tiny or huge values)
    size = 50 * (err - err.min()) / (err.ptp() + 1e-6) + 1e-6  # range ~10 to 110

    # Plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        x, y, c=err, s=size, cmap="jet", alpha=0.8, edgecolors="w", linewidths=0.5
    )
    plt.colorbar(scatter, label="Relative $\\rm{L_2}$ Error (Prediction)")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.grid(True)

    # Save the figure
    plt.savefig("results/scatter_error_prediction.png", format="png", dpi=300)
    plt.savefig("results/scatter_error_prediction.pdf", format="pdf")

    bins_pred = np.linspace(
        0.0,
        pred_err_all.max(initial=0.0),
        31,
    )
    fig = plt.figure()
    plt.hist(pred_err_all, bins=bins_pred, edgecolor="black")
    plt.xlabel("Relative $\\rm{L_2}$ Error (Prediction)")
    plt.ylabel("Count")
    plt.grid(True, which="major", linestyle="-", linewidth=0.75, color="gray")
    plt.tight_layout()
    pred_hist_all_png = os.path.join(Nonmesh["save_path"], "DMD_hist_pred_all.png")
    plt.savefig(pred_hist_all_png, dpi=250)
    plt.close(fig)

    print(f"Maximum error: {pred_err_all.max()}")

    # 12) Generate mode images
    modes_dir = os.path.join(Nonmesh["save_path"], "modes")
    os.makedirs(modes_dir, exist_ok=True)

    Levels[4]["T0"] = mean_field[:nn]
    saveResult(Levels[4], "MeanField_", 1, f"{modes_dir}/", 0)
    plot_modes(input_file)

    for savenum, _u_m_np1 in enumerate(u_m_np1.T, start=1):
        print(f"Saving u_m_np1 mode {savenum}")
        saveSVDImages(
            Levels[4]["node_coords"],
            "u_m_np1_",
            savenum,
            f"{modes_dir}/",
            0,
            _u_m_np1[:nn],
            "Mode",
        )

    for savenum, _u_m in enumerate(u_m.T, start=1):
        print(f"Saving u_m mode {savenum}")
        saveSVDImages(
            Levels[4]["node_coords"],
            "u_m_",
            savenum,
            f"{modes_dir}/",
            0,
            _u_m[:nn],
            "Mode",
        )

    metric_npz = os.path.join(Nonmesh["save_path"], "MetricData.npz")
    jnp.savez(
        metric_npz,
        reconstruction_error=error_per_sample_all,
        prediction_error=pred_err_all,
        xmean=mean_field,
        s_input=s,
        s_output=s_np1,
        s_norm_input=s_norm,
        s_norm_output=s_norm_np1,
        s_comp_input=s_comp,
        s_comp_output=s_comp_np1,
        input_mask=mask,
        output_mask=mask_np1,
        reduced_rank=rank_r,
    )
    # --- End clean DMD post-processing -----------------------------------------------


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 run_go_melt.py DEVICE_ID input_file number_of_samples")
        DEVICE_ID = 0
        input_file = "examples/training_triangle_20x20.json"
        DMD(DEVICE_ID, input_file)
    else:
        DEVICE_ID = int(sys.argv[1])
        input_file = sys.argv[2]
        total_number_of_samples = int(sys.argv[3])
        DMD(DEVICE_ID, input_file, total_number_of_samples)

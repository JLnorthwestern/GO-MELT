import sys
import os
from operator import getitem

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from go_melt.computeFunctions import *


def calculateSubsectionElements(Levels, ne_nn, laser_position, Properties):
    # Get shape functions and weights
    coords = getSampleCoords(Levels[3])
    Nf, _, wqf = computeQuad3dFemShapeFunctions_jax(coords)

    def stepcomputeCoarseSource(ieltf):
        # Get the nodal indices for that element
        ix, iy, iz, idx = convert2XYZ(
            ieltf,
            Levels[3]["elements"][0],
            Levels[3]["elements"][1],
            Levels[3]["nodes"][0],
            Levels[3]["nodes"][1],
        )
        # Get nodal coordinates for the fine element
        x, y, z = getQuadratureCoords(Levels[3], ix, iy, iz, Nf)

        x_bool = (
            1.0
            * (x.min() > (laser_position[0] - 2 * Properties["laser_radius"]))
            * (x.max() < (laser_position[0] + 2 * Properties["laser_radius"]))
        )
        y_bool = (
            1.0
            * (y.min() > laser_position[1] - 2 * Properties["laser_radius"])
            * (y.max() < laser_position[1] + 2 * Properties["laser_radius"])
        )
        z_bool = 1.0 * (z.min() > laser_position[2] - 2 * Properties["laser_depth"])
        # Compute the source at the quadrature point location
        return (x_bool * y_bool) * z_bool

    vstepcomputeCoarseSource = jax.vmap(stepcomputeCoarseSource)
    _data = vstepcomputeCoarseSource(jnp.arange(ne_nn[1]))

    return _data


def update_level_center_indices(Levels, ne_nn, Properties):
    """
    Updates Levels[3] with CenterIdx and SurroSubIdx based on laser center properties.

    Parameters:
    - Levels: dict containing mesh level data
    - ne_nn: some parameter used by calculateSubsectionElements
    - Properties: dict containing simulation properties, including 'laser_center'
    """
    # Identify center indices where subsection elements are affected
    Levels[3]["CenterIdx"] = jnp.arange(Levels[3]["ne"])[
        calculateSubsectionElements(
            Levels, ne_nn, Properties["laser_center"], Properties
        )
        > 0
    ]

    # Deep copy for surrounding subsection index
    Levels[3]["SurroSubIdx"] = copy.deepcopy(Levels[3]["CenterIdx"])

    # Compute the coordinate index of the laser center
    _xc = jnp.stack(
        [
            jnp.where(
                Levels[3]["node_coords"][_] >= Properties["laser_center"][_], size=1
            )[0][0]
            for _ in range(2)
        ]
    )

    # Adjust CenterIdx based on coordinate offset
    Levels[3]["CenterIdx"] = Levels[3]["CenterIdx"] - (
        (_xc[0])
        + (_xc[1]) * Levels[3]["elements"][0]
        + (Levels[3]["elements"][2] - 1)
        * Levels[3]["elements"][0]
        * Levels[3]["elements"][1]
    )

    return Levels


def handle_surrogate_mode(Levels, Nonmesh):
    """
    Handles surrogate model setup based on Nonmesh flags.

    Parameters:
    - Levels: dict containing mesh level data
    - Nonmesh: dict with keys 'haste' and 'training'
    """
    if Nonmesh.get("haste"):
        T_DA_orig = get_transforms(Levels[4])

        if not Nonmesh.get("training"):
            data = jnp.load("DMD_ROM.npz")
            Levels[4]["A"] = data["A"]
            Levels[4]["C"] = data["B"]
            Levels[4]["mean_orig"] = data["xmean"]
            Levels[4]["meanA"] = data["xmean"][: Levels[4]["nn"]]
            Levels[4]["meanC"] = data["xmean"][Levels[4]["nn"] :]
            Levels[4]["u"] = data["u"]

        return T_DA_orig
    return None


def get_transforms(Level):
    """
    Generates a set of transformed index arrays for data augmentation or symmetry
    operations.

    Parameters:
    - Level: dict containing mesh level data, including 'nodes' and 'nn'

    Returns:
    - jnp.ndarray: stacked array of flattened transformed indices
    """

    def to_3d(T):
        """
        Reshape a flat array into a 3D array with shape (z, y, x), then transpose
        to (x, y, z).
        """
        return np.array(
            T.reshape(Level["nodes"][2], Level["nodes"][1], Level["nodes"][0])
        ).transpose((2, 1, 0))

    def to_flat(T):
        """
        Flatten a 3D array back to 1D after transposing back to (z, y, x).
        """
        return T.transpose((2, 1, 0)).reshape(-1)

    # Original index array reshaped to 3D
    T_orig = to_3d(jnp.arange(Level["nn"]))

    # Generate flipped and rotated versions
    transforms = [
        T_orig,
        jnp.fliplr(T_orig),
        jnp.flipud(T_orig),
        jnp.fliplr(jnp.flipud(T_orig)),
        jnp.rot90(T_orig, k=1),
        jnp.rot90(T_orig, k=2),
        jnp.rot90(T_orig, k=3),
        jnp.rot90(T_orig),  # same as k=1, but kept for clarity
    ]

    # Flatten all transformed arrays and stack them
    return jnp.stack([to_flat(t) for t in transforms])


@partial(
    jax.jit,
    static_argnames=["ne_nn", "tmp_ne_nn", "substrate", "subcycle", "record_accum"],
)
def HASTE(
    Levels,
    ne_nn,
    substrate,
    LInterp,
    tmp_ne_nn,
    laser_position_all,
    properties,
    subcycle,
    max_accum_time,
    accum_time,
    flip_flag,
    record_accum,
    laser_start,
    move_hist,
    L1L2Eratio,
    L2L3Eratio,
):
    for _tool_path_loop in range(subcycle[-1]):
        _tool_path_idx = _tool_path_loop * subcycle[2] + jnp.arange(subcycle[2])
        laser_position = laser_position_all[_tool_path_idx, :]
        (Levels, Shapes, LInterp, move_hist) = moveEverything(
            laser_position[0, :],
            laser_start,
            Levels,
            move_hist,
            LInterp,
            L1L2Eratio,
            L2L3Eratio,
            properties["layer_height"],
        )

        laserP = laser_position[:, 6]
        if record_accum == 1:
            IC = Levels[4]["T0"][None, :]
            IC3 = Levels[3]["T0"][None, :]

        Levels = computeLevel3SubIdx(Levels, laser_position)
        ShapeF1 = computeSurrogateCoarseFineShapeFunctions(Levels[1], Levels[3])
        ShapeF2 = computeSurrogateCoarseFineShapeFunctions(Levels[2], Levels[3])

        L4InterpMatrices = interpolateSurrogatePointsMatrix(
            Levels[4], Levels[2], Levels[3]["node_coords"], laser_position
        )
        L4InterpMatrices[2] = L4InterpMatrices[2][:, Levels[4]["ProbeIdx"], :]
        L4InterpMatrices[3] = L4InterpMatrices[3][:, Levels[4]["ProbeIdx"], :]

        # --- Level 1 Material Properties ---
        _, _, L3k_L1, L3rhocp_L1 = computeStateProperties(
            Levels[3]["T0"], Levels[3]["S1"], properties, substrate[3]
        )
        _, _, L2k_L1, L2rhocp_L1 = computeStateProperties(
            Levels[2]["T0"], Levels[2]["S1"], properties, substrate[2]
        )

        # Update Level 1 S1 from Level 2 overlap
        _val = interpolatePoints(Levels[2], Levels[2]["S1"], Levels[2]["overlapCoords"])
        _idx = getOverlapRegion(
            Levels[2]["overlapNodes"], Levels[1]["nodes"][0], Levels[1]["nodes"][1]
        )
        Levels[1]["S1"] = Levels[1]["S1"].at[_idx].set(_val)
        Levels[1]["S1"] = Levels[1]["S1"].at[: substrate[1]].set(1)

        _, _, L1k, L1rhocp = computeStateProperties(
            Levels[1]["T0"], Levels[1]["S1"], properties, substrate[1]
        )

        # Compute Level 1 Source #
        L1F = computeLevelSurrogateSource(
            Levels,
            ne_nn[2],
            laser_position,
            ShapeF1[0],
            ShapeF1[1],
            properties,
            laserP,
            Levels[3]["SurroSubIdx"],
        )
        L1F = computeConvRadBC(
            Levels[1], Levels[1]["T0"], tmp_ne_nn[0], ne_nn[2], properties, L1F
        )
        L1V = computeL1TprimeTerms_Part1(Levels, ne_nn, L3k_L1, Shapes, L2k_L1)

        # --- Level 1 Temperature Predictor ---
        L1T = computeL1Temperature(
            Levels,
            ne_nn,
            tmp_ne_nn,
            L1F,
            L1V,
            L1k,
            L1rhocp,
            laser_position[:, 5].sum(),
            properties,
        )
        L1T = jnp.maximum(properties["T_amb"], L1T)  # TFSP

        ## Subcycle Level 2 ##
        def subcycleL2_Part1(_L2carry, _L2sub):
            # Compute interpolation weights for Level 2 boundary conditions
            alpha_L2 = (_L2sub + 1) / subcycle[3]
            beta_L2 = 1 - alpha_L2

            # Determine the laser substeps for this Level 2 subcycle
            Lidx = _L2sub * subcycle[1] + jnp.arange(subcycle[1])

            # --- Material Properties ---
            # Compute Level 3 properties using current Level 3 temperature and phase
            L3S1, _, L3k_L2, _ = computeStateProperties(
                _L2carry[2], _L2carry[4], properties, substrate[3]
            )

            # Compute Level 2 properties using current Level 2 temperature and phase
            L2S1, _, L2k, L2rhocp = computeStateProperties(
                _L2carry[0], _L2carry[1], properties, substrate[2]
            )

            # --- Source Term ---
            ## Compute Level 2 Source ##
            _L2F = computeLevelSurrogateSource(
                Levels,
                ne_nn[3],
                laser_position[Lidx, :],
                ShapeF2[0][Lidx],
                ShapeF2[1][Lidx],
                properties,
                laserP[Lidx],
                Levels[3]["SurroSubIdx"][Lidx],
            )

            # Add convection, radiation, and evaporation boundary conditions
            L2F = computeConvRadBC(
                Levels[2],
                _L2carry[0],
                ne_nn[0],
                ne_nn[3],
                properties,
                _L2F,
            )

            # --- Subgrid Correction ---
            # Compute divergence of subgrid heat flux from Level 3 to Level 2
            L2V = computeL2TprimeTerms_Part1(Levels, ne_nn, _L2carry[3], L3k_L2, Shapes)

            # --- Temperature Solve ---
            # Interpolate Level 1 temperature to Level 2 boundary using alpha-beta blend
            _BC = alpha_L2 * L1T + beta_L2 * Levels[1]["T0"]

            # Solve Level 2 temperature using matrix-free FEM
            L2T = computeL2Temperature(
                _BC,
                LInterp[0],
                Levels,
                ne_nn,
                _L2carry[0],
                L2F,
                L2V,
                L2k,
                L2rhocp,
                laser_position[Lidx, 5].sum(),
            )
            L2T = jnp.maximum(properties["T_amb"], L2T)  # TFSP

            ### Subcycle Level 3 ###
            ### Solve Level 3 Temperature ###
            L4_L2sub = (_L2sub + 1) * subcycle[1] - 1
            Probe_T = multiply(
                L4InterpMatrices[2][L4_L2sub], L2T[L4InterpMatrices[3][L4_L2sub]]
            ).sum(axis=1)
            L4M = Levels[4]["A"] @ _L2carry[5] + Levels[4]["C"] @ (
                Probe_T - Levels[4]["meanC"]
            )
            L4T = (Levels[4]["u"] @ L4M + Levels[4]["meanA"])[flip_flag]
            L4T = jnp.maximum(properties["T_amb"], L4T)

            L3T = interpolate_w_matrix(LInterp[1], L2T)
            # This is the same as interpolate_w_matrix
            _L3T = multiply(
                L4InterpMatrices[0][L4_L2sub], L4T[L4InterpMatrices[1][L4_L2sub]]
            ).sum(axis=1)
            L3T = L3T - jnp.logical_or(
                (_L3T > properties["T_solidus"]), L3T > properties["T_liquidus"]
            ) * (L3T - _L3T)

            # L3T = interpolatePointsLevel4(Levels[4], L4nc, L4T, Levels[3]["node_coords"])
            L3T = jnp.maximum(properties["T_amb"], L3T)  # TFPD

            ## Compute Updated Level 3 Tprime ##
            L3Tp, L2T = getNewTprime(Levels[3], L3T, L2T, Levels[2], LInterp[1])

            return (
                [L2T, L2S1, L3T, L3Tp, L3S1, L4M, _L2F],
                [L2T, L2S1, L3T, L3Tp, L3S1, L4M, _L2F],
            )

        # Run Level 2 subcycling loop
        [L2T, _, _, L3Tp, _, _, _], [_, _, _, L3Tp_L2, _, _, L2F_all] = jax.lax.scan(
            subcycleL2_Part1,
            [
                Levels[2]["T0"],
                Levels[2]["S1"],
                Levels[3]["T0"],
                Levels[3]["Tprime0"],
                Levels[3]["S1"],
                Levels[4]["surrogateA"],
                jnp.zeros_like(Levels[2]["T0"]),
            ],
            jnp.arange(subcycle[0]),
        )

        # --- Level 2 Tprime and Level 1 Corrector Update ---
        L2Tp, L1T = getNewTprime(Levels[2], L2T, L1T, Levels[1], LInterp[0])
        L1V = computeL1TprimeTerms_Part2(
            Levels,
            ne_nn,
            L3Tp,
            L2Tp,
            L3rhocp_L1,
            L2rhocp_L1,
            laser_position[:, 5].sum(),
            Shapes,
            L1V,
        )
        L1T = computeL1Temperature(
            Levels,
            ne_nn,
            tmp_ne_nn,
            L1F,
            L1V,
            L1k,
            L1rhocp,
            laser_position[:, 5].sum(),
            properties,
        )
        L1T = jnp.maximum(properties["T_amb"], L1T)  # TFSP

        ## Subcycle Level 2 ##
        def subcycleL2_Part2(_L2carry, _L2sub):
            # Compute interpolation weights for Level 2 boundary conditions
            alpha_L2 = (_L2sub + 1) / subcycle[3]
            beta_L2 = 1 - alpha_L2

            # Determine the laser substeps for this Level 2 subcycle
            Lidx = _L2sub * subcycle[1] + jnp.arange(subcycle[1])

            # --- Material Properties ---
            # Compute Level 3 properties using current Level 3 temperature and phase
            L3S1, L3S2, L3k_L2, L3rhocp_L2 = computeStateProperties(
                _L2carry[2], _L2carry[4], properties, substrate[3]
            )

            # Compute Level 2 properties using current Level 2 temperature and phase
            L2S1, _, L2k, L2rhocp = computeStateProperties(
                _L2carry[0], _L2carry[1], properties, substrate[2]
            )

            # --- Source Term ---
            L2F = L2F_all[_L2sub]
            L2F = computeConvRadBC(
                Levels[2], _L2carry[0], ne_nn[0], ne_nn[3], properties, L2F
            )

            # --- Subgrid Correction ---
            # Compute divergence of subgrid heat flux from Level 3 to Level 2
            L2V = computeL2TprimeTerms_Part1(Levels, ne_nn, _L2carry[3], L3k_L2, Shapes)

            # Add time derivative correction from Level 3 to Level 2
            L2V = computeL2TprimeTerms_Part2(
                Levels,
                ne_nn,
                L3Tp_L2[_L2sub],
                _L2carry[3],
                L3rhocp_L2,
                laser_position[Lidx, 5].sum(),
                Shapes,
                L2V,
            )

            # --- Temperature Solve ---
            # Interpolate Level 1 temperature to Level 2 boundary using alpha-beta blend
            _BC = alpha_L2 * L1T + beta_L2 * Levels[1]["T0"]

            # Solve Level 2 temperature using matrix-free FEM
            L2T = computeL2Temperature(
                _BC,
                LInterp[0],
                Levels,
                ne_nn,
                _L2carry[0],
                L2F,
                L2V,
                L2k,
                L2rhocp,
                laser_position[Lidx, 5].sum(),
            )
            L2T = jnp.maximum(properties["T_amb"], L2T)  # TFSP

            ### Subcycle Level 3 ###
            ### Solve Level 3 Temperature ###
            L4_L2sub = (_L2sub + 1) * subcycle[1] - 1

            Probe_T = multiply(
                L4InterpMatrices[2][L4_L2sub], L2T[L4InterpMatrices[3][L4_L2sub]]
            ).sum(axis=1)
            L4M = Levels[4]["A"] @ _L2carry[5] + Levels[4]["C"] @ (
                Probe_T - Levels[4]["meanC"]
            )
            L4T = (Levels[4]["u"] @ L4M + Levels[4]["meanA"])[flip_flag]
            L4T = jnp.maximum(properties["T_amb"], L4T)

            L3T = interpolate_w_matrix(LInterp[1], L2T)
            # This is the same as interpolate_w_matrix
            _L3T = multiply(
                L4InterpMatrices[0][L4_L2sub], L4T[L4InterpMatrices[1][L4_L2sub]]
            ).sum(axis=1)
            L3T = L3T - jnp.logical_or(
                (_L3T > properties["T_solidus"]), L3T > properties["T_liquidus"]
            ) * (L3T - _L3T)

            # L3T = interpolatePointsLevel4(Levels[4], L4nc, L4T, Levels[3]["node_coords"])
            L3T = jnp.maximum(properties["T_amb"], L3T)  # TFPD

            ## Compute Updated Level 3 Tprime ##
            L3Tp, L2T = getNewTprime(Levels[3], L3T, L2T, Levels[2], LInterp[1])

            return (
                [L2T, L2S1, L3T, L3Tp, L3S1, L4M, L3S2, L4T],
                [L2T, L2S1, L3T, L3Tp, L3S1, L4M, L3S2, L4T],
            )

        [
            Levels[2]["T0"],
            Levels[2]["S1"],
            Levels[3]["T0"],
            Levels[3]["Tprime0"],
            Levels[3]["S1"],
            Levels[4]["surrogateA"],
            Levels[3]["S2"],
            Levels[4]["T0"],
        ], [_, _, L3T_all, _, _, _, _, L4T_all] = jax.lax.scan(
            subcycleL2_Part2,
            [
                Levels[2]["T0"],
                Levels[2]["S1"],
                Levels[3]["T0"],
                Levels[3]["Tprime0"],
                Levels[3]["S1"],
                Levels[4]["surrogateA"],
                Levels[3]["S2"],
                Levels[4]["T0"],
            ],
            jnp.arange(subcycle[0]),
        )

        # --- Final Tprime and Phase Updates ---
        Levels[2]["Tprime0"], Levels[1]["T0"] = getNewTprime(
            Levels[2], Levels[2]["T0"], L1T, Levels[1], LInterp[0]
        )
        Levels[0]["S1"] = Levels[0]["S1"].at[Levels[0]["idx"]].set(Levels[3]["S1"])
        Levels[0]["S2"] = Levels[0]["S2"].at[:].set(False)
        Levels[0]["S2"] = Levels[0]["S2"].at[Levels[0]["idx"]].set(Levels[3]["S2"])

        if record_accum == 1:
            L3T_all = jnp.vstack((IC3, L3T_all[:-1]))
            L4T_all = jnp.vstack((IC, L4T_all))

            # Precompute alpha_L3
            alpha_L3 = jnp.linspace(1 / subcycle[4], 1.0, subcycle[1])[
                None, :, None, None
            ]  # shape (M,1,1)

            # Compute idx for all steps
            idx = jnp.arange(subcycle[0] * subcycle[1]).reshape(
                subcycle[0], subcycle[1]
            )  # shape (N, M)

            # Gather indices
            index_1 = L4InterpMatrices[1][idx]  # shape (N, M, ...)

            # Interpolate L4T_all
            L4T_0 = L4T_all[:-1]  # shape (N, ...)
            L4T_1 = L4T_all[1:]  # shape (N, ...)
            L4T_0_gathered = jax.vmap(lambda x, i: getitem(x, i))(L4T_0, index_1)
            L4T_1_gathered = jax.vmap(lambda x, i: getitem(x, i))(L4T_1, index_1)

            # Interpolate
            _L4T = (
                alpha_L3 * L4T_1_gathered + (1 - alpha_L3) * L4T_0_gathered
            )  # shape (M, N, ...)

            # Multiply and sum
            drag_result = jnp.sum(
                multiply(L4InterpMatrices[0][idx], _L4T), axis=-1
            )  # shape (N, M)

            # Concatenate with L3T_all
            drag_result = jnp.concatenate(
                [L3T_all[:, None], drag_result], axis=1
            )  # shape (N, M+1)

            accum_L3 = accum_time[Levels[0]["idx"]]
            max_accum_L3 = max_accum_time[Levels[0]["idx"]]

            accum_L3 += (
                jnp.minimum(
                    jnp.maximum(
                        jnp.maximum(drag_result[:, :-1], drag_result[:, 1:])
                        - properties["T_liquidus"],
                        0,
                    )
                    / (jnp.abs(jnp.diff(drag_result, axis=1)) + 1e-6),
                    1,
                ).sum(axis=(1, 0))
                * laser_position[0, 5]
            )

            max_accum_L3 = jnp.maximum(max_accum_L3, accum_L3)

            # Reset where L3T_all < T_liquidus
            accum_L3 = jnp.where(
                Levels[3]["T0"] < properties["T_liquidus"], 0, accum_L3
            )

            max_accum_time = max_accum_time.at[Levels[0]["idx"]].set(max_accum_L3)
            accum_time = accum_time.at[Levels[0]["idx"]].set(accum_L3)

    return Levels, move_hist, LInterp, max_accum_time, accum_time


def evaluate_surrogate_run(
    Levels,
    Nonmesh,
    laser_all,
    laser_last,
    Test2pos,
    SVD_ready,
    surrogate_count,
    surro_track,
    track,
    anti_flip_flag,
):
    """
    Evaluates whether to run the surrogate model based on laser input and system state.

    Returns:
        Updated values for:
        - Levels
        - Test2pos
        - laser_last
        - SVD_ready
        - surrogate_count
        - surro_track
        - track
        - run_surrogate (bool)
    """
    run_surrogate = False

    if Nonmesh.get("haste"):
        laser_diff = jnp.diff(laser_all, axis=0)

        if (jnp.abs(laser_diff - laser_last) < 1e-4).all() and (
            laser_all[:, 3:5] == 1
        ).all():
            if SVD_ready:
                Test2pos = Test2pos.at[:].set(laser_all[-1, :])
                Levels, _ = getLevel4Pos(Levels, laser_all, Test2pos)

                SurroLInterpTestL0 = interpolatePointsMatrix(
                    Levels[0], Levels[4]["node_coords"]
                )
                Levels[4]["S1"] = (
                    interpolate_w_matrix(SurroLInterpTestL0, Levels[0]["S1"]) > 1e-3
                ) * 1
                _flipped = Levels[4]["S1"][anti_flip_flag]

                _line = [_flipped[_] for _ in Levels[4]["BC_top"]]
                _line = [line.sum() / line.shape[0] for line in _line]

                _idx0 = jnp.greater(_line[1], _line[0])
                _idx1 = _line[2] == 1

                if _idx0 and _idx1:
                    run_surrogate = True
                else:
                    run_surrogate = False
                    surro_track = False
                    SVD_ready = False
            else:
                run_surrogate = False

            surrogate_count += 1

        else:
            if surrogate_count > 0:
                surro_track = True
                track += 1

            surrogate_count = 0
            run_surrogate = False
            SVD_ready = False

        laser_last = laser_diff[-1, :]

    else:
        run_surrogate = False

    return (
        Levels,
        Test2pos,
        laser_last,
        SVD_ready,
        surrogate_count,
        surro_track,
        track,
        run_surrogate,
    )


def getLevel4Pos(Levels, laser_all, Test2pos):
    """
    Updates Levels[4] mesh and node coordinates based on laser movement direction.

    Parameters:
    - Levels: dict containing mesh level data
    - laser_all: jnp.ndarray of laser positions over time
    - Test2pos: jnp.ndarray of the current laser position
    Returns:
    - Updated Levels dictionary
    """
    _diff = jnp.diff(laser_all, axis=0)

    def update_mesh(node_order, bounds_order, flip_axes, coord_signs):
        Levels[4]["nodes"] = [Levels[4]["orig_nodes"][i] for i in node_order]
        bounds = [Levels[4]["bounds"][axis] for axis in bounds_order]
        nodes = Levels[4]["nodes"]

        Levels[4]["node_coords"], Levels[4]["connect"] = createMesh3D(
            (bounds[0][0], bounds[0][1], nodes[0]),
            (bounds[1][0], bounds[1][1], nodes[1]),
            (bounds[2][0], bounds[2][1], nodes[2]),
        )

        # Apply flipping if needed
        Levels[4]["new_node_coords"] = [
            -Levels[4]["node_coords"][i] if flip else Levels[4]["node_coords"][i]
            for i, flip in enumerate(flip_axes)
        ]

        # Translate coordinates based on Test2pos and sign
        Levels[4]["node_coords"] = [
            Test2pos[i] + coord_signs[i] * Levels[4]["node_coords"][i] for i in range(3)
        ]

    if (_diff[:, 0] > 1e-4).all():
        update_mesh(
            node_order=[0, 1, 2],
            bounds_order=["x", "y", "z"],
            flip_axes=[True, False, False],
            coord_signs=[-1, 1, 1],
        )
        direction = "east"
    elif (_diff[:, 1] > 1e-4).all():
        update_mesh(
            node_order=[1, 0, 2],
            bounds_order=["y", "x", "z"],
            flip_axes=[False, True, False],
            coord_signs=[1, -1, 1],
        )
        direction = "north"
    elif (_diff[:, 0] < -1e-4).all():
        update_mesh(
            node_order=[0, 1, 2],
            bounds_order=["x", "y", "z"],
            flip_axes=[False, False, False],
            coord_signs=[1, 1, 1],
        )
        direction = "west"
    elif (_diff[:, 1] < -1e-4).all():
        update_mesh(
            node_order=[1, 0, 2],
            bounds_order=["y", "x", "z"],
            flip_axes=[False, False, False],
            coord_signs=[1, 1, 1],
        )
        direction = "south"
    else:
        print("Velocities do not line up for training")
        sys.exit(1)

    return Levels, direction


def getflipflag(T_DA_orig, Levels):
    # Determine grid dimensions
    nx, ny = max(Levels[4]["nodes"][0], Levels[4]["nodes"][1]), min(
        Levels[4]["nodes"][0], Levels[4]["nodes"][1]
    )
    nz = Levels[4]["nodes"][2]
    nn = Levels[4]["nn"]

    # Reshape helpers
    def Horizontal(T):
        return np.array(T.reshape(nz, ny, nx)).transpose((2, 1, 0))

    def HorizontalTranspose(T):
        return np.array(T.reshape(nx, ny, nz)).transpose((2, 1, 0))

    def Vertical(T):
        return np.array(T.reshape(nz, nx, ny)).transpose((2, 1, 0))

    def VerticalTranspose(T):
        return np.array(T.reshape(ny, nx, nz)).transpose((2, 1, 0))

    # Boundary indices
    bidx = jnp.arange(0, nx * ny)
    tidx = jnp.arange(nx * ny * (nz - 1), nn)
    widx = jnp.arange(0, nn, nx)
    eidx = jnp.arange(nx - 1, nn, nx)
    sidx = (
        jnp.arange(0, nx)[:, None] + (nx * ny * jnp.arange(0, nz))[None, :]
    ).reshape(-1)
    nidx = (
        jnp.arange(nx * (ny - 1), nx * ny)[:, None]
        + (nx * ny * jnp.arange(0, nz))[None, :]
    ).reshape(-1)

    Levels[4]["BC"] = [widx, eidx, sidx, nidx, bidx, tidx]
    Levels[4]["BC_top"] = [jnp.intersect1d(_, tidx) for _ in Levels[4]["BC"][:3]]

    # Generate rotated and flipped configurations
    Orig = jnp.arange(nn)
    T_orig = Horizontal(Orig)
    rotations = [
        T_orig,
        jnp.fliplr(T_orig),
        jnp.flipud(T_orig),
        jnp.fliplr(jnp.flipud(T_orig)),
        jnp.rot90(T_orig, k=1),
        jnp.rot90(T_orig, k=2),
        jnp.rot90(T_orig, k=3),
        jnp.rot90(T_orig),
    ]
    T_DA_orig = jnp.stack([HorizontalTranspose(r).reshape(-1) for r in rotations])

    # Determine rotation mappings
    if Levels[4]["nodes"][1] > Levels[4]["nodes"][0]:
        rotateit = HorizontalTranspose(jnp.rot90(Vertical(Orig), k=1)).reshape(-1)
        rotateitback = VerticalTranspose(jnp.rot90(Horizontal(Orig), k=3)).reshape(-1)
    else:
        rotateit = rotateitback = Orig

    # Flip flag configurations
    flip_flag_list = [
        T_DA_orig[2][rotateitback],  # Flip across y
        T_DA_orig[0][rotateitback],  # No change
        T_DA_orig[5][rotateitback],  # 180° rotation
        T_DA_orig[2][T_DA_orig[5][rotateitback]],  # Flip + 180° rotation
    ]

    # Anti-flip flag configurations
    anti_flip_flag_list = [
        rotateit[T_DA_orig[2]],
        rotateit[T_DA_orig[0]],
        rotateit[T_DA_orig[5]],
        rotateit[T_DA_orig[2][T_DA_orig[5]]],
    ]

    flip_flag_list = jnp.stack(flip_flag_list)
    anti_flip_flag_list = jnp.stack(anti_flip_flag_list)

    # Evaluate boundary conditions
    flipped = Levels[4]["S1"][anti_flip_flag_list]
    line_scores = [flipped[:, _] for _ in Levels[4]["BC_top"]]
    line_scores = [
        line_scores[_].sum(axis=1) / Levels[4]["BC_top"][_].shape[0]
        for _ in range(len(line_scores))
    ]

    # Determine surrogate run condition
    idx0 = jnp.where(line_scores[1] > line_scores[0])[0]
    idx1 = jnp.where(line_scores[2] == 1)[0]

    if len(idx0) == 0 or len(idx1) == 0:
        idx = 1
        run_surrogate = False
    else:
        intersection = np.intersect1d(idx0, idx1)
        if len(intersection) == 0:
            idx = 1
            run_surrogate = False
        else:
            idx = intersection[0]
            run_surrogate = True

    return flip_flag_list[idx], anti_flip_flag_list[idx], run_surrogate, Levels, idx


def check_surrogate_conditions(
    laser_all,
    laser_last,
    SVD_ready,
    Test2pos,
    Levels,
    surrogate_count,
    track,
    num,
    surro_track,
    remaining_HASTE_iterations,
):
    run_surrogate = False
    laser_diff = jnp.diff(laser_all, axis=0)

    # Check for consistent laser movement and flags
    if (jnp.abs(laser_diff - laser_last) < 1e-4).all() and (
        laser_all[:, 3:5] == 1
    ).all():
        if SVD_ready:
            # Update test position and interpolate surrogate matrix
            Test2pos = Test2pos.at[:].set(laser_all[-1, :])
            Levels, _ = getLevel4Pos(Levels, laser_all, Test2pos)

            if remaining_HASTE_iterations > 0:
                remaining_HASTE_iterations -= 1
                run_surrogate = True
            else:
                run_surrogate = False

            if not run_surrogate:
                surro_track = False
                SVD_ready = False
        else:
            run_surrogate = False

        surrogate_count += 1

    else:
        # Reset tracking if movement is inconsistent
        if surrogate_count > 0:
            surro_track = True
            track += 1

        surrogate_count = 0
        run_surrogate = False
        SVD_ready = False
        num = 0

    laser_last = laser_diff[-1, :]

    return (
        run_surrogate,
        surro_track,
        SVD_ready,
        surrogate_count,
        track,
        num,
        laser_last,
        Test2pos,
        Levels,
        remaining_HASTE_iterations,
    )


@jax.jit
def interpolatePointsLevel4(Level, node_coords, u, node_coords_new):
    """interpolatePoints interpolate solutions located on Level["node_coords"]
    and connected with Level["connect"] to new coordinates node_coords_new. Values
    that are later bin counted are the output
    :param Level["node_coords"]: source nodal coordinates
    :param Level["connect"]: connectivity matrix
    :param node_coords_new: output nodal coordinates
    :return _val: nodal values that need to be bincounted
    """
    ne_x = Level["connect"][0].shape[0]
    ne_y = Level["connect"][1].shape[0]
    ne_z = Level["connect"][2].shape[0]
    nn_x, nn_y = ne_x + 1, ne_y + 1

    nn_xn = len(node_coords_new[0])
    nn_yn = len(node_coords_new[1])
    nn_zn = len(node_coords_new[2])
    tmp_ne_nn = nn_xn * nn_yn * nn_zn
    h_x = node_coords[0][1] - node_coords[0][0]
    h_y = node_coords[1][1] - node_coords[1][0]
    h_z = node_coords[2][1] - node_coords[2][0]

    def stepInterpolatePoints(ielt):
        # Get nodal indices
        izn, _ = jnp.divmod(ielt, (nn_xn) * (nn_yn))
        iyn, _ = jnp.divmod(ielt, nn_xn)
        iyn -= izn * nn_yn
        ixn = jnp.mod(ielt, nn_xn)

        _x = node_coords_new[0][ixn, None]
        _y = node_coords_new[1][iyn, None]
        _z = node_coords_new[2][izn, None]

        x_comp = (ne_x - 1) * jnp.ones_like(_x)
        y_comp = (ne_y - 1) * jnp.ones_like(_y)
        z_comp = (ne_z - 1) * jnp.ones_like(_z)

        x_comp2 = jnp.zeros_like(_x)
        y_comp2 = jnp.zeros_like(_y)
        z_comp2 = jnp.zeros_like(_z)

        # Figure out which coarse element we are in
        _floorx = jnp.floor((_x - node_coords[0][0]) / h_x)
        _conx = jnp.concatenate((_floorx, x_comp))
        _ielt_x = jnp.min(_conx)
        _conx = jnp.concatenate((_ielt_x[None], x_comp2))
        ielt_x = jnp.max(_conx).T.astype(int)

        _floory = jnp.floor((_y - node_coords[1][0]) / h_y)
        _cony = jnp.concatenate((_floory, y_comp))
        _ielt_y = jnp.min(_cony)
        _cony = jnp.concatenate((_ielt_y[None], y_comp2))
        ielt_y = jnp.max(_cony).T.astype(int)

        _floorz = jnp.floor((_z - node_coords[2][0]) / h_z)
        _conz = jnp.concatenate((_floorz, z_comp))
        _ielt_z = jnp.min(_conz).T.astype(int)
        _conz = jnp.concatenate((_ielt_z[None], z_comp2))
        ielt_z = jnp.max(_conz).T.astype(int)

        nodex = Level["connect"][0][ielt_x, :]
        nodey = Level["connect"][1][ielt_y, :]
        nodez = Level["connect"][2][ielt_z, :]
        node = nodex + nodey * nn_x + nodez * (nn_x * nn_y)

        xx = node_coords[0][nodex]
        yy = node_coords[1][nodey]
        zz = node_coords[2][nodez]

        xc0, xc1 = xx[0], xx[1]
        yc0, yc3 = yy[0], yy[3]
        zc0, zc5 = zz[0], zz[5]

        # Evaluate shape functions associated with coarse nodes
        Nc = jnp.concatenate(
            compute3DN(
                [_x, _y, _z], [xc0, xc1], [yc0, yc3], [zc0, zc5], [h_x, h_y, h_z]
            )
        )
        # We need the 2e-2 tolerance for surrogate, more likely to fail
        # The where makes it zero outside the interpolation range
        Nc = ((Nc >= -5e-2).all() & (Nc <= 1 + 5e-2).all()) * Nc
        Nc = jnp.clip(Nc, 0.0, 1.0)
        # Nc = (
        #     Nc
        #     * (Nc >= -2e-2).all().astype(float)
        #     * (Nc <= 1 + 2e-2).all().astype(float)
        # )
        return Nc @ u[node]

    vstepInterpolatePoints = jax.vmap(stepInterpolatePoints)
    return vstepInterpolatePoints(jnp.arange(tmp_ne_nn))


@partial(jax.jit, static_argnames=["ne_nn_tmp"])
def computeLevelSurrogateSource(
    Levels,
    ne_nn_tmp,
    laser_position,
    LevelShape0,
    LevelShape2,
    properties,
    laserP,
    SurroSubIdx,
):
    # Get shape functions and weights
    coords = getSampleCoords(Levels[3])
    Nf, _, wqf = computeQuad3dFemShapeFunctions_jax(coords)

    def stepLaserPosition(ilaser):
        def stepcomputeCoarseSource(ieltf):
            # Get the nodal indices for that element
            ix, iy, iz, idx = convert2XYZ(
                ieltf,
                Levels[3]["elements"][0],
                Levels[3]["elements"][1],
                Levels[3]["nodes"][0],
                Levels[3]["nodes"][1],
            )
            # Get nodal coordinates for the fine element
            x, y, z = getQuadratureCoords(Levels[3], ix, iy, iz, Nf)
            w = wqf
            # Compute the source at the quadrature point location
            Q = computeSourceFunction_jax(
                x, y, z, laser_position[ilaser], properties, laserP[ilaser]
            )
            return Q * w

        vstepcomputeCoarseSource = jax.vmap(stepcomputeCoarseSource)

        _data = vstepcomputeCoarseSource(SurroSubIdx[ilaser, :])
        _data1tmp = multiply(LevelShape0[ilaser, SurroSubIdx[ilaser, :]], _data).sum(
            axis=1
        )
        return _data1tmp.reshape(-1)

    vstepLaserPosition = jax.vmap(stepLaserPosition)
    # Find the average heat source value
    lshape = laser_position.shape[0]
    _data1 = vstepLaserPosition(jnp.arange(lshape)) / lshape

    return bincount(LevelShape2.reshape(-1), _data1.reshape(-1), ne_nn_tmp)


@jax.jit
def interpolateSurrogatePointsMatrix(
    Level4, Level2, Level3_node_coords, laser_pos_input
):
    # Get Level 4 connectivities
    ne_x = Level4["connect"][0].shape[0]
    ne_y = Level4["connect"][1].shape[0]
    ne_z = Level4["connect"][2].shape[0]
    nn_x, nn_y = ne_x + 1, ne_y + 1
    h_x = Level4["new_node_coords"][0][1] - Level4["new_node_coords"][0][0]
    h_y = Level4["new_node_coords"][1][1] - Level4["new_node_coords"][1][0]
    h_z = Level4["new_node_coords"][2][1] - Level4["new_node_coords"][2][0]

    # Get Level 2 connectivities
    ne_x2 = Level2["connect"][0].shape[0]
    ne_y2 = Level2["connect"][1].shape[0]
    ne_z2 = Level2["connect"][2].shape[0]
    nn_x2, nn_y2 = ne_x2 + 1, ne_y2 + 1
    h_x2 = Level2["node_coords"][0][1] - Level2["node_coords"][0][0]
    h_y2 = Level2["node_coords"][1][1] - Level2["node_coords"][1][0]
    h_z2 = Level2["node_coords"][2][1] - Level2["node_coords"][2][0]

    # Get Level 3 node numbers
    nn_xn = len(Level3_node_coords[0])
    nn_yn = len(Level3_node_coords[1])
    nn_zn = len(Level3_node_coords[2])
    tmp_ne_nn = nn_xn * nn_yn * nn_zn

    # Get Level 4 node numbers
    nn_xn4 = len(Level4["new_node_coords"][0])
    nn_yn4 = len(Level4["new_node_coords"][1])
    nn_zn4 = len(Level4["new_node_coords"][2])
    tmp_ne_nn4 = nn_xn4 * nn_yn4 * nn_zn4

    def stepLaserPos(laser_pos):
        L4nc = [Level4["new_node_coords"][_] + laser_pos[_] for _ in range(3)]

        def stepInterpolatePoints43(ielt):
            # Get nodal indices
            izn, _ = jnp.divmod(ielt, (nn_xn) * (nn_yn))
            iyn, _ = jnp.divmod(ielt, nn_xn)
            iyn -= izn * nn_yn
            ixn = jnp.mod(ielt, nn_xn)

            _x = Level3_node_coords[0][ixn, None]
            _y = Level3_node_coords[1][iyn, None]
            _z = Level3_node_coords[2][izn, None]

            x_comp = (ne_x - 1) * jnp.ones_like(_x)
            y_comp = (ne_y - 1) * jnp.ones_like(_y)
            z_comp = (ne_z - 1) * jnp.ones_like(_z)

            x_comp2 = jnp.zeros_like(_x)
            y_comp2 = jnp.zeros_like(_y)
            z_comp2 = jnp.zeros_like(_z)

            # Figure out which coarse element we are in
            _floorx = jnp.floor((_x - L4nc[0][0]) / h_x)
            _conx = jnp.concatenate((_floorx, x_comp))
            _ielt_x = jnp.min(_conx)
            _conx = jnp.concatenate((_ielt_x[None], x_comp2))
            ielt_x = jnp.max(_conx).T.astype(int)

            _floory = jnp.floor((_y - L4nc[1][0]) / h_y)
            _cony = jnp.concatenate((_floory, y_comp))
            _ielt_y = jnp.min(_cony)
            _cony = jnp.concatenate((_ielt_y[None], y_comp2))
            ielt_y = jnp.max(_cony).T.astype(int)

            _floorz = jnp.floor((_z - L4nc[2][0]) / h_z)
            _conz = jnp.concatenate((_floorz, z_comp))
            _ielt_z = jnp.min(_conz).T.astype(int)
            _conz = jnp.concatenate((_ielt_z[None], z_comp2))
            ielt_z = jnp.max(_conz).T.astype(int)

            nodex = Level4["connect"][0][ielt_x, :]
            nodey = Level4["connect"][1][ielt_y, :]
            nodez = Level4["connect"][2][ielt_z, :]
            node = nodex + nodey * nn_x + nodez * (nn_x * nn_y)

            xx = L4nc[0][nodex]
            yy = L4nc[1][nodey]
            zz = L4nc[2][nodez]

            xc0, xc1 = xx[0], xx[1]
            yc0, yc3 = yy[0], yy[3]
            zc0, zc5 = zz[0], zz[5]

            # Evaluate shape functions associated with coarse nodes
            Nc = jnp.concatenate(
                compute3DN(
                    [_x, _y, _z], [xc0, xc1], [yc0, yc3], [zc0, zc5], [h_x, h_y, h_z]
                )
            )
            # The where makes it zero outside the interpolation range
            Nc = ((Nc >= -5e-2).all() & (Nc <= 1 + 5e-2).all()) * Nc
            Nc = jnp.clip(Nc, 0.0, 1.0)

            return Nc, node

        def stepInterpolatePoints24(ielt):
            # Get nodal indices
            izn, _ = jnp.divmod(ielt, (nn_xn4) * (nn_yn4))
            iyn, _ = jnp.divmod(ielt, nn_xn4)
            iyn -= izn * nn_yn4
            ixn = jnp.mod(ielt, nn_xn4)

            _x = L4nc[0][ixn, None]
            _y = L4nc[1][iyn, None]
            _z = L4nc[2][izn, None]

            x_comp = (ne_x2 - 1) * jnp.ones_like(_x)
            y_comp = (ne_y2 - 1) * jnp.ones_like(_y)
            z_comp = (ne_z2 - 1) * jnp.ones_like(_z)

            x_comp2 = jnp.zeros_like(_x)
            y_comp2 = jnp.zeros_like(_y)
            z_comp2 = jnp.zeros_like(_z)

            # Figure out which coarse element we are in
            _floorx = jnp.floor((_x - Level2["node_coords"][0][0]) / h_x2)
            _conx = jnp.concatenate((_floorx, x_comp))
            _ielt_x = jnp.min(_conx)
            _conx = jnp.concatenate((_ielt_x[None], x_comp2))
            ielt_x = jnp.max(_conx).T.astype(int)

            _floory = jnp.floor((_y - Level2["node_coords"][1][0]) / h_y2)
            _cony = jnp.concatenate((_floory, y_comp))
            _ielt_y = jnp.min(_cony)
            _cony = jnp.concatenate((_ielt_y[None], y_comp2))
            ielt_y = jnp.max(_cony).T.astype(int)

            _floorz = jnp.floor((_z - Level2["node_coords"][2][0]) / h_z2)
            _conz = jnp.concatenate((_floorz, z_comp))
            _ielt_z = jnp.min(_conz).T.astype(int)
            _conz = jnp.concatenate((_ielt_z[None], z_comp2))
            ielt_z = jnp.max(_conz).T.astype(int)

            nodex = Level2["connect"][0][ielt_x, :]
            nodey = Level2["connect"][1][ielt_y, :]
            nodez = Level2["connect"][2][ielt_z, :]
            node = nodex + nodey * nn_x2 + nodez * (nn_x2 * nn_y2)

            xx = Level2["node_coords"][0][nodex]
            yy = Level2["node_coords"][1][nodey]
            zz = Level2["node_coords"][2][nodez]

            xc0, xc1 = xx[0], xx[1]
            yc0, yc3 = yy[0], yy[3]
            zc0, zc5 = zz[0], zz[5]

            # Evaluate shape functions associated with coarse nodes
            Nc = jnp.concatenate(
                compute3DN(
                    [_x, _y, _z], [xc0, xc1], [yc0, yc3], [zc0, zc5], [h_x2, h_y2, h_z2]
                )
            )
            # The where makes it zero outside the interpolation range
            Nc = ((Nc >= -5e-2).all() & (Nc <= 1 + 5e-2).all()) * Nc
            Nc = jnp.clip(Nc, 0.0, 1.0)

            return Nc, node

        vstepInterpolatePoints43 = jax.vmap(stepInterpolatePoints43)
        _Nc43, _node43 = vstepInterpolatePoints43(jnp.arange(tmp_ne_nn))
        vstepInterpolatePoints24 = jax.vmap(stepInterpolatePoints24)
        _Nc24, _node24 = vstepInterpolatePoints24(jnp.arange(tmp_ne_nn4))
        return _Nc43, _node43, _Nc24, _node24

    vstepLaserPos = jax.vmap(stepLaserPos)
    __Nc43, __node43, __Nc24, __node24 = vstepLaserPos(laser_pos_input)
    return [__Nc43, __node43, __Nc24, __node24]


@jax.jit
def computeLevel3SubIdx(Levels, laser_all):
    def compute_element_number(_laser_pos):
        _xc = jnp.stack(
            [
                jnp.where(Levels[3]["node_coords"][_] >= _laser_pos[_], size=1)[0][0]
                for _ in range(2)
            ]
        )
        element_number = (
            (_xc[0])
            + (_xc[1]) * Levels[3]["elements"][0]
            + (Levels[3]["elements"][2] - 1)
            * Levels[3]["elements"][0]
            * Levels[3]["elements"][1]
        )
        return Levels[3]["CenterIdx"] + element_number

    vcompute_element_number = jax.vmap(compute_element_number)
    Levels[3]["SurroSubIdx"] = vcompute_element_number(laser_all)
    return Levels


@jax.jit
def computeSurrogateCoarseFineShapeFunctions(Coarse, Fine):
    """computeCoarseFineShapeFunctions finds the shape functions of
    the fine scale quadrature points for the coarse element
    :param Coarse["node_coords"]: nodal coordinates of global coarse
    :param Coarse["connect"]: indices to get coordinates of nodes of coarse element
    :param Fine["node_coords"]: nodal coordinates of global fine
    :param Fine["connect"]: indices to get x coordinates of nodes of fine element
    :return Nc, dNcdx, dNcdy, dNcdz, _nodes.reshape(-1)
    :return Nc: (Num fine elements, 8 quadrature, 8), coarse shape function for fine element
    :return dNcdx: (Num fine elements, 8 quadrature, 8), coarse x-derivate shape function for fine element
    :return dNcdy: (Num fine elements, 8 quadrature, 8), coarse y-derivate shape function for fine element
    :return dNcdz: (Num fine elements, 8 quadrature, 8), coarse z-derivate shape function for fine element
    :return _nodes: (Num fine elements * 8 * 8), coarse nodal indices
    """
    # Get number of elements and nodes for both coarse and fine
    nec_x, nec_y, nec_z = [Coarse["connect"][i].shape[0] for i in range(3)]
    nnc_x, nnc_y, nnc_z = [Coarse["node_coords"][i].shape[0] for i in range(3)]
    nnc = nnc_x * nnc_y * nnc_z
    nef_x, nef_y, nef_z = [Fine["connect"][i].shape[0] for i in range(3)]
    nef = nef_x * nef_y * nef_z
    nnf_x, nnf_y = [Fine["node_coords"][i].shape[0] for i in range(2)]

    # Assume constant mesh sizes
    hc_x = Coarse["node_coords"][0][1] - Coarse["node_coords"][0][0]
    hc_y = Coarse["node_coords"][1][1] - Coarse["node_coords"][1][0]
    hc_z = Coarse["node_coords"][2][1] - Coarse["node_coords"][2][0]

    # Get lower bounds of meshes
    xminc_x, xminc_y, xminc_z = [Coarse["node_coords"][i][0] for i in range(3)]

    # Get shape functions and weights
    coords_x = Fine["node_coords"][0][Fine["connect"][0][0, :]].reshape(-1, 1)
    coords_y = Fine["node_coords"][1][Fine["connect"][1][0, :]].reshape(-1, 1)
    coords_z = Fine["node_coords"][2][Fine["connect"][2][0, :]].reshape(-1, 1)
    coords = jnp.concatenate([coords_x, coords_y, coords_z], axis=1)
    Nf, _, _ = computeQuad3dFemShapeFunctions_jax(coords)

    def stepComputeCoarseFineTerm(ieltf):
        ix, iy, iz, _ = convert2XYZ(ieltf, nef_x, nef_y, nnf_x, nnf_y)
        coords_x = Fine["node_coords"][0][Fine["connect"][0][ix, :]].reshape(-1, 1)
        coords_y = Fine["node_coords"][1][Fine["connect"][1][iy, :]].reshape(-1, 1)
        coords_z = Fine["node_coords"][2][Fine["connect"][2][iz, :]].reshape(-1, 1)

        # Do all of the quadrature points simultaneously
        x = Nf @ coords_x
        y = Nf @ coords_y
        z = Nf @ coords_z

        x_comp = (nec_x - 1) * jnp.ones_like(x)
        y_comp = (nec_y - 1) * jnp.ones_like(y)
        z_comp = (nec_z - 1) * jnp.ones_like(z)

        # Figure out which coarse element we are in
        _floorx = jnp.floor((x - xminc_x) / hc_x)
        _conx = jnp.concatenate((_floorx, x_comp), axis=1)
        ieltc_x = jnp.min(_conx, axis=1).T.astype(int)
        _floory = jnp.floor((y - xminc_y) / hc_y)
        _cony = jnp.concatenate((_floory, y_comp), axis=1)
        ieltc_y = jnp.min(_cony, axis=1).T.astype(int)
        _floorz = jnp.floor((z - xminc_z) / hc_z)
        _conz = jnp.concatenate((_floorz, z_comp), axis=1)
        ieltc_z = jnp.min(_conz, axis=1).T.astype(int)

        x = x.reshape(-1)
        y = y.reshape(-1)
        z = z.reshape(-1)

        def iqLoopMass(iq):
            nodec_x = Coarse["connect"][0][ieltc_x[iq], :].astype(int)
            nodec_y = Coarse["connect"][1][ieltc_y[iq], :].astype(int)
            nodec_z = Coarse["connect"][2][ieltc_z[iq], :].astype(int)
            nodes = nodec_x + nodec_y * nnc_x + nodec_z * nnc_x * nnc_y

            _x = x[iq]
            _y = y[iq]
            _z = z[iq]

            xc0 = Coarse["node_coords"][0][Coarse["connect"][0][ieltc_x[iq], 0]]
            xc1 = Coarse["node_coords"][0][Coarse["connect"][0][ieltc_x[iq], 1]]
            yc0 = Coarse["node_coords"][1][Coarse["connect"][1][ieltc_y[iq], 0]]
            yc3 = Coarse["node_coords"][1][Coarse["connect"][1][ieltc_y[iq], 3]]
            zc0 = Coarse["node_coords"][2][Coarse["connect"][2][ieltc_z[iq], 0]]
            zc5 = Coarse["node_coords"][2][Coarse["connect"][2][ieltc_z[iq], 5]]

            # Evaluate shape functions associated with coarse nodes
            Nc = compute3DN(
                [_x, _y, _z], [xc0, xc1], [yc0, yc3], [zc0, zc5], [hc_x, hc_y, hc_z]
            )
            return Nc, nodes

        viqLoopMass = jax.vmap(iqLoopMass)
        return viqLoopMass(jnp.arange(8))

    vstepComputeCoarseFineTerm = jax.vmap(stepComputeCoarseFineTerm)

    def compute_terms(_SubIdx):
        Nc, _nodes = vstepComputeCoarseFineTerm(_SubIdx)
        _nodes = _nodes[:, 0, :]
        indices = jnp.concatenate(
            [_nodes.reshape(-1, 1), jnp.arange(_nodes.size).reshape(-1, 1)], axis=1
        )
        return [Nc, indices]

    vcompute_terms = jax.vmap(compute_terms)
    [Nc, indices] = vcompute_terms(Fine["SurroSubIdx"])
    return [Nc, indices[:, :, 0]]


def process_levels_and_save(
    Nonmesh: dict,
    Test2pos,
    laser_all,
    Levels: dict,
    track,
    run_surrogate,
    t_output,
    savenum: int,
    flip_flag=None,
):
    """
    Processes level data and saves results if training is not active.
    """

    if not Nonmesh.get("training"):
        # Update Test2pos with the last laser_all position
        Test2pos = Test2pos.at[:].set(laser_all[-1, :])

        # Update node coordinates for Level 4
        Levels[4]["node_coords"] = [
            Levels[4]["orig_node_coords"][i] + Test2pos[i] for i in range(3)
        ]

        # Interpolate temperature values
        L4T = interpolatePoints(Levels[2], Levels[2]["T0"], Levels[4]["node_coords"])
        L4T += interpolatePoints(
            Levels[3], Levels[3]["Tprime0"], Levels[4]["node_coords"]
        )

        # Interpolate surrogate matrix
        SurroLInterpTestL0 = interpolatePointsMatrix(
            Levels[0], Levels[4]["node_coords"]
        )
        Levels[4]["S1"] = (
            interpolate_w_matrix(SurroLInterpTestL0, Levels[0]["S1"]) > 1e-2
        ) * 1

        # Save results
        save_data = {
            "L2T": Levels[2]["T0"],
            "L3T": Levels[3]["T0"],
            "L4T": L4T,
            "L4S": Levels[4]["S1"],
            "track": track,
            "run_surrogate": run_surrogate,
            "time": t_output,
            "flip_flag": flip_flag if flip_flag is not None else -1,
        }

        jnp.savez(f"{Nonmesh['save_path']}check_error_{savenum:07}", **save_data)

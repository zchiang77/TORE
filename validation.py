from src.forward import run_forward, external_validation, specify_use_model
from src.ringHelper import *
from src.visualize import model_testing_plot, model_testing_plot_chip
from src.topology import plot_persistence_diagram, TopoConfig, get_topo_config, plot_latent_space
from src.data import (
    training_chip_data, build_chip_chrom_offsets,
    extract_blocks_by_indices, get_chip_labels,
    compute_oe_matrix, inverse_oe_matrix,
    compute_spatial_distance_matrix,
    inverse_transform_chip_batch
)
from src.ringHelper import Log1pMinMaxScaler, knockout_protein
from arguments import *
from torch.utils.data import TensorDataset
from datetime import datetime
import cooler
import os
import logging
import random
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
config_file = load_config("config.json")
ring_platform = config_file["info"]["platform"]
ring_version = config_file["info"]["version"]
now = datetime.now()
date_str = str(now.date())
hour_str = str(now.hour)
logging.basicConfig(filename=f'validation.log', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)
logger.info(f"Running Ring model (version {ring_version}) on platform {ring_platform}")
has_cuda = torch.cuda.is_available()
if has_cuda:
    logger.info(f"Available GPUs are as follows:")
    for i in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_name(i)
        logger.info(f" - ({i + 1}) {gpu_name}")
else:
    logger.info("No CUDA GPUs available  running on CPU")
os.environ["USE_LIBUV"] = "0"
rs = 123
set_seed(rs)
logger.info(f"[MAIN] Random seed: {rs}")
train_cfg = config_file.get("ring", {})
topo_cfg = get_topo_config(train_cfg)
logger.info(f"[MAIN] TopoConfig: threshold={topo_cfg.threshold}, "
            f"h0_weight={topo_cfg.h0_weight}, "
            f"smooth_weight={topo_cfg.smooth_weight}, gamma={topo_cfg.gamma}")
parser = test_args()
args = parser.parse_args()
x_resolution = args.x_resolution
y_resolution = args.y_resolution if args.y_resolution is not None else x_resolution
logger.info(f"[MAIN] Hi-C (X) Resolution: {x_resolution}")
if y_resolution != x_resolution:
    logger.info(f"[MAIN] ChIP-seq (Y) Resolution: {y_resolution}")
patch_size = 1e+6 * args.patch
patch_bins = int(patch_size / x_resolution)
logger.info(f"[MAIN] Each patch has {patch_bins} samples equal to {patch_size}bp")
patch_overlap = 1e+6 * args.patch_overlap
patch_overlap_bins = int(patch_overlap / x_resolution)
logger.info(f"[MAIN] Each two patches have {patch_overlap_bins} samples overlap")
world_size = 1
device = args.device
if device == "cpu":
    rank = "cpu"
    logger.info(f"[MAIN] Using CPU (--device cpu)")
elif device == "cuda":
    if not has_cuda:
        logger.error("[MAIN] --device cuda requested but no CUDA GPU available")
        sys.exit(1)
    rank = 0
    logger.info(f"[MAIN] Using GPU (--device cuda)")
else:          
    if has_cuda:
        rank = 0
        logger.info(f"[MAIN] Using GPU (auto-detected)")
    else:
        rank = "cpu"
        logger.info(f"[MAIN] Using CPU (no GPU detected)")
if args.threads is not None:
    safe_threads = args.threads
    os.environ["NUMEXPR_NUM_THREADS"] = str(safe_threads)
else:
    logger.info(f"[MAIN] Missing parameters --threads")
    max_cores = os.cpu_count()
    safe_threads = int(max_cores * 0.5)
    os.environ["NUMEXPR_MAX_THREADS"] = str(safe_threads)
logger.info(f"[MAIN] Using {safe_threads} threads")
os.environ['MASTER_ADDR'] = 'localhost'
port_use = args.port
os.environ['MASTER_PORT'] = port_use
logger.info(f"[MAIN] Port: {port_use}")
experiment = args.test_x
chrom = args.chr
testing_range = chrom + ":" + args.range1 + "-" + args.range2
logger.info(f"[MAIN] Data X: {args.test_x} ({testing_range})")
recon_test_label = args.test_x_label if args.test_x_label is not None else None
logger.info(f"[MAIN] Label X: {recon_test_label if args.test_x_label is not None else ''}")
topo_enabled = args.use_topology
if topo_enabled:
    logger.info(f"[MAIN] Topological loss enabled")
else:
    logger.info(f"[MAIN] Topological loss disabled")
save_path_dir = args.model_save
if not os.path.exists(save_path_dir):
    os.makedirs(save_path_dir)
hic_norm = args.hic_normalization
logger.info(f"[MAIN] Hi-C normalization: {hic_norm}")
previous_model_path = args.use_model
logger.info(f"[MAIN] Using model {previous_model_path}")
whole_chrom_mode = not args.no_whole_chrom
logger.info(f"[MAIN] Whole chromosome mode: {whole_chrom_mode}")
if whole_chrom_mode:
    logger.info(f"[MAIN] Will scan all blocks of chr{chrom} and stitch results")
def stitch_blocks(block_list, local_starts, chrom_nbins, patch_bins, overlap_bins):
    accum = np.zeros((chrom_nbins, chrom_nbins), dtype=np.float64)
    weight = np.zeros((chrom_nbins, chrom_nbins), dtype=np.float64)
    for block, s in zip(block_list, local_starts):
        e = min(s + patch_bins, chrom_nbins)
        actual = e - s
        accum[s:e, s:e] += block[:actual, :actual]
        weight[s:e, s:e] += 1.0
    weight[weight == 0] = 1.0
    return (accum / weight).astype(np.float32)
if __name__ == '__main__':
    logger.info(f"[MAIN] Current Hi-C file {experiment}")
    cooler_path = "../data/HiC.GRCh38/" + experiment + "_" + str(x_resolution) + ".cool"
    cooler_file = cooler.Cooler(cooler_path)
    cooler_mat = cooler_file.matrix()
    if whole_chrom_mode:
        import sys
        chrom_extent = cooler_file.extent(chrom)
        chrom_w_start = chrom_extent[0]
        chrom_w_end = chrom_extent[1]
        chrom_nbins = chrom_w_end - chrom_w_start
        full_raw = np.array(
            cooler_mat[chrom_w_start:chrom_w_end, chrom_w_start:chrom_w_end],
            dtype=np.float32
        )
        full_raw[np.isnan(full_raw)] = 0
        logger.info(f"[MAIN] Whole chr{chrom}: {chrom_nbins}x{chrom_nbins} bins "
                     f"({chrom_nbins * x_resolution / 1e6:.1f} Mb)")
        if hic_norm == "oe":
            full_normed = compute_oe_matrix(full_raw).astype(np.float32)
        elif hic_norm == "log1p":
            full_normed = np.log1p(full_raw).astype(np.float32)
        else:
            full_normed = full_raw.copy()
        step = patch_bins - patch_overlap_bins
        blocks = []
        for local_s in range(0, chrom_nbins - patch_bins + 1, step):
            b = full_normed[local_s:local_s + patch_bins, local_s:local_s + patch_bins]
            blocks.append((b, local_s, chrom_w_start + local_s))
        M = len(blocks)
        logger.info(f"[MAIN] Sliced into {M} blocks (window={patch_bins}, "
                     f"overlap={patch_overlap_bins}, step={step})")
        if M == 0:
            raise ValueError(
                f"Chromosome {chrom} too small ({chrom_nbins} bins) "
                f"for patch size {patch_bins} bins."
            )
        base_name = f"Validation_Experiment={experiment}_WholeChr{chrom}"
        if args.reconstruction:
            model_type = "x_reconstruction"
            y_p = 0
            chip_meta_block = None
            chip_labels_val = None
            debug_mode = False
            save_path = save_path_dir + (
                f"{base_name}_Topological" if topo_enabled else base_name)
        else:
            chip_file_path = (
                "../data/ChIP.GRCh38.simp/resolution" + str(y_resolution) + "/"
                if args.test_y is None else args.test_y
            )
            logger.info(f"[MAIN] ChIP file path: {chip_file_path}")
            entire_chip, chip_scaler = training_chip_data(chip_file_path, return_scaler=True)
            chrom_offsets, chrom_nbins_dict, _ = build_chip_chrom_offsets(chip_file_path)
            chip_labels_val = get_chip_labels(chip_file_path)
            y_p = entire_chip.shape[1]
            ko_tag = ""
            if args.knockout is not None:
                ko_proteins = [p.strip() for p in args.knockout.split(",")]
                ko_indices = [chip_labels_val.index(p) for p in ko_proteins if p in chip_labels_val]
                not_found = [p for p in ko_proteins if p not in chip_labels_val]
                if not_found:
                    logger.warning(f"[MAIN] Knockout proteins not found: {not_found}")
                    logger.warning(f"[MAIN] Available proteins: {chip_labels_val}")
                if ko_indices:
                    entire_chip = knockout_protein(entire_chip, ko_indices)
                    ko_tag = "_KO-" + "+".join([p for p in ko_proteins if p in chip_labels_val])
                    logger.info(f"[MAIN] Knocked out: {[p for p in ko_proteins if p in chip_labels_val]} "
                                 f"at indices {ko_indices}")
            bins_table = cooler_file.bins()[:]
            chip_meta_block = {
                'bins_table': bins_table,
                'entire_chip': entire_chip,
                'chip_scaler': chip_scaler,
                'chrom_offsets': chrom_offsets,
                'chrom_nbins': chrom_nbins_dict,
                'patch_bins': patch_bins,
                'x_resolution': x_resolution,
                'y_resolution': y_resolution,
            }
            logger.info(f"[MAIN] ChIP-seq shape: {entire_chip.shape}, "
                        f"{len(chrom_offsets)} chromosomes, {y_p} features")
            if args.x_pred_y:
                model_type = "y_prediction"
            elif args.y_pred_x:
                model_type = "x_prediction"
            elif args.ring:
                model_type = "ring"
            elif args.y_reconstruction:
                model_type = "y_reconstruction"
            elif args.halfring:
                model_type = "halfring"
            else:
                raise NotImplementedError(
                    "Must specify one of: --reconstruction, --x_pred_y, "
                    "--y_pred_x, --ring, --y_reconstruction, --halfring"
                )
            save_path = save_path_dir + base_name + ko_tag
            if topo_enabled:
                save_path += "_Topological"
        logger.info(f"[MAIN] Model type: {model_type}")
        if args.reconstruction and recon_test_label is not None:
            logger.warning(
                f"[MAIN] --test_x_label is ignored in whole-chromosome mode."
            )
        from src.model_builder import build_model
        y_length = None
        if chip_meta_block is not None:
            y_length = patch_bins * x_resolution // y_resolution
        model = build_model(model_type, config_file, patch_bins, y_p, rank,
                            y_length=y_length, preset=args.preset)
        model = specify_use_model(rank, previous_model_path, model)
        model.eval()
        all_recon = []
        all_local_starts = []
        all_cross_y_raw = []                                          
        all_cross_y_norm = []                                                   
        all_cross_y_pos = []                                               
        all_cross_x = []
        all_cross_x_starts = []
        all_true_y_norm = []                                            
        all_true_y_raw = []                                      
        first_raw_block = full_raw[0:patch_bins, 0:patch_bins]
        D_x_whole = compute_spatial_distance_matrix(first_raw_block, gamma=topo_cfg.gamma)
        for i, (block_mat, local_s, global_s) in enumerate(blocks):
            block_hic = torch.tensor(block_mat, dtype=torch.float32)
            if args.reconstruction:
                block_dataset = TensorDataset(
                    block_hic,
                    torch.zeros(block_hic.shape[0], dtype=torch.float32)
                )
                block_pos_repeated = None
            else:
                block_pos = torch.tensor(
                    [[global_s, global_s]], dtype=torch.int32
                )
                block_pos_repeated = block_pos.repeat_interleave(patch_bins, dim=0)
                block_dataset = TensorDataset(block_hic, block_pos_repeated)
            x_pred, y_pred, x_recon, y_recon, u1, u2, y_gt, y_from_u1, x_from_u2 = external_validation(
                rank, model, model_type,
                block_dataset, batch_size=patch_bins,
                chip_meta=chip_meta_block,
                cross_reconstruction=args.cross_reconstruction
            )
            if x_recon is not None:
                recon_np = x_recon.detach().cpu().numpy()
                recon_np = recon_np.reshape(patch_bins, patch_bins)
                all_recon.append(recon_np)
                all_local_starts.append(local_s)
            if y_from_u1 is not None:
                y_from_u1_norm = y_from_u1.detach().cpu()
                all_cross_y_norm.append(y_from_u1_norm.numpy())
                all_cross_y_pos.append(block_pos_repeated)
                if (chip_meta_block is not None
                        and block_pos_repeated is not None):
                    y_from_u1_raw = inverse_transform_chip_batch(
                        y_from_u1_norm, block_pos_repeated, chip_meta_block, 'cpu'
                    ).numpy()
                else:
                    y_from_u1_raw = y_from_u1_norm.numpy()
                all_cross_y_raw.append(y_from_u1_raw)
            if x_from_u2 is not None:
                cross_x_np = x_from_u2.detach().cpu().numpy()
                cross_x_np = cross_x_np.reshape(patch_bins, patch_bins)
                all_cross_x.append(cross_x_np)
                all_cross_x_starts.append(local_s)
            if y_gt is not None:
                y_gt_norm = y_gt.detach().cpu()
                all_true_y_norm.append(y_gt_norm.numpy())
                if (chip_meta_block is not None
                        and block_pos_repeated is not None):
                    y_gt_raw = inverse_transform_chip_batch(
                        y_gt_norm, block_pos_repeated, chip_meta_block, 'cpu'
                    ).numpy()
                else:
                    y_gt_raw = y_gt_norm.numpy()
                all_true_y_raw.append(y_gt_raw)
            if M <= 10 or (i + 1) % max(1, M // 10) == 0 or i + 1 == M:
                logger.info(f"[MAIN] Block {i + 1}/{M} completed")
        stitched = stitch_blocks(
            all_recon, all_local_starts,
            chrom_nbins, patch_bins, patch_overlap_bins
        )
        if hic_norm == "oe":
            stitched_display = inverse_oe_matrix(stitched, full_raw)
        elif hic_norm == "log1p":
            stitched_display = np.expm1(stitched)
        else:
            stitched_display = stitched
        np.savetxt(save_path + "_reconstructed-x.txt", stitched, fmt="%.6f")
        np.savetxt(save_path + "_true-x.txt", full_raw, fmt="%.6f")
        model_testing_plot(
            x=full_raw,
            y=stitched_display,
            range_1=0,
            range_2=chrom_nbins,
            caption_model=previous_model_path,
            caption_data=f"{experiment}:{chrom} (whole, {M} blocks)",
            save_name=save_path + "_x-reconstruction-whole-chromosome.png",
            log_op=True
        )
        if all_cross_y_raw:
            cross_y_raw = np.vstack(all_cross_y_raw)
            cross_y_norm = np.vstack(all_cross_y_norm)
            np.savetxt(save_path + "_cross-reconstructed-y.txt", cross_y_raw, fmt="%.6f")
            np.savetxt(save_path + "_cross-reconstructed-y-norm.txt", cross_y_norm, fmt="%.6f")
            logger.info(f"[MAIN] Cross-reconstructed Y saved ({M} blocks concatenated)")
        if all_true_y_norm:
            true_y_norm = np.vstack(all_true_y_norm)
            np.savetxt(save_path + "_true-y-norm.txt", true_y_norm, fmt="%.6f")
            if all_true_y_raw:
                true_y_raw = np.vstack(all_true_y_raw)
                np.savetxt(save_path + "_true-y.txt", true_y_raw, fmt="%.6f")
            logger.info(f"[MAIN] Ground-truth Y saved ({M} blocks concatenated)")
        if all_cross_x:
            cross_x_stitched = stitch_blocks(
                all_cross_x, all_cross_x_starts,
                chrom_nbins, patch_bins, patch_overlap_bins
            )
            if hic_norm == "oe":
                cross_x_display = inverse_oe_matrix(cross_x_stitched, full_raw)
            elif hic_norm == "log1p":
                cross_x_display = np.expm1(cross_x_stitched)
            else:
                cross_x_display = cross_x_stitched
            np.savetxt(save_path + "_cross-reconstructed-x.txt", cross_x_stitched, fmt="%.6f")
            model_testing_plot(
                x=full_raw,
                y=cross_x_display,
                range_1=0,
                range_2=chrom_nbins,
                caption_model=previous_model_path + " [u2→x_reconstructor]",
                caption_data=f"{experiment}:{chrom} (whole, {M} blocks)",
                save_name=save_path + "_x-cross-reconstruction-whole-chromosome.png",
                log_op=True
            )
            logger.info(f"[MAIN] Cross-reconstructed X saved ({M} blocks stitched)")
        if topo_enabled and all_recon and u1 is not None:
            embed_1 = u1.detach().cpu().numpy()
            plot_persistence_diagram(
                D_x=D_x_whole,
                z=embed_1,
                output_filename=save_path + "_latent-representation.png",
                topo_cfg=topo_cfg
            )
            logger.info(f"[MAIN] Persistence diagram saved (via distance matrix)")
            if u1 is not None and u1.shape[1] <= 3:
                plot_latent_space(u1.detach().cpu().numpy(),
                                  save_path + "_latent-space-u1.png",
                                  title_prefix="u1")
                logger.info(f"[MAIN] u1 latent space interactive HTML saved "
                            f"(dim={u1.shape[1]})")
            if u2 is not None and u2.shape[1] <= 3:
                plot_latent_space(u2.detach().cpu().numpy(),
                                  save_path + "_latent-space-u2.png",
                                  title_prefix="u2")
                logger.info(f"[MAIN] u2 latent space interactive HTML saved "
                            f"(dim={u2.shape[1]})")
        logger.info(f"[MAIN] All outputs saved to {save_path_dir}")
        logger.info(f"[MAIN] Whole-chromosome validation completed "
                     f"({M} blocks stitched → {chrom_nbins}x{chrom_nbins})")
        sys.exit(0)
    if args.ranges is not None:
        regions = []
        for seg in args.ranges.split(","):
            r1, r2 = seg.split("-")
            regions.append((int(r1.strip()), int(r2.strip())))
        logger.info(f"[MAIN] Multi-region validation: {len(regions)} regions: {regions}")
    else:
        regions = [(int(args.range1), int(args.range2))]
    base_name = f"Validation_Experiment={experiment}"
    if args.reconstruction:                       
        logger.info(f"[MAIN] Running in reconstruction (pretrain) mode")
        save_path = save_path_dir + (
            f"{base_name}_Topological" if topo_enabled else base_name)
        model_type = "x_reconstruction"
        y_p = 0
        chip_meta = None
        chip_labels = None
        debug_mode = False
    else:                           
        chip_file_path = (
            "../data/ChIP.GRCh38.simp/resolution" + str(y_resolution) + "/"
            if args.test_y is None else args.test_y
        )
        logger.info(f"[MAIN] ChIP file path: {chip_file_path}")
        entire_chip, chip_scaler = training_chip_data(chip_file_path, return_scaler=True)
        chrom_offsets, chrom_nbins, _ = build_chip_chrom_offsets(chip_file_path)
        chip_labels = get_chip_labels(chip_file_path)
        y_p = entire_chip.shape[1]
        ko_tag = ""
        if args.knockout is not None:
            ko_proteins = [p.strip() for p in args.knockout.split(",")]
            ko_indices = [chip_labels.index(p) for p in ko_proteins if p in chip_labels]
            not_found = [p for p in ko_proteins if p not in chip_labels]
            if not_found:
                logger.warning(f"[MAIN] Knockout proteins not found: {not_found}")
                logger.warning(f"[MAIN] Available proteins: {chip_labels}")
            if ko_indices:
                entire_chip = knockout_protein(entire_chip, ko_indices)
                ko_tag = "_KO-" + "+".join([p for p in ko_proteins if p in chip_labels])
                logger.info(f"[MAIN] Knocked out: {[p for p in ko_proteins if p in chip_labels]} "
                             f"at indices {ko_indices}")
        bins_table = cooler_file.bins()[:]
        chip_meta = {
            'bins_table': bins_table,
            'entire_chip': entire_chip,
            'chip_scaler': chip_scaler,
            'chrom_offsets': chrom_offsets,
            'chrom_nbins': chrom_nbins,
            'patch_bins': patch_bins,
            'x_resolution': x_resolution,
            'y_resolution': y_resolution,
        }
        logger.info(f"[MAIN] ChIP-seq shape: {entire_chip.shape}, "
                    f"{len(chrom_offsets)} chromosomes detected, "
                    f"{y_p} features")
        debug_mode = args.debug
        if args.x_pred_y:
            model_type = "y_prediction"
        elif args.y_pred_x:
            model_type = "x_prediction"
        elif args.ring:
            model_type = "ring"
        elif args.y_reconstruction:
            model_type = "y_reconstruction"
        elif args.halfring:
            model_type = "halfring"
        else:
            raise NotImplementedError(
                "Must specify one of: --reconstruction, --x_pred_y, --y_pred_x, --ring, --y_reconstruction, --halfring"
            )
        save_path = save_path_dir + base_name + ko_tag
        if topo_enabled:
            save_path += "_Topological"
    logger.info(f"[MAIN] Model type: {model_type}")
    logger.info(f"[MAIN] Debug mode: {debug_mode}")
    from src.model_builder import build_model
    y_length = None
    if chip_meta is not None:
        y_length = patch_bins * x_resolution // y_resolution
    model = build_model(model_type, config_file, patch_bins, y_p, rank,
                        y_length=y_length, preset=args.preset)
    model = specify_use_model(rank, previous_model_path, model)
    model.eval()
    for (r1_bp, r2_bp) in regions:
        region_tag = f"{r1_bp}-{r2_bp}"
        testing_range = chrom + ":" + str(r1_bp) + "-" + str(r2_bp)
        logger.info(f"[MAIN] === Validating region {testing_range} ===")
        testing_bins = cooler_file.extent(testing_range)
        testing_hic_mat = cooler_mat[testing_bins[0]:testing_bins[1],
                         testing_bins[0]:testing_bins[1]]
        testing_hic_mat[np.isnan(testing_hic_mat)] = 0
        if testing_hic_mat.shape[0] != patch_bins or testing_hic_mat.shape[1] != patch_bins:
            logger.warning(
                f"[MAIN] Region {region_tag} dim {testing_hic_mat.shape} "
                f"!= {patch_bins}x{patch_bins}, skipping. "
                f"Bins extracted: {testing_bins}"
            )
            continue
        testing_hic_mat_raw = testing_hic_mat.copy()
        D_x_val = compute_spatial_distance_matrix(testing_hic_mat_raw, gamma=topo_cfg.gamma)
        if hic_norm == "log1p":
            testing_hic_mat = np.log1p(testing_hic_mat)
        elif hic_norm == "oe":
            testing_hic_mat = compute_oe_matrix(testing_hic_mat).astype(np.float32)
        elif hic_norm == "none":
            pass
        testing_hic = torch.tensor(np.array(testing_hic_mat), dtype=torch.float32)
        if args.reconstruction:
            testing_chip_dummy = torch.zeros(testing_hic.shape[0], dtype=torch.float32)
            testing_dataset = TensorDataset(testing_hic, testing_chip_dummy)
            testing_pos_repeated = None
        else:
            testing_pos = torch.tensor(
                [[testing_bins[0], testing_bins[0]]], dtype=torch.int32
            )
            testing_pos_repeated = testing_pos.repeat_interleave(patch_bins, dim=0)
            testing_dataset = TensorDataset(testing_hic, testing_pos_repeated)
        use_label = False
        testing_hic_label = None
        reconstructed_x_display = None
        if args.reconstruction and recon_test_label is not None:
            cooler_label = cooler.Cooler(recon_test_label)
            cooler_label_mat = cooler_label.matrix()
            testing_hic_label = extract_blocks_by_indices(
                cooler_label_mat,
                [(testing_bins[0], testing_bins[0])],
                patch_bins
            )
            testing_hic_label = testing_hic_label.reshape(-1, patch_bins)
            testing_hic_label = testing_hic_label.numpy()
            testing_hic_label_raw = testing_hic_label.copy()
            if hic_norm == "log1p":
                testing_hic_label = np.log1p(testing_hic_label)
            elif hic_norm == "oe":
                testing_hic_label = compute_oe_matrix(testing_hic_label).astype(np.float32)
            logger.info(f"[MAIN] Reconstruction label prepared for {region_tag}")
            use_label = True
        logger.info(f"[MAIN] Starting forward pass on {testing_range}")
        x_pred, y_pred, x_recon, y_recon, u1, u2, y_gt, y_from_u1, x_from_u2 = \
            external_validation(
                rank=rank, model=model, model_type=model_type,
                dataset=testing_dataset, batch_size=patch_bins,
                chip_meta=chip_meta,
                cross_reconstruction=args.cross_reconstruction
            )
        logger.info(f"[MAIN] Forward pass completed for {region_tag}")
        save_path_region = save_path + f"_{region_tag}"
        embed_1 = None
        if x_recon is not None:
            reconstructed_x = x_recon.detach().cpu().numpy()
            embed_1 = u1.detach().cpu().numpy()
            reconstructed_x_sym = symmetry_correction(reconstructed_x)
            if hic_norm == "oe":
                reconstructed_x_display = inverse_oe_matrix(reconstructed_x_sym, testing_hic_mat_raw)
            elif hic_norm == "log1p":
                reconstructed_x_display = np.expm1(reconstructed_x_sym)
            else:
                reconstructed_x_display = reconstructed_x_sym
            np.savetxt(save_path_region + "_reconstructed-x.txt", reconstructed_x_sym, fmt="%.6f")
            np.savetxt(save_path_region + "_embed-u1.txt", embed_1, fmt="%.6f")
        if x_pred is not None:
            predicted_x = x_pred.detach().cpu().numpy()
            predicted_x_sym = symmetry_correction(predicted_x)
            if hic_norm == "oe":
                predicted_x_display = inverse_oe_matrix(predicted_x_sym, testing_hic_mat_raw)
            elif hic_norm == "log1p":
                predicted_x_display = np.expm1(predicted_x_sym)
            else:
                predicted_x_display = predicted_x_sym
            np.savetxt(save_path_region + "_predicted-x.txt", predicted_x_sym, fmt="%.6f")
            model_testing_plot(
                x=testing_hic_mat_raw,
                y=predicted_x_display,
                range_1=0,
                range_2=patch_bins,
                caption_model=previous_model_path,
                caption_data=experiment + ":" + testing_range,
                save_name=save_path_region + "_x-prediction.png",
                log_op=True
            )
        if y_pred is not None:
            predicted_y = y_pred.detach().cpu()
            if chip_meta is not None and testing_pos_repeated is not None:
                predicted_y_raw = inverse_transform_chip_batch(
                    predicted_y, testing_pos_repeated, chip_meta, 'cpu').numpy()
            else:
                predicted_y_raw = predicted_y.numpy()
            predicted_y_norm = predicted_y.numpy()
            np.savetxt(save_path_region + "_predicted-y.txt", predicted_y_raw, fmt="%.6f")
            np.savetxt(save_path_region + "_predicted-y-norm.txt", predicted_y_norm, fmt="%.6f")
            if y_gt is not None:
                y_gt_np = y_gt.detach().cpu()
                if chip_meta is not None and testing_pos_repeated is not None:
                    y_gt_raw = inverse_transform_chip_batch(
                        y_gt_np, testing_pos_repeated, chip_meta, 'cpu').numpy()
                else:
                    y_gt_raw = y_gt_np.numpy()
                y_gt_norm = y_gt_np.numpy()
            else:
                y_gt_raw = None
                y_gt_norm = None
            model_testing_plot_chip(
                x=y_gt_raw,
                y=predicted_y_raw,
                range_1=0,
                range_2=patch_bins,
                xticks=chip_labels,
                caption_model=previous_model_path,
                caption_data=experiment + ":" + testing_range,
                save_name=save_path_region + "_ChIP-prediction.png",
                log_op=False
            )
        if y_recon is not None:
            reconstructed_y = y_recon.detach().cpu()
            if chip_meta is not None and testing_pos_repeated is not None:
                reconstructed_y_raw = inverse_transform_chip_batch(
                    reconstructed_y, testing_pos_repeated, chip_meta, 'cpu').numpy()
            else:
                reconstructed_y_raw = reconstructed_y.numpy()
            reconstructed_y_norm = reconstructed_y.numpy()
            np.savetxt(save_path_region + "_reconstructed-y.txt", reconstructed_y_raw, fmt="%.6f")
            np.savetxt(save_path_region + "_reconstructed-y-norm.txt", reconstructed_y_norm, fmt="%.6f")
            if y_gt is not None:
                y_gt_np = y_gt.detach().cpu()
                if chip_meta is not None and testing_pos_repeated is not None:
                    y_gt_raw_for_recon = inverse_transform_chip_batch(
                        y_gt_np, testing_pos_repeated, chip_meta, 'cpu').numpy()
                else:
                    y_gt_raw_for_recon = y_gt_np.numpy()
                np.savetxt(save_path_region + "_true-y.txt",
                           y_gt_raw_for_recon, fmt="%.6f")
                np.savetxt(save_path_region + "_true-y-norm.txt",
                           y_gt_np.numpy(), fmt="%.6f")
            else:
                y_gt_raw_for_recon = None
            model_testing_plot_chip(
                x=y_gt_raw_for_recon,
                y=reconstructed_y_raw,
                range_1=0,
                range_2=patch_bins,
                xticks=chip_labels,
                caption_model=previous_model_path,
                caption_data=experiment + ":" + testing_range,
                save_name=save_path_region + "_ChIP-reconstruction.png",
                log_op=False
            )
        if y_from_u1 is not None:
            cross_y = y_from_u1.detach().cpu()
            if chip_meta is not None and testing_pos_repeated is not None:
                cross_y_raw = inverse_transform_chip_batch(
                    cross_y, testing_pos_repeated, chip_meta, 'cpu').numpy()
            else:
                cross_y_raw = cross_y.numpy()
            cross_y_norm = cross_y.numpy()
            np.savetxt(save_path_region + "_cross-reconstructed-y.txt", cross_y_raw, fmt="%.6f")
            np.savetxt(save_path_region + "_cross-reconstructed-y-norm.txt", cross_y_norm, fmt="%.6f")
            if y_gt is not None:
                y_gt_np = y_gt.detach().cpu()
                if chip_meta is not None and testing_pos_repeated is not None:
                    y_gt_raw_cross = inverse_transform_chip_batch(
                        y_gt_np, testing_pos_repeated, chip_meta, 'cpu').numpy()
                else:
                    y_gt_raw_cross = y_gt_np.numpy()
            else:
                y_gt_raw_cross = None
            model_testing_plot_chip(
                x=y_gt_raw_cross,
                y=cross_y_raw,
                range_1=0,
                range_2=patch_bins,
                xticks=chip_labels,
                caption_model=previous_model_path + " [u1→y_reconstructor]",
                caption_data=experiment + ":" + testing_range,
                save_name=save_path_region + "_ChIP-cross-reconstruction.png",
                log_op=False
            )
        if x_from_u2 is not None:
            cross_x = x_from_u2.detach().cpu().numpy()
            cross_x_sym = symmetry_correction(cross_x)
            if hic_norm == "oe":
                cross_x_display = inverse_oe_matrix(cross_x_sym, testing_hic_mat_raw)
            elif hic_norm == "log1p":
                cross_x_display = np.expm1(cross_x_sym)
            else:
                cross_x_display = cross_x_sym
            np.savetxt(save_path_region + "_cross-reconstructed-x.txt", cross_x_sym, fmt="%.6f")
            model_testing_plot(
                x=testing_hic_mat_raw,
                y=cross_x_display,
                range_1=0,
                range_2=patch_bins,
                caption_model=previous_model_path + " [u2→x_reconstructor]",
                caption_data=experiment + ":" + testing_range,
                save_name=save_path_region + "_x-cross-reconstruction.png",
                log_op=True
            )
        np.savetxt(save_path_region + "_true-x.txt", testing_hic_mat_raw, fmt="%.6f")
        if use_label:
            np.savetxt(save_path_region + "_true-x-label.txt", testing_hic_label_raw, fmt="%.6f")
            model_testing_plot(
                x=testing_hic_label_raw,
                y=reconstructed_x_display,
                range_1=0,
                range_2=patch_bins,
                caption_model=previous_model_path,
                caption_data=experiment + ":" + testing_range,
                save_name=save_path_region + "_x-reconstruction-with-labels.png",
                log_op=True
            )
        elif reconstructed_x_display is not None:
            model_testing_plot(
                x=testing_hic_mat_raw,
                y=reconstructed_x_display,
                range_1=0,
                range_2=patch_bins,
                caption_model=previous_model_path,
                caption_data=experiment + ":" + testing_range,
                save_name=save_path_region + "_x-reconstruction.png",
                log_op=True
            )
        if topo_enabled and embed_1 is not None:
            plot_persistence_diagram(
                D_x=D_x_val,
                z=embed_1,
                output_filename=save_path_region + "_latent-representation.png",
                topo_cfg=topo_cfg
            )
            logger.info(f"[MAIN] Persistence diagram saved for {region_tag} "
                        f"(threshold={topo_cfg.threshold})")
            if embed_1 is not None and embed_1.shape[1] <= 3:
                plot_latent_space(embed_1,
                                  save_path_region + "_latent-space-u1.png",
                                  title_prefix="u1")
                logger.info(f"[MAIN] u1 latent space interactive HTML saved "
                            f"(dim={embed_1.shape[1]}) in {region_tag}")
            if u2 is not None and u2.shape[1] <= 3:
                plot_latent_space(u2.detach().cpu().numpy(),
                                  save_path_region + "_latent-space-u2.png",
                                  title_prefix="u2")
                logger.info(f"[MAIN] u2 latent space interactive HTML saved "
                            f"(dim={u2.shape[1]}) in {region_tag}")
        logger.info(f"[MAIN] Region {region_tag} outputs saved to {save_path_dir}")
    logger.info(f"[MAIN] All outputs saved to {save_path_dir}")
    logger.info(f"[MAIN] Validation completed successfully ({len(regions)} region(s))")

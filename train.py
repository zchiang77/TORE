from src.backward import *
from src.data import *
from src.ringHelper import *
from src.lossCompute import *
from arguments import *
from torch.utils.data import TensorDataset
from datetime import datetime
import cooler
import torch.multiprocessing as mp
import os
import sys
import logging
import random
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def select_best_lr(search_prefix, candidates, n_gpus, logger):
    best_lr, best_score = None, -float('inf')
    summary_path = f"{search_prefix}_summary.txt"
    with open(summary_path, "w") as sf:
        sf.write("Rank\tLR\tTrainFirst\tTrainLast\tTestFirst\tTestLast\t"
                 "TestReduction\tStatus\n")
        for i in range(n_gpus):
            path = f"{search_prefix}_lr{i}.txt"
            train_losses, test_losses = [], []
            try:
                with open(path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("Epoch"):
                            continue
                        parts = line.split("\t")
                        train_losses.append(float(parts[1]))
                        if len(parts) > 2 and parts[2] != "":
                            test_losses.append(float(parts[2]))
            except Exception as ex:
                logger.warning(f"[MAIN] Failed to read LR search record {path}: {ex}")
            train_finite = len(train_losses) >= 2 and all(np.isfinite(train_losses))
            test_ok = len(test_losses) >= 2 and all(np.isfinite(test_losses))
            status = "ok" if (train_finite and test_ok) else "diverged"
            if not (train_finite and test_ok):
                logger.warning(f"[MAIN] Candidate LR {candidates[i]} "
                               f"diverged/insufficient test points (file {path})")
                sf.write(f"{i}\t{candidates[i]}\t-\t-\t-\t-\t-\t{status}\n")
                continue
            reduction = (test_losses[0] - test_losses[-1]) / test_losses[0]
            logger.info(f"[MAIN] LR {candidates[i]}: test first={test_losses[0]:.6f} "
                        f"last={test_losses[-1]:.6f} reduction={reduction:.4%}")
            sf.write(f"{i}\t{candidates[i]}\t{train_losses[0]:.8f}\t"
                     f"{train_losses[-1]:.8f}\t{test_losses[0]:.8f}\t"
                     f"{test_losses[-1]:.8f}\t{reduction:.4%}\t{status}\n")
            if reduction > best_score:
                best_score, best_lr = reduction, candidates[i]
    if best_lr is None:
        best_lr = min(candidates)
        logger.warning(f"[MAIN] All candidates diverged, "
                       f"fallback to smallest LR {best_lr}")
    return best_lr
config_file = load_config("config.json")
ring_platform = config_file["info"]["platform"]
ring_version = config_file["info"]["version"]
now = datetime.now()
date_str = str(now.date())
hour_str = str(now.hour)
logging.basicConfig(filename=os.environ.get('RING_LOG_FILE', f'log/{ring_version}_{date_str}_train.log'), level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)
logger.info(f"Running Ring model (version {ring_version}) on platform {ring_platform}")
logger.info(f"Available GPUs are as follows:")
for i in range(torch.cuda.device_count()):
    gpu_name = torch.cuda.get_device_name(i)
    logger.info(f" - ({i+1}) {gpu_name}")
os.environ["USE_LIBUV"] = "0"
rs = 123
set_seed(rs)
logger.info(f"[MAIN] Random seed: {rs}")
parser = train_args()
args = parser.parse_args()
resolution = args.x_resolution
y_resolution = args.y_resolution if args.y_resolution is not None else resolution
logger.info(f"[MAIN] Hi-C (X) Resolution: {resolution}")
if y_resolution != resolution:
    logger.info(f"[MAIN] ChIP-seq (Y) Resolution: {y_resolution}")
patch_size = 1e+6 * args.patch
patch_bins = int(patch_size / resolution)
logger.info(f"[MAIN] Each patch has {patch_bins} samples equal to {patch_size}bp")
patch_overlap = 1e+6 * args.patch_overlap
patch_overlap_bins = int(patch_overlap / resolution)
logger.info(f"[MAIN] Each two patches have {patch_overlap_bins} samples overlap")
if args.num_gpus is not None:
    if args.num_gpus > torch.cuda.device_count():
        logger.info(f"[MAIN] Requested number of GPUs exceeds the available quantity.")
    world_size = min(args.num_gpus, torch.cuda.device_count())
else:
    logger.info(f"[MAIN] Missing parameters --num_gpus (-G)")
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
logger.info(f"[MAIN] Using {world_size} GPUs")
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
experiment1 = args.train_x1
experiment2 = args.train_x2
logger.info(f"[MAIN] Data X: {experiment1} and {experiment2}")
topo_enabled = args.use_topology
if topo_enabled:
    logger.info(f"[MAIN] Topological loss enabled")
else:
    logger.info(f"[MAIN] Topological loss disabled")
save_path_dir = args.model_save
if not os.path.exists(save_path_dir):
    os.makedirs(save_path_dir)
use_bins = args.use_bins
logger.info(f"[MAIN] Using {use_bins:.2%} bins")
topo_use_bins = args.topo_use_bins
logger.info(f"[MAIN] topo_use_bins (train-only topo block fraction): {topo_use_bins}")
hic_norm = args.hic_normalization
logger.info(f"[MAIN] Hi-C normalization: {hic_norm}")
ring_cfg = config_file.get("ring", {})
skip_zero_threshold = ring_cfg.get("skip_zero_threshold", 0.9)
logger.info(f"[MAIN] Zero-skip threshold: {skip_zero_threshold}")
if __name__ == '__main__':
    logger.info(f"[MAIN] First Hi-C file {experiment1}")
    cooler_path = "../data/HiC.GRCh38/" + experiment1 + "_" + str(resolution) + ".cool"
    cooler_file = cooler.Cooler(cooler_path)
    training_hic1, testing_hic1, training_pos, testing_pos, topology_threshold, \
        train_dist, test_dist = \
        training_hic_data(world_size, cooler_file, patch_bins, patch_overlap_bins,
                          use_bins, normalization=hic_norm,
                          zero_threshold=skip_zero_threshold)
    training_hic_flat1 = training_hic1.reshape(-1, patch_bins)
    testing_hic_flat1 = testing_hic1.reshape(-1, patch_bins)
    n_train_blocks = train_dist.shape[0]
    n_test_blocks = test_dist.shape[0]
    logger.info(f"[MAIN] Spatial distance matrices: "
                f"{n_train_blocks} train blocks, {n_test_blocks} test blocks, "
                f"shape={train_dist.shape[1:]} each")
    training_dataset = testing_dataset = None
    chip_meta = None
    if args.reconstruction:                       
        if args.use_model is not None:
            logger.info(f"[MAIN] Using an existing pretrained network {args.use_model}")
            previous_model_path = args.use_model
        else:
            previous_model_path = None
        pretrained_model_path = None
        if experiment2 is not None:
            logger.info(f"[MAIN] Second Hi-C file {experiment2}")
            cooler_path = "../data/HiC.GRCh38/" + experiment2 + "_" + str(resolution) + ".cool"
            cooler_file = cooler.Cooler(cooler_path)
            training_hic2, testing_hic2, _, _, topology_threshold, \
                train_dist2, test_dist2 = \
                training_hic_data(world_size, cooler_file, patch_bins, patch_overlap_bins,
                                  use_bins, normalization=hic_norm,
                                  zero_threshold=skip_zero_threshold)
            training_hic_flat2 = training_hic2.reshape(-1, patch_bins)
            testing_hic_flat2 = testing_hic2.reshape(-1, patch_bins)
            training_hic_flat = torch.cat([training_hic_flat1, training_hic_flat2], dim=0)
            testing_hic_flat = torch.cat([testing_hic_flat1, testing_hic_flat2], dim=0)
            train_dist = torch.cat([train_dist, train_dist2], dim=0)
            test_dist = torch.cat([test_dist, test_dist2], dim=0)
            base_name = f"Recon_Experiment={experiment1}+{experiment2}"
        else:
            logger.info(f"[MAIN] Second Hi-C file is not set")
            training_hic_flat = training_hic_flat1
            testing_hic_flat = testing_hic_flat1
            base_name = f"Recon_Experiment={experiment1}"
        logger.info(f"[MAIN] Data for training has {training_hic_flat.shape[0]} samples")
        logger.info(f"[MAIN] Data for testing has {testing_hic_flat.shape[0]} samples")
        training_hic_label = torch.zeros(training_hic_flat.shape[0], dtype=torch.float32)
        testing_hic_label = torch.zeros(testing_hic_flat.shape[0], dtype=torch.float32)
        use_label = False
        training_chip = torch.zeros(training_hic_flat.shape[0], dtype=torch.float32)
        train_dist_flat = train_dist.reshape(-1, patch_bins)
        test_dist_flat = test_dist.reshape(-1, patch_bins)
        training_dataset = TensorDataset(training_hic_flat, training_chip, training_hic_label, train_dist_flat)
        testing_chip = torch.zeros(testing_hic_flat.shape[0], dtype=torch.float32)
        testing_dataset = TensorDataset(testing_hic_flat, testing_chip, testing_hic_label, test_dist_flat)
        save_path = save_path_dir + (f"{base_name}_Topological" if topo_enabled else base_name)
        model_type = "x_reconstruction"
        y_p = 0
        debug_mode = False
    else:
        training_hic_label = torch.zeros(training_hic_flat1.shape[0], dtype=torch.float32)
        testing_hic_label = torch.zeros(testing_hic_flat1.shape[0], dtype=torch.float32)
        use_label = False
        if args.use_pretrained is not None:
            logger.info(f"[MAIN] Using pretrained network {args.use_pretrained}")
            pretrained_model_path = args.use_pretrained
        else:
            pretrained_model_path = None
        if args.use_model is not None:
            logger.info(f"[MAIN] Loading existing model {args.use_model}")
            previous_model_path = args.use_model
        else:
            previous_model_path = None
        if args.train_y is not None:
            chip_file_path = args.train_y
        else:
            chip_file_path = f"../data/ChIP.GRCh38/resolution{y_resolution}/"
        entire_chip, chip_scaler = training_chip_data(chip_file_path, return_scaler=True)
        chrom_offsets, chrom_nbins, _ = build_chip_chrom_offsets(chip_file_path)
        bins_table = cooler_file.bins()[:]
        chip_meta = {
            'bins_table':     bins_table,
            'entire_chip':    entire_chip,
            'chip_scaler':    chip_scaler,
            'chrom_offsets':  chrom_offsets,
            'chrom_nbins':    chrom_nbins,
            'patch_bins':     patch_bins,
            'x_resolution':   resolution,
            'y_resolution':   y_resolution,
        }
        train_dist_flat = train_dist.reshape(-1, patch_bins)
        test_dist_flat = test_dist.reshape(-1, patch_bins)
        training_dataset = TensorDataset(training_hic_flat1, training_pos.repeat_interleave(patch_bins, dim=0), train_dist_flat)
        testing_dataset  = TensorDataset(testing_hic_flat1, testing_pos.repeat_interleave(patch_bins, dim=0), test_dist_flat)
        y_p = entire_chip.shape[1]
        debug_mode = args.debug
        logger.info(f"[MAIN] ChIP-seq shape: {entire_chip.shape}, "
                    f"{len(chrom_offsets)} chromosomes detected")
        if args.x_pred_y:
            base_name = f"Pred_Experiment={experiment1}"
            model_type = "y_prediction"
        elif args.y_pred_x:
            base_name = f"Rev_Experiment={experiment1}"
            model_type = "x_prediction"
        elif args.ring:
            base_name = f"Ring_Experiment={experiment1}"
            model_type = "ring"
        elif args.y_reconstruction:
            base_name = f"YRecon_Experiment={experiment1}"
            model_type = "y_reconstruction"
        elif args.halfring:
            base_name = f"HalfRing_Experiment={experiment1}"
            model_type = "halfring"
        else:
            raise NotImplementedError
        save_path = save_path_dir + base_name
    logger.info(f"[MAIN] Debug mode: {debug_mode}")
    save_tensor_txt(training_pos, save_path_dir + base_name + "_trainingPos.txt")
    save_tensor_txt(testing_pos, save_path_dir + base_name + "_testingPos.txt")
    if args.lr_search:
        if sys.platform == 'win32' and world_size > 1:
            logger.warning("[MAIN] Windows  LR spawn "
                           " --learning_rate "
                           "")
        else:
            assert args.lr_candidates is not None, \
                "--lr_search  --lr_candidates"
            candidates = [float(x) for x in args.lr_candidates.split(",")]
            assert len(candidates) == world_size, \
                f"--lr_candidates ({len(candidates)}) GPU ({world_size})"
            logger.info(f"[MAIN] LR search phase: candidates={candidates}, "
                        f"{args.lr_search_iters} iters each")
            search_prefix = save_path + "_lrsearch"
            mp.spawn(run_backward,
                     args=(1, model_type, config_file, training_dataset, testing_dataset,
                           use_label, patch_bins, topo_enabled, topology_threshold, patch_bins, y_p,
                           None, save_path, pretrained_model_path, args.pretrained_module,
                           None,                                                     
                           args.learning_strategy, args.model_freeze, args.loss_components,
                           args.optimizer, debug_mode, chip_meta, args.hic_base_loss, hic_norm,
                           args.pretrained_modules, args, args.preset,
                           train_dist, test_dist,
                           True, candidates, args.lr_search_iters, search_prefix),
                     nprocs=world_size)
            best_lr = select_best_lr(search_prefix, candidates, world_size, logger)
            logger.info(f"[MAIN] LR search complete, best LR = {best_lr}, "
                        f"starting formal DDP training from scratch")
            args.learning_rate = best_lr
            previous_model_path = None            
    if world_size > 1:
        mp.spawn(run_backward,
                 args=(world_size, model_type, config_file, training_dataset, testing_dataset,
                       use_label, patch_bins, topo_enabled, topology_threshold, patch_bins, y_p,
                       previous_model_path, save_path, pretrained_model_path, args.pretrained_module,
                       args.learning_rate,
                       args.learning_strategy, args.model_freeze, args.loss_components,
                       args.optimizer, debug_mode,
                       chip_meta, args.hic_base_loss, hic_norm,
                       args.pretrained_modules,
                       args, args.preset,
                       train_dist, test_dist),                                                          
                 nprocs=world_size)         
    else:
        run_backward(rank=0, world_size=1, model_type=model_type, config=config_file,
                     dataset=training_dataset, test_dataset=testing_dataset, dataset_label=use_label,
                     batch_size=patch_bins, topological=topo_enabled, topo_thresh=topology_threshold,
                     x_p=patch_bins, y_p=y_p, load_path=previous_model_path, save_path=save_path,
                     pretrained=pretrained_model_path, pretrained_module=args.pretrained_module,
                     customized_lr=args.learning_rate,
                     lr_strategy=args.learning_strategy, freeze=args.model_freeze,
                     loss_components=args.loss_components,
                     optimizer=args.optimizer, debug_mode=debug_mode,
                     chip_meta=chip_meta, hic_base_loss=args.hic_base_loss, hic_norm=hic_norm,
                     pretrained_modules=args.pretrained_modules,
                     train_args=args, preset=args.preset,
                     train_dist=train_dist, test_dist=test_dist)                                                          
    previous_model_path = save_path
    torch.cuda.empty_cache()
    logger.info(f"[MAIN] " + monitor_resources())

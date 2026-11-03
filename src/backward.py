from torch.amp import autocast, GradScaler
from model import *
from src.test import *
from src.visualize import *
from src.lossCompute import *
from src.ringHelper import *
from src.data import inverse_transform_chip_batch
from src.topology import count_h1_from_dist_matrix, count_h0_h1_from_dist_matrix, TopoConfig, get_topo_config, is_degenerate_distance_matrix
from src.model_builder import build_model
from torch.nn.parallel  import DistributedDataParallel as DDP
from torch.utils.data.distributed  import DistributedSampler
import torch.optim
import numpy as np
import logging
import time
import subprocess
import shlex
import sys
import os
logging.basicConfig(filename=os.environ.get('RING_LOG_FILE', 'train.log'), level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)
def _to_device_block(D_x_flat, batch_size, rank):
    D_x = D_x_flat.to(rank)
    if D_x.shape != (batch_size, batch_size):
        raise RuntimeError(
            f"[RANK {rank}] D_x_flat shape {tuple(D_x.shape)} != "
            f"({batch_size}, {batch_size}); batch_size  patch_bins"
            f" batch  block "
        )
    return D_x
def _format_batch_pos(pos_batch, chip_meta):
    if pos_batch is None or chip_meta is None:
        return "N/A (reconstruction mode)"
    hic_bin = int(pos_batch[0][0].item()) if torch.is_tensor(pos_batch) else int(pos_batch[0][0])
    patch_bins = chip_meta['patch_bins']
    try:
        bins_table = chip_meta['bins_table']
        chrom = str(bins_table.iloc[hic_bin]['chrom'])
        g_start = int(bins_table.iloc[hic_bin]['start'])
        g_end = int(bins_table.iloc[hic_bin + patch_bins - 1]['end'])
        return f"{chrom}:{g_start}-{g_end} (hic_bin={hic_bin})"
    except Exception:
        return f"hic_bin={hic_bin}"
def _extract_hic_bin(pos_batch):
    if pos_batch is None:
        return None
    try:
        val = pos_batch[0][0]
        return int(val.item()) if torch.is_tensor(val) else int(val)
    except Exception:
        return None
def _block_in_topo_subset(hic_bin, topo_use_bins, seed=42):
    if topo_use_bins >= 1.0:
        return True
    if hic_bin is None:
        return True
    rng = np.random.default_rng(seed + int(hic_bin))
    return float(rng.random()) < topo_use_bins
def backward_initialize(world_size, rank, record_file_prefix,
                        lr_search=False, search_save_prefix=None):
    record_file_training = record_file_prefix + "_loss.train.txt"
    record_file_testing = record_file_prefix + "_loss.test.txt"
    device_id = rank
    if world_size > 1:
        setup(rank, world_size)
        torch.cuda.set_device(rank)
    else:
        torch.cuda.set_device(rank)
        rank = "cuda:" + str(rank)
    search_record_file = None
    if lr_search:
        search_record_file = search_save_prefix + f"_lr{device_id}.txt"
        try:
            with open(search_record_file, "a") as f:
                if f.tell() == 0:
                    f.write("Epoch\tTrainLoss\tTestLoss\n")
            logger.info(f"[RANK {device_id}] LR search record: {search_record_file}")
        except Exception as ex:
            logger.info(f"[RANK {device_id}] Search file initialization error: {ex}")
        return search_record_file, record_file_testing
    try:
        if rank == 0 or world_size == 1:
            with open(record_file_training, "a") as f:
                if f.tell() == 0:
                    f.write("Epoch\tLR\tLoss\t"
                            "x_pred_loss\ty_pred_loss\t"
                            "x_recon_loss\ty_recon_loss\t"
                            "sim_loss\ttopo_loss\n")
            with open(record_file_testing, "a") as f:
                if f.tell() == 0:
                    f.write("Epoch\tLoss\t"
                            "x_pred_loss\ty_pred_loss\t"
                            "x_recon_loss\ty_recon_loss\t"
                            "sim_loss\ttopo_loss\n")
    except Exception as ex:
        logger.info(f"[RANK {rank}] File initialization error: {ex}")
    else:
        logger.info(f"[RANK {rank}] Initialized successfully")
    return record_file_training, record_file_testing
def specify_use_model(world_size, rank, load_path, pretrained, pretrained_module, model,
                      pretrained_modules=None):
    load_pretrained_is_or_not = False                                                                              
    if load_path is not None:
        logger.info(f"[RANK {rank}] Existing model {load_path} from the previous Hi-C file")
        exist_model_use = load_path
    elif pretrained is not None:
        logger.info(f"[RANK {rank}] Existing model {pretrained} from pretraining")
        exist_model_use = pretrained
        load_pretrained_is_or_not = True
    else:
        logger.info(f"[RANK {rank}] No existing model to use")
        exist_model_use = None
    if exist_model_use is not None:
        exist_model_use = os.path.abspath(exist_model_use.replace("\\", "/"))
        if world_size > 1:
            torch.distributed.barrier(device_ids=[rank])
            logger.info(f"[RANK {rank}] All processes synchronized")
        if rank == 0:
            if not os.path.exists(exist_model_use):
                logger.info(f"[RANK {rank}] Model {exist_model_use} not found.")
            checkpoint = torch.load(exist_model_use, map_location="cpu", weights_only=True)
            state_dict = checkpoint.get('model', None)
            logger.info(f"[RANK {rank}] Loading {exist_model_use}")
            if load_pretrained_is_or_not:
                if pretrained_module is None:
                    raise ValueError(
                        "--pretrained_module must be specified when --use_pretrained is set. "
                        "E.g. 'y_prediction', 'x_prediction', 'x_reconstruction'. "
                        "Check the model structure output for available module names."
                    )
                target_module = model.get_submodule(pretrained_module)
                target_keys = set(target_module.state_dict().keys())
                filtered_state_dict = {
                    k: v for k, v in state_dict.items() if k in target_keys
                }
                skipped_keys = set(state_dict.keys()) - target_keys
                if skipped_keys:
                    logger.info(
                        f"[RANK {rank}] Skipped {len(skipped_keys)} keys not in "
                        f"'{pretrained_module}': {sorted(skipped_keys)[:5]}..."
                    )
                missing_keys = target_keys - set(filtered_state_dict.keys())
                if missing_keys:
                    logger.error(
                        f"[RANK {rank}] FATAL: {len(missing_keys)} keys missing in checkpoint "
                        f"for '{pretrained_module}' (strict=True will raise RuntimeError): "
                        f"{sorted(missing_keys)[:5]}..."
                    )
                target_module.load_state_dict(filtered_state_dict, strict=True)
                logger.info(
                    f"[RANK {rank}] Loaded pretrained weights into module: {pretrained_module} "
                    f"({len(filtered_state_dict)}/{len(state_dict)} keys from checkpoint)"
                )
            else:
                model.load_state_dict(state_dict)
            logger.info(f"[RANK {rank}] Model {exist_model_use} loaded on host process")
        if world_size > 1:
            for param in model.parameters():
                torch.distributed.broadcast(param.data, src=0)
            if rank != 0:
                logger.info(f"[RANK {rank}] Model {exist_model_use} broadcasted")
    if pretrained_modules is not None and rank == 0:
        pairs = [p.strip() for p in pretrained_modules.split(",") if p.strip()]
        logger.info(f"[RANK {rank}] Loading {len(pairs)} pretrained modules: {pairs}")
        for pair in pairs:
            parts = [x.strip() for x in pair.split(";")]
            if len(parts) == 3:
                module_name, source_prefix, path = parts
            elif len(parts) == 2:
                module_name, path = parts
                source_prefix = None
            else:
                raise ValueError(
                    f"Invalid pretrained_modules format: '{pair}'. "
                    f"Expected 'module_name;path' or "
                    f"'module_name;source_prefix;path'."
                )
            abs_path = os.path.abspath(path.replace("\\", "/"))
            if not os.path.exists(abs_path):
                logger.error(f"[RANK {rank}] Pretrained model {abs_path} not found, skipping.")
                continue
            checkpoint = torch.load(abs_path, map_location="cpu", weights_only=True)
            state_dict = checkpoint.get('model', None)
            target_module = model.get_submodule(module_name)
            target_keys = set(target_module.state_dict().keys())
            if source_prefix is not None:
                if not source_prefix.endswith("."):
                    prefix_dot = source_prefix + "."
                else:
                    prefix_dot = source_prefix
                filtered_state_dict = {}
                remapped_count = 0
                for k, v in state_dict.items():
                    if k.startswith(prefix_dot):
                        stripped_key = k[len(prefix_dot):]
                        if stripped_key in target_keys:
                            filtered_state_dict[stripped_key] = v
                            remapped_count += 1
                prefixed_in_ckpt = [k for k in state_dict.keys() if k.startswith(prefix_dot)]
                prefixed_loaded = [prefix_dot + k for k in filtered_state_dict.keys()]
                skipped_prefixed = set(prefixed_in_ckpt) - set(prefixed_loaded)
                non_prefix_keys = set(k for k in state_dict.keys() if not k.startswith(prefix_dot))
                if non_prefix_keys:
                    logger.info(
                        f"[RANK {rank}] Ignored {len(non_prefix_keys)} keys not matching "
                        f"prefix '{source_prefix}': {sorted(non_prefix_keys)[:5]}..."
                    )
                if skipped_prefixed:
                    logger.info(
                        f"[RANK {rank}] Skipped {len(skipped_prefixed)} prefix-matched "
                        f"keys not in '{module_name}': "
                        f"{sorted(skipped_prefixed)[:5]}..."
                    )
                missing = target_keys - set(filtered_state_dict.keys())
                if missing:
                    logger.warning(
                        f"[RANK {rank}] Missing {len(missing)} keys for "
                        f"'{module_name}': {sorted(missing)[:5]}..."
                    )
                key_info = (f" {remapped_count} keys remapped "
                            f"[prefix: '{source_prefix}']")
            else:
                filtered_state_dict = {
                    k: v for k, v in state_dict.items() if k in target_keys
                }
                skipped = set(state_dict.keys()) - target_keys
                if skipped:
                    logger.info(
                        f"[RANK {rank}] Skipped {len(skipped)} keys for "
                        f"'{module_name}': {sorted(skipped)[:5]}..."
                    )
                missing = target_keys - set(filtered_state_dict.keys())
                if missing:
                    logger.warning(
                        f"[RANK {rank}] Missing {len(missing)} keys for "
                        f"'{module_name}': {sorted(missing)[:5]}..."
                    )
                key_info = ""
            target_module.load_state_dict(filtered_state_dict, strict=True)
            logger.info(
                f"[RANK {rank}] Loaded pretrained weights into module: "
                f"{module_name} from {abs_path} "
                f"({len(filtered_state_dict)}/{len(state_dict)} keys){key_info}"
            )
    if pretrained_modules is not None and world_size > 1:
        torch.distributed.barrier(device_ids=[rank])
        for param in model.parameters():
            torch.distributed.broadcast(param.data, src=0)
    return model
def apply_freeze(rank, model, freeze):
    if freeze is None:
        logger.info(f"[RANK {rank}] No weights frozen.")
    else:
        modules = [m.strip() for m in freeze.split(",") if m.strip()]
        frozen_count = 0
        for module_path in modules:
            try:
                sub_module = model.get_submodule(module_path)
            except AttributeError:
                logger.error(f"[RANK {rank}] Module path '{module_path}' not found in model, skipping.")
                continue
            for param in sub_module.parameters():
                param.requires_grad = False
            frozen_count += 1
            logger.info(f"[RANK {rank}] Frozen module: {module_path}")
        logger.info(f"[RANK {rank}] Total frozen modules: {frozen_count}")
    return model
def model_initialize(world_size, rank, load_path, pretrained, pretrained_module, model, config,
                     customized_lr, dataset, batch_size, freeze=None, optimize='SGD',
                     lr_strategy='step', pretrained_modules=None):
    model = specify_use_model(world_size, rank, load_path, pretrained, pretrained_module, model,
                              pretrained_modules=pretrained_modules)
    summarize_parameters_size(rank, model, logger)
    model = apply_freeze(rank, model, freeze)
    if world_size > 1:
        if not dist.is_initialized():
            raise RuntimeError(
                "Distributed process group is not initialized. "
                "Ensure dist.init_process_group() is called before wrapping the model with DDP."
            )
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    else:
        model = model
    sampler = BlockDistributedSampler(dataset,
                                       num_replicas=world_size,
                                       rank=rank,
                                       block_size=batch_size,
                                       drop_last=True) if world_size > 1 else None
    train_cfg = config["ring"]
    if customized_lr is not None:
        current_lr = customized_lr
        if rank == 0:
            logger.info(f"[RANK {rank}] Learning rate {current_lr} from the console")
    else:
        current_lr = train_cfg["learning_rate"]
        if rank == 0:
            logger.info(f"[RANK {rank}] Learning rate {current_lr} from config")
    logger.info(f"[RANK {rank}] Using {optimize} optimizer with {lr_strategy} strategy")
    if optimize == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=current_lr,
            momentum=0.9,
            weight_decay=train_cfg["weight_decay"]
        )
    elif optimize == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=current_lr,
            weight_decay=train_cfg["weight_decay"]
        )
    elif optimize == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=current_lr,
            betas=(0.9, 0.95),
            weight_decay=train_cfg["weight_decay"]
        )
    elif optimize == 'RAdam':
        optimizer = torch.optim.RAdam(
            model.parameters(),
            lr=current_lr,
            weight_decay=train_cfg["weight_decay"]
        )
    else:
        logger.error(f"[RANK {rank}] Optimizer is not supported")
        optimizer = None
    scaler = GradScaler(device='cuda', growth_interval=1024)
    if lr_strategy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=train_cfg["lr_decay"]["step_size"],
            gamma=train_cfg["lr_decay"]["gamma"]
        )
    elif lr_strategy == 'warm':
        warm_steps = train_cfg["lr_warm"]["warm_steps"]
        def lr_lambda(step):
            if step < warm_steps:
                return step / warm_steps
            else:
                return 1.0
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        logger.error(f"[RANK {rank}] LR strategy is not supported")
        scheduler = None
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=False
    )
    return model, scaler, optimizer, dataloader, scheduler, train_cfg
def model_save(epoch, model, optimizer, save_path):
    state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save({
        'epoch': epoch,
        'model': state_dict,
        'optimizer': optimizer.state_dict()
    }, save_path + "_epoch" + str(epoch + 1) + ".pth")
_validation_process = None
def _build_validation_cmd(checkpoint_path, val_output_dir, train_args):
    model_flags = []
    if getattr(train_args, 'reconstruction', False):
        model_flags.append("--reconstruction")
    elif getattr(train_args, 'x_pred_y', False):
        model_flags.append("--x_pred_y")
    elif getattr(train_args, 'y_pred_x', False):
        model_flags.append("--y_pred_x")
    elif getattr(train_args, 'ring', False):
        model_flags.append("--ring")
    elif getattr(train_args, 'y_reconstruction', False):
        model_flags.append("--y_reconstruction")
    elif getattr(train_args, 'halfring', False):
        model_flags.append("--halfring")
    else:
        raise ValueError("Cannot determine model_type from train_args")
    patch_val = getattr(train_args, 'patch', 4)
    block_bp = int(patch_val * 1000000)                                                         
    starts = [22000000, 24000000, 26000000, 28000000]
    ranges_str = ",".join(f"{s}-{s + block_bp}" for s in starts)
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "..", "validation.py"),
        "--use_model",        checkpoint_path,
        "--test_x",           getattr(train_args, 'train_x1', '4DNFIPAF4PYT'),
        "--chr",              "22",
        "--x_resolution",     str(getattr(train_args, 'x_resolution', 50000)),
        "--patch",            str(patch_val),
        "--patch_overlap",    str(getattr(train_args, 'patch_overlap', 0)),
        "--model_save",       val_output_dir,
        "--hic_normalization", getattr(train_args, 'hic_normalization', 'oe'),
        "--port",             str(getattr(train_args, 'port', '30000')),
        "--preset",           str(getattr(train_args, 'preset', 'SMALL')),
        "--device",           "cpu",
        "--no_whole_chrom",
        "--ranges",           ranges_str,
    ] + model_flags
    if getattr(train_args, 'use_topology', False):
        cmd.append("--use_topology")
    y_res = getattr(train_args, 'y_resolution', None)
    if y_res is not None:
        cmd.extend(["--y_resolution", str(y_res)])
    test_y = getattr(train_args, 'train_y', None)
    if test_y is not None and not getattr(train_args, 'reconstruction', False):
        cmd.extend(["--test_y", test_y])
    return cmd
def trigger_external_validation(checkpoint_path, epoch, train_args, detach=False):
    global _validation_process
    if _validation_process is not None:
        prev_pid = _validation_process.pid
        prev_status = _validation_process.poll()
        if prev_status is None:                                    
            logger.warning(
                f"[RANK 0] Previous external validation (PID {prev_pid}) is still "
                f"running  killing it to launch validation for epoch {epoch + 1}."
            )
            _kill_process_tree(_validation_process)
        else:
            logger.info(
                f"[RANK 0] Previous external validation (PID {prev_pid}) "
                f"finished with exit code {prev_status}."
            )
        _validation_process = None
    checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    val_output_dir = os.path.join(
        os.path.dirname(checkpoint_path),
        f"Validation_{checkpoint_name}/"
    )
    os.makedirs(val_output_dir, exist_ok=True)
    try:
        cmd = _build_validation_cmd(checkpoint_path, val_output_dir, train_args)
    except ValueError as e:
        logger.warning(f"[RANK 0] Skipping external validation: {e}")
        return
    logger.info(
        f"[RANK 0] Launching external validation for epoch {epoch + 1} "
        f"(checkpoint: {checkpoint_path}, output → {val_output_dir})"
    )
    val_log_fd = None
    try:
        val_log_path = os.path.join(val_output_dir, "validation.log")
        val_log_fd = open(val_log_path, "w")
        _validation_process = subprocess.Popen(
            cmd,
            stdout=val_log_fd,
            stderr=subprocess.STDOUT,
            start_new_session=detach,                                                
        )
        val_log_fd.close()
        cont_msg = ("Training will exit; validation runs detached."
                    if detach else "Training continues.")
        logger.info(
            f"[RANK 0] Validation subprocess spawned (PID {_validation_process.pid}). "
            f"{cont_msg}"
        )
    except Exception as e:
        logger.error(f"[RANK 0] Failed to launch external validation: {e}")
        if val_log_fd is not None and not val_log_fd.closed:
            val_log_fd.close()
        _validation_process = None
def _kill_process_tree(proc):
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True, timeout=10
            )
        else:
            proc.kill()
            proc.wait(timeout=5)
    except Exception as e:
        logger.warning(f"[RANK 0] Could not cleanly kill PID {proc.pid}: {e}")
def model_loss_plot(rank, txt_file):
    logger.info(f"[RANK {rank}] Reading log {txt_file}, "
                f"plot saving to {os.path.dirname(txt_file)}")
    visualizer = TrainingMetricsVisualizer(
        file_path=txt_file,             
        output_dir=os.path.dirname(txt_file),           
        figsize=(12, 7),           
        dpi=120          
    )
    visualizer.generate_plots()
_BN_COLLAPSE_TOKEN_STD = 1e-3
@torch.no_grad()
def _bottleneck_stats(z):
    if z is None:
        return None
    z = z.detach()
    if z.dim() < 2:
        token_std = z.std().item()
        dim_std = token_std
        n_dims = 1
        dead_dims = int(token_std < _BN_COLLAPSE_TOKEN_STD)
        return {
            'shape': tuple(z.shape),
            'abs_mean': z.abs().mean().item(),
            'token_std': token_std,
            'dim_std': dim_std,
            'dead_dims': dead_dims,
            'n_dims': n_dims,
        }
    per_dim_std = z.std(dim=0)                                   
    token_std = per_dim_std.mean().item()                             
    dim_std = z.std(dim=1).mean().item()                              
    n_dims = z.shape[1]
    dead_dims = int((per_dim_std < _BN_COLLAPSE_TOKEN_STD).sum().item())
    return {
        'shape': tuple(z.shape),
        'abs_mean': z.abs().mean().item(),
        'token_std': token_std,
        'dim_std': dim_std,
        'dead_dims': dead_dims,
        'n_dims': n_dims,
    }
def monitor_bottleneck(u1, u2, rank, epoch, batch_idx, logger, model_type):
    has_u1 = u1 is not None
    has_u2 = u2 is not None
    if not (has_u1 or has_u2):
        return
    log_this = (epoch == 0 and batch_idx == 0) or \
               ((epoch + 1) % 50 == 0 and batch_idx == 0)
    if not log_this:
        return
    s1 = _bottleneck_stats(u1)
    s2 = _bottleneck_stats(u2)
    for name, s in (("u1(x)", s1), ("u2(y)", s2)):
        if s is None:
            continue
        logger.info(
            f"[RANK {rank}] [BOTTLENECK] epoch {epoch + 1} batch {batch_idx + 1} "
            f"{name}: shape={s['shape']}, "
            f"abs_mean={s['abs_mean']:.6f}, "
            f"token_std={s['token_std']:.6f} (across tokens), "
            f"dim_std={s['dim_std']:.6f} (across dims), "
            f"dead_dims={s['dead_dims']}/{s['n_dims']}"
        )
    if has_u1 and has_u2 and s1 is not None and s2 is not None:
        try:
            u1d, u2d = u1.detach(), u2.detach()
            if u1d.shape == u2d.shape:
                diff = (u1d - u2d).norm().item() / (u1d.norm().item() + 1e-12)
                logger.info(
                    f"[RANK {rank}] [BOTTLENECK] epoch {epoch + 1} batch {batch_idx + 1} "
                    f"u1-u2 rel_L2={diff:.6f} "
                    f"( + token_std  = ;  + token_std  = )"
                )
        except Exception as ex:
            logger.warning(f"[RANK {rank}] [BOTTLENECK] cross-modal stats failed: {ex}")
    for name, s in (("u1", s1), ("u2", s2)):
        if s is None:
            continue
        if s['token_std'] < _BN_COLLAPSE_TOKEN_STD:
            logger.warning(
                f"[RANK {rank}] [BOTTLENECK] ⚠ {name} : "
                f"token_std={s['token_std']:.6f} < {_BN_COLLAPSE_TOKEN_STD}, "
                f"dead_dims={s['dead_dims']}/{s['n_dims']}"
                f" sim_loss  0  sim_weight "
                f" sim "
            )
def run_backward(rank, world_size, model_type, config, dataset, test_dataset, dataset_label,
                 batch_size, topological, topo_thresh, x_p=0, y_p=0, load_path=None,
                 save_path=None, pretrained=None, pretrained_module=None,
                 customized_lr=None, lr_strategy="step", freeze=None,
                 loss_components=None, optimizer=None, debug_mode=False,
                 chip_meta=None, hic_base_loss="mse", hic_norm=None,
                 pretrained_modules=None,
                 train_args=None, preset="SMALL",
                 train_dist=None, test_dist=None,
                 lr_search=False, lr_candidates=None, max_iters=None,
                 search_save_prefix=None):
    record_file, record_test = backward_initialize(
        world_size, rank, save_path,
        lr_search=lr_search, search_save_prefix=search_save_prefix)
    if lr_search:
        customized_lr = lr_candidates[rank]
        logger.info(f"[RANK {rank}] LR search worker: lr={customized_lr}, "
                    f"max_iters={max_iters}")
    if loss_components is not None:
        loss_components = [c.strip() for c in loss_components.split(",") if c.strip()]
    logger.info(f"[RANK {rank}] Loss components: {loss_components}")
    logger.info(f"[RANK {rank}] Hi-C base loss: {hic_base_loss}")
    logger.info(f"[RANK {rank}] Hi-C normalization: {hic_norm}")
    sim_mode = getattr(train_args, 'sim_mode', 'mse') if train_args is not None else "mse"
    logger.info(f"[RANK {rank}] Sim mode: {sim_mode}")
    if (not lr_search) and train_dist is not None and train_dist.shape[0] > 0:
        h0_counts_thresh = []
        h1_counts_thresh = []
        h0_counts_full = []
        h1_counts_full = []
        for bi in range(train_dist.shape[0]):
            D_block = train_dist[bi].numpy()
            n_h0_thresh, n_h1_thresh = count_h0_h1_from_dist_matrix(D_block, topo_thresh)
            n_h0_full, n_h1_full = count_h0_h1_from_dist_matrix(D_block, threshold=1e10)
            h0_counts_thresh.append(n_h0_thresh)
            h1_counts_thresh.append(n_h1_thresh)
            h0_counts_full.append(n_h0_full)
            h1_counts_full.append(n_h1_full)
        if rank == 0:
            n_blocks = train_dist.shape[0]
            logger.info(f"[RANK {rank}] H0 & H1 counts per distance matrix block "
                        f"(block shape: {train_dist.shape[1]}*{train_dist.shape[2]}):")
            for bi in range(n_blocks):
                logger.info(f"  Block {bi}: {h0_counts_thresh[bi]} H0 components, "
                            f"{h1_counts_thresh[bi]} H1 loops "
                            f"(thresh={topo_thresh}), "
                            f"{h0_counts_full[bi]} H0 components, "
                            f"{h1_counts_full[bi]} H1 loops (no filter)")
            logger.info(f"[RANK {rank}] H0 summary (thresh={topo_thresh}): "
                        f"total={sum(h0_counts_thresh)}, "
                        f"mean={np.mean(h0_counts_thresh):.1f}, "
                        f"min={min(h0_counts_thresh)}, max={max(h0_counts_thresh)}")
            logger.info(f"[RANK {rank}] H1 summary (thresh={topo_thresh}): "
                        f"total={sum(h1_counts_thresh)}, "
                        f"mean={np.mean(h1_counts_thresh):.1f}, "
                        f"min={min(h1_counts_thresh)}, max={max(h1_counts_thresh)}")
            logger.info(f"[RANK {rank}] H0 summary (no filter): "
                        f"total={sum(h0_counts_full)}, "
                        f"mean={np.mean(h0_counts_full):.1f}, "
                        f"min={min(h0_counts_full)}, max={max(h0_counts_full)}")
            logger.info(f"[RANK {rank}] H1 summary (no filter): "
                        f"total={sum(h1_counts_full)}, "
                        f"mean={np.mean(h1_counts_full):.1f}, "
                        f"min={min(h1_counts_full)}, max={max(h1_counts_full)}")
    else:
        if (not lr_search) and rank == 0:
            logger.info(f"[RANK {rank}] No spatial distance matrices available, "
                        f"skipping H0 & H1 count")
    y_length = None
    if chip_meta is not None:
        y_length = chip_meta['patch_bins'] * chip_meta['x_resolution'] // chip_meta['y_resolution']
    model = build_model(model_type, config, x_p, y_p, rank, y_length=y_length, preset=preset)
    model, scaler, optimizer, dataloader, scheduler, train_cfg = model_initialize(
        world_size, rank, load_path, pretrained, pretrained_module, model, config, customized_lr,
        dataset, batch_size, freeze=freeze, optimize=optimizer, lr_strategy=lr_strategy,
        pretrained_modules=pretrained_modules
    )
    topo_cfg = get_topo_config(train_cfg) if topological else None
    if topo_cfg is not None and train_args is not None:
        tub = float(getattr(train_args, 'topo_use_bins', 1.0))
        topo_cfg.topo_use_bins = float(min(max(tub, 1e-6), 1.0))
    if topo_cfg is not None and rank == 0:
        logger.info(f"[RANK {rank}] TopoConfig: threshold={topo_cfg.threshold}, "
                    f"h0_weight={topo_cfg.h0_weight}, "
                    f"smooth_weight={topo_cfg.smooth_weight}, gamma={topo_cfg.gamma}, "
                    f"topo_use_bins={topo_cfg.topo_use_bins} (train-only)")
    ext_val_cfg = train_cfg.get("external_validation", {})
    ext_val_enabled = ext_val_cfg.get("enabled", False) if isinstance(ext_val_cfg, dict) else bool(ext_val_cfg)
    if rank == 0:
        logger.info(f"[RANK {rank}] External validation enabled: {ext_val_enabled}")
    if rank == 0 and not lr_search:
        model_str_print_path = save_path + "_model-structure.txt"
        with open(model_str_print_path, 'w') as f:
            f.write(str(model))
        logger.info(f"[RANK {rank}] Model structure printed to {model_str_print_path}")
    itr = train_cfg["epochs_topology"] if topological else train_cfg["epochs"]
    if lr_search:
        itr = min(itr, max_iters)
    max_duration_hours = getattr(train_args, 'max_duration_hours', None) \
        if train_args is not None else None
    train_start_time = time.time()
    if rank == 0 and max_duration_hours is not None and max_duration_hours > 0:
        logger.info(f"[RANK {rank}] Max training duration set to "
                    f"{max_duration_hours}h")
    for epoch in range(itr):
        optimizer.zero_grad()
        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"[RANK {rank}] Iteration {epoch + 1}/{itr} LR = {current_lr:.6f}")
        logger.info(f"[RANK {rank}] Iteration {epoch + 1}/{itr} begins")
        start_time = time.time()
        model.train()
        total_loss = total_pred_x = total_pred_y = total_sim = \
            total_recon_x = total_recon_y = total_topo = 0
        batch_idx = 0
        for batch_data in dataloader:
            if epoch == 0:
                logger.info(f"[RANK {rank}] Batch {batch_idx + 1}/{len(dataloader)} running")
            with (autocast(device_type="cuda")):
                if chip_meta is not None:
                    x, pos_batch, D_x_flat = batch_data
                    x = x.to(rank).requires_grad_(False)
                    y = extract_chip_from_batch(pos_batch, chip_meta, rank)
                    z = None
                    D_x = _to_device_block(D_x_flat, batch_size, rank)
                else:
                    x, y, z, D_x_flat = batch_data
                    x = x.to(rank).requires_grad_(False)
                    y = y.to(rank).requires_grad_(False)
                    if dataset_label:
                        z = z.to(rank).requires_grad_(False)
                    else:
                        z = None
                    pos_batch = None                           
                    D_x = _to_device_block(D_x_flat, batch_size, rank)
                if x.shape[0] != batch_size:
                    raise RuntimeError(f"[Error] Dimension of x ({x.shape[0]}) "
                                       f"is not expected as {batch_size}")
                optimizer.zero_grad()
                x_pred = y_pred = x_recon = y_recon = \
                    u1 = u2 = None
                y_input = y.unsqueeze(0).permute(0, 2, 1) if y is not None and chip_meta is not None else y
                if model_type == 'x_reconstruction':
                    x_recon, u1 = model(x)
                elif model_type == 'y_prediction':
                    y_pred, x_recon, u1 = model(x)
                elif model_type == 'x_prediction':
                    x_pred, y_recon, u2 = model(y_input)
                elif model_type == 'y_reconstruction':
                    y_recon, u2 = model(y_input)
                elif model_type == 'halfring':
                    x_recon, y_recon, u1, u2 = model(x, y_input)
                elif model_type == 'ring':
                    x_pred, y_pred, x_recon, y_recon, u1, u2 = model(x, y_input)
                else:
                    raise ValueError(f"[RANK {rank}] Model type is not supported")
                monitor_bottleneck(u1, u2, rank, epoch, batch_idx,
                                   logger, model_type)
                effective_penalty = train_cfg["zero_penalty"]
                if hic_norm == "oe":
                    effective_penalty = 1
                x_recon_loss, y_pred_loss, x_pred_loss, y_recon_loss, sim = \
                compute_normal_loss(x, y, z, x_pred, y_pred, x_recon, y_recon,
                                    u1, u2, effective_penalty,
                                    hic_base_loss=hic_base_loss,
                                    huber_delta=train_cfg.get("huber_delta", 2.0),
                                    sim_mode=sim_mode)
                if topological:
                    hic_bin = _extract_hic_bin(pos_batch)
                    if not _block_in_topo_subset(hic_bin, topo_cfg.topo_use_bins):
                        topo_total = torch.tensor(0.0, device=D_x.device)
                        if epoch == 0:
                            logger.info(f"[RANK {rank}] Batch {batch_idx + 1}/{len(dataloader)} "
                                        f"topo skipped by topo_use_bins={topo_cfg.topo_use_bins} "
                                        f"(hic_bin={hic_bin})")
                    else:
                        is_deg, rel_range = is_degenerate_distance_matrix(
                            D_x, topo_cfg.degenerate_bandwidth,
                            topo_cfg.degenerate_threshold)
                        if epoch == 0:
                            pos_str = _format_batch_pos(pos_batch, chip_meta)
                            tag = "[DEGENERATE→topo skipped]" if is_deg else "[ok]"
                            logger.info(f"[RANK {rank}] Batch {batch_idx + 1}/{len(dataloader)} "
                                        f"topo degeneracy: {tag} "
                                        f"band_rel_range={rel_range:.4f} "
                                        f"(bw={topo_cfg.degenerate_bandwidth}, "
                                        f"thr={topo_cfg.degenerate_threshold}) "
                                        f"pos={pos_str}")
                        topo_total = compute_topological_loss(
                            x, u1, u2, topo_thresh, D_x=D_x,
                            topo_cfg=topo_cfg,
                            degenerate_info=(is_deg, rel_range),
                        )
                else:
                    topo_total = 0
                total = combine_loss(
                    x_recon_loss=x_recon_loss,
                    y_pred_loss=y_pred_loss,
                    x_pred_loss=x_pred_loss,
                    y_recon_loss=y_recon_loss,
                    sim=sim,
                    topo_total=topo_total,
                    loss_components=loss_components,
                    train_cfg=train_cfg
                )
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            total.detach()
            batch_idx += 1
            total_loss += total.item()
            total_pred_x += x_pred_loss.item() if x_pred_loss > 0 else 0
            total_pred_y += y_pred_loss.item() if y_pred_loss > 0 else 0
            total_recon_x += x_recon_loss.item() if x_recon_loss > 0 else 0
            total_recon_y += y_recon_loss.item() if y_recon_loss > 0 else 0
            total_sim += sim.item() if sim > 0 else 0
            if topological:
                total_topo += topo_total.item()
        end_time = time.time()
        logger.info(f"[RANK {rank}] Iteration {epoch + 1}/{itr} "
                    f"completed in {end_time - start_time:.2f} seconds")
        if (epoch + 1) % 50 == 0 and rank == 0:
            seconds_needed = (itr - epoch - 1) * (end_time - start_time)
            compute_complete_time(rank, seconds_needed, logger)
        scheduler.step()
        stopping = False
        search_test_loss = None
        if (epoch + 1) % 50 == 0 and lr_search:
            logger.info(f"[RANK {rank}] Initiate test at epoch {epoch + 1}/{itr}")
            search_test_loss, _, _ = test_on_master(
                rank=rank, epoch=epoch, model=model, model_type=model_type,
                loss_components=loss_components,
                test_dataset=test_dataset,
                batch_size=batch_size, extra_label=dataset_label, topological=topological,
                topo_thresh=topo_thresh, zero_penalty=train_cfg["zero_penalty"],
                debug=False, file_path=save_path,
                chip_meta=chip_meta, train_cfg=train_cfg,
                hic_base_loss=hic_base_loss, hic_norm=hic_norm,
                huber_delta=train_cfg.get("huber_delta", 2.0),
                topo_cfg=topo_cfg,
                sim_mode=sim_mode)
            if search_test_loss is not None:
                logger.info(f"[RANK {rank}] Search test loss {search_test_loss:.8f}")
        elif (epoch + 1) % 50 == 0 and not lr_search:
            logger.info(f"[RANK {rank}] Initiate test at epoch {epoch + 1}/{itr}")
            test_loss, test_loss_dict, viz_outputs = test_on_master(rank=rank, epoch=epoch, model=model, model_type=model_type,
                                       loss_components=loss_components,
                                       test_dataset=test_dataset,
                                       batch_size=batch_size, extra_label=dataset_label, topological=topological,
                                       topo_thresh=topo_thresh, zero_penalty=train_cfg["zero_penalty"],
                                       debug=debug_mode, file_path=save_path,
                                       chip_meta=chip_meta, train_cfg=train_cfg,
                                       hic_base_loss=hic_base_loss, hic_norm=hic_norm,
                                       huber_delta=train_cfg.get("huber_delta", 2.0),
                                       topo_cfg=topo_cfg,
                                       sim_mode=sim_mode)
            if test_loss is not None:
                logger.info(f"[RANK {rank}] Test loss {test_loss:.8f}")
            if rank == 0:
                with open(record_test, mode="a") as f:
                    f.write(f"{epoch + 1}\t{test_loss:.8f}\t"
                            f"{test_loss_dict['x_pred_loss']:.8f}\t"
                            f"{test_loss_dict['y_pred_loss']:.8f}\t"
                            f"{test_loss_dict['x_recon_loss']:.8f}\t"
                            f"{test_loss_dict['y_recon_loss']:.8f}\t"
                            f"{test_loss_dict['sim']:.8f}\t"
                            f"{test_loss_dict['topo']:.8f}\n")
                stopping = early_stopping(record_test)
                if not ext_val_enabled:
                  try:
                    if model_type == 'x_reconstruction':
                        if viz_outputs['x_recon'] is not None:
                            x_gt_np = viz_outputs['x_gt'].numpy()
                            x_recon_np = viz_outputs['x_recon'].numpy()
                            x_recon_np = symmetry_correction(x_recon_np)
                            model_testing_plot(
                                x=x_gt_np, y=x_recon_np,
                                range_1=0, range_2=batch_size,
                                caption_model=save_path,
                                caption_data=f"Epoch {epoch+1}",
                                save_name=f"{save_path}_epoch{epoch+1}_x-reconstruction.png",
                                log_op=False
                            )
                    elif model_type == 'y_prediction':
                        if viz_outputs['y_pred'] is not None:
                            y_pred_tensor = viz_outputs['y_pred']
                            y_gt_tensor = viz_outputs['y_gt']
                            if chip_meta is not None and viz_outputs.get('pos_batch') is not None:
                                y_pred_np = inverse_transform_chip_batch(
                                    y_pred_tensor, viz_outputs['pos_batch'], chip_meta, 'cpu').numpy()
                                y_gt_np = inverse_transform_chip_batch(
                                    y_gt_tensor, viz_outputs['pos_batch'], chip_meta, 'cpu').numpy()
                            else:
                                y_pred_np = y_pred_tensor.numpy()
                                y_gt_np = y_gt_tensor.numpy()
                            pcc_val = pearson_corr_np(y_gt_np.flatten(), y_pred_np.flatten())
                            mse_val = np.mean((y_gt_np - y_pred_np) ** 2)
                            model_testing_plot_chip(
                                x=y_gt_np, y=y_pred_np,
                                range_1=0, range_2=batch_size,
                                xticks=None,
                                caption_model=save_path,
                                caption_data=f"Epoch {epoch+1}",
                                caption_mse=float(mse_val),
                                caption_pcc=float(pcc_val),
                                save_name=f"{save_path}_epoch{epoch+1}_ChIP-prediction.png",
                                log_op=False
                            )
                        if viz_outputs['x_recon'] is not None:
                            x_gt_np = viz_outputs['x_gt'].numpy()
                            x_recon_np = viz_outputs['x_recon'].numpy()
                            x_recon_np = symmetry_correction(x_recon_np)
                            model_testing_plot(
                                x=x_gt_np, y=x_recon_np,
                                range_1=0, range_2=batch_size,
                                caption_model=save_path,
                                caption_data=f"Epoch {epoch+1}",
                                save_name=f"{save_path}_epoch{epoch+1}_x-reconstruction.png",
                                log_op=False
                            )
                    elif model_type == 'x_prediction':
                        if viz_outputs['x_pred'] is not None:
                            x_gt_np = viz_outputs['x_gt'].numpy()
                            x_pred_np = viz_outputs['x_pred'].numpy()
                            x_pred_np = symmetry_correction(x_pred_np)
                            model_testing_plot(
                                x=x_gt_np, y=x_pred_np,
                                range_1=0, range_2=batch_size,
                                caption_model=save_path,
                                caption_data=f"Epoch {epoch+1}",
                                save_name=f"{save_path}_epoch{epoch+1}_x-prediction.png",
                                log_op=False
                            )
                        if viz_outputs['y_recon'] is not None:
                            y_recon_tensor = viz_outputs['y_recon']
                            y_gt_tensor = viz_outputs['y_gt']
                            if chip_meta is not None and viz_outputs.get('pos_batch') is not None:
                                y_recon_np = inverse_transform_chip_batch(
                                    y_recon_tensor, viz_outputs['pos_batch'], chip_meta, 'cpu').numpy()
                                y_gt_np = inverse_transform_chip_batch(
                                    y_gt_tensor, viz_outputs['pos_batch'], chip_meta, 'cpu').numpy()
                            else:
                                y_recon_np = y_recon_tensor.numpy()
                                y_gt_np = y_gt_tensor.numpy()
                            pcc_val = pearson_corr_np(y_gt_np.flatten(), y_recon_np.flatten())
                            mse_val = np.mean((y_gt_np - y_recon_np) ** 2)
                            model_testing_plot_chip(
                                x=y_gt_np, y=y_recon_np,
                                range_1=0, range_2=batch_size,
                                xticks=None,
                                caption_model=save_path,
                                caption_data=f"Epoch {epoch+1}",
                                caption_mse=float(mse_val),
                                caption_pcc=float(pcc_val),
                                save_name=f"{save_path}_epoch{epoch+1}_ChIP-reconstruction.png",
                                log_op=False
                            )
                    elif model_type == 'y_reconstruction':
                        if viz_outputs['y_recon'] is not None:
                            y_recon_tensor = viz_outputs['y_recon']
                            y_gt_tensor = viz_outputs['y_gt']
                            if chip_meta is not None and viz_outputs.get('pos_batch') is not None:
                                y_recon_np = inverse_transform_chip_batch(
                                    y_recon_tensor, viz_outputs['pos_batch'], chip_meta, 'cpu').numpy()
                                y_gt_np = inverse_transform_chip_batch(
                                    y_gt_tensor, viz_outputs['pos_batch'], chip_meta, 'cpu').numpy()
                            else:
                                y_recon_np = y_recon_tensor.numpy()
                                y_gt_np = y_gt_tensor.numpy()
                            pcc_val = pearson_corr_np(y_gt_np.flatten(), y_recon_np.flatten())
                            mse_val = np.mean((y_gt_np - y_recon_np) ** 2)
                            model_testing_plot_chip(
                                x=y_gt_np, y=y_recon_np,
                                range_1=0, range_2=batch_size,
                                xticks=None,
                                caption_model=save_path,
                                caption_data=f"Epoch {epoch+1}",
                                caption_mse=float(mse_val),
                                caption_pcc=float(pcc_val),
                                save_name=f"{save_path}_epoch{epoch+1}_ChIP-reconstruction.png",
                                log_op=False
                            )
                    elif model_type == 'halfring':
                        if viz_outputs['x_recon'] is not None:
                            x_gt_np = viz_outputs['x_gt'].numpy()
                            x_recon_np = viz_outputs['x_recon'].numpy()
                            x_recon_np = symmetry_correction(x_recon_np)
                            model_testing_plot(
                                x=x_gt_np, y=x_recon_np,
                                range_1=0, range_2=batch_size,
                                caption_model=save_path,
                                caption_data=f"Epoch {epoch+1}",
                                save_name=f"{save_path}_epoch{epoch+1}_x-reconstruction.png",
                                log_op=False
                            )
                        if viz_outputs['y_recon'] is not None:
                            y_recon_tensor = viz_outputs['y_recon']
                            y_gt_tensor = viz_outputs['y_gt']
                            if chip_meta is not None and viz_outputs.get('pos_batch') is not None:
                                y_recon_np = inverse_transform_chip_batch(
                                    y_recon_tensor, viz_outputs['pos_batch'], chip_meta, 'cpu').numpy()
                                y_gt_np = inverse_transform_chip_batch(
                                    y_gt_tensor, viz_outputs['pos_batch'], chip_meta, 'cpu').numpy()
                            else:
                                y_recon_np = y_recon_tensor.numpy()
                                y_gt_np = y_gt_tensor.numpy()
                            pcc_val = pearson_corr_np(y_gt_np.flatten(), y_recon_np.flatten())
                            mse_val = np.mean((y_gt_np - y_recon_np) ** 2)
                            model_testing_plot_chip(
                                x=y_gt_np, y=y_recon_np,
                                range_1=0, range_2=batch_size,
                                xticks=None,
                                caption_model=save_path,
                                caption_data=f"Epoch {epoch+1}",
                                caption_mse=float(mse_val),
                                caption_pcc=float(pcc_val),
                                save_name=f"{save_path}_epoch{epoch+1}_ChIP-reconstruction.png",
                                log_op=False
                            )
                    elif model_type == 'ring':
                        if viz_outputs['x_recon'] is not None:
                            x_gt_np = viz_outputs['x_gt'].numpy()
                            x_recon_np = viz_outputs['x_recon'].numpy()
                            x_recon_np = symmetry_correction(x_recon_np)
                            model_testing_plot(
                                x=x_gt_np, y=x_recon_np,
                                range_1=0, range_2=batch_size,
                                caption_model=save_path,
                                caption_data=f"Epoch {epoch+1}",
                                save_name=f"{save_path}_epoch{epoch+1}_x-reconstruction.png",
                                log_op=False
                            )
                        if viz_outputs['y_pred'] is not None:
                            y_pred_tensor = viz_outputs['y_pred']
                            y_gt_tensor = viz_outputs['y_gt']
                            if chip_meta is not None and viz_outputs.get('pos_batch') is not None:
                                y_pred_np = inverse_transform_chip_batch(
                                    y_pred_tensor, viz_outputs['pos_batch'], chip_meta, 'cpu').numpy()
                                y_gt_np = inverse_transform_chip_batch(
                                    y_gt_tensor, viz_outputs['pos_batch'], chip_meta, 'cpu').numpy()
                            else:
                                y_pred_np = y_pred_tensor.numpy()
                                y_gt_np = y_gt_tensor.numpy()
                            pcc_val = pearson_corr_np(y_gt_np.flatten(), y_pred_np.flatten())
                            mse_val = np.mean((y_gt_np - y_pred_np) ** 2)
                            model_testing_plot_chip(
                                x=y_gt_np, y=y_pred_np,
                                range_1=0, range_2=batch_size,
                                xticks=None,
                                caption_model=save_path,
                                caption_data=f"Epoch {epoch+1}",
                                caption_mse=float(mse_val),
                                caption_pcc=float(pcc_val),
                                save_name=f"{save_path}_epoch{epoch+1}_ChIP-prediction.png",
                                log_op=False
                            )
                    logger.info(f"[RANK {rank}] Epoch {epoch+1} visualization plots saved to {save_path}")
                  except Exception as e:
                    logger.warning(f"[RANK {rank}] Visualization failed at epoch {epoch+1}: {e}")
                else:
                    logger.info(f"[RANK {rank}] Epoch {epoch+1} visualization skipped "
                                f"(external validation is enabled)")
        epoch_dict = {
            "total": total_loss,
            "pred_x": total_pred_x,
            "pred_y": total_pred_y,
            "recon_x": total_recon_x,
            "recon_y": total_recon_y,
            "sim": total_sim,
            "topo": total_topo
        }
        if world_size > 1:
            loss_dict = log_reduced_losses_ring(epoch_dict, world_size)           
            if rank == 0:
                logger.info(f"[RANK {rank}] Iteration {epoch + 1}/{itr} loss: "
                            f"{loss_dict.get('total')}")
                if epoch % 1000 == 0:
                    logger.info(f"[RANK {rank}] " + monitor_resources())
                with open(record_file, mode='a') as f:
                    f.write(f"{epoch + 1}\t{current_lr:.8f}\t{loss_dict.get('total'):.8f}\t"
                            f"{loss_dict.get('pred_x'):.8f}\t{loss_dict.get('pred_y'):.8f}\t"
                            f"{loss_dict.get('recon_x'):.8f}\t{loss_dict.get('recon_y'):.8f}\t"
                            f"{loss_dict.get('sim'):.8f}\t{loss_dict.get('topo'):.8f}\n")
            save_epoch = train_cfg["model_save"]["topo"] if topological else train_cfg["model_save"]["normal"]
            time_stop = False
            elapsed_h = 0.0
            if max_duration_hours is not None and max_duration_hours > 0:
                elapsed_h = (time.time() - train_start_time) / 3600.0
                if rank == 0:
                    time_stop = elapsed_h >= max_duration_hours
                _flag = torch.tensor([1 if time_stop else 0], device=rank)
                dist.broadcast(_flag, src=0)
                time_stop = bool(_flag.item())
            if rank == 0:
                if (epoch + 1) % save_epoch == 0:
                    model_save(epoch, model, optimizer, save_path)
                    logger.info(f"[RANK {rank}] Model cached after {epoch + 1} iterations")
                    if train_args is not None and ext_val_enabled:
                        ckpt_path = save_path + "_epoch" + str(epoch + 1) + ".pth"
                        trigger_external_validation(ckpt_path, epoch, train_args)
                if time_stop:
                    model_save(epoch, model, optimizer, save_path)
                    logger.info(f"[RANK {rank}] Max duration {max_duration_hours}h reached "
                                f"after {elapsed_h:.2f}h (epoch {epoch + 1}); saving final "
                                f"checkpoint and launching validation")
                    if train_args is not None and ext_val_enabled:
                        ckpt_path = save_path + "_epoch" + str(epoch + 1) + ".pth"
                        trigger_external_validation(ckpt_path, epoch, train_args, detach=True)
                if stopping and (not debug_mode):
                    model_save(epoch, model, optimizer, save_path)
                    logger.info(f"[RANK {rank}] Model stopped after {epoch + 1} iterations "
                                f"due to early stopping")
                    if train_args is not None and ext_val_enabled:
                        ckpt_path = save_path + "_epoch" + str(epoch + 1) + ".pth"
                        trigger_external_validation(ckpt_path, epoch, train_args)
                    break
            if time_stop:
                if rank == 0:
                    logger.info(f"[RANK {rank}] Breaking training loop due to time limit")
                break
        else:
            logger.info(f"[RANK {rank}] Iteration {epoch + 1}/{itr} loss: {total_loss:.8f}")
            if lr_search:
                tl = f"{search_test_loss:.8f}" if search_test_loss is not None else ""
                with open(record_file, "a") as f:
                    f.write(f"{epoch + 1}\t{total_loss:.8f}\t{tl}\n")
            if max_duration_hours is not None and max_duration_hours > 0:
                elapsed_h = (time.time() - train_start_time) / 3600.0
                if elapsed_h >= max_duration_hours:
                    model_save(epoch, model, optimizer, save_path)
                    logger.info(f"[RANK {rank}] Max duration {max_duration_hours}h reached "
                                f"after {elapsed_h:.2f}h (epoch {epoch + 1}); saving final "
                                f"checkpoint and launching validation")
                    if train_args is not None and ext_val_enabled:
                        ckpt_path = save_path + "_epoch" + str(epoch + 1) + ".pth"
                        trigger_external_validation(ckpt_path, epoch, train_args, detach=True)
                    break
    if lr_search:
        logger.info(f"[RANK {rank}] LR search worker finished")
        return
    if rank == 0:
        model_loss_plot(rank, record_file)
    dist.destroy_process_group() if world_size > 1 else None

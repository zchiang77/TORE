from src.lossCompute import *
from torch.utils.data import DataLoader
from src.data import extract_chip_from_batch
import torch
import torch.distributed as dist
import pandas as pd
import numpy as np
def _to_device_block(D_x_flat, batch_size, rank):
    D_x = D_x_flat.to(rank)
    if D_x.shape != (batch_size, batch_size):
        raise RuntimeError(
            f"[test] D_x_flat shape {tuple(D_x.shape)} != "
            f"({batch_size}, {batch_size}); batch_size  patch_bins"
            f" batch  block "
        )
    return D_x
def test_on_master(
        rank,
        epoch,
        model,
        model_type,
        loss_components,
        test_dataset,
        batch_size,
        extra_label,
        topological,
        topo_thresh,
        zero_penalty,
        debug,
        file_path=None,
        chip_meta=None,
        train_cfg=None,
        hic_base_loss="mse",
        hic_norm=None,
        huber_delta=2.0,
        topo_cfg=None,
        sim_mode="mse"
):
    if dist.is_initialized():
        dist.barrier()
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    model.eval()
    total_loss = 0
    total_x_recon_loss = 0
    total_y_pred_loss = 0
    total_x_pred_loss = 0
    total_y_recon_loss = 0
    total_sim = 0
    total_topo = 0
    accumulated_loss = None
    viz_outputs = None                                                 
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            pos_batch = None
            if chip_meta is not None:
                x, pos_batch, D_x_flat = batch_data
                x = x.to(rank)
                y = extract_chip_from_batch(pos_batch, chip_meta, rank)
                y_gt = y.clone()
                z = None
                D_x = _to_device_block(D_x_flat, batch_size, rank)
            elif model_type == 'ring':
                x, pos_batch, D_x_flat = batch_data
                x = x.to(rank)
                y = extract_chip_from_batch(pos_batch, chip_meta, rank)
                y_gt = y.clone()
                z = None
                D_x = _to_device_block(D_x_flat, batch_size, rank)
            elif model_type == 'x_reconstruction':
                x = batch_data[0]
                x = x.to(rank)
                y, z, y_gt = None, None, None
                D_x = _to_device_block(batch_data[3], batch_size, rank)
            else:
                x, y, z, D_x_flat = batch_data
                x, y = x.to(rank), y.to(rank)
                y_gt = y.clone()
                z = z.to(rank) if extra_label else None
                D_x = _to_device_block(D_x_flat, batch_size, rank)
            if x.shape[0] != batch_size:
                raise ValueError(
                    f"[test_on_master] Batch size mismatch: "
                    f"expected {batch_size}, got {x.shape[0]}. "
                    f"This should not happen; check DataLoader drop_last setting."
                )
            y_input = y.unsqueeze(0).permute(0, 2, 1) if y is not None and chip_meta is not None else y
            x_pred = y_pred = x_recon = y_recon = u1 = u2 = None
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
            if batch_idx == 0:
                viz_outputs = {
                    'x_gt':      x.detach().cpu() if x is not None else None,
                    'y_gt':      y_gt.detach().cpu() if y_gt is not None else None,
                    'x_pred':    x_pred.detach().cpu() if x_pred is not None else None,
                    'y_pred':    y_pred.detach().cpu() if y_pred is not None else None,
                    'x_recon':   x_recon.detach().cpu() if x_recon is not None else None,
                    'y_recon':   y_recon.detach().cpu() if y_recon is not None else None,
                    'u1':        u1.detach().cpu() if u1 is not None else None,
                    'u2':        u2.detach().cpu() if u2 is not None else None,
                    'pos_batch': (pos_batch.detach().cpu()
                                  if (pos_batch is not None
                                      and torch.is_tensor(pos_batch)) else None),
                }
            effective_penalty = zero_penalty
            if hic_norm == "oe":
                effective_penalty = 1
            x_recon_loss, y_pred_loss, x_pred_loss, y_recon_loss, sim = \
            compute_normal_loss(x, y, z, x_pred, y_pred, x_recon, y_recon,
                                u1, u2, effective_penalty,
                                hic_base_loss=hic_base_loss,
                                huber_delta=huber_delta,
                                sim_mode=sim_mode)
            if debug and rank == 0:
                debug_loss = feature_mse_loss(y_pred, y)
                if accumulated_loss is None:
                    n_features = len(debug_loss)
                    accumulated_loss = [0.0] * n_features
                for i, loss in enumerate(debug_loss):
                    accumulated_loss[i] += loss
            if topological:
                topo_total = compute_topological_loss(
                    x, u1, u2, topo_thresh, D_x=D_x, topo_cfg=topo_cfg,
                    topo_h0_weight=train_cfg.get("topo_h0_weight", 0.1) if train_cfg is not None else 0.1,
                )
            else:
                topo_total = 0
            loss = combine_loss(
                x_recon_loss=x_recon_loss,
                y_pred_loss=y_pred_loss,
                x_pred_loss=x_pred_loss,
                y_recon_loss=y_recon_loss,
                sim=sim,
                topo_total=topo_total,
                loss_components=loss_components,
                train_cfg=train_cfg
            )
            total_loss += loss
            total_x_recon_loss += x_recon_loss.item() if isinstance(x_recon_loss, torch.Tensor) else 0
            total_y_pred_loss  += y_pred_loss.item()  if isinstance(y_pred_loss, torch.Tensor)  else 0
            total_x_pred_loss  += x_pred_loss.item()  if isinstance(x_pred_loss, torch.Tensor)  else 0
            total_y_recon_loss += y_recon_loss.item() if isinstance(y_recon_loss, torch.Tensor) else 0
            total_sim          += sim.item()          if isinstance(sim, torch.Tensor)          else 0
            total_topo         += topo_total.item()   if isinstance(topo_total, torch.Tensor)   else 0
    avg_loss = total_loss / len(test_loader)
    avg_x_recon_loss = total_x_recon_loss / len(test_loader)
    avg_y_pred_loss  = total_y_pred_loss  / len(test_loader)
    avg_x_pred_loss  = total_x_pred_loss  / len(test_loader)
    avg_y_recon_loss = total_y_recon_loss / len(test_loader)
    avg_sim          = total_sim          / len(test_loader)
    avg_topo         = total_topo         / len(test_loader)
    loss_dict = {
        'x_pred_loss':  avg_x_pred_loss,
        'y_pred_loss':  avg_y_pred_loss,
        'x_recon_loss': avg_x_recon_loss,
        'y_recon_loss': avg_y_recon_loss,
        'sim':          avg_sim,
        'topo':         avg_topo,
    }
    if debug and rank == 0:
        avg_feature_loss = [acc / len(test_loader) for acc in accumulated_loss]
        with open(file_path + "_loss.debug.txt", "a") as f:
            f.write(f"{epoch + 1}\t" + "\t".join(f"{val:.6f}" for val in avg_feature_loss) + "\n")
    if dist.is_initialized():
        dist.barrier()
    model.train()
    if rank == 0 or not dist.is_initialized():
        return avg_loss, loss_dict, viz_outputs
    else:
        return None, None, None
def early_stopping(records, patience=5):
    data = pd.read_csv(records, sep='\t')
    loss_curve = data['Loss'].tail(patience)
    if len(loss_curve) < patience:
        return False
    return np.all(np.diff(loss_curve) > 0)

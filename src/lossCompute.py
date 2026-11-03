from torch.nn.functional import mse_loss
from src.topology import (distance_matrix_loss, HomologyLoss,
                          distance_matrix_smooth_loss, normalize_distance_matrix,
                          is_degenerate_distance_matrix)
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
class SignedExpMSELoss(nn.Module):
    def __init__(self):
        super(SignedExpMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='mean')
    def forward(self, pred, target):
        y_pred = torch.where(
            pred >= 0,
            1 - torch.exp(-torch.abs(pred)),
            torch.exp(torch.abs(pred)) - 1
        )
        y_target = torch.where(
            target >= 0,
            1 - torch.exp(-torch.abs(target)),
            torch.exp(torch.abs(target)) - 1
        )
        loss = self.mse_loss(y_pred, y_target)
        return loss
def pearson_corr(x, y):
    x_centered = x - x.mean(dim=-1,  keepdim=True)
    y_centered = y - y.mean(dim=-1,  keepdim=True)
    cov = (x_centered * y_centered).sum(dim=-1)
    x_std = x_centered.norm(dim=-1)
    y_std = y_centered.norm(dim=-1)
    corr = cov / (x_std * y_std + 1e-8)
    return corr.mean()
def pearson_corr_np(x, y):
    x_centered = x - np.mean(x, axis=-1, keepdims=True)
    y_centered = y - np.mean(y, axis=-1, keepdims=True)
    cov = np.sum(x_centered * y_centered, axis=-1)
    x_std = np.linalg.norm(x_centered, axis=-1)
    y_std = np.linalg.norm(y_centered, axis=-1)
    corr = cov / (x_std * y_std + 1e-8)
    return np.mean(corr)
def weighted_mse_loss(prediction, true, eta):
    base_loss = (prediction - true) ** 2
    weights = torch.ones_like(true)
    penalty_condition = (true > 1e-4) & (prediction < 1e-4)
    weights[penalty_condition] = eta
    weighted_loss = base_loss * weights
    return torch.mean(weighted_loss)
def weighted_huber_loss(prediction, true, eta, delta=1.0):
    diff = prediction - true
    abs_diff = torch.abs(diff)
    base_loss = torch.where(
        abs_diff <= delta,
        0.5 * diff ** 2,
        delta * (abs_diff - 0.5 * delta)
    )
    weights = torch.ones_like(true)
    penalty_condition = (true > 1e-4) & (prediction < 1e-4)
    weights[penalty_condition] = eta
    weighted_loss = base_loss * weights
    return torch.mean(weighted_loss)
def weighted_mae_loss(prediction, true, eta):
    base_loss = torch.abs(prediction - true)
    weights = torch.ones_like(true)
    penalty_condition = (true > 1e-4) & (prediction < 1e-4)
    weights[penalty_condition] = eta
    weighted_loss = base_loss * weights
    return torch.mean(weighted_loss)
def chip_peak_weighted_mse_loss(prediction, true, alpha=2.0, base_weight=1.0):
    weights = base_weight + alpha * (true ** 2)
    squared_error = (prediction - true) ** 2
    return torch.mean(weights * squared_error)
def chip_prediction_loss(prediction, true, alpha=2.0, base_weight=1.0):
    mse_term = chip_peak_weighted_mse_loss(prediction, true, alpha=alpha, base_weight=base_weight)
    pcc_term = 1 - pearson_corr(prediction, true)
    return mse_term + pcc_term
def batch_diagonal_weighted_mse(y_pred, y_true, pos, zero_penalty,
                                bandwidth=400, high_weight=0.7, low_weight=0.3):
    if zero_penalty != 1:
        raise ValueError("This version does not support zero_penalty not equal to 1.")
    mse = (y_pred - y_true) ** 2
    rows, cols = y_pred.shape
    device = y_pred.device
    i = torch.arange(pos.item(), pos.item() + rows, device=device).view(-1, 1)
    j = torch.arange(1, 1+cols, device=device).view(1, -1)
    distance = torch.abs(i - j)
    weights = torch.where(distance <= bandwidth, high_weight, low_weight)
    weighted_loss = mse * weights
    return torch.mean(weighted_loss)
def cosine_similarity_loss(x1, x2, dim=0):
    cos_sim = F.cosine_similarity(x1, x2, dim=dim)
    cos_sim_mapped = (cos_sim + 1) / 2                   
    loss = 1 - cos_sim_mapped
    return loss.mean()
def loss_weight_scale_factor(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    ratio = A / (B + 1e-12)                                               
    log_val = torch.log10(ratio)
    exponent = torch.floor(log_val)
    return torch.pow(10, exponent + 1)
class WassersteinAlignmentLoss(nn.Module):
    def __init__(self, mu_weight=1.0, sigma_weight=1.0, reduction='mean', eps=1e-6):
        super(WassersteinAlignmentLoss, self).__init__()
        self.mu_weight = mu_weight
        self.sigma_weight = sigma_weight
        self.reduction = reduction
        self.eps = eps
        if reduction not in ['mean', 'sum']:
            raise ValueError(f"reduction'mean''sum'{reduction}")
    def forward(self, mu1, log_var1, mu2, log_var2):
        sigma1 = torch.exp(0.5 * log_var1) + self.eps
        sigma2 = torch.exp(0.5 * log_var2) + self.eps
        mu_diff = mu1 - mu2
        mu_distance = torch.sum(mu_diff ** 2, dim=1)           
        sigma_diff = sigma1 - sigma2
        sigma_distance = torch.sum(sigma_diff ** 2, dim=1)           
        per_sample_loss = self.mu_weight * mu_distance + self.sigma_weight * sigma_distance
        if self.reduction == 'mean':
            return per_sample_loss.mean()
        else:         
            return per_sample_loss.sum()
    def extra_repr(self):
        return f'mu_weight={self.mu_weight}, sigma_weight={self.sigma_weight}, reduction={self.reduction}'
def cka_loss(u1, u2, eps=1e-8):
    u1c = u1 - u1.mean(dim=0, keepdim=True)
    u2c = u2 - u2.mean(dim=0, keepdim=True)
    num = (u1c.t() @ u2c).pow(2).sum()
    den = ((u1c.t() @ u1c).pow(2).sum() * (u2c.t() @ u2c).pow(2).sum() + eps).sqrt()
    return 1.0 - num / den
def compute_normal_loss(x, y, z, x_pred, y_pred, x_recon, y_recon,
                        u1, u2, penalty, hic_base_loss="mse", huber_delta=2.0, sim_mode="mse"):
    mse_loss_fn = torch.nn.MSELoss()
    x_recon_loss = y_pred_loss = x_pred_loss = y_recon_loss = sim = 0
    if hic_base_loss == "huber":
        hic_loss_fn = lambda prediction, true, eta: weighted_huber_loss(
            prediction, true, eta, delta=huber_delta)
    elif hic_base_loss == "mae":
        hic_loss_fn = lambda prediction, true, eta: 100 * weighted_mae_loss(
            prediction, true, eta)
    else:                
        hic_loss_fn = lambda prediction, true, eta: 100 * weighted_mse_loss(
            prediction, true, eta)
    if x_recon is not None:
        if z is None:                             
            base_x_recon = hic_loss_fn(prediction=x_recon, true=x, eta=penalty)
            pcc_x_recon = 1 - pearson_corr(x_recon, x)
        else:                                          
            base_x_recon = hic_loss_fn(prediction=x_recon, true=z, eta=penalty)
            pcc_x_recon = 1 - pearson_corr(x_recon, z)
        x_recon_loss = base_x_recon + pcc_x_recon
    if y_pred is not None:
        y_pred_loss = chip_prediction_loss(y_pred, y)
    if x_pred is not None:
        if z is None:                                           
            base_x_pred = hic_loss_fn(prediction=x_pred, true=x, eta=penalty)
            pcc_x_pred = 1 - pearson_corr(x_pred, x)
        else:                       
            base_x_pred = hic_loss_fn(prediction=x_pred, true=z, eta=penalty)
            pcc_x_pred = 1 - pearson_corr(x_pred, z)
        x_pred_loss = base_x_pred + pcc_x_pred
    if y_recon is not None:
        y_recon_loss = chip_prediction_loss(y_recon, y)
    if u1 is not None and u2 is not None:
        sim = cka_loss(u1, u2) if sim_mode == "cka" else mse_loss_fn(u1, u2)
    return (x_recon_loss, y_pred_loss, x_pred_loss, y_recon_loss, sim)
_homology_loss_fn = None
_homology_loss_params = None
_degenerate_warned = False
def _get_homology_loss(h0_weight=0.1, smooth_weight=0.1):
    global _homology_loss_fn, _homology_loss_params
    params = (h0_weight, smooth_weight)
    if _homology_loss_fn is None or _homology_loss_params != params:
        _homology_loss_fn = HomologyLoss(
            h0_weight=h0_weight,
            smooth_weight=smooth_weight)
        _homology_loss_params = params
    return _homology_loss_fn
def compute_topological_loss(x, u1, u2, topo_thresh, D_x=None,
                              topo_cfg=None,
                              topo_h0_weight=0.1,
                              degenerate_info=None):
    if topo_cfg is not None:
        topo_thresh = topo_cfg.threshold
        topo_h0_weight = topo_cfg.h0_weight
        smooth_weight = topo_cfg.smooth_weight
    else:
        smooth_weight = 0.1
    if D_x is None:
        D_x = torch.cdist(x, x)
    bw = topo_cfg.degenerate_bandwidth if topo_cfg is not None else 5
    thr_deg = topo_cfg.degenerate_threshold if topo_cfg is not None else 0.05
    if degenerate_info is not None:
        is_deg, rel_range = degenerate_info
    else:
        is_deg, rel_range = is_degenerate_distance_matrix(D_x, bw, thr_deg)
    if is_deg:
        if degenerate_info is None:
            global _degenerate_warned
            if not _degenerate_warned:
                import logging as _logging
                _logging.getLogger(__name__).info(
                    f"[TOPO] Degenerate block detected (skipping its topo loss): "
                    f"near-diagonal band rel_range={rel_range:.4f} < {thr_deg} "
                    f"(band saturated → collapse gradient avoided). "
                    f"Further occurrences will be skipped silently.")
                _degenerate_warned = True
        device = D_x.device if hasattr(D_x, "device") else (u1.device if u1 is not None else 'cpu')
        return torch.tensor(0.0, device=device)
    topology_fn = _get_homology_loss(h0_weight=topo_h0_weight,
                                      smooth_weight=smooth_weight)
    topo_x_latent = topology_fn(D_x, u1, topo_thresh)
    topo_x_latent = _apply_z_degeneracy_safeguard(
        topo_x_latent, D_x, u1, topo_thresh, topology_fn,
        bw, thr_deg, smooth_weight)
    if u2 is not None:
        topo_sim = topology_fn(D_x, u2, topo_thresh)
        topo_sim = _apply_z_degeneracy_safeguard(
            topo_sim, D_x, u2, topo_thresh, topology_fn,
            bw, thr_deg, smooth_weight)
    else:
        topo_sim = 0
    return topo_x_latent + topo_sim
def _apply_z_degeneracy_safeguard(homology_loss, D_x, z, topo_thresh,
                                   topology_fn, bw, thr_deg, smooth_weight):
    if z is None:
        return homology_loss
    z_dist = torch.cdist(z, z)
    is_z_deg, _ = is_degenerate_distance_matrix(z_dist, bw, thr_deg)
    if not is_z_deg:
        return homology_loss
    device = D_x.device if hasattr(D_x, "device") else z.device
    D_x_norm = normalize_distance_matrix(D_x)
    z_dist_norm = normalize_distance_matrix(z_dist)
    smooth = distance_matrix_smooth_loss(D_x_norm, z_dist_norm,
                                          eps=topology_fn.eps if hasattr(topology_fn, 'eps') else 1e-8)
    return smooth_weight * smooth
def combine_loss(x_recon_loss, y_pred_loss, x_pred_loss, y_recon_loss,
                 sim, topo_total, loss_components=None, train_cfg=None):
    topo_weight = train_cfg["topo_weight"] if train_cfg is not None else 1
    sim_weight = train_cfg["sim_weight"] if train_cfg is not None else 1
    ALL_COMPONENTS = {
        'x_recon_loss': x_recon_loss,
        'y_pred_loss': y_pred_loss,
        'x_pred_loss': x_pred_loss,
        'y_recon_loss': y_recon_loss,
        'sim': sim_weight * sim,
        'topo': topo_weight * topo_total,
    }
    if loss_components is None:
        raise ValueError(
            "loss_components must be explicitly specified. "
            f"Available components: {list(ALL_COMPONENTS.keys())}"
        )
    total = 0
    for comp in loss_components:
        if comp not in ALL_COMPONENTS:
            raise ValueError(
                f"Unknown loss component: '{comp}'. "
                f"Available components: {list(ALL_COMPONENTS.keys())}"
            )
        total = total + ALL_COMPONENTS[comp]
    return total
def feature_mse_loss(pred, target):
    diff = (pred - target) ** 2
    mse_per_feature = diff.mean(dim=0)
    losses = [loss.item() for loss in mse_per_feature]
    if len(losses) == pred.shape[1]:
        return list(losses)
    else:
        raise RuntimeError

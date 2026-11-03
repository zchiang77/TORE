import os
import sys
import json
import numpy as np
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import cooler
from torch.utils.data import TensorDataset
from src.data import (
    compute_oe_matrix, inverse_oe_matrix,
    compute_spatial_distance_matrix,
)
from src.ringHelper import symmetry_correction
from src.topology import get_topo_config, normalize_distance_matrix
from src.model_builder import build_model
from src.forward import external_validation, specify_use_model
from exp.insulation import insulation_score_numpy
def load_config():
    with open(os.path.join(ROOT, "config.json"), encoding="utf-8") as f:
        return json.load(f)
def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
def get_topo_cfg(cfg=None):
    if cfg is None:
        cfg = load_config()
    return get_topo_config(cfg.get("ring", {}))
def load_model(model_path, preset="SMALL", patch_bins=40, device="cpu"):
    cfg = load_config()
    rank = 0 if (device != "cpu" and torch.cuda.is_available()) else "cpu"
    if rank == "cpu":
        device_str = "cpu"
    else:
        device_str = f"cuda:{rank}"
    model = build_model("x_reconstruction", cfg, patch_bins, 0, rank,
                        y_length=None, preset=preset)
    model = specify_use_model(rank, model_path, model)
    model.eval()
    return model, rank
def extract_region(clr, chrom, start_bin, patch_bins=40):
    mat_bal = clr.matrix(balance=True)
    mat_raw = clr.matrix(balance=False)
    s = start_bin
    e = s + patch_bins
    balanced = np.array(mat_bal[s:e, s:e], dtype=np.float64)
    raw = np.array(mat_raw[s:e, s:e], dtype=np.float64)
    balanced[np.isnan(balanced)] = 0
    raw[np.isnan(raw)] = 0
    return balanced, raw
def forward_reconstruct(model, rank, balanced_mat, patch_bins=40, hic_norm="oe", oe_reference=None):
    if hic_norm == "oe":
        normed = compute_oe_matrix(balanced_mat).astype(np.float32)
    elif hic_norm == "log1p":
        normed = np.log1p(balanced_mat).astype(np.float32)
    else:
        normed = balanced_mat.astype(np.float32)
    x_tensor = torch.tensor(normed, dtype=torch.float32)
    dummy = torch.zeros(x_tensor.shape[0], dtype=torch.float32)
    dataset = TensorDataset(x_tensor, dummy)
    x_pred, y_pred, x_recon, y_recon, u1, u2, y_gt, y_from_u1, x_from_u2 = \
        external_validation(rank, model, "x_reconstruction",
                            dataset, batch_size=patch_bins,
                            chip_meta=None, cross_reconstruction=False)
    recon_oe = x_recon.detach().cpu().numpy().reshape(patch_bins, patch_bins)
    recon_oe = symmetry_correction(recon_oe)
    embed = u1.detach().cpu().numpy()
    ref = oe_reference if oe_reference is not None else balanced_mat
    if hic_norm == "oe":
        recon_balanced = inverse_oe_matrix(recon_oe, ref)
    elif hic_norm == "log1p":
        recon_balanced = np.expm1(recon_oe)
    else:
        recon_balanced = recon_oe
    return recon_balanced, recon_oe, embed
_chr22_pixel_cache = {}
def _get_chr22_pixels(clr):
    key = clr.uri
    if key in _chr22_pixel_cache:
        return _chr22_pixel_cache[key]
    ext = clr.extent("22")
    pix = clr.matrix(as_pixels=True, balance=False).fetch("22", "22")
    pix = pix.copy()
    pix['bin1_id'] -= ext[0]
    pix['bin2_id'] -= ext[0]
    _chr22_pixel_cache[key] = pix
    return pix
def _get_chr22_bins(clr):
    ext = clr.extent("22")
    bins = clr.bins()[ext[0]:ext[1]]
    bins = bins.reset_index(drop=True).drop(columns=['weight'], errors='ignore')
    return bins
def downsample_and_balance(clr, fraction, seed=42, ignore_diags=2,
                           temp_dir=None):
    import tempfile
    if temp_dir is None:
        temp_dir = os.path.join(ROOT, "exp", "results", "_tmp_cool")
    os.makedirs(temp_dir, exist_ok=True)
    pix = _get_chr22_pixels(clr).copy()
    total = int(pix['count'].sum())
    if fraction >= 1.0:
        pix_ds = pix
    else:
        rng = np.random.RandomState(seed)
        n_sample = int(round(fraction * total))
        p = pix['count'].values.astype(np.float64) / total
        sampled = rng.multinomial(n_sample, p)
        pix_ds = pix.copy()
        pix_ds['count'] = sampled
        pix_ds = pix_ds[pix_ds['count'] > 0]
    bins_c22 = _get_chr22_bins(clr)
    cool_path = os.path.join(
        temp_dir, f"chr22_ds_frac{fraction:.4f}_seed{seed}.cool")
    if os.path.exists(cool_path):
        os.remove(cool_path)
    cooler.create_cooler(cool_path, bins_c22, pix_ds,
                         symmetric_upper=True, ordered=False)
    clr_ds = cooler.Cooler(cool_path)
    weights, stats = cooler.balance_cooler(
        clr_ds, store=False, ignore_diags=ignore_diags)
    return cool_path, weights
def balanced_block_from_weights(clr_ds, weights, start_bin, patch_bins=40):
    mat = clr_ds.matrix(balance=False)
    raw = np.array(mat[start_bin:start_bin + patch_bins,
                       start_bin:start_bin + patch_bins], dtype=np.float64)
    raw[np.isnan(raw)] = 0
    w = weights[start_bin:start_bin + patch_bins]
    w = np.nan_to_num(w, nan=0.0)
    balanced = raw * np.outer(w, w)
    zero_idx = np.where(w == 0)[0]
    if len(zero_idx):
        balanced[zero_idx, :] = 0
        balanced[:, zero_idx] = 0
    return balanced
def compute_mse(m1, m2):
    return float(np.mean((m1 - m2) ** 2))
def compute_mse_triu(m1, m2):
    n = m1.shape[0]
    triu = np.triu_indices(n, k=1)
    return float(np.mean((m1[triu] - m2[triu]) ** 2))
def compute_rmse(m1, m2):
    return float(np.sqrt(np.mean((m1 - m2) ** 2)))
def compute_mae(m1, m2):
    return float(np.mean(np.abs(m1 - m2)))
def compute_r2(m1, m2):
    y_true = m1.ravel()
    y_pred = m2.ravel()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)
def _gaussian_window(win=7, sigma=1.5):
    coords = np.arange(win) - win // 2
    g = np.exp(-(coords ** 2) / (2 * sigma ** 2))
    g2 = np.outer(g, g)
    g2 /= g2.sum()
    return g2
def compute_eigenvector(mat, ignore_diags=2):
    n = mat.shape[0]
    oe = compute_oe_matrix(mat).astype(np.float64)
    oe[~np.isfinite(oe)] = 0
    for d in range(ignore_diags):
        for i in range(n - d):
            oe[i, i+d] = 0
            oe[i+d, i] = 0
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = np.corrcoef(oe)
    corr[~np.isfinite(corr)] = 0
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    order = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:, order[0]]
def phase_eigenvector(e1, mat, min_r=0.05):
    from scipy.stats import pearsonr
    row_sums = mat.sum(axis=1)
    valid = np.isfinite(e1) & np.isfinite(row_sums) & (row_sums > 0)
    if valid.sum() > 3 and np.std(e1[valid]) > 0 and np.std(row_sums[valid]) > 0:
        r = float(pearsonr(e1[valid], row_sums[valid])[0])
        if abs(r) >= min_r and r < 0:
            return -e1, r
        return e1, r
    return e1, float('nan')
def pixel_corr_mask(true_bal, threshold=50.0):
    oe_true = compute_oe_matrix(np.asarray(true_bal, dtype=np.float64))
    oe_true = np.nan_to_num(oe_true, nan=0.0, posinf=0.0, neginf=0.0)
    return np.isfinite(oe_true) & (np.abs(oe_true) < threshold)
def compute_ssim(m1, m2, win=7, data_range=None):
    from scipy.ndimage import uniform_filter               
    if data_range is None:
        combined = np.concatenate([m1.ravel(), m2.ravel()])
        data_range = combined.max() - combined.min()
        if data_range == 0:
            return 1.0
    m1 = m1.astype(np.float64)
    m2 = m2.astype(np.float64)
    w = _gaussian_window(win)
    from scipy.signal import fftconvolve
    mu1 = fftconvolve(m1, w, mode='same')
    mu2 = fftconvolve(m2, w, mode='same')
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = fftconvolve(m1 * m1, w, mode='same') - mu1_sq
    sigma2_sq = fftconvolve(m2 * m2, w, mode='same') - mu2_sq
    sigma12 = fftconvolve(m1 * m2, w, mode='same') - mu1_mu2
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return float(np.mean(ssim_map))
def insulation_similarity(true_mat, recon_mat, window=10, ignore_diags=2):
    from scipy.stats import pearsonr, spearmanr
    is_true, _, _ = insulation_score_numpy(true_mat, window=window,
                                           ignore_diags=ignore_diags)
    is_recon, _, _ = insulation_score_numpy(recon_mat, window=window,
                                            ignore_diags=ignore_diags)
    mask = np.isfinite(is_true) & np.isfinite(is_recon)
    if mask.sum() < 3:
        return {"true_is": is_true, "recon_is": is_recon,
                "pearson_r": float('nan'), "spearman_rho": float('nan'),
                "mse": float('nan')}
    t = is_true[mask]
    r = is_recon[mask]
    pr = pearsonr(t, r)[0]
    sr = spearmanr(t, r).correlation
    return {"true_is": is_true, "recon_is": is_recon,
            "pearson_r": float(pr), "spearman_rho": float(sr),
            "mse": float(np.mean((t - r) ** 2))}
def get_chr22_qarm_blocks(clr, patch_bins=40):
    chrom = "22"
    ext = clr.extent(chrom)
    chrom_start = ext[0]
    mat_raw = clr.matrix(balance=False)
    nbins = ext[1] - ext[0]
    blocks = []
    local_start = int(18e6 / clr.binsize)                              
    idx = 0
    for local_s in range(local_start, nbins - patch_bins + 1, patch_bins):
        s = chrom_start + local_s
        sub = np.array(mat_raw[s:s+patch_bins, s:s+patch_bins])
        sub[np.isnan(sub)] = 0
        if sub.sum() == 0:
            continue
        start_mb = local_s * clr.binsize / 1e6
        end_mb = (local_s + patch_bins) * clr.binsize / 1e6
        blocks.append({
            "idx": idx,
            "global_start_bin": int(s),
            "chrom": chrom,
            "start_mb": round(start_mb, 2),
            "end_mb": round(end_mb, 2),
        })
        idx += 1
    return blocks
def open_cooler(experiment="HIC01", resolution=50000):
    path = os.path.join(ROOT, "..", "data", "HiC.GRCh38",
                        f"{experiment}_{resolution}.cool")
    return cooler.Cooler(path)
def results_dir(step):
    d = os.path.join(ROOT, "exp", "results", step)
    os.makedirs(d, exist_ok=True)
    return d
def bh_adjust(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out

import os
import sys
import json
import time
import argparse
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from exp._helpers import results_dir
STEP1D_DIR = os.path.join(ROOT, "exp", "results", "section4_step1a")
STEP1E_DIR = os.path.join(ROOT, "exp", "results", "section4_step1b")
H5_V1_PATH = os.path.join(STEP1D_DIR, "genome_imputed.h5")
PATCH_BINS = 40
GAUSSIAN_SIGMA = 1.0
SELECTED_BLOCKS = [0, 1, 28, 37, 41, 45, 49, 53, 57, 6]
TRAIN_FRAC = 0.65
RANDOM_SEED = 42
TRIU = np.triu_indices(PATCH_BINS, k=1)
N_SPEC_CELLS_PER_TYPE = 100                      
N_RECOVERY_CELLS = 300                                  
def reproduce_split(labels, split, train_frac=TRAIN_FRAC, seed=RANDOM_SEED):
    rng = np.random.RandomState(seed)
    train_idx, val_idx = [], []
    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        rng.shuffle(idx)
        n_train = int(len(idx) * train_frac)
        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train:])
    return np.array(train_idx) if split == "train" else np.array(val_idx)
def coverage_normalize(mat):
    totals = mat.sum(axis=(1, 2), keepdims=True)
    return np.where(totals > 0, mat / np.maximum(totals, 1e-10), 0.0)
def knn_impute_k(count, knn_idx, knn_sim, k):
    idx_k = knn_idx[:, :k]
    sim_k = knn_sim[:, :k]
    neigh = count[idx_k]                                             
    w = np.maximum(sim_k, 0.0) + 1e-3
    w = w / w.sum(axis=1, keepdims=True)
    return np.einsum('nk,nkij->nij', w, neigh.astype(np.float64))
def smooth_and_clean(imputed):
    s = gaussian_filter(imputed, sigma=(0, GAUSSIAN_SIGMA, GAUSSIAN_SIGMA))
    s = (s + s.transpose(0, 2, 1)) / 2
    idx = np.arange(PATCH_BINS)
    s[:, idx, idx] = 0.0
    return s
def compute_d_x(imputed, gamma, eps):
    d = np.power(imputed.astype(np.float64) + eps, -1.0 / gamma)
    idx = np.arange(PATCH_BINS)
    d[:, idx, idx] = 0.0
    return d
def load_counts_and_meta():
    knn = np.load(os.path.join(STEP1E_DIR, "knn_v2.npz"))
    knn_idx, knn_sim = knn["knn_idx"], knn["knn_sim"]
    labels = knn["labels"]
    contact_totals = knn["contact_totals"]
    n_cells = len(labels)
    counts = np.zeros((len(SELECTED_BLOCKS), n_cells, PATCH_BINS,
                       PATCH_BINS), dtype=np.float32)
    with h5py.File(H5_V1_PATH, "r") as h5f:
        for bi, blk in enumerate(SELECTED_BLOCKS):
            counts[bi] = np.array(h5f[f"block_{blk}"]["count"],
                                  dtype=np.float32)
    return counts, labels, knn_idx, knn_sim, contact_totals
def metric_specificity(d_x_blocks, labels, seed=42):
    rng = np.random.RandomState(seed)
    corrs = []
    for t in np.unique(labels):
        idx_t = np.where(labels == t)[0]
        if len(idx_t) < 2:
            continue
        sel = rng.choice(idx_t, min(N_SPEC_CELLS_PER_TYPE, len(idx_t)),
                         replace=False)
        feats = np.concatenate([d_x_blocks[bi][sel][:, TRIU[0], TRIU[1]]
                                for bi in range(d_x_blocks.shape[0])], axis=1)
        feats = feats - feats.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12
        f = feats / norms
        sim = f @ f.T
        iu = np.triu_indices(len(sel), k=1)
        corrs.append(float(sim[iu].mean()))
    return float(np.mean(corrs))
def metric_recovery(imputed_blocks, counts, contact_totals):
    top = np.argsort(contact_totals)[::-1][:N_RECOVERY_CELLS]
    rs = []
    for bi in range(counts.shape[0]):
        raw_n = coverage_normalize(counts[bi][top])
        imp_n = coverage_normalize(imputed_blocks[bi][top])
        a = raw_n[:, TRIU[0], TRIU[1]]
        b = imp_n[:, TRIU[0], TRIU[1]]
        for j in range(len(top)):
            if a[j].std() > 0 and b[j].std() > 0:
                r = np.corrcoef(a[j], b[j])[0, 1]
                if np.isfinite(r):
                    rs.append(float(r))
    return float(np.median(rs)) if rs else 0.0
def metric_contrast(d_x_blocks, n_sample=500, seed=42):
    rng = np.random.RandomState(seed)
    n_cells = d_x_blocks.shape[1]
    sel = rng.choice(n_cells, min(n_sample, n_cells), replace=False)
    cs = []
    for bi in range(d_x_blocks.shape[0]):
        d = d_x_blocks[bi][sel][:, TRIU[0], TRIU[1]]
        cs.append((d.std(axis=1) / (d.mean(axis=1) + 1e-12)).mean())
    return float(np.mean(cs))
def analyze(out_dir):
    print("\n" + "=" * 70)
    print("[ANALYZE] D_x  (K × γ × eps)")
    print("=" * 70)
    t0 = time.time()
    counts, labels, knn_idx, knn_sim, contact_totals = load_counts_and_meta()
    print(f"  counts: {counts.shape}, cells: {len(labels)}")
    K_LIST = [5, 10, 20]
    GAMMA_LIST = [1.0, 2.0, 3.0]
    EPS_LIST = [0.01, 0.1]
    report = []
    for k in K_LIST:
        print(f"\n  --- K={k}:  +  (10 blocks) ---")
        imputed = np.stack([smooth_and_clean(
            knn_impute_k(counts[bi], knn_idx, knn_sim, k))
            for bi in range(counts.shape[0])])
        rec = metric_recovery(imputed, counts, contact_totals)
        print(f"    recovery r = {rec:.3f}")
        for gamma in GAMMA_LIST:
            for eps in EPS_LIST:
                d_x = np.stack([compute_d_x(imputed[bi], gamma, eps)
                                for bi in range(imputed.shape[0])])
                spec = metric_specificity(d_x, labels)
                contr = metric_contrast(d_x)
                row = {"K": k, "gamma": gamma, "eps": eps,
                       "specificity_within_type_corr": round(spec, 3),
                       "recovery_r": round(rec, 3),
                       "contrast": round(contr, 3),
                       "d_range": [round(float(d_x.min()), 2),
                                   round(float(d_x.max()), 2)]}
                report.append(row)
                print(f"    γ={gamma} eps={eps}: "
                      f"spec={spec:.3f} (=)  "
                      f"contrast={contr:.3f}  range={row['d_range']}")
                del d_x
        del imputed
    with open(os.path.join(out_dir, "dx_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    lines = ["=" * 78]
    lines.append("D_x ")
    lines.append("  specificity:  D_x  ()")
    lines.append("  recovery:     coverage  leave-self-out  r ()")
    lines.append("  contrast:    D_x  std/mean ( PH )")
    lines.append("=" * 78)
    lines.append(f"{'K':>3} {'γ':>4} {'eps':>5} {'spec↓':>7} {'recov↑':>7} "
                 f"{'contr↑':>7} {'D range':>16}")
    for r in report:
        lines.append(f"{r['K']:>3} {r['gamma']:>4} {r['eps']:>5} "
                     f"{r['specificity_within_type_corr']:>7.3f} "
                     f"{r['recovery_r']:>7.3f} {r['contrast']:>7.3f} "
                     f"{str(r['d_range']):>16}")
    txt = "\n".join(lines)
    with open(os.path.join(out_dir, "dx_report.txt"), "w") as f:
        f.write(txt)
    print(f"\n{txt}")
    print(f"\n[ANALYZE] Done in {time.time()-t0:.0f}s → dx_report.json/txt")
def prepare(out_dir, configs):
    print("\n" + "=" * 70)
    print(f"[PREPARE]  d_x : {configs}")
    print("=" * 70)
    t0 = time.time()
    counts, labels, knn_idx, knn_sim, contact_totals = load_counts_and_meta()
    train_idx = reproduce_split(labels, "train")
    n_train = len(train_idx)
    n_blocks = len(SELECTED_BLOCKS)
    print(f"  train cells: {n_train}, blocks: {n_blocks}")
    for (k, gamma, eps) in configs:
        tag = f"K{k}_g{gamma:g}_e{eps:g}"
        print(f"\n  --- {tag} ---")
        d_x = np.stack([smooth_and_clean(
            knn_impute_k(counts[bi], knn_idx, knn_sim, k))
            for bi in range(n_blocks)])
        d_x = np.stack([compute_d_x(d_x[bi], gamma, eps)
                        for bi in range(n_blocks)])
        d_x_flat = d_x[:, train_idx].reshape(
            n_blocks * n_train, PATCH_BINS, PATCH_BINS).astype(np.float32)
        out_path = os.path.join(out_dir, f"dx_{tag}.npz")
        np.savez(out_path, d_x=d_x_flat)
        print(f"    saved {out_path} ({os.path.getsize(out_path)/1e9:.2f} GB, "
              f"range=[{d_x_flat.min():.2f},{d_x_flat.max():.2f}])")
        del d_x, d_x_flat
    print(f"\n[PREPARE] Done in {time.time()-t0:.0f}s")
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--configs", type=str, default="",
                    help=" K:γ:eps,  5:1:0.01,10:1:0.01")
    args = ap.parse_args()
    out_dir = results_dir("section4_step1c")
    print(f"[Section4-Step1f] Output → {out_dir}")
    if args.analyze:
        analyze(out_dir)
    if args.prepare:
        if not args.configs:
            print("Error: --prepare  --configs ( 10:1:0.01)")
            return
        configs = []
        for item in args.configs.split(","):
            k, g, e = item.split(":")
            configs.append((int(k), float(g), float(e)))
        prepare(out_dir, configs)
if __name__ == "__main__":
    main()

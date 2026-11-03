import os
import sys
import json
import time
import argparse
import h5py
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from exp._helpers import results_dir
from src.data import compute_oe_matrix
STEP1D_DIR = os.path.join(ROOT, "exp", "results", "section4_step1a")
H5_V1_PATH = os.path.join(STEP1D_DIR, "genome_imputed.h5")
PATCH_BINS = 40
K_NN = 20
GAUSSIAN_SIGMA = 1.0
N_PC = 60                                     
PC_VARIANTS = {"pc1_20": (0, 20), "pc2_21": (1, 21), "pc1_50": (0, 50)}
FEATURE_SOURCES = ["raw"]                                                        
FEATURE_TRANSFORMS = ["cov", "oe"]                                   
EPS_DX = 0.01
GAMMA = 3.0
SELECTED_BLOCKS = [0, 1, 28, 37, 41, 45, 49, 53, 57, 6]                   
G3_TOP_CELLS = 500
TRIU = np.triu_indices(PATCH_BINS, k=1)                
def list_blocks(h5f):
    keys = [k for k in h5f.keys() if k.startswith("block_")]
    return sorted(keys, key=lambda k: int(k.split("_")[1]))
def coverage_normalize(mat):
    totals = mat.sum(axis=(1, 2), keepdims=True)
    return np.where(totals > 0, mat / np.maximum(totals, 1e-10), 0.0)
def torch_pca(X, q, center=True, seed=42):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    if center:
        mean = X_t.mean(dim=0, keepdim=True)
        X_t = X_t - mean
    U, S, V = torch.pca_lowrank(X_t, q=q, center=False, niter=2)
    scores = (U * S).cpu().numpy()
    total_var = float((X_t ** 2).sum() / (X_t.shape[0] - 1))
    evr = (S ** 2 / (X_t.shape[0] - 1) / max(total_var, 1e-12)).cpu().numpy()
    del X_t, U, S, V
    if device == "cuda":
        torch.cuda.empty_cache()
    return scores, evr
def cosine_knn(scores, k):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.tensor(scores, dtype=torch.float32, device=device)
    X = X / (X.norm(dim=1, keepdim=True) + 1e-10)
    sim = X @ X.T
    sim.fill_diagonal_(-1.0)
    knn_sim, knn_idx = torch.topk(sim, k, dim=1)
    return knn_idx.cpu().numpy(), knn_sim.cpu().numpy().astype(np.float32)
def same_type_fraction(knn_idx, labels):
    n = len(labels)
    neigh_labels = labels[knn_idx]                               
    frac = (neigh_labels == labels[:, None]).mean(axis=1)
    per_type = {}
    for t in np.unique(labels):
        mask = labels == t
        per_type[int(t)] = float(frac[mask].mean()) if mask.sum() > 0 else None
    return float(frac.mean()), per_type
def adjusted_rand_index(labels_true, labels_pred):
    n = len(labels_true)
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)
    contingency = np.zeros((len(classes), len(clusters)), dtype=int)
    for i, c in enumerate(classes):
        for j, k in enumerate(clusters):
            contingency[i, j] = np.sum((labels_true == c) & (labels_pred == k))
    sum_comb_c = sum(int(x * (x - 1) / 2) for x in contingency.flatten())
    sum_comb_a = sum(int(a * (a - 1) / 2) for a in contingency.sum(axis=1))
    sum_comb_b = sum(int(b * (b - 1) / 2) for b in contingency.sum(axis=0))
    total_comb = int(n * (n - 1) / 2)
    expected = sum_comb_a * sum_comb_b / total_comb if total_comb > 0 else 0
    max_index = 0.5 * (sum_comb_a + sum_comb_b)
    if max_index == expected:
        return 1.0
    return (sum_comb_c - expected) / (max_index - expected)
def kmeans(X, k, n_iter=200, seed=42):
    rng = np.random.RandomState(seed)
    n = len(X)
    centers = [X[rng.randint(n)]]
    for _ in range(1, k):
        dists = np.min([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0)
        probs = dists / dists.sum()
        centers.append(X[rng.choice(n, p=probs)])
    centers = np.array(centers)
    for _ in range(n_iter):
        dists = np.array([np.sum((X - c) ** 2, axis=1) for c in centers])
        labels = np.argmin(dists, axis=0)
        for j in range(k):
            if np.sum(labels == j) > 0:
                centers[j] = X[labels == j].mean(axis=0)
    return labels
def build_features(h5f, block_keys, n_cells, source, transform):
    feat_dim = len(block_keys) * len(TRIU[0])
    features = np.zeros((n_cells, feat_dim), dtype=np.float32)
    assert source == "raw", "v1 imputed "
    dset = "count"
    for bi, key in enumerate(block_keys):
        mat = np.array(h5f[key][dset], dtype=np.float32)[:n_cells]
        if transform == "cov":
            feat_in = coverage_normalize(mat)
        else:      
            feat_in = np.zeros_like(mat)
            for i in range(len(mat)):
                m = compute_oe_matrix(mat[i])
                m[np.isnan(m)] = 0
                m[np.isinf(m)] = 0
                feat_in[i] = m.astype(np.float32)
        features[:, bi * len(TRIU[0]):(bi + 1) * len(TRIU[0])] = \
            feat_in[:, TRIU[0], TRIU[1]].astype(np.float32)
        del mat, feat_in
    return features
def phase_a1(out_dir, smoke=False):
    print("\n" + "=" * 60)
    print("[A1]  (raw/imputed) + PCA + KNN")
    print("=" * 60)
    t0 = time.time()
    with h5py.File(H5_V1_PATH, "r") as h5f:
        labels = np.array(h5f["labels"])
        n_cells = int(h5f.attrs["n_cells"])
        block_keys = list_blocks(h5f)
        if smoke:
            block_keys = block_keys[:3]
            n_cells = 2000
            labels = labels[:n_cells]
        n_blocks = len(block_keys)
        print(f"  Cells: {n_cells}, Blocks: {n_blocks} (smoke={smoke})")
        contact_totals = None
        results = {}
        sources = FEATURE_SOURCES if not smoke else ["raw"]
        transforms = FEATURE_TRANSFORMS if not smoke else ["cov"]
        for source in sources:
            for transform in transforms:
                tag = f"{source}:{transform}"
                print(f"\n  --- : {tag} ---")
                features = build_features(h5f, block_keys, n_cells,
                                          source, transform)
                print(f"  PCA (q={N_PC})...")
                scores, evr = torch_pca(features, q=N_PC, center=True)
                print(f"  EVR top5: {[round(float(v), 4) for v in evr[:5]]}")
                del features
                for vname, (a, b) in PC_VARIANTS.items():
                    b = min(b, scores.shape[1])
                    knn_idx, knn_sim = cosine_knn(scores[:, a:b], K_NN)
                    frac, per_type = same_type_fraction(knn_idx, labels)
                    key = f"{tag}:{vname}"
                    results[key] = {"knn_idx": knn_idx, "knn_sim": knn_sim,
                                    "frac": frac, "per_type": per_type,
                                    "scores": scores, "evr": evr}
                    print(f"  [{key}] same-type frac = {frac:.3f}  "
                          f"(per type: { {k: round(v, 3) for k, v in per_type.items()} })")
                del scores
        print("\n   contact_totals ...")
        contact_totals = np.zeros(n_cells, dtype=np.float64)
        for key in block_keys:
            c = np.array(h5f[key]["count"], dtype=np.float32)[:n_cells]
            contact_totals += c.sum(axis=(1, 2)) / 2
            del c
    best = max(results, key=lambda k: results[k]["frac"])
    print(f"\n  → : {best} (frac={results[best]['frac']:.3f})")
    if not smoke:
        baseline_scores = None
        for k in ("raw:cov:pc1_20", "raw:cov:pc1_50"):
            if k in results:
                baseline_scores = results[k]["scores"]
                break
        np.savez(os.path.join(out_dir, "knn_v2.npz"),
                 knn_idx=results[best]["knn_idx"],
                 knn_sim=results[best]["knn_sim"],
                 pc_scores=results[best]["scores"],
                 pc_evr=results[best]["evr"],
                 pc_scores_rawcov=baseline_scores,
                 variant=best, labels=labels,
                 contact_totals=contact_totals,
                 same_type_frac=results[best]["frac"])
        qc = {"G1_same_type_frac": results[best]["frac"],
              "G1_pass": bool(results[best]["frac"] >= 0.60),
              "variant": best,
              "per_type": results[best]["per_type"],
              "pc_evr_top5": [float(v) for v in results[best]["evr"][:5]],
              "all_variants": {k: v["frac"] for k, v in results.items()}}
        with open(os.path.join(out_dir, "qc_a1.json"), "w") as f:
            json.dump(qc, f, indent=2)
        print(f"  Saved → knn_v2.npz, qc_a1.json")
    print(f"[A1] Done in {time.time()-t0:.0f}s")
def weighted_knn_impute(count, knn_idx, knn_sim):
    neigh = count[knn_idx]                                               
    w = np.maximum(knn_sim, 0.0) + 1e-3                          
    w = w / w.sum(axis=1, keepdims=True)
    imputed = np.einsum('nk,nkij->nij', w, neigh.astype(np.float64))
    return imputed
def phase_a2(out_dir, smoke=False):
    print("\n" + "=" * 60)
    print("[A2]  +  + QC")
    print("=" * 60)
    t0 = time.time()
    knn = np.load(os.path.join(out_dir, "knn_v2.npz"))
    knn_idx, knn_sim = knn["knn_idx"], knn["knn_sim"]
    labels = knn["labels"]
    contact_totals = knn["contact_totals"]
    n_cells = len(labels)
    h5_v2_path = os.path.join(out_dir, "genome_imputed_v2.h5")
    with h5py.File(H5_V1_PATH, "r") as h5v1, \
         h5py.File(h5_v2_path, "w") as h5out:
        block_keys = list_blocks(h5v1)
        if smoke:
            block_keys = block_keys[:3]
        h5out.attrs["n_cells"] = n_cells
        h5out.create_dataset("labels", data=labels)
        h5out.create_dataset(
            "barcodes", data=np.array(h5v1["barcodes"]))
        for bi, key in enumerate(block_keys):
            grp_v1 = h5v1[key]
            count = np.array(grp_v1["count"], dtype=np.float32)
            imputed = weighted_knn_impute(count, knn_idx, knn_sim)
            smoothed = gaussian_filter(imputed, sigma=(0, GAUSSIAN_SIGMA,
                                                       GAUSSIAN_SIGMA))
            smoothed = (smoothed + smoothed.transpose(0, 2, 1)) / 2
            idx = np.arange(PATCH_BINS)
            smoothed[:, idx, idx] = 0.0
            grp = h5out.create_group(f"block_{bi}")
            grp.create_dataset("imputed", data=smoothed.astype(np.float32),
                               compression="gzip")
            for attr in ("chrom", "block_idx", "start_bin", "end_bin"):
                grp.attrs[attr] = grp_v1.attrs[attr]
            grp.attrs["imputed_nonzero_frac"] = float((smoothed > 0).mean())
            if (bi + 1) % 10 == 0 or bi == 0:
                print(f"    [{bi+1}/{len(block_keys)}] {key} "
                      f"nz={float((smoothed > 0).mean())*100:.1f}%")
    print(f"  h5 → {h5_v2_path} ({os.path.getsize(h5_v2_path)/1e9:.1f} GB)")
    if smoke:
        print(f"[A2] smoke done in {time.time()-t0:.0f}s")
        return
    print("\n  [QC-G2]  PCA-10D + KMeans ...")
    with h5py.File(h5_v2_path, "r") as h5f:
        block_keys = list_blocks(h5f)
        feat_dim = len(block_keys) * len(TRIU[0])
        imp_features = np.zeros((n_cells, feat_dim), dtype=np.float32)
        for bi, key in enumerate(block_keys):
            imp = np.array(h5f[key]["imputed"], dtype=np.float32)
            cov = coverage_normalize(imp)
            imp_features[:, bi*len(TRIU[0]):(bi+1)*len(TRIU[0])] = \
                cov[:, TRIU[0], TRIU[1]].astype(np.float32)
            del imp, cov
    imp_scores, _ = torch_pca(imp_features, q=10, center=True)
    cl_imp = kmeans(imp_scores, k=5)
    ari_imp = adjusted_rand_index(labels, cl_imp)
    del imp_features
    raw_scores = knn["pc_scores_rawcov"]
    if raw_scores is None or raw_scores.ndim != 2:
        raw_scores = knn["pc_scores"]
    cl_raw = kmeans(raw_scores[:, :10], k=5)
    ari_raw = adjusted_rand_index(labels, cl_raw)
    print(f"    raw-covnorm PCA-10D ARI = {ari_raw:.4f} ()")
    print(f"    imputed    PCA-10D ARI = {ari_imp:.4f}")
    print(f"\n  [QC-G3] Top-{G3_TOP_CELLS}  coverage  ...")
    top_cells = np.argsort(contact_totals)[::-1][:G3_TOP_CELLS]
    rs = []
    with h5py.File(H5_V1_PATH, "r") as h5v1, \
         h5py.File(h5_v2_path, "r") as h5v2:
        for key in list_blocks(h5v1):
            raw = np.array(h5v1[key]["count"], dtype=np.float32)[top_cells]
            imp = np.array(h5v2[key]["imputed"], dtype=np.float32)[top_cells]
            raw_n = coverage_normalize(raw)
            imp_n = coverage_normalize(imp)
            for j in range(len(top_cells)):
                a = raw_n[j][TRIU]
                b = imp_n[j][TRIU]
                if a.std() > 0 and b.std() > 0:
                    r = np.corrcoef(a, b)[0, 1]
                    if np.isfinite(r):
                        rs.append(float(r))
            del raw, imp, raw_n, imp_n
    g3_r = float(np.median(rs)) if rs else 0.0
    print(f"    median recovery r = {g3_r:.4f} (n={len(rs)} cell×block)")
    qc = {"G2_ari_imputed": ari_imp, "G2_ari_raw_baseline": ari_raw,
          "G2_pass": bool(ari_imp >= 0.15),
          "G3_median_recovery_r": g3_r,
          "G3_pass": bool(g3_r >= 0.5),
          "G3_n_pairs": len(rs)}
    with open(os.path.join(out_dir, "qc_a2.json"), "w") as f:
        json.dump(qc, f, indent=2)
    print(f"  Saved → qc_a2.json")
    print(f"[A2] Done in {time.time()-t0:.0f}s")
def compute_d_x(imputed, eps=EPS_DX, gamma=GAMMA):
    d = np.power(imputed.astype(np.float64) + eps, -1.0 / gamma)
    idx = np.arange(PATCH_BINS)
    d[:, idx, idx] = 0.0
    return d.astype(np.float32)
def phase_a3(out_dir):
    print("\n" + "=" * 60)
    print(f"[A3] D_x  (blocks={SELECTED_BLOCKS}) + v2 npz")
    print("=" * 60)
    t0 = time.time()
    h5_v2_path = os.path.join(out_dir, "genome_imputed_v2.h5")
    d_x_blocks = []
    with h5py.File(h5_v2_path, "r") as h5f:
        for bi in SELECTED_BLOCKS:
            imp = np.array(h5f[f"block_{bi}"]["imputed"], dtype=np.float32)
            d_x_blocks.append(compute_d_x(imp))
            print(f"    block_{bi}: D_x range=[{d_x_blocks[-1].min():.3f}, "
                  f"{d_x_blocks[-1].max():.3f}]")
    d_x_all = np.stack(d_x_blocks, axis=0)                                 
    del d_x_blocks
    n_blocks = d_x_all.shape[0]
    for split in ("train", "val"):
        old = np.load(os.path.join(STEP1D_DIR, f"{split}_data.npz"))
        covnorm, labels = old["covnorm"], old["labels"]
        barcodes, block_ids = old["barcodes"], old["block_ids"]
        n_cells = len(covnorm) // n_blocks
        h5_labels = np.load(os.path.join(out_dir, "knn_v2.npz"))["labels"]
        cell_labels_in_npz = labels[:n_cells]
        all_idx = _reproduce_split(h5_labels, split)
        assert np.array_equal(h5_labels[all_idx], cell_labels_in_npz), \
            f"{split}:  v1 !"
        dx_flat = d_x_all[:, all_idx].reshape(
            n_blocks * n_cells, PATCH_BINS, PATCH_BINS).astype(np.float32)
        out_path = os.path.join(out_dir, f"{split}_data_v2.npz")
        np.savez(out_path,
                 covnorm=covnorm, d_x=dx_flat, labels=labels,
                 barcodes=barcodes, block_ids=block_ids)
        print(f"  {split}: {out_path} "
              f"({os.path.getsize(out_path)/1e9:.2f} GB, d_x float32)")
    print(f"[A3] Done in {time.time()-t0:.0f}s")
def _reproduce_split(labels, split, train_frac=0.65, seed=42):
    rng = np.random.RandomState(seed)
    train_idx, val_idx = [], []
    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        rng.shuffle(idx)
        n_train = int(len(idx) * train_frac)
        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train:])
    return np.array(train_idx) if split == "train" else np.array(val_idx)
def write_summary(out_dir):
    qc1 = json.load(open(os.path.join(out_dir, "qc_a1.json")))
    qc2 = json.load(open(os.path.join(out_dir, "qc_a2.json")))
    lines = ["=" * 60]
    lines.append("Section 4, Step 1e  KNN imputation v2")
    lines.append("=" * 60)
    lines.append(f"\n[G1] same-type neighbor frac = {qc1['G1_same_type_frac']:.3f} "
                 f"(≥0.60, {'PASS' if qc1['G1_pass'] else 'FAIL'})")
    lines.append(f"     variant={qc1['variant']}, "
                 f"per-type={ {k: round(v,3) for k,v in qc1['per_type'].items()} }")
    lines.append(f"     PC var top5={[round(v,4) for v in qc1['pc_evr_top5']]}")
    lines.append(f"[G2] imputed-PCA ARI = {qc2['G2_ari_imputed']:.4f} "
                 f"(raw  {qc2['G2_ari_raw_baseline']:.4f}, "
                 f"≥0.15, {'PASS' if qc2['G2_pass'] else 'FAIL'})")
    lines.append(f"[G3] median recovery r = {qc2['G3_median_recovery_r']:.4f} "
                 f"(≥0.5, {'PASS' if qc2['G3_pass'] else 'FAIL'})")
    lines.append(f"\n:")
    lines.append(f"  knn_v2.npz / genome_imputed_v2.h5")
    lines.append(f"  train_data_v2.npz / val_data_v2.npz (d_x )")
    lines.append(f"\n: Phase B  (step3  step1e  v2 npz)")
    txt = "\n".join(lines)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(txt)
    print(f"\n{txt}")
def smoke(out_dir):
    print("[SMOKE]  (3 blocks × 2000 cells)")
    phase_a1(out_dir, smoke=True)
    count = np.random.rand(50, PATCH_BINS, PATCH_BINS).astype(np.float32)
    idx = np.random.randint(0, 50, (50, K_NN))
    sim = np.random.rand(50, K_NN).astype(np.float32)
    imp = weighted_knn_impute(count, idx, sim)
    assert imp.shape == (50, PATCH_BINS, PATCH_BINS)
    dx = compute_d_x(imp)
    assert dx.shape == (50, PATCH_BINS, PATCH_BINS)
    assert np.allclose(dx[:, np.arange(40), np.arange(40)], 0.0)
    print("[SMOKE] ")
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["all", "a1", "a2", "a3", "summary"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out_dir = results_dir("section4_step1b")
    print(f"[Section4-Step1e] Output → {out_dir}")
    if args.smoke:
        smoke(out_dir)
        return
    if args.phase in ("all", "a1"):
        phase_a1(out_dir)
    if args.phase in ("all", "a2"):
        phase_a2(out_dir)
    if args.phase in ("all", "a3"):
        phase_a3(out_dir)
    if args.phase in ("all", "summary"):
        write_summary(out_dir)
    print(f"\n[Section4-Step1e] Done. → {out_dir}")
if __name__ == "__main__":
    main()

import os
import sys
import json
import argparse
import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from collections import defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from exp._helpers import results_dir
DATA_DIR = os.path.join(os.path.dirname(ROOT), "data", "scHiC.Kim2020", "input")
SPARSE_MATRIX_PATH = os.path.join(DATA_DIR, "sparse_matrix_500kb_10Mb.txt")
LP_NAMES_PATH = os.path.join(DATA_DIR, "LP_names.txt")
CELL_LABELS_PATH = os.path.join(DATA_DIR, "cell_labels.txt")
STEP1D_DIR = os.path.join(ROOT, "exp", "results", "section4_step1a")
H5_PATH = os.path.join(STEP1D_DIR, "genome_imputed.h5")
RESOLUTION_500K = 500_000
RESOLUTION_1M = 1_000_000
PATCH_BINS = 40
K_NN = 20
GAUSSIAN_SIGMA = 1.0
EPS_DX = 0.01
GAMMA = 3.0
TRAIN_FRAC = 0.65
RANDOM_SEED = 42
HG38_CHROM_SIZES = {
    "chr1": 248956422, "chr2": 242193529, "chr3": 198295559,
    "chr4": 190214555, "chr5": 181538259, "chr6": 170805979,
    "chr7": 159345973, "chr8": 145138636, "chr9": 138394717,
    "chr10": 133797422, "chr11": 135086622, "chr12": 133275309,
    "chr13": 114364328, "chr14": 107043718, "chr15": 101991189,
    "chr16": 90338345, "chr17": 83257441, "chr18": 80373285,
    "chr19": 58617616, "chr20": 64444167, "chr21": 46709983,
    "chr22": 50818468, "chrX": 156040895,
}
def load_lp_names(path):
    chrom_lps = defaultdict(dict)
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if ":" not in line:
                continue
            chrom, positions = line.split(":", 1)
            parts = positions.split("-")
            pos1, pos2 = int(parts[0]), int(parts[1])
            chrom_lps[chrom][i] = (pos1, pos2)
    return dict(chrom_lps)
def load_cell_labels(path):
    cell_names, cell_types = [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                cell_names.append(parts[0])
                cell_types.append(parts[1])
    types_sorted = sorted(set(cell_types))
    type_to_label = {t: i for i, t in enumerate(types_sorted)}
    return cell_names, cell_types, type_to_label
def get_block_positions(chrom_lps):
    blocks = []
    for chrom in sorted(HG38_CHROM_SIZES.keys()):
        if chrom not in chrom_lps:
            continue
        n_bins_1m = HG38_CHROM_SIZES[chrom] // RESOLUTION_1M + 1
        n_blocks = n_bins_1m // PATCH_BINS
        if n_blocks == 0:
            continue
        for b in range(n_blocks):
            start = b * PATCH_BINS
            end = start + PATCH_BINS
            blocks.append((chrom, b, start, end))
    return blocks
def build_all_count_matrices(sparse_matrix_path, chrom_lps, all_blocks, n_cells):
    lp_rows = []
    for chrom in sorted(chrom_lps.keys()):
        for lp_idx, (pos1, pos2) in chrom_lps[chrom].items():
            bin500_1 = (pos1 - RESOLUTION_500K // 2) // RESOLUTION_500K
            bin500_2 = (pos2 - RESOLUTION_500K // 2) // RESOLUTION_500K
            bin1m_1 = bin500_1 // 2
            bin1m_2 = bin500_2 // 2
            lp_rows.append((lp_idx, chrom, bin1m_1, bin1m_2))
    lp_df = pd.DataFrame(lp_rows, columns=['lp_idx', 'chrom', 'bin1m_1', 'bin1m_2'])
    print(f"  Reading sparse_matrix (full, ~66M rows)...")
    sm = pd.read_csv(sparse_matrix_path, sep=r'\s+', header=None,
                     names=['cell_idx', 'lp_idx', 'count'],
                     dtype={'cell_idx': np.int32, 'lp_idx': np.int32,
                            'count': np.float32})
    print(f"  sparse_matrix: {len(sm):,} rows")
    print(f"  Merging with LP_names...")
    merged = sm.merge(lp_df, on='lp_idx')
    del sm
    print(f"  Merged: {len(merged):,} rows")
    matrices = {}
    for bi, (chrom, b_idx, start, end) in enumerate(all_blocks):
        mask = (merged['chrom'] == chrom) & \
               (merged['bin1m_1'] >= start) & (merged['bin1m_1'] < end) & \
               (merged['bin1m_2'] >= start) & (merged['bin1m_2'] < end)
        block_data = merged[mask]
        mat = np.zeros((n_cells, PATCH_BINS, PATCH_BINS), dtype=np.float32)
        if len(block_data) > 0:
            cell_ids = (block_data['cell_idx'].values - 1).astype(np.int32)
            bin1s = (block_data['bin1m_1'].values - start).astype(np.int32)
            bin2s = (block_data['bin1m_2'].values - start).astype(np.int32)
            counts = block_data['count'].values.astype(np.float32)
            np.add.at(mat, (cell_ids, bin1s, bin2s), counts)
            np.add.at(mat, (cell_ids, bin2s, bin1s), counts)
        matrices[bi] = mat
        if (bi + 1) % 10 == 0:
            print(f"    built {bi+1}/{len(all_blocks)} blocks")
    del merged
    return matrices
def knn_indices(features, k=20):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized = features / (norms + 1e-10)
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, -1)
    knn_idx = np.argsort(similarity, axis=1)[:, -k:]
    return knn_idx
def knn_impute(matrices, knn_idx, k=20):
    n_blocks, n_cells, h, w = matrices.shape
    imputed = np.zeros((n_blocks, n_cells, h, w), dtype=np.float64)
    for b in range(n_blocks):
        imputed[b] = matrices[b][knn_idx].mean(axis=1)
    return imputed
def gaussian_smooth(matrices, sigma=1.0):
    n_blocks, n_cells, h, w = matrices.shape
    smoothed = np.zeros_like(matrices, dtype=np.float64)
    for b in range(n_blocks):
        for i in range(n_cells):
            s = gaussian_filter(matrices[b, i], sigma=sigma)
            s = (s + s.T) / 2
            np.fill_diagonal(s, 0)
            smoothed[b, i] = s
    return smoothed
def coverage_normalize(matrices):
    n_blocks, n_cells, h, w = matrices.shape
    normalized = np.zeros_like(matrices, dtype=np.float32)
    for b in range(n_blocks):
        for i in range(n_cells):
            total = matrices[b, i].sum()
            if total > 0:
                normalized[b, i] = matrices[b, i] / total
    return normalized
def compute_d_x(imputed, eps=EPS_DX, gamma=GAMMA):
    d_x = np.power(imputed + eps, -1.0 / gamma)
    n = d_x.shape[0]
    for b in range(d_x.shape[0]):
        for i in range(d_x.shape[1]):
            np.fill_diagonal(d_x[b, i], 0.0)
    return d_x
def phase1_analyze():
    os.makedirs(STEP1D_DIR, exist_ok=True)
    print("[Phase 1]  KNN imputation + ")
    print("\n=== Step 1: Load LP_names ===")
    chrom_lps = load_lp_names(LP_NAMES_PATH)
    for chrom in sorted(chrom_lps.keys()):
        print(f"  {chrom}: {len(chrom_lps[chrom])} LPs")
    print("\n=== Step 2: Load cell labels ===")
    cell_names, cell_types, type_to_label = load_cell_labels(CELL_LABELS_PATH)
    n_cells = len(cell_names)
    labels = np.array([type_to_label[t] for t in cell_types])
    print(f"  Cells: {n_cells}")
    print("\n=== Step 3: Determine block positions ===")
    all_blocks = get_block_positions(chrom_lps)
    print(f"  Total block positions: {len(all_blocks)}")
    chrom_blocks = defaultdict(list)
    for blk in all_blocks:
        chrom_blocks[blk[0]].append(blk)
    for chrom in sorted(chrom_blocks.keys()):
        print(f"  {chrom}: {len(chrom_blocks[chrom])} blocks")
    print("\n=== Step 4: Build count matrices (genome-wide, one-pass) ===")
    all_matrices = build_all_count_matrices(
        SPARSE_MATRIX_PATH, chrom_lps, all_blocks, n_cells)
    print(f"  Built {len(all_matrices)} block matrices")
    print("\n=== Step 5: KNN imputation + Gaussian smoothing (per chromosome) ===")
    block_quality = []
    with h5py.File(H5_PATH, "w") as h5f:
        h5f.attrs["n_cells"] = n_cells
        h5f.create_dataset("labels", data=labels)
        barcodes = np.array(cell_names, dtype="S64")
        h5f.create_dataset("barcodes", data=barcodes)
        block_global_idx = 0
        for chrom in sorted(chrom_blocks.keys()):
            print(f"\n  --- {chrom} ({len(chrom_blocks[chrom])} blocks) ---")
            chrom_blks = chrom_blocks[chrom]
            n_blocks_chrom = len(chrom_blks)
            chrom_matrices = np.stack(
                [all_matrices[block_global_idx + bi] for bi in range(n_blocks_chrom)])
            for bi in range(n_blocks_chrom):
                all_matrices[block_global_idx + bi] = None
            print(f"    Count matrices: {chrom_matrices.shape}")
            print(f"    Computing KNN...")
            triu = np.triu_indices(PATCH_BINS, k=1)
            features = np.zeros((n_cells, n_blocks_chrom * len(triu[0])),
                                dtype=np.float64)
            for b in range(n_blocks_chrom):
                feat = chrom_matrices[b][:, triu[0], triu[1]]
                features[:, b * len(triu[0]):(b + 1) * len(triu[0])] = feat
            knn_idx = knn_indices(features, k=K_NN)
            print(f"    KNN computed (K={K_NN})")
            same_type = np.array([np.mean(labels[knn_idx[i]] == labels[i])
                                  for i in range(n_cells)])
            print(f"    Same-type neighbor frac: {same_type.mean():.3f}")
            print(f"    Imputing...")
            imputed = knn_impute(chrom_matrices, knn_idx, k=K_NN)
            print(f"    Gaussian smoothing (σ={GAUSSIAN_SIGMA})...")
            smoothed = gaussian_smooth(imputed, sigma=GAUSSIAN_SIGMA)
            print(f"    Saving to h5 + quality analysis...")
            for bi, (c, b_idx, start, end) in enumerate(chrom_blks):
                orig = chrom_matrices[bi]
                imp = imputed[bi]
                sm = smoothed[bi]
                orig_nz = (orig > 0).mean()
                imp_nz = (imp > 0).mean()
                sm_nz = (sm > 0).mean()
                grp = h5f.create_group(f"block_{block_global_idx}")
                grp.create_dataset("count", data=orig.astype(np.float32),
                                   compression="gzip")
                grp.create_dataset("imputed", data=sm.astype(np.float64),
                                   compression="gzip")
                grp.attrs["chrom"] = c
                grp.attrs["block_idx"] = b_idx
                grp.attrs["start_bin"] = start
                grp.attrs["end_bin"] = end
                grp.attrs["orig_nonzero_frac"] = orig_nz
                grp.attrs["imputed_nonzero_frac"] = imp_nz
                grp.attrs["smoothed_nonzero_frac"] = sm_nz
                block_quality.append({
                    "global_idx": block_global_idx,
                    "chrom": c,
                    "block_idx": b_idx,
                    "start_bin": start,
                    "end_bin": end,
                    "start_mb": start * RESOLUTION_1M // 1_000_000,
                    "end_mb": end * RESOLUTION_1M // 1_000_000,
                    "orig_nonzero_frac": float(orig_nz),
                    "imputed_nonzero_frac": float(imp_nz),
                    "smoothed_nonzero_frac": float(sm_nz),
                })
                print(f"      block_{block_global_idx} ({c}:{start}-{end} bin, "
                      f"{start*RESOLUTION_1M//1000000}-{end*RESOLUTION_1M//1000000}Mb): "
                      f"orig={orig_nz*100:.1f}% imp={imp_nz*100:.1f}% "
                      f"smooth={sm_nz*100:.1f}%")
                block_global_idx += 1
            del chrom_matrices, imputed, smoothed, features
            import gc
            gc.collect()
    print(f"\n=== Step 5: Save quality report ===")
    with open(os.path.join(STEP1D_DIR, "block_quality.json"), "w") as f:
        json.dump(block_quality, f, indent=2)
    bq = sorted(block_quality, key=lambda x: x["smoothed_nonzero_frac"],
                reverse=True)
    lines = ["=" * 90]
    lines.append("Block quality report (sorted by smoothed nonzero fraction)")
    lines.append("=" * 90)
    lines.append(f"{'idx':>4} {'chrom':>6} {'start_Mb':>8} {'end_Mb':>8} "
                 f"{'orig%':>6} {'imp%':>6} {'smooth%':>8}")
    lines.append("-" * 90)
    for q in bq:
        lines.append(f"{q['global_idx']:>4} {q['chrom']:>6} "
                      f"{q['start_mb']:>8} {q['end_mb']:>8} "
                      f"{q['orig_nonzero_frac']*100:>5.1f}% "
                      f"{q['imputed_nonzero_frac']*100:>5.1f}% "
                      f"{q['smoothed_nonzero_frac']*100:>7.1f}%")
    txt = "\n".join(lines)
    with open(os.path.join(STEP1D_DIR, "block_quality.txt"), "w") as f:
        f.write(txt)
    print(f"\n{txt}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(14, 5))
        indices = [q["global_idx"] for q in block_quality]
        orig = [q["orig_nonzero_frac"] * 100 for q in block_quality]
        smooth = [q["smoothed_nonzero_frac"] * 100 for q in block_quality]
        x = range(len(indices))
        ax.bar([i - 0.15 for i in x], orig, 0.3, label="original", color="#1f77b4")
        ax.bar([i + 0.15 for i in x], smooth, 0.3, label="KNN+Gaussian", color="#ff7f0e")
        chroms = [q["chrom"] for q in block_quality]
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"{q['chrom']}\n{q['start_mb']}M" for q in block_quality],
                           fontsize=6, rotation=90)
        ax.set_ylabel("Nonzero fraction (%)")
        ax.set_title("Genome-wide imputation quality per block")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(STEP1D_DIR, "block_quality.png"), dpi=150)
        plt.close()
        print(f"\n  Plot → {os.path.join(STEP1D_DIR, 'block_quality.png')}")
    except Exception as e:
        print(f"  [WARN] plot failed: {e}")
    print(f"\n[Phase 1] Done. Results → {STEP1D_DIR}")
    print(f"  h5: {H5_PATH}")
    print(f"  quality: block_quality.json / .txt / .png")
    print(f"\n  Next: review block_quality.txt, select blocks, then run:")
    print(f"  python exp/paper/section4_step1a_knn_impute.py --prepare --blocks 0,5,10,...")
def phase2_prepare(block_indices):
    os.makedirs(STEP1D_DIR, exist_ok=True)
    print(f"[Phase 2] Prepare training data for blocks: {block_indices}")
    with h5py.File(H5_PATH, "r") as h5f:
        labels = np.array(h5f["labels"])
        n_cells = h5f.attrs["n_cells"]
        n_blocks_sel = len(block_indices)
        print(f"  Cells: {n_cells}, Selected blocks: {n_blocks_sel}")
        all_count = []
        all_imputed = []
        block_info = []
        for bi in block_indices:
            key = f"block_{bi}"
            if key not in h5f:
                print(f"  [WARN] {key} not found, skipping")
                continue
            grp = h5f[key]
            count = np.array(grp["count"])
            imputed = np.array(grp["imputed"])
            all_count.append(count)
            all_imputed.append(imputed)
            block_info.append({
                "global_idx": bi,
                "chrom": grp.attrs["chrom"],
                "block_idx": grp.attrs["block_idx"],
                "start_bin": int(grp.attrs["start_bin"]),
                "end_bin": int(grp.attrs["end_bin"]),
                "smoothed_nonzero_frac": float(grp.attrs["smoothed_nonzero_frac"]),
            })
    n_actual = len(all_count)
    if n_actual == 0:
        print("  No valid blocks found!")
        return
    count_matrices = np.stack(all_count, axis=0)                                
    imputed_matrices = np.stack(all_imputed, axis=0)
    print(f"\n  Computing D_x (eps={EPS_DX}, gamma={GAMMA})...")
    d_x = compute_d_x(imputed_matrices)
    print(f"  Computing coverage normalization...")
    covnorm = coverage_normalize(count_matrices)
    rng = np.random.RandomState(RANDOM_SEED)
    train_idx, val_idx = [], []
    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        rng.shuffle(idx)
        n_train = int(len(idx) * TRAIN_FRAC)
        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train:])
    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")
    def flatten_blocks(arrays, idx):
        n_blocks, n = len(arrays), len(idx)
        return arrays[:, idx].reshape(n_blocks * n, PATCH_BINS, PATCH_BINS)
    train_cov = flatten_blocks(covnorm, train_idx)
    train_dx = flatten_blocks(d_x, train_idx)
    val_cov = flatten_blocks(covnorm, val_idx)
    val_dx = flatten_blocks(d_x, val_idx)
    train_block_ids = np.array(
        [[bi] * len(train_idx) for bi in range(n_actual)]).flatten()
    val_block_ids = np.array(
        [[bi] * len(val_idx) for bi in range(n_actual)]).flatten()
    train_labels = np.array(
        [labels[i] for i in train_idx] * n_actual)
    val_labels = np.array(
        [labels[i] for i in val_idx] * n_actual)
    with h5py.File(H5_PATH, "r") as h5f:
        barcodes = np.array(h5f["barcodes"]).astype(str)
    train_barcodes = np.array(
        [barcodes[i] for i in train_idx] * n_actual)
    val_barcodes = np.array(
        [barcodes[i] for i in val_idx] * n_actual)
    np.savez(os.path.join(STEP1D_DIR, "train_data.npz"),
             covnorm=train_cov,
             d_x=train_dx,
             labels=train_labels,
             barcodes=train_barcodes,
             block_ids=train_block_ids)
    np.savez(os.path.join(STEP1D_DIR, "val_data.npz"),
             covnorm=val_cov,
             d_x=val_dx,
             labels=val_labels,
             barcodes=val_barcodes,
             block_ids=val_block_ids)
    lines = ["=" * 60]
    lines.append("Phase 2  Training data prepared")
    lines.append("=" * 60)
    lines.append(f"Selected blocks: {block_indices}")
    for bi in block_info:
        lines.append(f"  block_{bi['global_idx']} ({bi['chrom']}:{bi['start_bin']}-{bi['end_bin']}): "
                      f"smoothed_nz={bi['smoothed_nonzero_frac']*100:.1f}%")
    lines.append(f"\nTrain: {len(train_cov)} samples ({n_actual} blocks × {len(train_idx)} cells)")
    lines.append(f"Val: {len(val_cov)} samples ({n_actual} blocks × {len(val_idx)} cells)")
    txt = "\n".join(lines)
    with open(os.path.join(STEP1D_DIR, "prepare_summary.txt"), "w") as f:
        f.write(txt)
    print(f"\n{txt}")
    print(f"\n[Phase 2] Done. → {STEP1D_DIR}")
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true",
                    help="Phase 1:  imputation + ")
    ap.add_argument("--prepare", action="store_true",
                    help="Phase 2:  D_x + ")
    ap.add_argument("--blocks", type=str, default="",
                    help=" block  (Phase 2)")
    args = ap.parse_args()
    if args.analyze:
        phase1_analyze()
    elif args.prepare:
        if not args.blocks:
            print("Error: --prepare requires --blocks (e.g. --blocks 0,5,10)")
            return
        block_indices = [int(b.strip()) for b in args.blocks.split(",")]
        phase2_prepare(block_indices)
    else:
        ap.print_help()
if __name__ == "__main__":
    main()

import os
import sys
import json
import numpy as np
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from exp._helpers import results_dir, load_model
from src.data import compute_oe_matrix
SU_DIR = os.path.join(os.path.dirname(ROOT), "Su 2020 cell")
STEP3_DIR = os.path.join(ROOT, "exp", "results", "section4_step3_eval")
STEP1E_DIR = os.path.join(ROOT, "exp", "results", "section4_step1b")
KIM_DIR = os.path.join(os.path.dirname(ROOT), "data", "scHiC.Kim2020", "input")
PATCH_BINS = 40
CHR2_BLOCKS = [28, 29, 30, 31, 32, 33]
BLOCK_START = {28: 0, 29: 40, 30: 80, 31: 120, 32: 160, 33: 200}
IMR90_LABEL = 4
MIN_VALID_BINS = 25
ALL63_ORDER = [38, 8, 1, 48, 40, 49, 28, 20, 13, 4, 37, 62, 24, 33, 9, 5,
               29, 18, 56, 26, 59, 52, 14, 6, 53, 12, 32, 55, 60, 41, 0, 51,
               31, 2, 11, 45, 43, 44, 16, 47, 34, 7, 57, 23, 21, 27, 58, 46,
               39, 50, 10, 42, 25, 54, 30, 22, 35, 36, 15, 17, 19, 3, 61]
TOP20_ORDER = [1, 4, 5, 8, 9, 13, 18, 20, 24, 26, 28, 29, 33, 37, 38, 40,
               48, 49, 56, 62]                                           
P10_ORDER = [0, 1, 28, 37, 41, 45, 49, 53, 57, 6]                   
ALL63_MODEL = os.path.join(
    ROOT, "models", "TORE_finetune_1000",
    "TORE_finetune_1000_epoch{}.pth")
def spearman(a, b):
    from scipy.stats import spearmanr
    return spearmanr(a, b)[0]
def corr_stats(a, b):
    from scipy.stats import pearsonr, spearmanr
    n = a.shape[0]
    triu = np.triu_indices(n, k=1)
    x, y = a[triu], b[triu]
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 20:
        return np.nan, np.nan, np.nan, np.nan, int(ok.sum())
    pr, sr = pearsonr(x[ok], y[ok]), spearmanr(x[ok], y[ok])
    return pr[0], sr[0], pr[1], sr[1], int(ok.sum())
def build_imaging_reference():
    print("===  (Su chr2, ) ===", flush=True)
    cols = ["Z(nm)", "X(nm)", "Y(nm)", "Genomic coordinate",
            "Chromosome copy number"]
    df = pd.read_csv(os.path.join(SU_DIR, "chromosome2.tsv"), sep="\t",
                     usecols=cols)
    loci = sorted(df["Genomic coordinate"].unique(),
                  key=lambda s: int(s.split(":")[1].split("-")[0]))
    locus_hg19 = {}
    for lc in loci:
        start = int(lc.split(":")[1].split("-")[0])
        locus_hg19[lc] = int((start + 125_000) // 1_000_000)
    df["bin_hg19"] = df["Genomic coordinate"].map(locus_hg19)
    df = df.dropna(subset=["bin_hg19", "Z(nm)", "X(nm)", "Y(nm)"])
    df["bin_hg19"] = df["bin_hg19"].astype(np.int32)
    grp = df.groupby(["Chromosome copy number", "bin_hg19"])[
        ["Z(nm)", "X(nm)", "Y(nm)"]].mean().reset_index()
    copies = grp["Chromosome copy number"].unique()
    imaging, ok_bins_map = {}, {}
    for blk in CHR2_BLOCKS:
        s = BLOCK_START[blk]
        sub = grp[(grp["bin_hg19"] >= s) & (grp["bin_hg19"] < s + PATCH_BINS)]
        dists = []
        for c in copies:
            cs = sub[sub["Chromosome copy number"] == c]
            arr = np.full((PATCH_BINS, 3), np.nan)
            for _, row in cs.iterrows():
                arr[int(row["bin_hg19"]) - s] = [row["Z(nm)"], row["X(nm)"],
                                                 row["Y(nm)"]]
            ok = ~np.isnan(arr[:, 0])
            if ok.sum() < MIN_VALID_BINS:
                continue
            diff = arr[:, None, :] - arr[None, :, :]
            dists.append(np.sqrt(np.sum(diff ** 2, axis=2)))
        med = np.nanmedian(np.array(dists), axis=0)
        imaging[blk] = med
        ok_bins_map[blk] = ~np.isnan(med[0])
        print(f"  block_{blk}: {len(dists)} , valid bins="
              f"{ok_bins_map[blk].sum()}", flush=True)
    return imaging, ok_bins_map
def eval_arm(u1_3d, labels, barcodes, block_pos, imaging, ok_bins_map,
             imr_only=True):
    imr = np.where(labels == IMR90_LABEL)[0] if imr_only else \
        np.arange(len(labels))
    out, percell_detail = {}, {}
    rng = np.random.RandomState(0)
    for blk, ax in block_pos.items():
        if blk not in imaging:
            continue
        ok_bins = ok_bins_map[blk]
        U = u1_3d[ax][imr][:, ok_bins, :]
        diff = U[:, :, None, :] - U[:, None, :, :]
        D = np.sqrt(np.sum(diff ** 2, axis=3))
        med_u1 = np.median(D, axis=0)
        img_sub = imaging[blk][np.ix_(ok_bins, ok_bins)]
        p_agg, s_agg, p_agg_p, s_agg_p, npairs = corr_stats(med_u1, img_sub)
        triu = np.triu_indices(ok_bins.sum(), k=1)
        yv = img_sub[triu]
        per_cell = np.array([spearman(D[i][triu], yv)
                             for i in range(len(imr))])
        yv_null = yv[rng.permutation(len(yv))]
        per_cell_null = np.array([spearman(D[i][triu], yv_null)
                                  for i in range(len(imr))])
        out[f"block_{blk}"] = {
            "agg_pearson": float(p_agg), "agg_spearman": float(s_agg),
            "agg_pearson_p": float(p_agg_p), "agg_spearman_p": float(s_agg_p),
            "percell_spearman_mean": float(np.nanmean(per_cell)),
            "percell_spearman_std": float(np.nanstd(per_cell)),
            "percell_null_mean": float(np.nanmean(per_cell_null)),
            "n_pairs": int(npairs), "n_cells": int(len(imr)),
        }
        percell_detail[blk] = (imr, per_cell)
        print(f"    block_{blk}: agg ρ={s_agg:.3f}, "
              f" ρ={np.nanmean(per_cell):.3f}±{np.nanstd(per_cell):.3f}, "
              f"null={np.nanmean(per_cell_null):.3f}", flush=True)
    return out, percell_detail
def main():
    out_dir = results_dir("section4_step4b")
    imaging, ok_bins_map = build_imaging_reference()
    print("\n=== KNN  () ===", flush=True)
    lab_df = pd.read_csv(os.path.join(KIM_DIR, "cell_labels.txt"),
                         sep="\t", header=None, names=["barcode", "type"])
    type2id = {t: i for i, t in enumerate(
        ["GM12878", "H1Esc", "HAP1", "HFF", "IMR90"])}
    all_labels = lab_df["type"].map(type2id).values
    bc2row = {b: i for i, b in enumerate(lab_df["barcode"])}
    knn = np.load(os.path.join(STEP1E_DIR, "knn_v2.npz"))
    knn_idx = knn["knn_idx"]
    stf = (all_labels[knn_idx] == all_labels[:, None]).mean(axis=1)
    print(f"  ={stf.mean():.3f}; "
          f"IMR90 ={stf[all_labels == 4].mean():.3f}")
    report = {}
    all63_pos = {b: ALL63_ORDER.index(b) for b in CHR2_BLOCKS}
    arms_all63 = {
        "FINAL_all63ep20": "latent_finetune_all63ep20.npz",
        "ZEROSHOT_all63": "latent_finetune_zero_all63.npz",
        "FINAL_all63ep30": "latent_finetune_all63ep30.npz",
        "FINAL_all63ep20_VAL": "latent_finetune_all63ep20_val.npz",
        "K10g1_all63ep20": "latent_finetune_k10g1_ep20.npz",
        "K5g1_all63ep20": "latent_finetune_k5g1_ep20.npz",
        "K20g1_all63ep20_VAL": "latent_finetune_k20g1.npz",
        "K10g3_all63ep20_VAL": "latent_finetune_k10g3.npz",
        "K10input_g3_all63ep20_VAL": "latent_finetune_k10input_g3.npz",
    }
    percell_store = {}
    for arm, fname in arms_all63.items():
        path = os.path.join(STEP3_DIR, fname)
        if not os.path.exists(path):
            print(f"  [skip] {fname}")
            continue
        d = np.load(path)
        print(f"\n=== [{arm}] () ===", flush=True)
        res, detail = eval_arm(d["u1_3d"], d["labels"], d["barcodes"],
                               all63_pos, imaging, ok_bins_map)
        report[arm] = res
        percell_store[arm] = (d, detail)
    top20_pos = {b: TOP20_ORDER.index(b) for b in [28, 29, 33]}
    arms_top20 = {
        "FINAL_top20_ddp50": "latent_finetune_ddp50.npz",
        "RECON_only_top20_b4": "latent_finetune_b4_top20.npz",
        "ZEROSHOT_top20": "latent_finetune_zero_top20.npz",
    }
    for arm, fname in arms_top20.items():
        path = os.path.join(STEP3_DIR, fname)
        if not os.path.exists(path):
            continue
        d = np.load(path)
        print(f"\n=== [{arm}] ===", flush=True)
        res, _ = eval_arm(d["u1_3d"], d["labels"], d["barcodes"],
                          top20_pos, imaging, ok_bins_map)
        report[arm] = res
    p10_pos = {28: P10_ORDER.index(28)}
    for arm, fname in {"K10g1_top10_p1": "latent_finetune_p1.npz",
                       "K10g3_top10_p2": "latent_finetune_p2.npz"}.items():
        path = os.path.join(STEP3_DIR, fname)
        if not os.path.exists(path):
            continue
        d = np.load(path)
        print(f"\n=== [{arm}] (val ) ===", flush=True)
        res, _ = eval_arm(d["u1_3d"], d["labels"], d["barcodes"],
                          p10_pos, imaging, ok_bins_map)
        report[arm] = res
    ep10_path = ALL63_MODEL.format(10)
    if os.path.exists(ep10_path):
        import h5py
        import torch
        print("\n=== [D4] all63 ep10  (chr2 blocks) ===", flush=True)
        d_ref = np.load(os.path.join(STEP3_DIR, "latent_finetune_all63ep20.npz"))
        barcodes = d_ref["barcodes"].astype(str)
        labels = d_ref["labels"]
        with h5py.File(os.path.join(STEP1E_DIR, "genome_imputed_v2.h5"),
                       "r") as h5f:
            h5_barcodes = np.array(h5f["barcodes"]).astype(str)
            h5_row = {b: i for i, b in enumerate(h5_barcodes)}
            rows = np.array([h5_row[b] for b in barcodes])
            mats = []
            for blk in CHR2_BLOCKS:
                imp = np.array(h5f[f"block_{blk}"]["imputed"],
                               dtype=np.float32)
                mats.append(imp[rows])
        oe = []
        for blk_i in range(len(CHR2_BLOCKS)):
            for i in range(len(rows)):
                m = compute_oe_matrix(mats[blk_i][i])
                m[np.isnan(m)] = 0
                m[np.isinf(m)] = 0
                oe.append(m.astype(np.float32))
        oe = np.array(oe, dtype=np.float32)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _ = load_model(ep10_path, device=device, patch_bins=PATCH_BINS)
        model.eval()
        lat = np.zeros((len(oe), PATCH_BINS, 3), dtype=np.float32)
        with torch.no_grad():
            for i in range(len(oe)):
                x = torch.tensor(oe[i], dtype=torch.float32, device=device)
                _, u1 = model(x)
                lat[i] = u1.cpu().numpy()
        n_cells = len(rows)
        lat = lat.reshape(len(CHR2_BLOCKS), n_cells, PATCH_BINS, 3)
        pos = {b: i for i, b in enumerate(CHR2_BLOCKS)}
        res, _ = eval_arm(lat, labels, barcodes, pos, imaging, ok_bins_map)
        report["FINAL_all63ep10"] = res
    print("\n=== [D1] Δρ(-) vs KNN  ===", flush=True)
    d1 = {}
    if "FINAL_all63ep20" in percell_store and \
            "ZEROSHOT_all63" in percell_store:
        d_fin, det_fin = percell_store["FINAL_all63ep20"]
        d_zero, det_zero = percell_store["ZEROSHOT_all63"]
        bc_fin = d_fin["barcodes"].astype(str)
        bc_zero = d_zero["barcodes"].astype(str)
        assert np.array_equal(bc_fin, bc_zero)
        rows = np.array([bc2row[b] for b in bc_fin])
        deltas, stfs, cells = [], [], []
        for blk in CHR2_BLOCKS:
            imr_f, pc_f = det_fin[blk]
            imr_z, pc_z = det_zero[blk]
            assert np.array_equal(imr_f, imr_z)
            deltas.append(pc_z - pc_f)
            stfs.append(stf[rows[imr_f]])
            cells.append(imr_f)
        deltas = np.concatenate(deltas)
        stfs = np.concatenate(stfs)
        from scipy.stats import pearsonr, spearmanr, binomtest
        pr = pearsonr(stfs, deltas)[0]
        sr = spearmanr(stfs, deltas)[0]
        n_pos = int((deltas > 0).sum())
        n_neg = int((deltas < 0).sum())
        d1 = {"pearson_r": float(pr), "spearman_r": float(sr),
              "n": int(len(deltas)),
              "delta_mean": float(np.mean(deltas)),
              "stf_mean_imr90_cells": float(np.mean(stfs))}
        if n_pos + n_neg > 0:
            d1["sign_test"] = {
                "n_pos": n_pos, "n_neg": n_neg,
                "p_binom": float(binomtest(
                    n_pos, n_pos + n_neg, 0.5).pvalue)}
        q = np.quantile(stfs, [0.33, 0.67])
        for name, m in [("low", stfs < q[0]),
                        ("mid", (stfs >= q[0]) & (stfs < q[1])),
                        ("high", stfs >= q[1])]:
            d1[f"delta_{name}_tercile"] = {
                "mean": float(np.mean(deltas[m])), "n": int(m.sum()),
                "stf_mean": float(np.mean(stfs[m]))}
        print(f"  Pearson r={pr:.3f}, Spearman r={sr:.3f} (n={len(deltas)})")
        print(f"  Δρ ={np.mean(deltas):+.3f}")
        for t in ("low", "mid", "high"):
            print(f"    {t}: Δρ={d1[f'delta_{t}_tercile']['mean']:+.3f} "
                  f"(stf={d1[f'delta_{t}_tercile']['stf_mean']:.2f})")
    report["D1_delta_vs_knn_purity"] = d1
    lines = ["=" * 78,
             " ( block )", "=" * 78, ""]
    lines.append(f"{'arm':26s} {'block':>7s} {'aggρ':>7s} {'cellρ':>16s} "
                 f"{'null':>7s} {'cells':>6s}")
    for arm, res in report.items():
        if not isinstance(res, dict) or arm.startswith("D1"):
            continue
        for bk, r in sorted(res.items()):
            lines.append(
                f"{arm:26s} {bk:>7s} {r['agg_spearman']:7.3f} "
                f"{r['percell_spearman_mean']:6.3f}±{r['percell_spearman_std']:.3f}"
                f"  {r['percell_null_mean']:7.3f} {r['n_cells']:6d}")
    lines.append("")
    lines.append("[D1] Δρ(-) vs KNN : "
                 + json.dumps(d1, indent=1, ensure_ascii=False))
    txt = "\n".join(lines)
    with open(os.path.join(out_dir, "imaging_diagnostics.txt"), "w") as f:
        f.write(txt)
    with open(os.path.join(out_dir, "imaging_diagnostics.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(f"\n{txt}")
    print(f"\n[Step11d-fix] Done → {out_dir}/imaging_diagnostics.*")
if __name__ == "__main__":
    main()

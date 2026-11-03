import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from exp._helpers import load_model
from exp.paper.section4_step1c_dx_variants import knn_impute_k, smooth_and_clean
from exp.paper.section4_step3b_feature_sweep import simple_pca
from exp.paper.section4_step4d_imaging_diagnostics import (
    CHR2_BLOCKS, BLOCK_START)
SU_DIR = os.path.join(os.path.dirname(ROOT), "Su 2020 cell")
OUT_DIR = os.path.join(ROOT, "exp", "results", "section4_simrecovery_higashi")
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")             
PATCH = 40
CHR2_NBINS = 243
MIN_VALID_BINS_B = 25
MIN_VALID_BINS_A = 122
DIST_CUTOFF_NM = 2000.0
SEED_A = 42
SEEDS_B = [42, 43, 44]
DEPTHS_A = [(250, 500), (500, 1000), (1000, 2500), (2500, 5000)]
SCALE = (40 * 39 // 2) / (CHR2_NBINS * (CHR2_NBINS - 1) // 2)              
DEPTHS_B = [(max(1, int(round(lo * SCALE))), max(2, int(round(hi * SCALE))))
            for lo, hi in DEPTHS_A]
K_IMPUTE = 20
RWR_P = 0.5                           
RWR_TOL = 1e-6
HICREP_H_MAIN = 1
HICREP_H_SENS = 3
MODELS = {
    "zeroshot": os.path.join(ROOT, "models", "TORE_50kb",
        "TORE_50kb_epoch500.pth"),
    "ft1000": os.path.join(ROOT, "models", "TORE_finetune_1000",
        "TORE_finetune_1000_epoch20.pth"),
    "ft12k": os.path.join(ROOT, "models", "TORE_finetune_12599",
        "TORE_finetune_12599_epoch20.pth"),
    "selfweighted": os.path.join(ROOT, "models", "TORE_finetune_selfweighted",
        "TORE_finetune_selfweighted_epoch20.pth"),
}
RUN_T0 = time.time()
_LOG_PATH = os.path.join(OUT_DIR, "run.log")
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
def gate(gid, msg, ok, hard=True):
    tag = "PASS" if ok else ("HARD-STOP" if hard else "WARN")
    log(f"  GATE {gid} [{tag}] {msg}")
    if not ok and hard:
        raise RuntimeError(f" {gid} : {msg}")
def load_chr2_grouped():
    cols = ["Z(nm)", "X(nm)", "Y(nm)", "Genomic coordinate",
            "Chromosome copy number"]
    df = pd.read_csv(os.path.join(SU_DIR, "chromosome2.tsv"), sep="\t",
                     usecols=cols)
    loci = sorted(df["Genomic coordinate"].unique(),
                  key=lambda s: int(s.split(":")[1].split("-")[0]))
    locus_bin = {lc: int((int(lc.split(":")[1].split("-")[0]) + 125_000)
                         // 1_000_000) for lc in loci}
    df["bin1m"] = df["Genomic coordinate"].map(locus_bin)
    df = df.dropna(subset=["bin1m", "Z(nm)", "X(nm)", "Y(nm)"])
    df["bin1m"] = df["bin1m"].astype(np.int32)
    grp = df.groupby(["Chromosome copy number", "bin1m"])[
        ["Z(nm)", "X(nm)", "Y(nm)"]].mean().reset_index()
    n_loci, n_copies = len(loci), grp["Chromosome copy number"].nunique()
    log(f"  chromosome2.tsv: {len(df)} rows, {n_loci} loci, {n_copies} copies")
    gate("P0", f" ( 935/3029): {n_loci}/{n_copies}",
         n_loci == 935 and n_copies == 3029)
    return grp
def coords_to_prob(arr):
    valid = ~np.isnan(arr[:, 0])
    diff = arr[:, None, :] - arr[None, :, :]
    d = np.sqrt(np.sum(diff ** 2, axis=2))
    with np.errstate(divide="ignore"):
        p = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0)
    p[d > DIST_CUTOFF_NM] = 0.0
    idx = np.arange(arr.shape[0])
    p[idx, idx] = 0.0
    return p, d, valid
def truth_A(grp):
    keep, drop = [], 0
    for c, cs in grp.groupby("Chromosome copy number"):
        arr = np.full((CHR2_NBINS, 3), np.nan)
        for _, row in cs.iterrows():
            arr[int(row["bin1m"])] = [row["Z(nm)"], row["X(nm)"],
                                      row["Y(nm)"]]
        if (~np.isnan(arr[:, 0])).sum() < MIN_VALID_BINS_A:
            drop += 1
            continue
        keep.append((int(c), arr))
    log(f"  A : {len(keep)} , {drop}  (<{MIN_VALID_BINS_A} bins)")
    return keep, drop
def truth_B(grp):
    truth, drops = {}, {}
    for blk in CHR2_BLOCKS:
        s = BLOCK_START[blk]
        sub = grp[(grp["bin1m"] >= s) & (grp["bin1m"] < s + PATCH)]
        lst, drop = [], 0
        for c, cs in sub.groupby("Chromosome copy number"):
            arr = np.full((PATCH, 3), np.nan)
            for _, row in cs.iterrows():
                arr[int(row["bin1m"]) - s] = [row["Z(nm)"], row["X(nm)"],
                                              row["Y(nm)"]]
            if (~np.isnan(arr[:, 0])).sum() < MIN_VALID_BINS_B:
                drop += 1
                continue
            lst.append((int(c), arr))
        truth[blk], drops[blk] = lst, drop
    log(f"  B : 6  {sum(len(v) for v in truth.values())} (,), "
        f" {sum(drops.values())}")
    return truth, drops
def sample_counts(p, triu_idx, rng, lo, hi):
    probs = p[triu_idx[0], triu_idx[1]]
    nz = probs > 0
    if nz.sum() == 0:
        return None, 0
    n = int(rng.integers(lo, hi + 1))
    if n == 0:
        return None, 0
    pv = probs[nz] / probs[nz].sum()
    counts = rng.multinomial(n, pv)
    mat = np.zeros(p.shape, dtype=np.int16)
    sel = counts > 0
    ii, jj = triu_idx[0][nz][sel], triu_idx[1][nz][sel]
    mat[ii, jj] = counts[sel]
    mat[jj, ii] = counts[sel]
    return mat, n
def arm_raw(counts):
    return counts.astype(np.float64)
def arm_schicluster(counts, p_restart=RWR_P):
    n = counts.shape[0]
    b = counts.astype(np.float64).copy()
    b[1:, :] += counts[:-1, :]
    b[:-1, :] += counts[1:, :]
    b[:, 1:] += counts[:, :-1]
    b[:, :-1] += counts[:, 1:]
    b[1:, 1:] += counts[:-1, :-1]
    b[1:, :-1] += counts[:-1, 1:]
    b[:-1, 1:] += counts[1:, :-1]
    b[:-1, :-1] += counts[1:, 1:]
    rs = b.sum(axis=1)
    if rs.max() == 0:
        return np.zeros_like(b)
    c = b / np.where(rs > 0, rs, 1.0)[:, None]
    q = np.eye(n)
    for _ in range(1000):
        q_new = (1.0 - p_restart) * (q @ c) + p_restart * np.eye(n)
        if np.linalg.norm(q_new - q, 2) <= RWR_TOL:
            q = q_new
            break
        q = q_new
    return q
def knn_impute_batch(mats, k=K_IMPUTE):
    m = np.asarray(mats, dtype=np.float64)
    n = len(m)
    if n < k + 2:
        return np.zeros_like(m)
    triu = np.triu_indices(m.shape[-1], k=1)
    feat = m[:, triu[0], triu[1]]
    if feat.sum() == 0:
        return np.zeros_like(m)
    pc = np.nan_to_num(simple_pca(feat, min(20, n - 1)))
    norms = np.linalg.norm(pc, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    pcn = pc / norms
    sim = pcn @ pcn.T
    np.fill_diagonal(sim, -1)
    k_adj = min(k, n - 1)
    knn_idx = np.argsort(-sim, axis=1)[:, :k_adj]
    knn_sim = np.maximum(np.take_along_axis(sim, knn_idx, axis=1), 0) + 1e-3
    imp = smooth_and_clean(knn_impute_k(m, knn_idx, knn_sim, k_adj))
    return imp
W_SELF_POOL = 0.3                          
def knn_impute_batch_selfweighted(mats, k=K_IMPUTE,
                                   w_self=W_SELF_POOL):
    m = np.asarray(mats, dtype=np.float64)
    n = len(m)
    if n < k + 2:
        return np.zeros_like(m)
    triu = np.triu_indices(m.shape[-1], k=1)
    feat = m[:, triu[0], triu[1]]
    if feat.sum() == 0:
        return np.zeros_like(m)
    pc = np.nan_to_num(simple_pca(feat, min(20, n - 1)))
    norms = np.linalg.norm(pc, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    pcn = pc / norms
    sim = pcn @ pcn.T
    np.fill_diagonal(sim, -1)
    k_adj = min(k, n - 1)
    knn_idx = np.argsort(-sim, axis=1)[:, :k_adj]
    knn_sim = np.maximum(np.take_along_axis(sim, knn_idx, axis=1), 0) + 1e-3
    navg = knn_impute_k(m, knn_idx, knn_sim, k_adj)
    self_term = np.zeros_like(navg)
    for i in range(n):
        c = m[i]
        pos = c > 0
        if pos.sum() == 0:
            continue
        m_obs = float(c[pos].mean())
        if m_obs <= 0:
            continue
        navg_obs = navg[i][pos & (navg[i] > 0)]
        if len(navg_obs) == 0:
            navg_obs = navg[i][navg[i] > 0]
        if len(navg_obs) == 0:
            continue
        s = float(navg_obs.mean())
        self_term[i] = c * (s / m_obs)
    return w_self * self_term + (1.0 - w_self) * navg
def batch_oe(matrices):
    b, n, _ = matrices.shape
    dist = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    expected = np.zeros((b, n))
    for d in range(n):
        if d > 0:
            diag = np.concatenate([
                np.diagonal(matrices, offset=d, axis1=1, axis2=2),
                np.diagonal(matrices, offset=-d, axis1=1, axis2=2)], axis=1)
        else:
            diag = np.diagonal(matrices, offset=0, axis1=1, axis2=2)
        mask = diag > 0
        expected[:, d] = np.where(mask.any(axis=1),
            (diag * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1), 0)
    e = expected[:, dist]
    return np.where(e > 0, matrices / np.where(e > 0, e, 1), 0).astype(np.float32)
def arm_ring(model, x_mats, device):
    import torch
    oe = batch_oe(x_mats)
    recon = np.zeros_like(oe)
    u1 = np.zeros((len(oe), PATCH, 3), dtype=np.float32)
    with torch.no_grad():
        for i in range(len(oe)):
            x = torch.tensor(oe[i], dtype=torch.float32, device=device)
            r, u = model(x)
            recon[i] = r.cpu().numpy()
            u1[i] = u.cpu().numpy()
    return recon, u1
def ring_dists(u1_block):
    g = u1_block @ u1_block.transpose(0, 2, 1)
    sq = np.diagonal(g, axis1=1, axis2=2)
    return np.sqrt(np.maximum(sq[:, :, None] + sq[:, None, :] - 2 * g, 0))
def to_contact_from_dist(d, eps=0.01):
    c = 1.0 / (d + eps)
    idx = np.arange(d.shape[-1])
    c[..., idx, idx] = 0.0
    return c
def symmetrize(m):
    s = (m + m.T) / 2
    idx = np.arange(m.shape[0])
    s[idx, idx] = 0.0
    return s
def eval_index(n, valid):
    iu = np.triu_indices(n, k=2)
    vm = valid[iu[0]] & valid[iu[1]]
    return (iu[0][vm], iu[1][vm])
def quantile_normalize_stack(mats_list):
    a = np.stack([np.sort(m.ravel()) for m in mats_list])           
    ref = a.mean(axis=0)                                           
    L = ref.shape[0]
    x = np.arange(L)
    out = []
    for m in mats_list:
        v = m.ravel()
        order = np.argsort(v, kind="mergesort")
        sv = v[order]
        uniq, first, cnt = np.unique(sv, return_index=True,
                                     return_counts=True)
        avg_rank_sorted = np.repeat((first + first + cnt - 1) / 2.0, cnt)
        ranks = np.empty(L)
        ranks[order] = avg_rank_sorted
        q = np.interp(ranks, x, ref)                      
        out.append(q.reshape(m.shape))
    return out
def compute_hicrep_scc(m1, m2, h):
    import scipy.sparse as sp
    from hicrep.utils import meanFilterSparse
    from hicrep import hicrep as hicrep_mod
    s1, s2 = m1.sum(), m2.sum()
    if s1 == 0 or s2 == 0:
        return float("nan")
    n = m1.shape[0]
    scc = hicrep_mod.sccByDiag(
        meanFilterSparse(sp.coo_matrix(m1 / s1), h),
        meanFilterSparse(sp.coo_matrix(m2 / s2), h), n)
    return float(scc) if np.isfinite(scc) else float("nan")
def eval_maps(maps_dict, truth_p, ev_idx, hicrep_hs):
    for m in list(maps_dict.values()) + [truth_p]:
        if not np.all(np.isfinite(m)):
            raise RuntimeError(
                f"eval_maps  ( G5/G6 ): "
                f"nan={int((~np.isfinite(m)).sum())}")
    names = ["truth"] + list(maps_dict.keys())
    qn = quantile_normalize_stack(
        [truth_p] + [maps_dict[a] for a in maps_dict])
    tq = qn[0]
    i, j = ev_idx
    res = {}
    for k, a in enumerate(maps_dict.keys()):
        mq = qn[k + 1]
        x, y = mq[i, j], tq[i, j]
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            pr = sr = float("nan")
        else:
            pr = float(pearsonr(x, y)[0])
            sr = float(spearmanr(x, y)[0])
        entry = {"pearson": pr, "spearman": sr}
        for h in hicrep_hs:
            entry[f"hicrep_h{h}"] = compute_hicrep_scc(mq, tq, h)
        res[a] = entry
    return res
def boxstats(vals):
    v = np.asarray(vals, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"median": None, "q1": None, "q3": None,
                "whisk_lo": None, "whisk_hi": None, "n": 0}
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    iqr = q3 - q1
    inl = v[(v >= q1 - 1.5 * iqr) & (v <= q3 + 1.5 * iqr)]
    return {"median": float(med), "q1": float(q1), "q3": float(q3),
            "whisk_lo": float(inl.min()) if len(inl) else None,
            "whisk_hi": float(inl.max()) if len(inl) else None,
            "n": int(len(v))}
def ckpt_path(kind, key):
    safe = key.replace("|", "_").replace("-", "_")
    return os.path.join(CKPT_DIR, f"{kind}_{safe}.json")
def save_ckpt(kind, key, box_results, per_cell_spearman):
    os.makedirs(CKPT_DIR, exist_ok=True)
    with open(ckpt_path(kind, key), "w", encoding="utf-8") as f:
        json.dump({"boxstats": box_results,
                   "per_cell_spearman": per_cell_spearman},
                  f, ensure_ascii=False, default=str)
def load_ckpt(kind, key):
    p = ckpt_path(kind, key)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)
def run_A(grp, copy_limit, fast):
    log("=== A  ( chr2 243x243; raw + scHiCluster) ===")
    keep, _ = truth_A(grp)
    if copy_limit:
        keep = keep[:copy_limit]
    triu243 = np.triu_indices(CHR2_NBINS, k=1)
    hs = (HICREP_H_MAIN,) if fast else (HICREP_H_MAIN, HICREP_H_SENS)
    results, raw_med = {}, {}
    pnz_all = []
    for lo, hi in DEPTHS_A:
        name = f"N{lo}-{hi}"
        if not copy_limit:
            ck = load_ckpt("A", name)
            if ck is not None:
                results[name] = ck["boxstats"]
                raw_med[name] = results[name]["raw"]["spearman"]["median"]
                log(f"  A {name}: ,  (raw  {raw_med[name]})")
                continue
        per_arm = {"raw": [], "schicluster": []}
        for c, arr in keep:
            rng = np.random.default_rng([SEED_A, c, 0])
            p, d, valid = coords_to_prob(arr)
            pnz_all.append((p > 0).mean())
            mat, n = sample_counts(p, triu243, rng, lo, hi)
            if mat is None:
                continue
            ev = eval_index(CHR2_NBINS, valid)
            maps = {"raw": arm_raw(mat),
                    "schicluster": arm_schicluster(mat.astype(np.float64))}
            met = eval_maps(maps, p, ev, hs)
            for a in per_arm:
                per_arm[a].append(met[a])
        results[name] = {a: {m: boxstats([r[m] for r in vals])
                             for m in per_arm[a][0]}
                         for a, vals in per_arm.items()}
        raw_med[name] = results[name]["raw"]["spearman"]["median"]
        if not copy_limit:
            save_ckpt("A", name, results[name],
                      {a: [r["spearman"] for r in vals]
                       for a, vals in per_arm.items()})
        log(f"  A {name}: n={len(per_arm['raw'])}, raw spearman  "
            f"{raw_med[name]}")
    gate("G1", f"A >0", all(results[k]["raw"]["spearman"]["n"] > 0
         for k in results))
    med_pnz = float(np.median(pnz_all))
    gate("G3", f"A  {med_pnz:.3f} ( 5%-95%)",
         0.05 <= med_pnz <= 0.95, hard=False)
    meds = [raw_med[f"N{lo}-{hi}"] for lo, hi in DEPTHS_A]
    gate("G4", f"A raw  {['%.3f' % m for m in meds]} ",
         all(meds[i] < meds[i + 1] for i in range(len(meds) - 1)))
    return results
def run_B(grp, copy_limit, fast):
    import torch
    log("=== B  ( 40x40; raw/schicluster/knn/RING×3×2×2/"
        "pop_ref) ===")
    truth, _ = truth_B(grp)
    pop_ref = {}
    for blk in CHR2_BLOCKS:
        dstack = np.stack([coords_to_prob(arr)[1] for _, arr in truth[blk]])
        d_med = np.nanmedian(dstack, axis=0)
        n_nan = int(np.isnan(d_med).sum())
        if n_nan:
            log(f"  [pop_ref] block_{blk}:  {n_nan}  NaN "
                f" (),  0")
        pop_ref[blk] = to_contact_from_dist(
            np.nan_to_num(d_med, nan=0.0))
    log(f"  : 6  ( {len(truth[CHR2_BLOCKS[0]])} )")
    if copy_limit:
        truth = {b: v[:copy_limit] for b, v in truth.items()}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {}
    for name, path in MODELS.items():
        if not os.path.exists(path):
            log(f"   {name} checkpoint , : {path}")
            continue
        m, _ = load_model(path, device=device, patch_bins=PATCH)
        models[name] = m
        log(f"   {name}  ({device})")
    triu40 = np.triu_indices(PATCH, k=1)
    hs = (HICREP_H_MAIN,) if fast else (HICREP_H_MAIN, HICREP_H_SENS)
    results, per_cell_store = {}, {}
    for lo, hi in DEPTHS_B:
        for sd in SEEDS_B:
            name, t_ds = f"N{lo}-{hi}", time.time()
            per_arm = {}
            for blk in CHR2_BLOCKS:
                cells = truth[blk]
                mats, plist, vlist, cids = [], [], [], []
                for c, arr in cells:
                    rng = np.random.default_rng([sd, c, blk])
                    p, d, valid = coords_to_prob(arr)
                    mat, n = sample_counts(p, triu40, rng, lo, hi)
                    if mat is None:
                        continue
                    mats.append(mat)
                    plist.append(p)
                    vlist.append(valid)
                    cids.append(c)
                if not mats:
                    continue
                counts = np.stack([m.astype(np.float64) for m in mats])
                knn_imp = knn_impute_batch(counts)
                knn_imp_self = knn_impute_batch_selfweighted(counts) \
                    if "selfweighted" in models else None
                ring = {}
                for mname, model in models.items():
                    ring[(mname, "R1")] = arm_ring(model, counts, device)
                    imp_for_model = knn_imp_self if (
                        mname == "selfweighted") else knn_imp
                    ring[(mname, "R2")] = arm_ring(model, imp_for_model,
                                                   device)
                for ci in range(len(mats)):
                    ev = eval_index(PATCH, vlist[ci])
                    maps = {"raw": arm_raw(mats[ci]),
                            "schicluster": arm_schicluster(counts[ci]),
                            "knn": knn_imp[ci],
                            "pop_ref": pop_ref[blk]}
                    for mname in models:
                        for proto in ("R1", "R2"):
                            rec, u1 = ring[(mname, proto)]
                            maps[f"ring_{mname}_{proto}_coord"] = \
                                to_contact_from_dist(ring_dists(
                                    u1[ci:ci + 1])[0])
                            maps[f"ring_{mname}_{proto}_recon"] = \
                                symmetrize(rec[ci])
                    met = eval_maps(maps, plist[ci], ev, hs)
                    for a, mv in met.items():
                        per_arm.setdefault(a, []).append(mv)
            key = f"{name}|seed{sd}"
            if not copy_limit:
                ck = load_ckpt("B", key)
                if ck is not None:
                    results[key] = ck["boxstats"]
                    per_cell_store[key] = ck["per_cell_spearman"]
                    log(f"  B {key}: , ")
                    continue
            first_arm = next(iter(per_arm.values()))
            metric_keys = list(first_arm[0].keys())
            results[key] = {a: {m: boxstats([r[m] for r in vals])
                                for m in metric_keys}
                            for a, vals in per_arm.items()}
            per_cell_store[key] = {a: [r["spearman"] for r in vals]
                                   for a, vals in per_arm.items()}
            if not copy_limit:
                save_ckpt("B", key, results[key], per_cell_store[key])
            log(f"  B {key}: {len(per_arm)} , "
                f"raw n={results[key]['raw']['spearman']['n']}, "
                f"{time.time() - t_ds:.0f}s")
    meds = []
    for lo, hi in DEPTHS_B:
        vals = []
        for sd in SEEDS_B:
            k = f"N{lo}-{hi}|seed{sd}"
            if k in results:
                vals.append(results[k]["raw"]["spearman"]["median"])
        meds.append(float(np.mean(vals)))
    gate("G4", f"B raw  {['%.3f' % m for m in meds]} ",
         all(meds[i] < meds[i + 1] for i in range(len(meds) - 1)))
    return results, per_cell_store
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    tag = "smoke" if args.smoke else "full"
    os.makedirs(OUT_DIR, exist_ok=True)
    log(f"=====   [{tag}] "
        f"(fast={args.fast}) =====")
    log(f": p=1/d, cutoff={DIST_CUTOFF_NM}nm, , , "
        f"QN, Pearson/Spearman/HiCRep")
    log(f" A={DEPTHS_A}; B ={DEPTHS_B} (scale={SCALE:.5f})")
    copy_limit = 50 if args.smoke else None
    grp = load_chr2_grouped()
    resA = run_A(grp, copy_limit, args.fast)
    resB, per_cell = run_B(grp, copy_limit, args.fast)
    suffix = "_smoke" if args.smoke else ""
    out_json = os.path.join(OUT_DIR, f"simrec_higashi_stage1{suffix}.json")
    out_txt = os.path.join(OUT_DIR, f"simrec_higashi_stage1{suffix}.txt")
    payload = {
        "meta": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "protocol": "Higashi SI Note A.4 (Su 2020 chr2 )",
            "formula": "p=1/d, cutoff=2000nm, with replacement, "
                       "first-diagonal excluded from evaluation, "
                       "quantile-normalized",
            "depths_A": [list(d) for d in DEPTHS_A],
            "depths_B_folded": [list(d) for d in DEPTHS_B],
            "fold_scale": SCALE,
            "seeds_B": SEEDS_B, "rwr_p": RWR_P,
            "hicrep_h": [HICREP_H_MAIN] + ([] if args.fast else
                                           [HICREP_H_SENS]),
            "smoke_copy_limit": copy_limit,
            "interpretive_choices": {
                "1_N_sampling": "()",
                "2_rwr_p": f"p={RWR_P}; PNAS2019 , "
                           " 0.3/0.7 ()",
                "3_qn": " truth+ QN ",
                "4_hicrep_h": f"h={HICREP_H_MAIN} (scHiCluster  "
                              f"1Mb ); h={HICREP_H_SENS} ",
                "5_depth_fold": f" =  × {SCALE:.5f} "
                                "(780/29403 )",
                "6_ring_contact": "C=1/(D+0.01); ="
                                  "(OE , )"},
            "gpu_statement": "RING  only (no_grad+eval, ); "
                              "scHiCluster/KNN/QN/ CPU",
            "elapsed_sec": round(time.time() - RUN_T0, 1),
        },
        "A_arm": resA,
        "B_arm": resB,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    lines = ["=" * 72,
             "  " + ("()" if args.smoke else ""),
             ": " + payload["meta"]["protocol"],
             f": {payload['meta']['formula']}", "=" * 72, ""]
    lines.append("--- A  ( chr2 243x243): raw / scHiCluster ---")
    for name, arms in resA.items():
        lines.append(f"\n[{name}]")
        for a, ms in arms.items():
            for m, st in ms.items():
                if st["median"] is not None:
                    lines.append(f"  {a:<12} {m:<12} median="
                                 f"{st['median']:.4f} "
                                 f"(Q1={st['q1']:.4f} Q3={st['q3']:.4f} "
                                 f"n={st['n']})")
    lines.append("\n--- B  ( 40x40) ---")
    for key, arms in resB.items():
        lines.append(f"\n[{key}]")
        for a, ms in arms.items():
            row = f"  {a:<24}"
            for m in ("pearson", "spearman", "hicrep_h1"):
                st = ms.get(m)
                if st and st["median"] is not None:
                    row += f" {m}={st['median']:.4f}"
            lines.append(row + f" (n={ms['spearman']['n']})")
    txt = "\n".join(lines)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(txt)
    log(f" → {out_txt}")
    log(f" → {out_json}")
    log(f" {time.time() - RUN_T0:.0f}s")
    print("\n" + txt)
if __name__ == "__main__":
    main()

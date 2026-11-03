import os
import sys
import json
import time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from exp._helpers import results_dir
STEP4_DIR = os.path.join(ROOT, "exp", "results", "section4_step3_eval")
STEP1E_DIR = os.path.join(ROOT, "exp", "results", "section4_step1b")
ARMS = {                                  
    "B3_topo100_K20g3": "latent_finetune_b3_train.npz",
    "P1_topo100_K10g1": "latent_finetune_p1_train.npz",
    "P2_topo100_K10g3": "latent_finetune_p2_train.npz",
    "B4_recononly":     "latent_finetune_b4_train.npz",
}
PATCH_BINS = 40
TRIU = np.triu_indices(PATCH_BINS, k=1)
def simple_pca(X, n_components=10):
    X_c = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    q = min(n_components, Vt.shape[0])
    return X_c @ Vt[:q].T
def kmeans(X, k, n_iter=200, seed=42):
    rng = np.random.RandomState(seed)
    n = len(X)
    centers = [X[rng.randint(n)]]
    for _ in range(1, k):
        d = np.min([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0)
        probs = d / d.sum()
        centers.append(X[rng.choice(n, p=probs)])
    centers = np.array(centers)
    for _ in range(n_iter):
        d = np.array([np.sum((X - c) ** 2, axis=1) for c in centers])
        lab = np.argmin(d, axis=0)
        for j in range(k):
            if np.sum(lab == j) > 0:
                centers[j] = X[lab == j].mean(axis=0)
    return lab
def adjusted_rand_index(y, p):
    n = len(y)
    classes, clusters = np.unique(y), np.unique(p)
    cont = np.zeros((len(classes), len(clusters)), dtype=int)
    for i, c in enumerate(classes):
        for j, k in enumerate(clusters):
            cont[i, j] = np.sum((y == c) & (p == k))
    sc = sum(int(x * (x - 1) / 2) for x in cont.flatten())
    sa = sum(int(a * (a - 1) / 2) for a in cont.sum(axis=1))
    sb = sum(int(b * (b - 1) / 2) for b in cont.sum(axis=0))
    tot = int(n * (n - 1) / 2)
    exp = sa * sb / tot if tot > 0 else 0
    mx = 0.5 * (sa + sb)
    return 1.0 if mx == exp else (sc - exp) / (mx - exp)
def eval_feature(X, labels, tag=""):
    res = {}
    for nd in (2, 5, 10):
        cl = kmeans(simple_pca(X, nd), k=5)
        res[f"pca{nd}"] = float(adjusted_rand_index(labels, cl))
    best_nd = max(res, key=res.get)
    return {"aris": res, "best": res[best_nd], "best_nd": best_nd}
def feat_A1_flat(u1):
    nb, n = u1.shape[0], u1.shape[1]
    return np.concatenate([u1[b].reshape(n, -1) for b in range(nb)], axis=1)
def feat_A2_centered(u1):
    c = u1 - u1.mean(axis=2, keepdims=True)
    nb, n = u1.shape[0], u1.shape[1]
    return np.concatenate([c[b].reshape(n, -1) for b in range(nb)], axis=1)
def feat_A3_pcaframe(u1):
    nb, n = u1.shape[0], u1.shape[1]
    out = np.zeros_like(u1)
    for b in range(nb):
        for i in range(n):
            X = u1[b, i] - u1[b, i].mean(axis=0)
            _, _, Vt = np.linalg.svd(X, full_matrices=False)
            out[b, i] = X @ Vt.T
    return np.concatenate([out[b].reshape(n, -1) for b in range(nb)], axis=1)
def _cdist_batch(U):
    G = np.einsum('bnij,bnkj->bnik', U, U)
    sq = np.diagonal(G, axis1=2, axis2=3)
    D2 = sq[:, :, :, None] + sq[:, :, None, :] - 2 * G
    D2 = np.maximum(D2, 0.0)
    D = np.sqrt(D2)
    return D[:, :, TRIU[0], TRIU[1]]
def feat_A4_gram(u1):
    nb, n = u1.shape[0], u1.shape[1]
    c = u1 - u1.mean(axis=2, keepdims=True)
    G = np.einsum('bnij,bnkj->bnik', c, c)
    tri = G[:, :, TRIU[0], TRIU[1]]
    return np.concatenate([tri[b] for b in range(nb)], axis=1)
def feat_A5_distmat(u1):
    tri = _cdist_batch(u1)
    return np.concatenate([tri[b] for b in range(u1.shape[0])], axis=1)
def feat_B1_moments(u1):
    nb, n = u1.shape[0], u1.shape[1]
    tri = _cdist_batch(u1)                                 
    feats = []
    for b in range(nb):
        m = u1[b].mean(axis=1)                        
        s = u1[b].std(axis=1)                         
        dm = tri[b].mean(axis=1, keepdims=True)
        ds = tri[b].std(axis=1, keepdims=True)
        dx = tri[b].max(axis=1, keepdims=True)
        feats.append(np.concatenate([m, s, dm, ds, dx], axis=1))
    return np.concatenate(feats, axis=1)                  
def feat_B2_shape(u1):
    nb, n = u1.shape[0], u1.shape[1]
    feats = []
    for b in range(nb):
        c = u1[b] - u1[b].mean(axis=1, keepdims=True)
        cov = np.einsum('nij,nik->njk', c, c) / PATCH_BINS
        ev = np.linalg.eigvalsh(cov)[:, ::-1]            
        ev = np.maximum(ev, 1e-12)
        elong = ev[:, 1:2] / ev[:, 0:1]
        planar = ev[:, 2:3] / ev[:, 1:2]
        spher = ev[:, 2:3] / ev[:, 0:1]
        feats.append(np.concatenate([ev, elong, planar, spher], axis=1))
    return np.concatenate(feats, axis=1)                  
def feat_B3_decay(u1):
    nb, n = u1.shape[0], u1.shape[1]
    feats = np.zeros((n, nb, PATCH_BINS - 1), dtype=np.float32)
    for b in range(nb):
        G = np.einsum('nij,nkj->nik', u1[b], u1[b])
        sq = np.diagonal(G, axis1=1, axis2=2)
        D = np.sqrt(np.maximum(sq[:, :, None] + sq[:, None, :] - 2 * G, 0))
        for d in range(1, PATCH_BINS):
            feats[:, b, d - 1] = np.diagonal(D, offset=d, axis1=1,
                                             axis2=2).mean(axis=1)
    return feats.reshape(n, -1)
def _rips_diagram(U_flat):
    import gudhi as gd
    tri = _cdist_batch(U_flat[None, None])[0, 0]             
    Dm = np.zeros((PATCH_BINS, PATCH_BINS))
    Dm[TRIU] = tri
    Dm = Dm + Dm.T
    rips = gd.RipsComplex(distance_matrix=Dm, max_edge_length=float(Dm.max()))
    st = rips.create_simplex_tree(max_dimension=2)
    st.persistence()
    h0, h1 = [], []
    for dim, (birth, death) in st.persistence():
        if dim == 0 and np.isfinite(death):
            h0.append((birth, death))
        elif dim == 1 and np.isfinite(death):
            h1.append((birth, death))
    return h0, h1
def _pers_image(h0, h1, bins=10):
    imgs = []
    for diag in (h0, h1):
        if len(diag) == 0:
            imgs.append(np.zeros((bins, bins), dtype=np.float32))
            continue
        pts = np.array([[b, d - b] for b, d in diag])
        w = pts[:, 1]
        m = max(pts[:, 0].max(), pts[:, 1].max(), 1e-8)
        H, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=bins,
                                 range=[[0, m], [0, m]], weights=w)
        imgs.append(H.astype(np.float32) / (w.sum() + 1e-12))
    return np.concatenate([im.flatten() for im in imgs])
def _betti_curve(h0, h1, n_thr=50, tmax=None):
    allv = [b for b, _ in h0 + h1] + [d for _, d in h0 + h1]
    tmax = tmax or (max(allv) if allv else 1.0)
    thr = np.linspace(0, tmax, n_thr)
    b0 = np.array([np.sum([(b <= t < d) for b, d in h0]) for t in thr])
    b1 = np.array([np.sum([(b <= t < d) for b, d in h1]) for t in thr])
    return np.concatenate([b0, b1]).astype(np.float32)
def _ph_stats(h0, h1):
    def agg(diag):
        if len(diag) == 0:
            return np.zeros(5, dtype=np.float64)
        p = np.array([d - b for b, d in diag])
        return np.array([len(diag), p.sum(), p.mean(), p.max(),
                         np.std(p)])
    s0, s1 = agg(h0), agg(h1)
    return np.concatenate([s0, s1])
def feat_C_family(u1, kind, n_cells_limit=None):
    nb, n = u1.shape[0], u1.shape[1]
    if n_cells_limit:
        n = min(n, n_cells_limit)
    out = []
    for b in range(nb):
        for i in range(n):
            h0, h1 = _rips_diagram(u1[b, i])
            if kind == "C1":
                out.append(_pers_image(h0, h1))
            elif kind == "C2":
                out.append(_betti_curve(h0, h1))
            else:
                out.append(_ph_stats(h0, h1))
            if (i + 1) % 200 == 0:
                print(f"      {kind} block{b} [{i+1}/{n}]", flush=True)
    out = np.array(out).reshape(nb, n, -1)
    return np.concatenate([out[b] for b in range(nb)], axis=1)
def feat_C4_coupling(u1, dx_sel):
    nb, n = u1.shape[0], u1.shape[1]
    tri = _cdist_batch(u1)                                 
    feats = np.zeros((n, nb), dtype=np.float32)
    for b in range(nb):
        d = dx_sel[b][:, TRIU[0], TRIU[1]]              
        for i in range(n):
            a, c = tri[b, i], d[i]
            if a.std() > 0 and c.std() > 0:
                feats[i, b] = np.corrcoef(a, c)[0, 1]
    return feats
def load_global_e1():
    import h5py
    h5_path = os.path.join(STEP1E_DIR, "genome_imputed_v2.h5")
    blocks = [0, 1, 28, 37, 41, 45, 49, 53, 57, 6]
    e1s = []
    with h5py.File(h5_path, "r") as f:
        for bi in blocks:
            imp = np.array(f[f"block_{bi}"]["imputed"], dtype=np.float32)
            mean_mat = imp.mean(axis=0)
            corr = np.corrcoef(mean_mat)                 
            corr = np.nan_to_num(corr)
            _, evecs = np.linalg.eigh(corr)
            e1s.append(evecs[:, -1])                      
    return np.array(e1s)                                 
def feat_D1_compartment(u1, e1s):
    nb, n = u1.shape[0], u1.shape[1]
    feats = np.zeros((n, nb), dtype=np.float32)
    for b in range(nb):
        e1 = e1s[b]
        for i in range(n):
            X = u1[b, i] - u1[b, i].mean(axis=0)
            pc1 = np.linalg.svd(X, full_matrices=False)[2][0]        
            proj = X @ pc1                                              
            if proj.std() > 0 and e1.std() > 0:
                feats[i, b] = abs(np.corrcoef(proj, e1)[0, 1])
    return feats
def feat_D2_insulation(u1, w=3):
    nb, n = u1.shape[0], u1.shape[1]
    feats = []
    for b in range(nb):
        G = np.einsum('nij,nkj->nik', u1[b], u1[b])
        sq = np.diagonal(G, axis1=1, axis2=2)
        D = np.sqrt(np.maximum(sq[:, :, None] + sq[:, None, :] - 2 * G, 0))
        ins = np.zeros((n, PATCH_BINS - 2 * w), dtype=np.float32)
        for i in range(w, PATCH_BINS - w):
            ins[:, i - w] = D[:, i - w:i + w, i - w:i + w].mean(axis=(1, 2))
        feats.append(ins)
    return np.concatenate(feats, axis=1)
def feat_E1_mean_distmat(u1):
    tri = _cdist_batch(u1)                                 
    return tri.mean(axis=0)
def feat_E2_block_zscore(u1):
    nb, n = u1.shape[0], u1.shape[1]
    feats = []
    for b in range(nb):
        X = u1[b].reshape(n, -1)
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
        feats.append(X)
    return np.concatenate(feats, axis=1)
def spectral_clustering(X, k=5, k_nn=15, seed=42):
    G = X @ X.T
    sq = np.diag(G)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * G, 0)
    np.fill_diagonal(D2, np.inf)
    idx = np.argsort(D2, axis=1)[:, :k_nn]
    sigma = np.median(np.sqrt(D2[np.arange(len(X))[:, None], idx])) + 1e-10
    W = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        W[i, idx[i]] = np.exp(-D2[i, idx[i]] / (2 * sigma ** 2))
    W = np.maximum(W, W.T)
    d = W.sum(axis=1)
    L = np.eye(len(X)) - (d ** -0.5)[:, None] * W * (d ** -0.5)[None, :]
    _, evecs = np.linalg.eigh(L)
    emb = evecs[:, 1:k + 1]
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    return kmeans(emb, k, seed=seed)
def gmm_diag(X, k=5, n_iter=50, seed=42):
    rng = np.random.RandomState(seed)
    n, d = X.shape
    mu = X[rng.choice(n, k, replace=False)]
    var = np.ones((k, d))
    pi = np.ones(k) / k
    for _ in range(n_iter):
        logp = np.array([
            -0.5 * (((X - mu[j]) ** 2) / var[j]).sum(axis=1)
            - 0.5 * np.log(var[j]).sum() + np.log(pi[j])
            for j in range(k)]).T
        logp -= logp.max(axis=1, keepdims=True)
        resp = np.exp(logp)
        resp /= resp.sum(axis=1, keepdims=True)
        Nk = resp.sum(axis=0) + 1e-10
        pi = Nk / n
        mu = (resp.T @ X) / Nk[:, None]
        for j in range(k):
            diff = X - mu[j]
            var[j] = (resp[:, j:j+1] * diff ** 2).sum(axis=0) / Nk[j] + 1e-6
    return resp.argmax(axis=1)
def kmeans_multiseed(X, k=5, n_seeds=10, n_iter=200):
    best_lab, best_inertia = None, np.inf
    for s in range(n_seeds):
        lab = kmeans(X, k, n_iter=n_iter, seed=s)
        inertia = sum(np.sum((X[lab == j] -
                              X[lab == j].mean(axis=0)) ** 2)
                      for j in range(k) if (lab == j).sum() > 0)
        if inertia < best_inertia:
            best_inertia, best_lab = inertia, lab
    return best_lab
def main():
    out_dir = results_dir("section4_step3_sweep")
    res_path = os.path.join(out_dir, "sweep_results.json")
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}
    print(f"[Step5-Sweep] Output → {out_dir}")
    arms = {}
    labels_ref = None
    for name, fname in ARMS.items():
        d = np.load(os.path.join(STEP4_DIR, fname))
        arms[name] = d["u1_3d"]
        if labels_ref is None:
            labels_ref = d["labels"]
        else:
            assert np.array_equal(d["labels"], labels_ref), f"{name} "
    labels = labels_ref
    print(f"  Arms: {list(arms.keys())}, cells: {len(labels)}")
    dx_sel = None
    e1s = None
    def get_dx():
        nonlocal dx_sel
        if dx_sel is None:
            print("  Loading D_x (v2 train npz, 500 cells) ...")
            from exp.paper.section4_step2_finetune import get_subset
            dd = np.load(os.path.join(STEP1E_DIR, "train_data_v2.npz"))
            sub = get_subset(dd["labels"], dd["block_ids"], 100)
            n_cells_total = len(dd["labels"]) // 10
            dx = dd["d_x"][sub].reshape(10, -1, PATCH_BINS, PATCH_BINS)
            dx_sel = dx
        return dx_sel
    def get_e1():
        nonlocal e1s
        if e1s is None:
            print("  Computing global pseudo-bulk e1 ...")
            e1s = load_global_e1()
        return e1s
    FEATURES = [
        ("A1_flat",       feat_A1_flat,        None),
        ("A2_centered",   feat_A2_centered,    None),
        ("A3_pcaframe",   feat_A3_pcaframe,    None),
        ("B1_moments",    feat_B1_moments,     None),
        ("B2_shape",      feat_B2_shape,       None),
        ("B3_decay",      feat_B3_decay,       None),
        ("E1_mean_dist",  feat_E1_mean_distmat, None),
        ("E2_block_z",    feat_E2_block_zscore, None),
        ("A4_gram",       feat_A4_gram,        None),
        ("A5_distmat",    feat_A5_distmat,     None),
        ("D1_comp_e1",    feat_D1_compartment, "e1"),
        ("D2_insulation", feat_D2_insulation,  None),
        ("C3_ph_stats",   lambda u: feat_C_family(u, "C3"), None),
        ("C2_betti",      lambda u: feat_C_family(u, "C2"), None),
        ("C1_persimage",  lambda u: feat_C_family(u, "C1"), None),
        ("C4_coupling",   feat_C4_coupling,    "dx"),
    ]
    for fname, fn, extra in FEATURES:
        for arm_name, u1 in arms.items():
            key = f"{fname}::{arm_name}"
            if key in results and "error" not in results[key]:
                continue
            t0 = time.time()
            print(f"\n[{fname}] arm={arm_name} ...", flush=True)
            try:
                if extra == "e1":
                    X = fn(u1, get_e1())
                elif extra == "dx":
                    X = fn(u1, get_dx())
                else:
                    X = fn(u1)
                X = np.nan_to_num(X.astype(np.float32))
                r = eval_feature(X, labels)
                r["feat_dim"] = int(X.shape[1])
                r["time_s"] = round(time.time() - t0, 1)
                results[key] = r
                print(f"  dim={X.shape[1]} best ARI={r['best']:.4f} "
                      f"({r['best_nd']}) [{r['time_s']}s]", flush=True)
            except Exception as e:
                results[key] = {"error": str(e)}
                print(f"  [ERROR] {e}", flush=True)
            with open(res_path, "w") as f:
                json.dump(results, f, indent=2)
    print("\n=== F :  () ===")
    feat_cache = {}
    for arm_name, u1 in arms.items():
        cands = [(k.split("::")[0], v) for k, v in results.items()
                 if k.endswith(f"::{arm_name}") and "best" in v
                 and not k.startswith("F")]
        if not cands:
            continue
        best_feat, best_res = max(cands, key=lambda kv: kv[1]["best"])
        print(f"\n[{arm_name}] ={best_feat} "
              f"(ARI={best_res['best']:.4f}), ")
        fn = dict((n, f) for n, f, _ in FEATURES)[best_feat]
        extra = dict((n, e) for n, _, e in FEATURES)[best_feat]
        X = fn(u1, get_e1()) if extra == "e1" else \
            fn(u1, get_dx()) if extra == "dx" else fn(u1)
        X = np.nan_to_num(X.astype(np.float32))
        Xp = simple_pca(X, 10)
        for cname, cfn in [("F1_spectral", spectral_clustering),
                           ("F2_gmm", gmm_diag),
                           ("F3_kmeans_ms", kmeans_multiseed)]:
            key = f"{cname}@{best_feat}::{arm_name}"
            if key in results:
                continue
            try:
                cl = cfn(Xp, 5)
                ari = float(adjusted_rand_index(labels, cl))
                results[key] = {"best": ari, "feat": best_feat}
                print(f"  {cname}: ARI={ari:.4f}", flush=True)
            except Exception as e:
                results[key] = {"error": str(e)}
            with open(res_path, "w") as f:
                json.dump(results, f, indent=2)
    lines = ["=" * 84]
    lines.append("Feature Sweep  (train-set transductive ARI, best over PCA dims)")
    lines.append("=" * 84)
    arm_names = list(ARMS.keys())
    lines.append(f"{'feature':<18}" + "".join(f"{a[:16]:>18}" for a in arm_names))
    feat_names = [n for n, _, _ in FEATURES]
    for fn_name in feat_names:
        row = f"{fn_name:<18}"
        for a in arm_names:
            v = results.get(f"{fn_name}::{a}", {})
            row += f"{v.get('best', -1):>18.4f}" if "best" in v else \
                   f"{'ERR':>18}"
        lines.append(row)
    lines.append("\n--- F  () ---")
    for k, v in results.items():
        if k.startswith("F") and "best" in v:
            lines.append(f"  {k}: ARI={v['best']:.4f}")
    txt = "\n".join(lines)
    with open(os.path.join(out_dir, "sweep_report.txt"), "w") as f:
        f.write(txt)
    print(f"\n{txt}")
    print(f"\n[Step5-Sweep] Done. → {out_dir}")
if __name__ == "__main__":
    main()

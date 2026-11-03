import os, sys, json
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)
import numpy as np, cooler, torch, torch.nn as nn
from torch.utils.data import TensorDataset
from src.model_builder import build_model
from src.forward import specify_use_model, external_validation
from src.data import (training_chip_data, build_chip_chrom_offsets,
                      compute_oe_matrix, extract_chip_from_batch)
from exp._helpers import load_config, results_dir
HIC = os.path.join(ROOT, "..", "data", "HiC.GRCh38", "HIC01_50000.cool")
CHIP14 = os.path.join(ROOT, "..", "data", "ChIP.GRCh38.simp", "resolution50000", "")
MODEL = os.path.join(ROOT, "models", "TORELink_20Mb", "TORELink_20Mb_epoch500.pth")
TP_FILE = os.path.join(ROOT, "models", "TORELink_20Mb", "HalfRing_Experiment=HIC01_testingPos.txt")
TR_FILE = os.path.join(ROOT, "models", "TORELink_20Mb", "HalfRing_Experiment=HIC01_trainingPos.txt")
PATCH = 400; PRESET = "BIG"; N_CHIP = 14
CHIP_NAMES = ["CTCF", "RAD21", "SMC3", "H3K79me2", "H3K9ac", "H4K20me1",
              "H2AFZ", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1",
              "H3K4me2", "H3K4me3", "H3K9me3"]
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
BIN_EPOCHS = 150; CT_EPOCHS = 200; LR = 1e-3; BATCH = 8192
dev = "cuda:0"
def pearson(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if a.std() < 1e-12 or b.std() < 1e-12: return float('nan')
    return float(np.corrcoef(a, b)[0, 1])
class BinMLP(nn.Module):
    def __init__(self, input_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, 1))
    def forward(self, x): return self.net(x).squeeze(-1)
class CrossDecoder(nn.Module):
    def __init__(self, input_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, 1))
    def forward(self, x): return self.net(x).squeeze(-1)
def _train_mlp(net, X_tr, y_tr, epochs):
    net = net.to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR); loss_fn = nn.HuberLoss()
    X_t = torch.tensor(X_tr.astype(np.float32), device=dev)
    y_t = torch.tensor(y_tr.astype(np.float32), device=dev)
    n = X_t.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, BATCH):
            xb, yb = X_t[perm[i:i+BATCH]], y_t[perm[i:i+BATCH]]
            loss = loss_fn(net(xb), yb); opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    del X_t, y_t, opt, loss_fn
    return net
def loo_bin(regs, alpha):
    n = len(regs); rs = []
    for k in range(n):
        Xt, yt = [], []
        for r in range(n):
            if r == k: continue
            v = regs[r]['valid']
            if v.sum() < 5: continue
            Xt.append(np.concatenate([regs[r]['feat'][v], np.ones((v.sum(), 1))], axis=1).astype(np.float64))
            yt.append(regs[r]['target'][v])
        if not Xt or regs[k]['valid'].sum() < 5: rs.append(float('nan')); continue
        X_tr = np.concatenate(Xt); y_tr = np.concatenate(yt)
        w = np.linalg.solve(X_tr.T @ X_tr + alpha * np.eye(X_tr.shape[1]), X_tr.T @ y_tr)
        v = regs[k]['valid']
        rs.append(pearson(regs[k]['target'][v],
                          np.concatenate([regs[k]['feat'][v], np.ones((v.sum(), 1))], axis=1) @ w))
    return float(np.nanmean(rs))
def read_pos(path):
    pos = []
    with open(path) as f:
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 2: pos.append(int(p[0]))
    return pos
def main():
    out = results_dir("section3_step1")
    cfg = load_config()
    clr = cooler.Cooler(HIC); bins_table = clr.bins()[:]
    ec, sc = training_chip_data(CHIP14, return_scaler=True)
    co, cn, _ = build_chip_chrom_offsets(CHIP14)
    cm = {'bins_table': bins_table, 'entire_chip': ec, 'chip_scaler': sc,
          'chrom_offsets': co, 'chrom_nbins': cn, 'patch_bins': PATCH,
          'x_resolution': 50000, 'y_resolution': 50000}
    test_pos, train_pos = read_pos(TP_FILE), read_pos(TR_FILE)
    triu = np.triu_indices(PATCH, k=1); i_idx, j_idx = triu
    dist_all = (j_idx - i_idx).astype(np.float64) / PATCH
    print(f"[step1d] BIG bidirectional probe (train={len(train_pos)}, test={len(test_pos)}, GPU)\n", flush=True)
    model = build_model("halfring", cfg, PATCH, N_CHIP, "cpu", y_length=PATCH, preset=PRESET)
    model = specify_use_model("cpu", MODEL, model); model.eval()
    train_regs, test_regs = [], []
    for gb in train_pos:
        bal = np.array(clr.matrix(balance=True)[gb:gb+PATCH, gb:gb+PATCH], dtype=np.float64); bal[np.isnan(bal)] = 0.0
        oe = np.nan_to_num(compute_oe_matrix(bal), nan=0.0, posinf=0.0, neginf=0.0)
        x_in = np.log1p(np.clip(bal, 0, None)).astype(np.float32)
        ds = TensorDataset(torch.tensor(x_in), torch.tensor([[gb, gb]], dtype=torch.int32).repeat_interleave(PATCH, 0))
        outs = external_validation("cpu", model, "halfring", ds, batch_size=PATCH, chip_meta=cm, cross_reconstruction=True)
        chip = extract_chip_from_batch(torch.tensor([[gb, gb]], dtype=torch.int32), cm, "cpu").numpy().astype(np.float64)
        train_regs.append({'u1': outs[4].numpy().astype(np.float64), 'u2': outs[5].numpy().astype(np.float64),
                           'chip': chip, 'oe': oe})
    for gb in test_pos:
        bal = np.array(clr.matrix(balance=True)[gb:gb+PATCH, gb:gb+PATCH], dtype=np.float64); bal[np.isnan(bal)] = 0.0
        oe = np.nan_to_num(compute_oe_matrix(bal), nan=0.0, posinf=0.0, neginf=0.0)
        x_in = np.log1p(np.clip(bal, 0, None)).astype(np.float32)
        ds = TensorDataset(torch.tensor(x_in), torch.tensor([[gb, gb]], dtype=torch.int32).repeat_interleave(PATCH, 0))
        outs = external_validation("cpu", model, "halfring", ds, batch_size=PATCH, chip_meta=cm, cross_reconstruction=True)
        chip = extract_chip_from_batch(torch.tensor([[gb, gb]], dtype=torch.int32), cm, "cpu").numpy().astype(np.float64)
        test_regs.append({'u1': outs[4].numpy().astype(np.float64), 'u2': outs[5].numpy().astype(np.float64),
                          'chip': chip, 'oe': oe})
    del model
    print("  forward done (train+test)\n", flush=True)
    print("[bin-level] u1 -> ChIP / raw-Hi-C -> ChIP (per mark)", flush=True)
    bin_results = {'per_mark': {}}
    for ch_idx, ch_name in enumerate(CHIP_NAMES):
        br_u1 = [{'feat': r['u1'], 'target': r['chip'][:, ch_idx], 'valid': np.ones(PATCH, dtype=bool)} for r in test_regs]
        best_u1 = max(ALPHAS, key=lambda a: (r if not np.isnan(r := loo_bin(br_u1, a)) else -9))
        r_u1_lin = loo_bin(br_u1, best_u1)
        X_tr = np.concatenate([r['u1'] for r in train_regs], axis=0).astype(np.float32)
        y_tr = np.concatenate([r['chip'][:, ch_idx] for r in train_regs], axis=0).astype(np.float32)
        torch.manual_seed(0)
        net = _train_mlp(BinMLP(X_tr.shape[1], 64), X_tr, y_tr, BIN_EPOCHS)
        with torch.no_grad():
            r_u1_mlp = float(np.nanmean([pearson(r['chip'][:, ch_idx],
                              net(torch.tensor(r['u1'].astype(np.float32), device=dev)).cpu().numpy()) for r in test_regs]))
        del net
        del net
        bin_results['per_mark'][ch_name] = {'u1_linear': r_u1_lin, 'u1_mlp': r_u1_mlp}
        print(f"  {ch_name:<10} u1: lin={r_u1_lin:.4f} mlp={r_u1_mlp:.4f}", flush=True)
    u1l = np.mean([bin_results['per_mark'][c]['u1_linear'] for c in CHIP_NAMES])
    u1m = np.mean([bin_results['per_mark'][c]['u1_mlp'] for c in CHIP_NAMES])
    bin_results['mean'] = {'u1_linear': u1l, 'u1_mlp': u1m}
    print("\n[contact-level] 5 variants -> O/E (full contacts, GPU)", flush=True)
    def prep_contact(regs, feat_fn):
        Xs, ys = [], []
        for r in regs:
            y = r['oe'][i_idx, j_idx]; v = np.isfinite(y) & (np.abs(y) < 50)
            Xs.append(feat_fn(r, v)); ys.append(y[v])
        return np.concatenate(Xs).astype(np.float32), np.concatenate(ys).astype(np.float32)
    feat_fns = {
        "dist-only": (lambda r, v: dist_all[v, None].astype(np.float32), 1),
        "u2+dist": (lambda r, v: np.concatenate([r['u2'][i_idx[v]], r['u2'][j_idx[v]], dist_all[v, None]], axis=1).astype(np.float32), 33),
        "u1+dist": (lambda r, v: np.concatenate([r['u1'][i_idx[v]], r['u1'][j_idx[v]], dist_all[v, None]], axis=1).astype(np.float32), 33),
        "u1+u2+dist": (lambda r, v: np.concatenate([r['u1'][i_idx[v]], r['u1'][j_idx[v]], r['u2'][i_idx[v]], r['u2'][j_idx[v]], dist_all[v, None]], axis=1).astype(np.float32), 65),
        "raw-ChIP+dist": (lambda r, v: np.concatenate([r['chip'][i_idx[v]], r['chip'][j_idx[v]], dist_all[v, None]], axis=1).astype(np.float32), 29),
    }
    ct_results = {}
    for label, (fn, dim) in feat_fns.items():
        Xtr, ytr = prep_contact(train_regs, fn)
        print(f"  {label}: train={len(ytr)} contacts", flush=True)
        torch.manual_seed(0); torch.cuda.manual_seed(0)
        net = _train_mlp(CrossDecoder(dim, 64), Xtr, ytr, CT_EPOCHS)
        rs = []
        with torch.no_grad():
            for r in test_regs:
                y = r['oe'][i_idx, j_idx]; v = np.isfinite(y) & (np.abs(y) < 50)
                X_te = torch.tensor(fn(r, v), device=dev)
                rs.append(pearson(y[v], net(X_te).cpu().numpy()))
        del net, Xtr, ytr; torch.cuda.empty_cache()
        ct_results[label] = float(np.nanmean(rs))
        print(f"  {label:<16} (dim={dim}) R={ct_results[label]:.4f}", flush=True)
    gain = ct_results.get("u1+u2+dist", 0) - max(ct_results.get("u1+dist", 0), ct_results.get("u2+dist", 0))
    print(f"  gain(u1+u2 - best) = {gain:+.4f}", flush=True)
    results = {
        'bin_level': bin_results,
        'contact_level': ct_results,
        'complementarity_gain': gain,
        'config': {'patch': PATCH, 'preset': PRESET, 'n_chip': N_CHIP,
                   'bin_epochs': BIN_EPOCHS, 'ct_epochs': CT_EPOCHS,
                   'n_train': len(train_pos), 'n_test': len(test_pos)},
    }
    with open(os.path.join(out, "big_bidirectional.json"), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[Done] -> {out}/big_bidirectional.json", flush=True)
if __name__ == "__main__":
    main()

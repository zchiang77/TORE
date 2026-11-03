import os, sys, json, time, argparse, traceback
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)
import numpy as np, cooler, torch, torch.nn as nn
from torch.utils.data import TensorDataset
from src.model_builder import build_model
from src.forward import specify_use_model, external_validation
from src.data import (training_chip_data, build_chip_chrom_offsets,
                      compute_oe_matrix, extract_chip_from_batch)
from exp._helpers import load_config, results_dir
HIC01 = os.path.join(ROOT, "..", "data", "HiC.GRCh38", "HIC01_50000.cool")
HIC02 = os.path.join(ROOT, "..", "data", "HiC.GRCh38", "HIC02_50000.cool")
CHIP6 = os.path.join(ROOT, "..", "data", "ChIP.GRCh38.six", "resolution50000", "")
MSE_CKPT = os.path.join(ROOT, "models", "TORELink_2Mb", "TORELink_2Mb_epoch280.pth")
CKA_CKPT = os.path.join(ROOT, "models", "TORELink_2Mb_CKA", "TORELink_2Mb_CKA_epoch508.pth")
TP = os.path.join(ROOT, "models", "TORELink_2Mb", "HalfRing_Experiment=HIC01_testingPos.txt")
TR = os.path.join(ROOT, "models", "TORELink_2Mb", "HalfRing_Experiment=HIC01_trainingPos.txt")
PATCH = 40; DEV = 'cuda:0'; EPOCHS = 200; LR = 0.001; BATCH = 4096
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
def pearson(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if a.std() < 1e-12 or b.std() < 1e-12: return float('nan')
    return float(np.corrcoef(a, b)[0, 1])
class CrossDecoder(nn.Module):
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(hidden, 1))
    def forward(self, x): return self.net(x).squeeze(-1)
class BinMLP(nn.Module):
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(hidden, 1))
    def forward(self, x): return self.net(x).squeeze(-1)
def train_mlp(X_tr, y_tr, test_regs, feat_fn, ii, jj, dd, dim, seed=0):
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    net = CrossDecoder(dim).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=LR); loss_fn = nn.HuberLoss()
    Xt = torch.tensor(X_tr, dtype=torch.float32, device=DEV)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=DEV)
    n = Xt.shape[0]
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, BATCH):
            xb, yb = Xt[perm[i:i+BATCH]], yt[perm[i:i+BATCH]]
            loss = loss_fn(net(xb), yb); opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); rs = []
    with torch.no_grad():
        for r in test_regs:
            v = r['valid']
            Xte = torch.tensor(feat_fn(r, ii[v], jj[v], dd[v]).astype(np.float32), device=DEV)
            rs.append(pearson(r['y'][v], net(Xte).cpu().numpy()))
    del net, Xt, yt, opt; torch.cuda.empty_cache()
    return float(np.nanmean(rs)) if rs else float('nan')
def loo_ridge_best(regs, ii, jj, dd, mode='full'):
    def build_Xc(f, i, j, d):
        if mode == 'full': return np.concatenate([f[i], f[j], d.reshape(-1,1), np.ones((len(i),1))], axis=1).astype(np.float64)
        elif mode == 'distonly': return np.concatenate([d.reshape(-1,1), np.ones((len(i),1))], axis=1).astype(np.float64)
    best = None
    for a in ALPHAS:
        rs = []
        for k in range(len(regs)):
            Xt, yt = [], []
            for r in range(len(regs)):
                if r == k: continue
                v = regs[r]['valid']
                if v.sum() < 5: continue
                Xt.append(build_Xc(regs[r]['feat'], ii[v], jj[v], dd[v])); yt.append(regs[r]['y'][v])
            vk = regs[k]['valid']
            if not Xt or vk.sum() < 5: rs.append(float('nan')); continue
            X_tr = np.concatenate(Xt); y_tr = np.concatenate(yt)
            w = np.linalg.solve(X_tr.T @ X_tr + a * np.eye(X_tr.shape[1]), X_tr.T @ y_tr)
            rs.append(pearson(regs[k]['y'][vk], build_Xc(regs[k]['feat'], ii[vk], jj[vk], dd[vk]) @ w))
        r_val = float(np.nanmean(rs)) if rs else float('nan')
        if best is None or (not np.isnan(r_val) and (np.isnan(best[1]) or r_val > best[1])): best = (a, r_val)
    return best
def ridge_fixed(train_regs, test_regs, ii, jj, dd, alpha, mode='full'):
    def build_Xc(f, i, j, d):
        if mode == 'full': return np.concatenate([f[i], f[j], d.reshape(-1,1), np.ones((len(i),1))], axis=1).astype(np.float64)
        elif mode == 'distonly': return np.concatenate([d.reshape(-1,1), np.ones((len(i),1))], axis=1).astype(np.float64)
    Xt, yt = [], []
    for r in train_regs:
        v = r['valid']
        if v.sum() < 5: continue
        Xt.append(build_Xc(r['feat'], ii[v], jj[v], dd[v])); yt.append(r['y'][v])
    X_tr = np.concatenate(Xt); y_tr = np.concatenate(yt)
    w = np.linalg.solve(X_tr.T @ X_tr + alpha * np.eye(X_tr.shape[1]), X_tr.T @ y_tr)
    rs = []
    for r in test_regs:
        v = r['valid']
        if v.sum() < 5: rs.append(float('nan')); continue
        rs.append(pearson(r['y'][v], build_Xc(r['feat'], ii[v], jj[v], dd[v]) @ w))
    return float(np.nanmean(rs)) if rs else float('nan')
def forward_model(ckpt, hic_path, chip_path, positions, label):
    cfg = load_config()
    clr = cooler.Cooler(hic_path); bins = clr.bins()[:]
    ec, sc = training_chip_data(chip_path, return_scaler=True)
    co, cn, _ = build_chip_chrom_offsets(chip_path)
    cm = {'bins_table': bins, 'entire_chip': ec, 'chip_scaler': sc, 'chrom_offsets': co,
          'chrom_nbins': cn, 'patch_bins': PATCH, 'x_resolution': 50000, 'y_resolution': 50000}
    m = build_model("halfring", cfg, PATCH, 6, "cpu", y_length=PATCH, preset="SMALL8")
    m = specify_use_model("cpu", ckpt, m); m.eval()
    ii, jj = np.triu_indices(PATCH, k=1); dd = (jj - ii).astype(np.float64) / PATCH
    regs = []
    for k, gb in enumerate(positions):
        bal = np.array(clr.matrix(balance=True)[gb:gb+PATCH, gb:gb+PATCH], dtype=np.float64); bal[np.isnan(bal)] = 0
        oe = np.nan_to_num(compute_oe_matrix(bal), nan=0.0, posinf=0.0, neginf=0.0)
        y = oe[ii, jj]; v = np.isfinite(y) & (np.abs(y) < 50)
        ds = TensorDataset(torch.tensor(oe.astype(np.float32)), torch.tensor([[gb, gb]], dtype=torch.int32).repeat_interleave(PATCH, 0))
        o = external_validation("cpu", m, "halfring", ds, batch_size=PATCH, chip_meta=cm, cross_reconstruction=True)
        chip = extract_chip_from_batch(torch.tensor([[gb, gb]], dtype=torch.int32), cm, "cpu").numpy().astype(np.float64)
        regs.append(dict(u1=o[4].numpy().astype(np.float32), u2=o[5].numpy().astype(np.float32), x_recon=o[2].numpy().astype(np.float32),
                         x_from_u2=o[8].numpy().astype(np.float32) if o[8] is not None else None,
                         chip=chip, oe=oe, y=y, valid=v))
        if k % 300 == 0: print(f"    [{label}] {k}/{len(positions)}", flush=True)
    del m
    print(f"  [{label}] forward done ({len(regs)} regions)", flush=True)
    return regs, ii, jj, dd
def read_pos(f):
    return [int(l.split('\t')[0]) for l in open(f) if len(l.split('\t')) >= 2]
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cka-epoch', type=int, default=508)
    args = parser.parse_args()
    cka_ckpt = os.path.join(ROOT, "models", "TORELink_2Mb_CKA",
                             f"TORELink_2Mb_CKA_epoch{args.cka_epoch}.pth")
    t0 = time.time()
    test_pos = read_pos(TP); train_pos = read_pos(TR)
    o1 = results_dir("section3_step1")
    f_u2d = lambda r, i, j, d: np.concatenate([r['u2'][i], r['u2'][j], d[:, None]], axis=1)
    f_u2o = lambda r, i, j, d: np.concatenate([r['u2'][i], r['u2'][j]], axis=1)
    f_d = lambda r, i, j, d: d[:, None]
    f_cd = lambda r, i, j, d: np.concatenate([r['chip'][i], r['chip'][j], d[:, None]], axis=1)
    if args.cka_epoch != 508:
        print(f"[lightweight] CKA epoch{args.cka_epoch} sensitivity (skip MSE/HIC02/u1->ChIP)", flush=True)
        with open(os.path.join(o1, "cka_vs_mse_mlp.json")) as f: existing_mlp = json.load(f)
        with open(os.path.join(o1, "cka_vs_mse_linear.json")) as f: existing_lin = json.load(f)
        mse_ctrl = existing_mlp["MSE+topo (control)"]
        mse_lin = existing_lin["MSE+topo (control)"]
        mlp_u2d = mse_ctrl["mlp_u2_dist"]; mlp_do = mse_ctrl["mlp_dist_only"]; mlp_cd = mse_ctrl["mlp_chip_dist"]
        lin_best = (mse_lin["best_alpha"], mse_lin["linear_probe_r"])
        tr_cka, ii, jj, dd = forward_model(cka_ckpt, HIC01, CHIP6, train_pos, f"CKA{args.cka_epoch}-train")
        te_cka, _, _, _ = forward_model(cka_ckpt, HIC01, CHIP6, test_pos, f"CKA{args.cka_epoch}-test")
        Xtr_c = np.concatenate([f_u2d(r, ii[r['valid']], jj[r['valid']], dd[r['valid']]) for r in tr_cka]).astype(np.float32)
        ytr_c = np.concatenate([r['y'][r['valid']] for r in tr_cka]).astype(np.float32)
        cka_mlp = train_mlp(Xtr_c, ytr_c, te_cka, f_u2d, ii, jj, dd, dim=17)
        cr_cka = [{'feat': r['u2'], 'y': r['y'], 'valid': r['valid']} for r in te_cka]
        cka_lin = loo_ridge_best(cr_cka, ii, jj, dd, mode='full')
        suffix = f"_epoch{args.cka_epoch}"
        cka_mlp_out = {"MSE+topo (control)": dict(mlp_u2_dist=mlp_u2d, mlp_dist_only=mlp_do, mlp_chip_dist=mlp_cd),
                       "CKA+topo": dict(mlp_u2_dist=cka_mlp, mlp_dist_only=mlp_do, mlp_chip_dist=mlp_cd)}
        cka_lin_out = {"MSE+topo (control)": dict(linear_probe_r=lin_best[1], best_alpha=lin_best[0]),
                       "CKA+topo": dict(linear_probe_r=cka_lin[1], best_alpha=cka_lin[0])}
        with open(os.path.join(o1, f"cka_vs_mse_mlp{suffix}.json"), 'w') as f: json.dump(cka_mlp_out, f, indent=2)
        with open(os.path.join(o1, f"cka_vs_mse_linear{suffix}.json"), 'w') as f: json.dump(cka_lin_out, f, indent=2)
        print(f"  CKA epoch{args.cka_epoch} mlp={cka_mlp:.4f} linear={cka_lin[1]:.4f} (MSE: {mlp_u2d:.4f}/{lin_best[1]:.4f}) dmlp={cka_mlp-mlp_u2d:+.4f}", flush=True)
        print(f"\n[Done] {time.time()-t0:.0f}s")
        return
    print(f"[step1b] SMALL8 probes: train={len(train_pos)} test={len(test_pos)} epochs={EPOCHS}", flush=True)
    tr_mse, ii, jj, dd = forward_model(MSE_CKPT, HIC01, CHIP6, train_pos, "MSE-train")
    te_mse, _, _, _ = forward_model(MSE_CKPT, HIC01, CHIP6, test_pos, "MSE-test")
    print("[1] mlp_probe_small8 (linear + decoder + MLP variants)...", flush=True)
    cr_u2 = [{'feat': r['u2'], 'y': r['y'], 'valid': r['valid']} for r in te_mse]
    lin_best = loo_ridge_best(cr_u2, ii, jj, dd, mode='full')
    default_r = float(np.nanmean([pearson(r['oe'][ii[r['valid']], jj[r['valid']]],
        r['x_from_u2'][ii[r['valid']], jj[r['valid']]]) for r in te_mse if r['x_from_u2'] is not None]))
    Xtr = np.concatenate([f_u2d(r, ii[r['valid']], jj[r['valid']], dd[r['valid']]) for r in tr_mse]).astype(np.float32)
    ytr = np.concatenate([r['y'][r['valid']] for r in tr_mse]).astype(np.float32)
    mlp_u2d = train_mlp(Xtr, ytr, te_mse, f_u2d, ii, jj, dd, dim=17)
    mlp_u2o = train_mlp(np.concatenate([f_u2o(r, ii[r['valid']], jj[r['valid']], dd[r['valid']]) for r in tr_mse]).astype(np.float32),
                        ytr, te_mse, f_u2o, ii, jj, dd, dim=16)
    mlp_do = train_mlp(np.concatenate([f_d(r, ii[r['valid']], jj[r['valid']], dd[r['valid']]) for r in tr_mse]).astype(np.float32),
                       ytr, te_mse, f_d, ii, jj, dd, dim=1)
    mlp_cd = train_mlp(np.concatenate([f_cd(r, ii[r['valid']], jj[r['valid']], dd[r['valid']]) for r in tr_mse]).astype(np.float32),
                       ytr, te_mse, f_cd, ii, jj, dd, dim=13)
    mlp_out = dict(linear_probe_r=lin_best[1], default_decoder_r=default_r,
                   mlp_u2_dist=mlp_u2d, mlp_u2_only=mlp_u2o, mlp_dist_only=mlp_do, mlp_chip_dist=mlp_cd)
    o1 = results_dir("section3_step1")
    with open(os.path.join(o1, "mlp_probe_small8.json"), 'w') as f: json.dump(mlp_out, f, indent=2)
    print(f"  linear={lin_best[1]:.4f} default={default_r:.4f} u2+dist={mlp_u2d:.4f} u2-only={mlp_u2o:.4f} dist={mlp_do:.4f} chip={mlp_cd:.4f}", flush=True)
    print("[2] cka_vs_mse (CKA forward + MLP)...", flush=True)
    tr_cka, _, _, _ = forward_model(CKA_CKPT, HIC01, CHIP6, train_pos, "CKA-train")
    te_cka, _, _, _ = forward_model(CKA_CKPT, HIC01, CHIP6, test_pos, "CKA-test")
    Xtr_c = np.concatenate([f_u2d(r, ii[r['valid']], jj[r['valid']], dd[r['valid']]) for r in tr_cka]).astype(np.float32)
    ytr_c = np.concatenate([r['y'][r['valid']] for r in tr_cka]).astype(np.float32)
    cka_mlp = train_mlp(Xtr_c, ytr_c, te_cka, f_u2d, ii, jj, dd, dim=17)
    cr_cka = [{'feat': r['u2'], 'y': r['y'], 'valid': r['valid']} for r in te_cka]
    cka_lin = loo_ridge_best(cr_cka, ii, jj, dd, mode='full')
    cka_mlp_out = {"MSE+topo (control)": dict(mlp_u2_dist=mlp_u2d, mlp_dist_only=mlp_do, mlp_chip_dist=mlp_cd),
                   "CKA+topo": dict(mlp_u2_dist=cka_mlp, mlp_dist_only=mlp_do, mlp_chip_dist=mlp_cd)}
    cka_lin_out = {"MSE+topo (control)": dict(linear_probe_r=lin_best[1], best_alpha=lin_best[0]),
                   "CKA+topo": dict(linear_probe_r=cka_lin[1], best_alpha=cka_lin[0])}
    with open(os.path.join(o1, "cka_vs_mse_mlp.json"), 'w') as f: json.dump(cka_mlp_out, f, indent=2)
    with open(os.path.join(o1, "cka_vs_mse_linear.json"), 'w') as f: json.dump(cka_lin_out, f, indent=2)
    print(f"  CKA mlp={cka_mlp:.4f} linear={cka_lin[1]:.4f} (MSE: {mlp_u2d:.4f}/{lin_best[1]:.4f}) Δmlp={cka_mlp-mlp_u2d:+.4f}", flush=True)
    print("[3] HIC02 cross-experiment...", flush=True)
    te_h02, _, _, _ = forward_model(MSE_CKPT, HIC02, CHIP6, test_pos, "HIC02-test")
    within_h02 = float(np.nanmean([pearson(r['oe'][ii[r['valid']], jj[r['valid']]],
        r['x_recon'][ii[r['valid']], jj[r['valid']]]) for r in te_h02]))
    h02_mlp = train_mlp(Xtr, ytr, te_h02, f_u2d, ii, jj, dd, dim=17)
    Xtr_d = np.concatenate([f_d(r, ii[r['valid']], jj[r['valid']], dd[r['valid']]) for r in tr_mse]).astype(np.float32)
    ytr_d = np.concatenate([r['y'][r['valid']] for r in tr_mse]).astype(np.float32)
    h02_dist = train_mlp(Xtr_d, ytr_d, te_h02, f_d, ii, jj, dd, dim=1)
    h02_out = dict(within_oe_r=within_h02, mlp_u2_dist=h02_mlp, mlp_dist_only=h02_dist)
    o4 = results_dir("section3_step1")
    with open(os.path.join(o4, "hic02_validation.json"), 'w') as f: json.dump(h02_out, f, indent=2)
    print(f"  HIC02 within={within_h02:.4f} mlp={h02_mlp:.4f} dist={h02_dist:.4f}", flush=True)
    print("[4] SMALL8 HiC→ChIP MLP (u1 → 6ch ChIP)...", flush=True)
    CHIP6_NAMES = ["CTCF", "RAD21", "SMC3", "H3K27ac", "H3K27me3", "H3K4me1"]
    u1_chip_mlp = {}
    for ch_idx, ch_name in enumerate(CHIP6_NAMES):
        Xtr_u1 = np.concatenate([r['u1'] for r in tr_mse], axis=0).astype(np.float32)
        ytr_u1 = np.concatenate([r['chip'][:, ch_idx] for r in tr_mse], axis=0).astype(np.float32)
        torch.manual_seed(0); torch.cuda.manual_seed(0)
        net = BinMLP(Xtr_u1.shape[1], 64).to(DEV)
        opt = torch.optim.Adam(net.parameters(), lr=LR); loss_fn = nn.HuberLoss()
        Xt = torch.tensor(Xtr_u1, device=DEV); yt = torch.tensor(ytr_u1, device=DEV); n = Xt.shape[0]
        for ep in range(EPOCHS):
            perm = torch.randperm(n, device=DEV)
            for i in range(0, n, BATCH):
                xb, yb = Xt[perm[i:i+BATCH]], yt[perm[i:i+BATCH]]
                loss = loss_fn(net(xb), yb); opt.zero_grad(); loss.backward(); opt.step()
        net.eval(); rs = []
        with torch.no_grad():
            for r in te_mse:
                pred = net(torch.tensor(r['u1'].astype(np.float32), device=DEV)).cpu().numpy()
                rs.append(pearson(r['chip'][:, ch_idx], pred))
        del net, Xt, yt, opt; torch.cuda.empty_cache()
        u1_chip_mlp[ch_name] = float(np.nanmean(rs))
        print(f"  {ch_name}: R={u1_chip_mlp[ch_name]:.4f}", flush=True)
    mlp_u1_chip_mean = float(np.nanmean(list(u1_chip_mlp.values())))
    mlp_out['mlp_u1_chip_mean'] = mlp_u1_chip_mean
    mlp_out['mlp_u1_chip_per_channel'] = u1_chip_mlp
    with open(os.path.join(o1, "mlp_probe_small8.json"), 'w') as f: json.dump(mlp_out, f, indent=2)
    print(f"  u1→ChIP mean = {mlp_u1_chip_mean:.4f}", flush=True)
    print("\n[5] chr22 external validation (MSE + CKA, zero-shot)...", flush=True)
    clr_h01 = cooler.Cooler(HIC01)
    chr22_ext = clr_h01.extent('22')
    chr22_pos = [chr22_ext[0] + i * PATCH for i in range((chr22_ext[1] - chr22_ext[0]) // PATCH)]
    print(f"  chr22: {len(chr22_pos)} patches", flush=True)
    Xtr_cd = np.concatenate([f_cd(r, ii[r['valid']], jj[r['valid']], dd[r['valid']]) for r in tr_mse]).astype(np.float32)
    te_chr22_mse, _, _, _ = forward_model(MSE_CKPT, HIC01, CHIP6, chr22_pos, "MSE-chr22")
    te_chr22_cka, _, _, _ = forward_model(cka_ckpt, HIC01, CHIP6, chr22_pos, "CKA-chr22")
    within_mse = float(np.nanmean([pearson(r['oe'][ii[r['valid']], jj[r['valid']]],
        r['x_recon'][ii[r['valid']], jj[r['valid']]]) for r in te_chr22_mse]))
    mlp_u2d_mse = train_mlp(Xtr, ytr, te_chr22_mse, f_u2d, ii, jj, dd, dim=17)
    mlp_do_chr22 = train_mlp(Xtr_d, ytr_d, te_chr22_mse, f_d, ii, jj, dd, dim=1)
    mlp_cd_chr22 = train_mlp(Xtr_cd, ytr, te_chr22_mse, f_cd, ii, jj, dd, dim=13)
    cr_train_mse = [{'feat': r['u2'], 'y': r['y'], 'valid': r['valid']} for r in tr_mse]
    cr_chr22_mse = [{'feat': r['u2'], 'y': r['y'], 'valid': r['valid']} for r in te_chr22_mse]
    lin_mse = ridge_fixed(cr_train_mse, cr_chr22_mse, ii, jj, dd, alpha=10.0, mode='full')
    print(f"  MSE: within={within_mse:.4f} mlp_u2d={mlp_u2d_mse:.4f} lin_u2d={lin_mse:.4f} dist={mlp_do_chr22:.4f} chip={mlp_cd_chr22:.4f}", flush=True)
    within_cka = float(np.nanmean([pearson(r['oe'][ii[r['valid']], jj[r['valid']]],
        r['x_recon'][ii[r['valid']], jj[r['valid']]]) for r in te_chr22_cka]))
    mlp_u2d_cka = train_mlp(Xtr_c, ytr_c, te_chr22_cka, f_u2d, ii, jj, dd, dim=17)
    cr_train_cka = [{'feat': r['u2'], 'y': r['y'], 'valid': r['valid']} for r in tr_cka]
    cr_chr22_cka = [{'feat': r['u2'], 'y': r['y'], 'valid': r['valid']} for r in te_chr22_cka]
    lin_cka = ridge_fixed(cr_train_cka, cr_chr22_cka, ii, jj, dd, alpha=10.0, mode='full')
    print(f"  CKA: within={within_cka:.4f} mlp_u2d={mlp_u2d_cka:.4f} lin_u2d={lin_cka:.4f} dist={mlp_do_chr22:.4f} chip={mlp_cd_chr22:.4f}", flush=True)
    chr22_out = {
        "MSE+topo (control)": dict(within_oe_r=within_mse, mlp_u2_dist=mlp_u2d_mse, linear_u2_dist=lin_mse,
                                    mlp_dist_only=mlp_do_chr22, mlp_chip_dist=mlp_cd_chr22),
        "CKA+topo": dict(within_oe_r=within_cka, mlp_u2_dist=mlp_u2d_cka, linear_u2_dist=lin_cka,
                         mlp_dist_only=mlp_do_chr22, mlp_chip_dist=mlp_cd_chr22)}
    with open(os.path.join(o1, "chr22_validation_small8.json"), 'w') as f: json.dump(chr22_out, f, indent=2)
    print(f"  -> {o1}/chr22_validation_small8.json", flush=True)
    print(f"\n[Done] {time.time()-t0:.0f}s")
if __name__ == '__main__':
    main()

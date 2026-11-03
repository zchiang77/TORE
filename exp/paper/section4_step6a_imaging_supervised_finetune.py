import os
import sys
import json
import time
import argparse
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.data import compute_oe_matrix
from exp._helpers import load_model, load_config, get_topo_cfg
from src.lossCompute import compute_normal_loss, compute_topological_loss
import exp.paper.section4_simrec_higashi_stage1 as s1
from exp.paper.section4_simrec_higashi_stage1 import (
    load_chr2_grouped, truth_B, coords_to_prob, sample_counts,
    DEPTHS_B, PATCH)
MODEL_PATH = os.path.join(ROOT, "models", "TORE_50kb",
                          "TORE_50kb_epoch500.pth")
OUT_DIR = os.path.join(ROOT, "exp", "results",
                       "section4_step6_imaging_supervised")
SPLIT_SEED = 20260916                                 
TRAIN_FRAC = 0.80
EPOCH_SAMPLE_SEED = 42                                              
LR = 1e-5
WEIGHT_DECAY = 0.0005
GRAD_CLIP = 10.0
WARM_STEPS = 100
HIC_BASE_LOSS = "mse"                                            
TRIU40 = np.triu_indices(PATCH, k=1)                        
def build_split(truth, seed=SPLIT_SEED, frac=TRAIN_FRAC):
    all_copies = sorted({c for blk in s1.CHR2_BLOCKS for c, _ in truth[blk]})
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(all_copies))
    n_train = int(len(all_copies) * frac)
    train = sorted(int(all_copies[i]) for i in perm[:n_train])
    test = sorted(int(all_copies[i]) for i in perm[n_train:])
    return train, test
def ensure_split(truth):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "split_imgsup.json")
    train, test = build_split(truth)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            saved = json.load(f)
        assert saved["train_copies"] == train and \
            saved["test_copies"] == test, \
            "split_imgsup.json   "
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"seed": SPLIT_SEED, "train_frac": TRAIN_FRAC,
                       "train_copies": train, "test_copies": test,
                       "n_train": len(train), "n_test": len(test)},
                      f, indent=1)
    return train, test
def build_train_samples(truth, train_copies):
    tset = set(train_copies)
    samples = []
    for blk in s1.CHR2_BLOCKS:
        for c, arr in truth[blk]:
            if c not in tset:
                continue
            p, d, valid = coords_to_prob(arr)
            z = compute_oe_matrix(p.astype(np.float64))
            z[np.isnan(z)] = 0
            z[np.isinf(z)] = 0
            fin = d[np.isfinite(d)]
            d_fill = np.nan_to_num(d, nan=float(fin.max()) if len(fin) else 0.0)
            np.fill_diagonal(d_fill, 0.0)
            samples.append({"copy": c, "blk": blk, "p": p.astype(np.float32),
                            "z": z.astype(np.float32),
                            "d": d_fill.astype(np.float32),
                            "valid": valid})
    return samples
def epoch_inputs(samples, epoch, seed=EPOCH_SAMPLE_SEED):
    oe_list, z_list, d_list, idx_list = [], [], [], []
    for i, s in enumerate(samples):
        rng = np.random.default_rng([seed, epoch, s["copy"], s["blk"]])
        tier = DEPTHS_B[int(rng.integers(0, len(DEPTHS_B)))]
        mat, _ = sample_counts(s["p"], TRIU40, rng, tier[0], tier[1])
        if mat is None:
            continue
        oe = compute_oe_matrix(mat.astype(np.float64))
        oe[np.isnan(oe)] = 0
        oe[np.isinf(oe)] = 0
        oe_list.append(oe.astype(np.float32))
        z_list.append(s["z"])
        d_list.append(s["d"])
        idx_list.append(i)
    return (np.stack(oe_list), np.stack(z_list), np.stack(d_list),
            np.array(idx_list))
def run_datacheck():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    print("[datacheck]  () ...", flush=True)
    grp = load_chr2_grouped()
    truth, drops = truth_B(grp)
    train, test = ensure_split(truth)
    print(f"  : {len(train)}  / {len(test)}  "
          f"( {SPLIT_SEED}, 80/20)")
    assert not (set(train) & set(test)), "/  "
    samples = build_train_samples(truth, train)
    n_expect = sum(1 for blk in s1.CHR2_BLOCKS
                   for c, _ in truth[blk] if c in set(train))
    print(f"   (×): {len(samples)} ( {n_expect})")
    assert len(samples) == n_expect
    oe1, z1, d1, _ = epoch_inputs(samples[:20], epoch=0)
    oe2, _, _, _ = epoch_inputs(samples[:20], epoch=0)
    assert np.array_equal(oe1, oe2), "epoch "
    oe3, _, _, _ = epoch_inputs(samples[:20], epoch=1)
    assert not np.array_equal(oe1, oe3[:len(oe1)]), \
        " epoch  ()"
    nz_frac = []
    for i, s in enumerate(samples[:400]):
        rng = np.random.default_rng([EPOCH_SAMPLE_SEED, 0, s["copy"],
                                     s["blk"]])
        tier = DEPTHS_B[int(rng.integers(0, len(DEPTHS_B)))]
        mat, _ = sample_counts(s["p"], TRIU40, rng, tier[0], tier[1])
        if mat is not None:
            nz_frac.append((mat[TRIU40[0], TRIU40[1]] > 0).mean())
    z_all = np.stack([s["z"] for s in samples[:400]])
    d_all = np.stack([s["d"] for s in samples[:400]])
    fin_d = d_all[np.isfinite(d_all)]
    report = {
        "n_train_copies": len(train), "n_test_copies": len(test),
        "n_train_samples": len(samples),
        "truth_dropped_pairs": sum(drops.values()),
        "input_nz_frac_median": float(np.median(nz_frac)),
        "input_nz_frac_range": [float(np.min(nz_frac)),
                                float(np.max(nz_frac))],
        "z_oe_range": [float(z_all.min()), float(z_all.max())],
        "z_nz_frac": float((z_all > 0).mean()),
        "d_fill_range_nm": [float(fin_d.min()), float(fin_d.max())],
        "d_fill_median_offdiag": float(np.median(
            fin_d[fin_d > 0])),
        "determinism_check": "PASS (epoch0 ; epoch1 )",
        "split_overlap": 0,
        "elapsed_s": round(time.time() - t0, 1),
    }
    p_all = np.stack([s["p"] for s in samples[:100]])
    for eps in (0.01, 1e-6):
        dx = np.power(p_all + eps, -1.0 / 3.0)
        report[f"rejected_dx_powerlaw_eps{eps}_range"] = \
            [float(dx.min()), float(dx.max())]
    path = os.path.join(OUT_DIR, "datacheck_imgsup.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"[datacheck] Done → {path}", flush=True)
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-interval", type=int, default=10)
    ap.add_argument("--smoke", action="store_true",
                    help="200  × 2 epoch,  _imgsup_smoke ")
    ap.add_argument("--datacheck", action="store_true",
                    help=" CPU ,  GPU")
    ap.add_argument("--resume", action="store_true",
                    help=" checkpoint  ()")
    args = ap.parse_args()
    if args.smoke:
        args.epochs = 2
    return args
def main():
    args = parse_args()
    if args.datacheck:
        run_datacheck()
        return
    import torch
    suffix = "_smoke" if args.smoke else ""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Step6a-imgsup] device={device}, smoke={args.smoke}, "
          f"epochs={args.epochs}", flush=True)
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(123)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    save_dir = os.path.join(ROOT, "models", f"superTORE_single_view{suffix}")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    grp = load_chr2_grouped()
    truth, _ = truth_B(grp)
    train, test = ensure_split(truth)
    samples = build_train_samples(truth, train)
    print(f"  : {len(train)}/{len(test)} ; "
          f" {len(samples)} (×6)", flush=True)
    cfg = load_config()
    topo_weight = float(cfg.get("ring", {}).get("topo_weight", 100))
    topo_cfg = get_topo_cfg(cfg)
    model, _ = load_model(MODEL_PATH, device=args.device, patch_bins=PATCH)
    model = model.to(device)
    model.train()
    print(f"   ({MODEL_PATH})", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))
    def warm_lr_lambda(step):
        return (step + 1) / WARM_STEPS if step < WARM_STEPS else 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lr_lambda=warm_lr_lambda)
    log = {"epoch": [], "x_recon_loss": [], "topo_loss": [], "total_loss": [],
           "time_per_epoch": [], "n_zero_topo": [],
           "mode": f"recon(mse,z=OE(trunc_p))+topo(w={topo_weight:g},"
                   f"D_x=imaging_d)",
           "supervision": "z=OE(); D_x=(bin"
                          ")",
           "split_seed": SPLIT_SEED, "epoch_sample_seed": EPOCH_SAMPLE_SEED,
           "hic_base_loss": HIC_BASE_LOSS, "lr": args.lr,
           "n_train_copies": len(train), "n_test_copies": len(test),
           "n_train_samples": len(samples), "seed": 123,
           "warm_steps": WARM_STEPS, "betas": [0.9, 0.95],
           "data": "Su2020 chr2 ; epoch rng=[42,epoch,c,blk], "
                   ""}
    start_epoch = 0
    if args.resume:
        ckpts = sorted([fn for fn in os.listdir(save_dir)
                        if fn.endswith(".pth")],
                       key=lambda fn: int(fn.split("epoch")[-1].split(".")[0]))
        if ckpts:
            ck = torch.load(os.path.join(save_dir, ckpts[-1]),
                            map_location=device)
            model.load_state_dict(ck["model"])
            optimizer.load_state_dict(ck["optimizer"])
            start_epoch = int(ck["epoch"])
            print(f"  [resume]  {ckpts[-1]} : epoch={start_epoch}",
                  flush=True)
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        oe_mats, z_mats, d_mats, _ = epoch_inputs(samples, epoch)
        if args.smoke:
            oe_mats = oe_mats[:200]
            z_mats = z_mats[:200]
            d_mats = d_mats[:200]
        n_samples = len(oe_mats)
        print(f"  epoch {epoch+1}:  {n_samples}  "
              f"({time.time()-t0:.1f}s)", flush=True)
        perm = np.random.RandomState(1234 + epoch).permutation(n_samples)
        epoch_recon = epoch_topo = epoch_total = 0.0
        n_valid = n_skipped = n_zero_topo = 0
        model.train()
        for i in perm:
            x = torch.tensor(oe_mats[i], dtype=torch.float32, device=device)
            z = torch.tensor(z_mats[i], dtype=torch.float32, device=device)
            D_x = torch.tensor(d_mats[i], dtype=torch.float32, device=device)
            try:
                x_recon, u1 = model(x)
                x_recon_loss, _, _, _, _ = compute_normal_loss(
                    x=x, y=None, z=z, x_pred=None, y_pred=None,
                    x_recon=x_recon, y_recon=None, u1=u1, u2=None,
                    penalty=1, hic_base_loss=HIC_BASE_LOSS, huber_delta=2.0)
                topo_loss = compute_topological_loss(
                    x=x, u1=u1, u2=None, topo_thresh=topo_cfg.threshold,
                    D_x=D_x, topo_cfg=topo_cfg)
                total_loss = x_recon_loss + topo_weight * topo_loss
                if float(topo_loss.detach()) == 0.0:
                    n_zero_topo += 1                          
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    raise
                n_skipped += 1
                if n_skipped <= 5:
                    print(f"  [WARN] sample {i} skipped: {e}", flush=True)
                zero_head = sum(p.sum() for p in model.parameters())
                total_loss = zero_head * 0.0
                x_recon_loss = total_loss
                topo_loss = torch.tensor(0.0)
                optimizer.zero_grad()
                total_loss.backward()
                continue
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            epoch_recon += float(x_recon_loss.detach())
            epoch_topo += float(topo_loss.detach())
            epoch_total += float(total_loss.detach())
            n_valid += 1
        dt = time.time() - t0
        log["epoch"].append(epoch + 1)
        log["x_recon_loss"].append(epoch_recon / max(n_valid, 1))
        log["topo_loss"].append(epoch_topo / max(n_valid, 1))
        log["total_loss"].append(epoch_total / max(n_valid, 1))
        log["time_per_epoch"].append(dt)
        log["n_zero_topo"].append(n_zero_topo)
        print(f"  epoch {epoch+1}/{args.epochs}  "
              f"recon={epoch_recon/max(n_valid,1):.4f}  "
              f"topo={epoch_topo/max(n_valid,1):.6f}  "
              f"total={epoch_total/max(n_valid,1):.4f}  "
              f"(n={n_valid}, skipped={n_skipped}, "
              f"zero_topo={n_zero_topo})  {dt:.1f}s", flush=True)
        if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
            ckpt_path = os.path.join(
                save_dir, f"superTORE_single_view_epoch{epoch+1}.pth")
            torch.save({"epoch": epoch + 1, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict()}, ckpt_path)
            print(f"  → saved {ckpt_path} ()", flush=True)
    log_path = os.path.join(OUT_DIR, f"train_log{suffix}.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    lines = ["=" * 60,
             f" (imgsup)   smoke={args.smoke}",
             f"/epoch: {log['n_train_samples']} ( 80/20 )",
             f": {log['supervision']}",
             f"Final: recon={log['x_recon_loss'][-1]:.4f} "
             f"topo={log['topo_loss'][-1]:.6f} "
             f"total={log['total_loss'][-1]:.4f}",
             f": {sum(log['time_per_epoch']):.0f}s"]
    with open(os.path.join(OUT_DIR, f"summary{suffix}.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"[Step6a-imgsup] Done → {OUT_DIR}", flush=True)
if __name__ == "__main__":
    main()

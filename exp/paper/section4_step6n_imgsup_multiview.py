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
EPOCH_SAMPLE_SEED = 42
LR = 1e-5
WEIGHT_DECAY = 0.0005
GRAD_CLIP = 10.0
WARM_STEPS = 100
HIC_BASE_LOSS = "mse"
TRIU40 = np.triu_indices(PATCH, k=1)
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
                            "d": d_fill.astype(np.float32)})
    return samples
def epoch_inputs_multiview(samples, epoch, K, seed=EPOCH_SAMPLE_SEED):
    oe_list, z_list, d_list = [], [], []
    for s in samples:
        for k in range(K):
            rng = np.random.default_rng([seed, epoch, k, s["copy"],
                                         s["blk"]])
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
    return (np.stack(oe_list), np.stack(z_list), np.stack(d_list))
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", type=int, default=10,
                    help=" epoch  K")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-interval", type=int, default=1)
    ap.add_argument("--smoke", action="store_true",
                    help="200  × 2 epoch")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.views = 2
        args.epochs = 2
    import torch
    ddp = "LOCAL_RANK" in os.environ
    if ddp:
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        local_rank, rank, world = 0, 0, 1
        device = torch.device(args.device if torch.cuda.is_available()
                              else "cpu")
    K = args.views
    suffix = "_smoke" if args.smoke else f"_mv{K}"
    print(f"[Step6n-multiview] ddp={ddp} world={world} K={K} "
          f"epochs={args.epochs} device={device}", flush=True)
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(123)
    save_dir = os.path.join(ROOT, "models", f"superTORE{suffix}")
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(OUT_DIR, exist_ok=True)
    if ddp:
        import torch.distributed as dist
        dist.barrier()
    grp = load_chr2_grouped()
    truth, _ = truth_B(grp)
    with open(os.path.join(OUT_DIR, "split_imgsup.json"),
              encoding="utf-8") as f:
        split = json.load(f)
    train, test = split["train_copies"], split["test_copies"]
    samples = build_train_samples(truth, train)
    print(f"  : {len(train)}/{len(test)} ;  "
          f"{len(samples)} (×6) × K={K} ", flush=True)
    cfg = load_config()
    topo_weight = float(cfg.get("ring", {}).get("topo_weight", 100))
    topo_cfg = get_topo_cfg(cfg)
    if ddp:
        from src.model_builder import build_model
        from src.forward import specify_use_model
        model = build_model("x_reconstruction", cfg, PATCH_BINS := PATCH, 0,
                            local_rank, y_length=None, preset="SMALL")
        model = specify_use_model(local_rank, MODEL_PATH, model)
        model = model.to(device)
        model = DDP(model, device_ids=[local_rank])
    else:
        model, _ = load_model(MODEL_PATH, device=args.device,
                              patch_bins=PATCH)
        model = model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        (model.module if ddp else model).parameters(), lr=args.lr,
        weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))
    def warm_lr_lambda(step):
        return (step + 1) / WARM_STEPS if step < WARM_STEPS else 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=warm_lr_lambda)
    log = {"epoch": [], "x_recon_loss": [], "topo_loss": [],
           "total_loss": [], "time_per_epoch": [], "n_zero_topo": [],
           "mode": f"multiview(K={K}), recon(mse,z)+topo(w={topo_weight:g})",
           "views": K, "epochs": args.epochs, "lr": args.lr,
           "n_train_copies": len(train), "n_test_copies": len(test),
           "n_base_samples": len(samples), "seed": 123,
           "hic_base_loss": HIC_BASE_LOSS}
    start_epoch = 0
    if args.resume and rank == 0:
        ckpts = sorted([fn for fn in os.listdir(save_dir)
                        if fn.endswith(".pth")],
                       key=lambda fn: int(fn.split("epoch")[-1].split(".")[0]))
        if ckpts:
            ck = torch.load(os.path.join(save_dir, ckpts[-1]),
                            map_location=device)
            (model.module if ddp else model).load_state_dict(ck["model"])
            optimizer.load_state_dict(ck["optimizer"])
            start_epoch = int(ck["epoch"])
            print(f"  [resume] {ckpts[-1]}: epoch={start_epoch}", flush=True)
    if ddp:
        import torch.distributed as dist
        dist.barrier()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        oe_mats, z_mats, d_mats = epoch_inputs_multiview(samples, epoch, K)
        if args.smoke:
            oe_mats, z_mats, d_mats = oe_mats[:200], z_mats[:200], \
                d_mats[:200]
        n_samples = len(oe_mats)
        if rank == 0:
            print(f"  epoch {epoch+1}:  {n_samples}  "
                  f"({time.time()-t0:.1f}s)", flush=True)
        perm = np.random.RandomState(1234 + epoch).permutation(n_samples)
        my = perm[rank::world]
        epoch_recon = epoch_topo = epoch_total = 0.0
        n_valid = n_skipped = n_zero_topo = 0
        for i in my:
            x = torch.tensor(oe_mats[i], dtype=torch.float32, device=device)
            z = torch.tensor(z_mats[i], dtype=torch.float32, device=device)
            D_x = torch.tensor(d_mats[i], dtype=torch.float32, device=device)
            try:
                x_recon, u1 = model(x)
                x_recon_loss, _, _, _, _ = compute_normal_loss(
                    x=x, y=None, z=z, x_pred=None, y_pred=None,
                    x_recon=x_recon, y_recon=None, u1=u1, u2=None,
                    penalty=1, hic_base_loss=HIC_BASE_LOSS,
                    huber_delta=2.0)
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
                zero_head = sum(p.sum() for p in
                                (model.module if ddp else model).parameters())
                total_loss = zero_head * 0.0
                optimizer.zero_grad()
                total_loss.backward()
                continue
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (model.module if ddp else model).parameters(),
                max_norm=GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            epoch_recon += float(x_recon_loss.detach())
            epoch_topo += float(topo_loss.detach())
            epoch_total += float(total_loss.detach())
            n_valid += 1
        if ddp:
            import torch.distributed as dist
            stats = torch.tensor([epoch_recon, epoch_topo, epoch_total,
                                  n_valid, n_skipped], dtype=torch.float64,
                                 device=device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            epoch_recon, epoch_topo, epoch_total, n_valid, n_skipped = \
                stats.tolist()
            n_valid = max(int(n_valid), 1)
        if rank == 0:
            dt = time.time() - t0
            log["epoch"].append(epoch + 1)
            log["x_recon_loss"].append(epoch_recon / max(n_valid, 1))
            log["topo_loss"].append(epoch_topo / max(n_valid, 1))
            log["total_loss"].append(epoch_total / max(n_valid, 1))
            log["time_per_epoch"].append(dt)
            log["n_zero_topo"].append(n_zero_topo)
            print(f"  epoch {epoch+1}/{args.epochs} "
                  f"recon={epoch_recon/max(n_valid,1):.4f} "
                  f"topo={epoch_topo/max(n_valid,1):.6f} "
                  f"total={epoch_total/max(n_valid,1):.4f} "
                  f"(n={int(n_valid)}, skip={int(n_skipped)}, "
                  f"zero_topo={n_zero_topo}) {dt:.1f}s", flush=True)
            if (epoch + 1) % args.save_interval == 0 or \
                    epoch == args.epochs - 1:
                ckpt_path = os.path.join(
                    save_dir, f"superTORE_mv{K}_epoch{epoch+1}.pth")
                torch.save({"epoch": epoch + 1,
                            "model": (model.module if ddp else model)
                            .state_dict(),
                            "optimizer": optimizer.state_dict()}, ckpt_path)
                print(f"  → saved {ckpt_path}", flush=True)
        if ddp:
            import torch.distributed as dist
            dist.barrier()
    if rank == 0:
        log_path = os.path.join(OUT_DIR, f"train_log{suffix}.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2, ensure_ascii=False)
        print(f"[Step6n-multiview] Done → {log_path}", flush=True)
if __name__ == "__main__":
    main()

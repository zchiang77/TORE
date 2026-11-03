import os
import sys
import json
import time
import argparse
import h5py
import numpy as np
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from exp._helpers import load_model, load_config, get_topo_cfg, results_dir
from src.data import compute_oe_matrix
from src.lossCompute import compute_normal_loss, compute_topological_loss
from src.model_builder import build_model
from src.forward import specify_use_model
from exp.paper.section4_step1c_dx_variants import knn_impute_k, smooth_and_clean
MODEL_PATH = os.path.join(
    ROOT, "models", "TORE_50kb",
    "TORE_50kb_epoch500.pth")
H5_V1_PATH = os.path.join(ROOT, "exp", "results", "section4_step1a",
                          "genome_imputed.h5")
KNN_V2_PATH = os.path.join(ROOT, "exp", "results", "section4_step1b",
                           "knn_v2.npz")
PATCH_BINS = 40
LR = 1e-5
WEIGHT_DECAY = 0.0005
GRAD_CLIP = 10.0
SELECTED_BLOCKS = [0, 1, 28, 37, 41, 45, 49, 53, 57, 6]
def reproduce_split(labels, split, train_frac=0.65, seed=42):
    rng = np.random.RandomState(seed)
    train_idx, val_idx = [], []
    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        rng.shuffle(idx)
        n_train = int(len(idx) * train_frac)
        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train:])
    return np.array(train_idx) if split == "train" else np.array(val_idx)
def load_imputed_cells(h5_path, blocks, cell_idx):
    mats = []
    with h5py.File(h5_path, "r") as h5f:
        for bi in blocks:
            imp = np.array(h5f[f"block_{bi}"]["imputed"], dtype=np.float32)
            mats.append(imp[cell_idx])
    return np.concatenate(mats, axis=0)
def get_subset(labels, block_ids, max_per_type, seed=42):
    n_blocks = len(np.unique(block_ids))
    n_samples = len(labels)
    n_cells = n_samples // n_blocks
    cell_labels = labels[:n_cells]
    rng = np.random.RandomState(seed)
    cell_idx = []
    for label in np.unique(cell_labels):
        t_idx = np.where(cell_labels == label)[0]
        n = min(max_per_type, len(t_idx))
        chosen = rng.choice(t_idx, n, replace=False)
        cell_idx.extend(chosen)
    cell_idx = np.sort(cell_idx)
    sample_idx = []
    for b in range(n_blocks):
        for ci in cell_idx:
            sample_idx.append(b * n_cells + ci)
    return np.array(sample_idx)
def prepare_oe(matrices):
    n = len(matrices)
    oe = np.zeros_like(matrices, dtype=np.float32)
    for i in range(n):
        m = compute_oe_matrix(matrices[i])
        m[np.isnan(m)] = 0
        m[np.isinf(m)] = 0
        oe[i] = m.astype(np.float32)
    return oe
def select_blocks_from_h5(h5_path, blocks_arg):
    with h5py.File(h5_path, "r") as h5f:
        quality = []
        for k in h5f.keys():
            if k.startswith("block_"):
                g = h5f[k]
                quality.append((int(k.split("_")[1]),
                                float(g.attrs.get("imputed_nonzero_frac", 0))))
    quality.sort(key=lambda x: (-x[1], x[0]))
    if blocks_arg == "all":
        return [q[0] for q in quality]
    if blocks_arg.startswith("top"):
        return sorted(q[0] for q in quality[:int(blocks_arg[3:])])
    return [int(x) for x in blocks_arg.split(",")]
def load_training_data(args):
    if args.blocks:
        assert args.input_mode == "imputed", \
            "--blocks  --input-mode imputed"
        h5_path = os.path.join(args.data_dir, "genome_imputed_v2.h5")
        blocks = select_blocks_from_h5(h5_path, args.blocks)
        n_blocks = len(blocks)
        with h5py.File(h5_path, "r") as h5f:
            h5_labels = np.array(h5f["labels"])
        print(f"  blocks: {args.blocks} → {n_blocks}  {blocks}")
        train_idx_all = reproduce_split(h5_labels, "train")
        train_labels = h5_labels[train_idx_all]
        rng = np.random.RandomState(42)
        cell_sel = []
        for t in np.unique(train_labels):
            t_idx = np.where(train_labels == t)[0]
            n_ = min(args.max_per_type, len(t_idx))
            cell_sel.extend(rng.choice(t_idx, n_, replace=False))
        cell_sel = np.sort(cell_sel)
        print(f"  Cells: {len(cell_sel)} (max {args.max_per_type}/type), "
              f"samples={len(cell_sel) * n_blocks}")
        input_mats = load_imputed_cells(h5_path, blocks,
                                        train_idx_all[cell_sel])
        labels = np.concatenate([train_labels[cell_sel]
                                 for _ in range(n_blocks)])
        print(f"  Imputed input: {input_mats.shape}, "
              f"nz={float((input_mats > 0).mean())*100:.1f}%")
        if args.dx_k and args.dx_k < 20:
            print(f"  D_x : K={args.dx_k} impute ( K=20)...")
            knn = np.load(KNN_V2_PATH)
            knn_idx, knn_sim = knn["knn_idx"], knn["knn_sim"]
            n_sel = len(cell_sel)
            sel_global = train_idx_all[cell_sel]                
            d_x_list = []
            with h5py.File(H5_V1_PATH, "r") as v1f:
                for bi, blk in enumerate(blocks):
                    cnt_full = np.array(v1f[f"block_{blk}"]["count"],
                                        dtype=np.float32)
                    idx_k = knn_idx[sel_global, :args.dx_k]
                    sim_k = knn_sim[sel_global, :args.dx_k]
                    neigh = cnt_full[idx_k]                      
                    w = np.maximum(sim_k, 0.0) + 1e-3
                    w = w / w.sum(axis=1, keepdims=True)
                    imp_k = np.einsum('nk,nkij->nij', w,
                                      neigh.astype(np.float64))
                    del cnt_full, neigh
                    imp_k = smooth_and_clean(imp_k)
                    dx = np.power(imp_k.astype(np.float64) + args.dx_eps,
                                  -1.0 / args.dx_gamma).astype(np.float32)
                    idx40 = np.arange(PATCH_BINS)
                    dx[:, idx40, idx40] = 0.0
                    d_x_list.append(dx)
                    del imp_k, dx
                    if (bi + 1) % 10 == 0:
                        print(f"    block {bi+1}/{n_blocks} D_x done",
                              flush=True)
            d_x = np.concatenate(d_x_list, axis=0)
            print(f"  D_x (K={args.dx_k}): {d_x.shape} "
                  f"range=[{d_x.min():.2f},{d_x.max():.2f}]")
        else:
            if args.dx_k and args.dx_k >= 20:
                print(f"  D_x = (C+{args.dx_eps})^(-1/{args.dx_gamma}) "
                      f" (dx_k={args.dx_k} >= 20,  D_x, )...")
            else:
                print(f"  D_x = (C+{args.dx_eps})^(-1/{args.dx_gamma}) ...")
            d_x = np.power(input_mats.astype(np.float64) + args.dx_eps,
                           -1.0 / args.dx_gamma).astype(np.float32)
            idx40 = np.arange(PATCH_BINS)
            d_x[:, idx40, idx40] = 0.0
        desc = f"{h5_path} (blocks={args.blocks})"
        return input_mats, d_x, labels, desc, n_blocks
    data_path = os.path.join(args.data_dir,
                             f"train_data{args.data_suffix}.npz")
    train_data = np.load(data_path)
    covnorm = train_data["covnorm"]
    d_x = train_data["d_x"]
    labels = train_data["labels"]
    block_ids = train_data["block_ids"]
    n_blocks = len(np.unique(block_ids))
    n_cells_total = len(covnorm) // n_blocks
    print(f"  Train data: {len(covnorm)} samples "
          f"({n_blocks} blocks × {n_cells_total} cells)")
    if args.dx_path:
        print(f"  d_x override: {args.dx_path}")
        d_x = np.load(args.dx_path)["d_x"]
        assert d_x.shape == covnorm.shape, \
            f"d_x : {d_x.shape} vs {covnorm.shape}"
    if args.max_per_type > 0:
        subset_idx = get_subset(labels, block_ids, args.max_per_type)
        d_x = d_x[subset_idx]
        labels = labels[subset_idx]
        n_subset_cells = len(subset_idx) // n_blocks
        print(f"  Subset: {len(subset_idx)} samples ({n_subset_cells} cells × "
              f"{n_blocks} blocks, max {args.max_per_type}/type)")
    else:
        subset_idx = np.arange(len(covnorm))
    if args.input_mode == "imputed":
        h5_path = os.path.join(args.data_dir, "genome_imputed_v2.h5")
        with h5py.File(h5_path, "r") as h5f:
            h5_labels = np.array(h5f["labels"])
        train_idx_all = reproduce_split(h5_labels, "train")
        assert np.array_equal(h5_labels[train_idx_all],
                              train_data["labels"][:n_cells_total]), \
            "h5  npz  train !"
        cell_sel = subset_idx[subset_idx < n_cells_total]
        input_mats = load_imputed_cells(h5_path, SELECTED_BLOCKS,
                                        train_idx_all[cell_sel])
        print(f"  Imputed input: {input_mats.shape}, "
              f"nz={float((input_mats > 0).mean())*100:.1f}%")
    else:
        input_mats = covnorm[subset_idx]
    return input_mats, d_x, labels, data_path, n_blocks
def build_ddp_model(local_rank, device):
    cfg = load_config()
    model = build_model("x_reconstruction", cfg, PATCH_BINS, 0, local_rank,
                        y_length=None, preset="SMALL")
    model = specify_use_model(local_rank, MODEL_PATH, model)
    model = model.to(device)
    model.train()
    return model
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--max-per-type", type=int, default=400)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-interval", type=int, default=10)
    ap.add_argument("--data-dir",
                    default=os.path.join(ROOT, "exp", "results",
                                         "section4_step1b"))
    ap.add_argument("--data-suffix", default="_v2")
    ap.add_argument("--topo-weight", type=float, default=None,
                    help=" config.json ring.topo_weight=100")
    ap.add_argument("--recon-only", action="store_true")
    ap.add_argument("--input-mode", default="covnorm",
                    choices=["covnorm", "imputed"])
    ap.add_argument("--dx-path", default=None)
    ap.add_argument("--blocks", default=None,
                    help="'topN'/'all'/;  h5  D_x")
    ap.add_argument("--dx-gamma", type=float, default=3.0)
    ap.add_argument("--dx-eps", type=float, default=0.01)
    ap.add_argument("--dx-k", type=int, default=None,
                    help="D_x  KNN K  ( K=20); "
                         " 10/5  count K=dx_k  D_x")
    ap.add_argument("--save-suffix", default="")
    ap.add_argument("--log-suffix", default="")
    ap.add_argument("--ddp", action="store_true",
                    help=" DDP ( torchrun --nproc_per_node=2 )")
    args = ap.parse_args()
    return args
def main():
    args = parse_args()
    ddp = args.ddp
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
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    out_dir = results_dir("section4_step2")
    save_dir = os.path.join(
        ROOT, "models", f"TORE_finetune{args.save_suffix}")
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
    cfg = load_config()
    topo_weight = args.topo_weight
    if topo_weight is None:
        topo_weight = float(cfg.get("ring", {}).get("topo_weight", 100))
    mode = "recon-only" if args.recon_only else f"recon+topo(w={topo_weight:g})"
    mode = f"{mode},input={args.input_mode}"
    if rank == 0:
        print(f"[Section4-Step2] Output → {out_dir}")
        print(f"  mode={mode}, ddp={ddp} (world={world})")
        print(f"  save_dir={save_dir}")
        print(f"  epochs={args.epochs}, lr={args.lr}, "
              f"max_per_type={args.max_per_type}", flush=True)
    if rank == 0:
        print("\n=== Load data ===", flush=True)
    input_mats, d_x, labels, data_desc, n_blocks = load_training_data(args)
    if rank == 0:
        print("\n=== OE normalization ===", flush=True)
    oe_matrices = prepare_oe(input_mats)
    del input_mats
    if rank == 0:
        print(f"  OE matrices: {oe_matrices.shape}, "
              f"range=[{oe_matrices.min():.4f}, {oe_matrices.max():.4f}]",
              flush=True)
    if ddp:
        model = build_ddp_model(local_rank, device)
        model = DDP(model, device_ids=[local_rank])
    else:
        model, _ = load_model(MODEL_PATH, device=args.device,
                              patch_bins=PATCH_BINS)
        model = model.to(device)
        model.train()
    if rank == 0:
        print(f"\n  Model loaded ({'DDP' if ddp else 'single'} on {device})",
              flush=True)
    topo_cfg = get_topo_cfg(cfg)
    if rank == 0:
        print(f"  topo_cfg: threshold={topo_cfg.threshold}, "
              f"gamma={topo_cfg.gamma}, h0_weight={topo_cfg.h0_weight}, "
              f"smooth_weight={topo_cfg.smooth_weight}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=WEIGHT_DECAY)
    n_samples = len(oe_matrices)
    log = {"epoch": [], "x_recon_loss": [], "topo_loss": [], "total_loss": [],
           "time_per_epoch": [],
           "mode": mode, "ddp": ddp, "world_size": world,
           "topo_weight": (0 if args.recon_only else topo_weight),
           "data": data_desc, "dx_path": args.dx_path,
           "dx_k": args.dx_k, "dx_gamma": args.dx_gamma, "dx_eps": args.dx_eps,
           "lr": args.lr, "max_per_type": args.max_per_type, "blocks": args.blocks}
    if rank == 0:
        print(f"\n=== Training ({n_samples} samples, {args.epochs} epochs, "
              f"{mode}) ===", flush=True)
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        epoch_recon = 0.0
        epoch_topo = 0.0
        epoch_total = 0.0
        n_valid = 0
        n_skipped = 0
        skip_reasons = {}
        perm = np.random.RandomState(1234 + epoch).permutation(n_samples)
        my_samples = perm[rank::world]
        for i in my_samples:
            x = torch.tensor(oe_matrices[i], dtype=torch.float32,
                             device=device)
            skipped = False
            try:
                x_recon, u1 = model(x)
                x_recon_loss, _, _, _, _ = compute_normal_loss(
                    x=x, y=None, z=None, x_pred=None, y_pred=None,
                    x_recon=x_recon, y_recon=None, u1=u1, u2=None,
                    penalty=1, hic_base_loss="mse")
                if args.recon_only:
                    total_loss = x_recon_loss
                    topo_val = 0.0
                else:
                    D_x = torch.tensor(d_x[i], dtype=torch.float32,
                                       device=device)
                    topo_loss = compute_topological_loss(
                        x=x, u1=u1, u2=None,
                        topo_thresh=topo_cfg.threshold,
                        D_x=D_x, topo_cfg=topo_cfg)
                    total_loss = x_recon_loss + topo_weight * topo_loss
                    topo_val = float(topo_loss.detach())
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    raise
                etype = type(e).__name__
                skip_reasons[etype] = skip_reasons.get(etype, 0) + 1
                n_skipped += 1
                skipped = True
                if n_skipped <= 5 or n_skipped % 100 == 0:
                    print(f"  [WARN rank{rank}] sample {i} skipped ({etype}): {e}",
                          flush=True)
                zero_head = sum(p.sum() for p in model.parameters())
                total_loss = zero_head * 0.0
                x_recon_loss = total_loss
                topo_val = 0.0
            except Exception as e:
                etype = type(e).__name__
                skip_reasons[etype] = skip_reasons.get(etype, 0) + 1
                n_skipped += 1
                print(f"  [ERROR rank{rank}] sample {i} unexpected ({etype}): {e}",
                      flush=True)
                raise
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=GRAD_CLIP)
            if not skipped:
                optimizer.step()
            if not skipped:
                epoch_recon += float(x_recon_loss.detach())
                epoch_topo += topo_val
                epoch_total += float(total_loss.detach())
                n_valid += 1
        stats = torch.tensor([epoch_recon, epoch_topo, epoch_total, n_valid, n_skipped],
                             dtype=torch.float64, device=device)
        if ddp:
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        epoch_recon, epoch_topo, epoch_total, n_valid, n_skipped = stats.tolist()
        n_valid = max(int(n_valid), 1)
        n_skipped = int(n_skipped)
        if ddp:
            gathered = [None] * world
            dist.all_gather_object(
                gathered, {k: int(v) for k, v in skip_reasons.items()})
            merged = {}
            for g in gathered:
                for k, v in g.items():
                    merged[k] = merged.get(k, 0) + v
            skip_reasons = merged
        dt = time.time() - t0
        avg_recon = epoch_recon / n_valid
        avg_topo = epoch_topo / n_valid
        avg_total = epoch_total / n_valid
        if rank == 0:
            log["epoch"].append(epoch + 1)
            log["x_recon_loss"].append(avg_recon)
            log["topo_loss"].append(avg_topo)
            log["total_loss"].append(avg_total)
            log["time_per_epoch"].append(dt)
            log.setdefault("n_skipped", []).append(n_skipped)
            log.setdefault("skip_reasons", []).append(skip_reasons)
            print(f"  epoch {epoch+1}/{args.epochs}  "
                  f"recon={avg_recon:.4f}  topo={avg_topo:.6f}  "
                  f"total={avg_total:.4f}  (n={n_valid}"
                  f"{', skipped=' + str(n_skipped) if n_skipped else ''})  "
                  f"{dt:.1f}s", flush=True)
            if (epoch + 1) % args.save_interval == 0 or \
                    epoch == args.epochs - 1:
                ckpt_path = os.path.join(
                    save_dir,
                    f"TORE_finetune{args.save_suffix}_epoch{epoch+1}.pth")
                state = model.module.state_dict() if ddp else \
                    model.state_dict()
                torch.save({'model': state}, ckpt_path)
                print(f"  → saved {ckpt_path}", flush=True)
        if ddp:
            dist.barrier()
    if rank == 0:
        log_path = os.path.join(out_dir, f"train_log{args.log_suffix}.json")
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
        lines = ["=" * 60]
        lines.append("Section 4, Step 2  scHi-C finetune")
        lines.append(f"Mode: {mode}, ddp={ddp}, world={world}")
        lines.append(f"Model: TORE_50kb (50kb pretrained)")
        lines.append(f"Data: {data_desc}")
        lines.append(f"Samples: {n_samples} (max {args.max_per_type}/type)")
        lines.append(f"Epochs: {args.epochs}, LR: {args.lr}")
        lines.append("=" * 60)
        lines.append(f"\nFinal loss: recon={log['x_recon_loss'][-1]:.4f}  "
                     f"topo={log['topo_loss'][-1]:.6f}  "
                     f"total={log['total_loss'][-1]:.4f}")
        lines.append(f"Total time: {sum(log['time_per_epoch']):.0f}s")
        lines.append(f"\nCheckpoints: {save_dir}")
        txt = "\n".join(lines)
        with open(os.path.join(out_dir,
                               f"summary{args.log_suffix}.txt"), "w") as f:
            f.write(txt)
        print(f"\n{txt}")
        print(f"\n[Section4-Step2] Done. → {out_dir}")
    if ddp:
        dist.destroy_process_group()
if __name__ == "__main__":
    main()

import os, sys, time, argparse
import numpy as np
import h5py
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from exp._helpers import load_model
from exp.paper.section4_step4d_imaging_diagnostics import ALL63_ORDER
MODEL = os.path.join(ROOT, "models", "TORE_finetune_1000",
    "TORE_finetune_1000_epoch20.pth")
H5 = os.path.join(ROOT, "exp/results/section4_step1b/genome_imputed_v2.h5")
OUT_DIR = os.path.join(ROOT, "exp/results/section4_step3_eval")
PATCH = 40
def reproduce_split(labels, train_frac=0.65, seed=42):
    rng = np.random.RandomState(seed)
    train_idx = []
    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        rng.shuffle(idx)
        n_train = int(len(idx) * train_frac)
        train_idx.extend(idx[:n_train])
    return np.array(train_idx)
def batch_oe(matrices):
    B, N, _ = matrices.shape
    D = np.abs(np.arange(N)[:, None] - np.arange(N)[None, :])
    expected = np.zeros((B, N))
    for d in range(N):
        if d > 0:
            diag = np.concatenate([
                np.diagonal(matrices, offset=d, axis1=1, axis2=2),
                np.diagonal(matrices, offset=-d, axis1=1, axis2=2)
            ], axis=1)
        else:
            diag = np.diagonal(matrices, offset=0, axis1=1, axis2=2)
        mask = diag > 0
        expected[:, d] = np.where(mask.any(axis=1),
            (diag * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1), 0)
    E = expected[:, D]
    oe = np.where(E > 0, matrices / np.where(E > 0, E, 1), 0)
    return oe.astype(np.float32)
def run_inference(model_path, block_start, block_end, gpu_id=0, out_suffix=""):
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    print(f"=== : blocks [{block_start}:{block_end}) on {device} ===", flush=True)
    with h5py.File(H5, "r") as h5:
        h5_labels = np.array(h5["labels"])
        h5_barcodes = np.array(h5["barcodes"]).astype(str)
    train_idx = reproduce_split(h5_labels)
    train_labels = h5_labels[train_idx]
    train_barcodes = h5_barcodes[train_idx]
    n_cells = len(train_idx)
    n_blocks = block_end - block_start
    print(f"  cells={n_cells}, blocks={n_blocks}, model={os.path.basename(model_path)}", flush=True)
    model, _ = load_model(model_path, device=device, patch_bins=PATCH)
    model = model.to(device)
    model.eval()
    u1_3d = np.zeros((n_blocks, n_cells, PATCH, 3), dtype=np.float32)
    t0 = time.time()
    with h5py.File(H5, "r") as h5:
        for bi, ax in enumerate(range(block_start, block_end)):
            blk = ALL63_ORDER[ax]
            imp = np.array(h5[f"block_{blk}"]["imputed"], dtype=np.float32)
            sub = imp[train_idx]
            del imp
            oe = batch_oe(sub)
            del sub
            with torch.no_grad():
                for i in range(n_cells):
                    x = torch.tensor(oe[i], device=device)
                    _, u1 = model(x)
                    u1_3d[bi, i] = u1.cpu().numpy()
            del oe
            elapsed = time.time() - t0
            eta = elapsed / (bi + 1) * n_blocks - elapsed
            if (bi + 1) % 5 == 0 or bi == 0:
                print(f"  blk {bi+1}/{n_blocks} (ax={ax},g={blk}) "
                      f"{elapsed:.0f}s ETA {eta:.0f}s", flush=True)
    out = os.path.join(OUT_DIR,
        f"latent_alltrain{out_suffix}_b{block_start}-{block_end}.npz")
    np.savez(out, u1_3d=u1_3d, labels=train_labels, barcodes=train_barcodes,
             block_range=(block_start, block_end))
    print(f"done in {time.time()-t0:.0f}s → {out}", flush=True)
def merge(out_suffix=""):
    import glob
    pattern = os.path.join(OUT_DIR,
        f"latent_alltrain{out_suffix}_b*.npz")
    shards = sorted(glob.glob(pattern),
        key=lambda s: int(s.rsplit("_b", 1)[1].split("-")[0]))
    if not shards:
        print(f"ERROR: no shards matching {pattern}")
        return
    print(f"===  {len(shards)}  ===")
    all_u1 = []
    ref = np.load(shards[0])
    prev_end = None
    for s in shards:
        d = np.load(s)
        assert np.array_equal(d["barcodes"], ref["barcodes"]), \
            f"barcodes mismatch in {s}"
        bs, be = d["block_range"]
        if prev_end is not None:
            assert bs == prev_end, \
                f"shard gap/overlap: prev_end={prev_end} vs start={bs} in {s}"
        prev_end = be
        all_u1.append(d["u1_3d"])
    u1 = np.concatenate(all_u1, axis=0)                                          
    n_cells = u1.shape[1]
    u1_flat = u1.transpose(1, 0, 2, 3).reshape(n_cells, -1)                   
    out = os.path.join(OUT_DIR, f"latent_finetune_alltrain{out_suffix}.npz")
    np.savez(out, u1_flat=u1_flat, u1_3d=u1,
             labels=ref["labels"], barcodes=ref["barcodes"])
    print(f"  u1_3d: {u1.shape} → {out}")
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=MODEL,
                    help=" checkpoint ")
    ap.add_argument("--gpu", type=int, default=0,
                    help="GPU  ID")
    ap.add_argument("--block-start", type=int, default=0)
    ap.add_argument("--block-end", type=int, default=63)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--out-suffix", default="",
                    help=" ( _k10g1)")
    a = ap.parse_args()
    if a.merge:
        merge(a.out_suffix)
    else:
        run_inference(a.model_path, a.block_start, a.block_end, a.gpu, a.out_suffix)

# TORE

TORE (Topology-aware Representation Encoder) is a deep representation-learning framework for chromatin organization. It also includes the multimodal TORELink and single-cell superTORE extensions.

The associated manuscript has not yet been published. This repository currently provides research code and selected final model checkpoints.

## Contents

- `model.py` and `src/`: model architectures, losses, training, and inference components.
- `train.py`, `validation.py`, `arguments.py`, and `config.json`: main entry points and configuration.
- `exp/`: task-specific preprocessing, fine-tuning, and predictor code.
- `R/BigWig.R`: ChIP-seq bigWig preprocessing.
- `models/`: TORE, TORELink, and superTORE checkpoints, no-topology controls, and frozen-latent MLP predictors.

## Requirements and use

The Python code requires PyTorch and common scientific packages, including NumPy, SciPy, pandas, GUDHI, cooler, h5py, matplotlib, and psutil. R with `rtracklayer` and `GenomicRanges` is needed only for bigWig preprocessing.

Keep the directory structure unchanged and configure the input-data paths before running training or inference. Raw datasets are not included.

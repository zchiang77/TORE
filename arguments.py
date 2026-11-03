import argparse
def train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', '-G', type=int, default=None,
                        help='Number of GPUs to use (default: all available)')
    parser.add_argument('--port', type=str, default='30000',
                        help='Port to use (default: 30000)')
    parser.add_argument('--threads', type=int, default=None,
                        help='Number of threads to use (default: half of all available)')
    parser.add_argument('--learning_strategy', '-LRS', type=str, default=None,
                        help='Learning rate adjustment strategy. Available options include '
                             '"step" or "warm" (default: "step")')
    parser.add_argument('--learning_rate', '-LR', type=float, default=0.1,
                        help='If learning strategy is set to "step", then this parameter '
                             'is the initial learning rate. If learning strategy is set to '
                             '"warm", then this parameter is the max learning rate. Note '
                             'that the initial learning rate under "warm" setting is '
                             '0.000001. (default: 0.1)')
    parser.add_argument('--optimizer', type=str, default="AdamW",
                        help='SGD, Adam or AdamW (default: "AdamW")')
    parser.add_argument('--reconstruction', action='store_true',
                        help='Enable pretrain mode to pretrain a network for Hi-C reconstructing')
    parser.add_argument('--x_pred_y', action='store_true',
                        help='Enable prediction mode to predict y using x '
                             '(need a pretrained x reconstruction model)')
    parser.add_argument('--y_pred_x', action='store_true',
                        help='Enable prediction mode to predict x using y')
    parser.add_argument('--ring', action='store_true',
                        help='Train full RingModel (both autoencoders + latent alignment + '
                             'cross-modal prediction)')
    parser.add_argument('--y_reconstruction', action='store_true',
                        help='Pretrain mode for Y (ChIP-seq) reconstruction only')
    parser.add_argument('--halfring', action='store_true',
                        help='Train HalfRingModel (x_recon + y_recon + latent alignment, no prediction)')
    parser.add_argument('--use_pretrained', type=str, default=None,
                        help='Use a pretrained network for Hi-C reconstructing. '
                             'This parameter is ignored if --reconstruction is used. (default: None)')
    parser.add_argument('--use_model', type=str, default=None,
                        help='Use an existing model.'
                             'The model to which this parameter is assigned must be consistent with '
                             'the model intended for training. (default: None)')
    parser.add_argument('--use_topology', action='store_true',
                        help='Enable topological loss')
    parser.add_argument('--train_x1', '-TX1', type=str, default='HIC01',
                        help='Data X for training. Separate with spaces. (default: "HIC01")')
    parser.add_argument('--train_x2', '-TX2', type=str, default=None,
                        help='Data X for training. Separate with spaces. This parameter will be '
                             'ignored if --x_pred_y or --y_pred_x is used. (default: None)')
    parser.add_argument('--patch', '-PS', type=float, default=2,
                        help='Patch size (Mb); accepts decimals, e.g. 0.4. (default: 2)')
    parser.add_argument('--patch_overlap', '-PSO', type=float, default=0,
                        help='Overlap between patches (Mb); accepts decimals. (default: 0)')
    parser.add_argument('--x_resolution', '-XR', type=int, default=50000,
                        help='Hi-C (X) bin resolution in bp. '
                             'The model latent layer dimensions are aligned '
                             'with this resolution. (default: 50000)')
    parser.add_argument('--y_resolution', '-YR', type=int, default=None,
                        help='ChIP-seq (Y) bin resolution in bp. '
                             'If not set, defaults to x_resolution. '
                             'Genomic coordinate mapping is used to bridge '
                             'the two resolutions. (default: None)')
    parser.add_argument('--use_bins', '-UB', type=float, default=1,
                        help='How many percent bins use. (default: 1)')
    parser.add_argument('--topo_use_bins', type=float, default=1.0,
                        help='Fraction of Hi-C blocks that contribute to the '
                             'topological loss during TRAINING (0,1]. '
                             '1.0 = all blocks (default, backward-compatible). '
                             'Set <1.0 (e.g. 0.15) under BIG preset to cut the '
                             'O(N^3) persistent-homology cost: only this fraction '
                             'of blocks (selected deterministically by global '
                             'hic_bin, fixed across epochs and DDP ranks) compute '
                             'the topo loss; reconstruction/prediction/sim still '
                             'use ALL blocks. Test-set and standalone validation '
                             'are unaffected and compute topo on all blocks. '
                             '(default: 1.0)')
    parser.add_argument('--hic_normalization', '-HN', type=str, default='oe',
                        choices=['log1p', 'oe', 'none'],
                        help='Hi-C normalization: "log1p" = log(1+x) (recommended), '
                             '"oe" = O/E (Observed/Expected), removes distance decay, '
                             '"none" = raw counts (default: oe)')
    parser.add_argument('--train_y', '-TY', type=str, default=None,
                        help='Data Y for training. Separate with spaces. '
                             'This parameter is ignored if --pretrain is used. '
                             '(default: None)')
    parser.add_argument('--model_save', '-MS', type=str, default='models/',
                        help='Specify a directory for storing the running results (default: "models/").')
    parser.add_argument('--model_freeze', '-MF', type=str, default=None,
                        help='Specify which part of the ring model you want to freeze.')
    parser.add_argument('--hic_base_loss', type=str, default='huber',
                        choices=['mse', 'huber', 'mae'],
                        help='Hi-C base loss: "mse" = MSE (coefficient 100), '
                             '"huber" = Huber loss (coefficient 1, delta from config.json), '
                             '"mae" = MAE / L1 loss (coefficient 100). '
                             '(default: huber)')
    parser.add_argument('--sim_mode', type=str, default='mse',
                        choices=['mse', 'cka'],
                        help='Latent similarity (u1-u2 alignment) loss mode: '
                             '"mse" = MSE (hard pointwise match, original behavior), '
                             '"cka" = 1 - linear CKA (soft statistical alignment, '
                             'preserves non-linear structure). (default: mse)')
    parser.add_argument('--loss_components', type=str, default=None,
                        help='Comma-separated loss components to optimize. '
                             'Available: x_recon_loss, y_pred_loss, x_pred_loss, '
                             'y_recon_loss, sim, topo, x_recon_long_loss, x_pred_long_loss. '
                             'If not set, defaults are inferred from model_type.')
    parser.add_argument('--pretrained_module', type=str, default=None,
                        help='Module name to load pretrained parameters into. '
                             'E.g. "y_prediction", "x_prediction", '
                             '"x_prediction.y_encoder". '
                             'Required when --use_pretrained is set.')
    parser.add_argument('--pretrained_modules', type=str, default=None,
                        help='Comma-separated list of module;path pairs for multi-module '
                             'pretrained loading. E.g. '
                             '"x_reconstruction;path1,y_reconstruction;path2". '
                             'Overrides --use_pretrained if both are set.')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode while using Hi-C to predict Y. It allows you to '
                             'examine the Loss of each feature in Y. However, it will not allow '
                             'early stopping.')
    parser.add_argument('--preset', type=str, default='SMALL',
                        choices=['SMALL', 'BIG', 'SMALL8', 'BIG64', 'SMALL16'],
                        help='Model hyperparameter preset: SMALL, BIG, SMALL8, BIG64 or SMALL16 '
                             '(default: SMALL)')
    parser.add_argument('--lr_search', action='store_true',
                        help='Enable parallel learning-rate search across GPUs before '
                             'formal training. Each GPU runs an independent single-card '
                             'process with a different candidate LR for --lr_search_iters '
                             'iterations; the LR with the largest test-loss reduction is '
                             'selected for formal DDP training. Skipped on Windows (see '
                             '--lr_candidates).')
    parser.add_argument('--lr_candidates', type=str, default=None,
                        help='Comma-separated N candidate learning rates for --lr_search, '
                             'one per GPU. E.g. "0.0001,0.0005,0.001,0.005". '
                             'Required when --lr_search is set. The number of candidates '
                             'must equal --num_gpus.')
    parser.add_argument('--lr_search_iters', type=int, default=200,
                        help='Iterations per candidate in the LR search phase '
                             '(default: 200). Test loss is evaluated every 50 iterations.')
    parser.add_argument('--max_duration_hours', type=float, default=240,
                        help='Max training wall-clock hours. When elapsed, rank 0 saves a '
                             'final checkpoint, triggers external validation (chr22, CPU), '
                             'and all ranks stop. Set <=0 to disable the time limit. '
                             '(default: 240)')
    return parser
def test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=str, default='30000',
                        help='Port to use (default: 30000)')
    parser.add_argument('--threads', type=int, default=None,
                        help='Number of threads to use (default: half of all available)')
    parser.add_argument('--reconstruction', action='store_true',
                        help='Enable pretrain mode to pretrain a network for Hi-C reconstructing')
    parser.add_argument('--x_pred_y', action='store_true',
                        help='Enable prediction mode to predict y using x '
                             '(need a pretrained x reconstruction model)')
    parser.add_argument('--y_pred_x', action='store_true',
                        help='Enable prediction mode to predict x using y')
    parser.add_argument('--ring', action='store_true',
                        help='Train full RingModel (both autoencoders + latent alignment + '
                             'cross-modal prediction)')
    parser.add_argument('--y_reconstruction', action='store_true',
                        help='Validate Y reconstruction model')
    parser.add_argument('--halfring', action='store_true',
                        help='Validate HalfRing model')
    parser.add_argument('--cross_reconstruction', action='store_true',
                        help='Cross-reconstruction in halfring mode: feed u1 into '
                             'y_reconstructor AND u2 into x_reconstructor. '
                             'Only works with --halfring mode.')
    parser.add_argument('--use_model', type=str, default=None,
                        help='Use an existing model. (default: None)')
    parser.add_argument('--use_topology', action='store_true',
                        help='Enable topological loss')
    parser.add_argument('--test_x', '-TX', type=str, default='HIC01',
                        help='Data X for testing. Separate with spaces. (default: "HIC01")')
    parser.add_argument('--test_x_label', '-TXL', type=str, default=None,
                        help='Label for reconstruction. (default: None)')
    parser.add_argument('--patch', '-PS', type=float, default=2,
                        help='Patch size (Mb); accepts decimals, e.g. 0.4. (default: 2)')
    parser.add_argument('--patch_overlap', '-PSO', type=float, default=0,
                        help='Overlap between patches (Mb); accepts decimals. (default: 0)')
    parser.add_argument('--x_resolution', '-XR', type=int, default=50000,
                        help='Hi-C (X) bin resolution in bp. '
                             'The model latent layer dimensions are aligned '
                             'with this resolution. (default: 50000)')
    parser.add_argument('--y_resolution', '-YR', type=int, default=None,
                        help='ChIP-seq (Y) bin resolution in bp. '
                             'If not set, defaults to x_resolution. '
                             'Genomic coordinate mapping is used to bridge '
                             'the two resolutions. (default: None)')
    parser.add_argument('--test_y', '-TY', type=str, default=None,
                        help='Data Y for testing. Separate with spaces. '
                             'This parameter is ignored if --reconstruction is used. '
                             '(default: None)')
    parser.add_argument('--chr', type=str, default="22",
                        help='Specify one chromosome for testing (default: "22").')
    parser.add_argument('--range1', type=str, default="20000000",
                        help='(default: "20000000").')
    parser.add_argument('--range2', type=str, default="24000000",
                        help='(default: "24000000").')
    parser.add_argument('--ranges', type=str, default=None,
                        help='Comma-separated start-end bp ranges for multi-region '
                             'single-block validation, e.g. '
                             '"22000000-26000000,24000000-28000000,26000000-30000000,28000000-32000000". '
                             'Overrides --range1/--range2 when set. Only used when '
                             '--no_whole_chrom is active.')
    parser.add_argument('--model_save', '-MS', type=str, default='test/',
                        help='Specify a directory for storing the results (default: "test/").')
    parser.add_argument('--hic_normalization', '-HN', type=str, default='oe',
                        choices=['log1p', 'oe', 'none'],
                        help='Hi-C normalization: "log1p" = log(1+x) (recommended), '
                             '"oe" = O/E (Observed/Expected), removes distance decay, '
                             '"none" = raw counts (default: oe)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode while using Hi-C to predict Y. '
                             'Outputs per-feature loss breakdown to a debug file.')
    parser.add_argument('--no_whole_chrom', action='store_true', default=False,
                        help='Disable whole-chromosome mode. When set, only the region '
                             'specified by --range1/--range2 is processed. '
                             'By default, the entire chromosome is scanned block-by-block '
                             'and stitched back into a chromosome-level output.')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device to run on. "auto" = GPU if available, else CPU. '
                             '"cpu" = CPU only. "cuda" = GPU only. (default: auto)')
    parser.add_argument('--knockout', '-KO', type=str, default=None,
                        help='Comma-separated protein names to knock out (set to 0). '
                             'E.g. "CTCF" or "CTCF,RAD21". Only effective in '
                             'ring/halfring/y_pred_x/x_pred_y modes. (default: None)')
    parser.add_argument('--preset', type=str, default='SMALL',
                        choices=['SMALL', 'BIG', 'SMALL8', 'BIG64', 'SMALL16'],
                        help='Model hyperparameter preset: SMALL, BIG, SMALL8, BIG64 or SMALL16 '
                             '(default: SMALL)')
    return parser

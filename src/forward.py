from src.data import extract_chip_from_batch
from src.model_builder import build_model
from torch.utils.data import DataLoader
import torch
import os
import logging
logging.basicConfig(filename=os.environ.get('RING_LOG_FILE', 'validation.log'), level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)
def specify_use_model(rank, exist_model_use, model):
    if exist_model_use is not None:
        exist_model_use = os.path.abspath(exist_model_use.replace("\\", "/"))
        if not os.path.exists(exist_model_use):
            raise FileNotFoundError(
                f"[RANK {rank}] Model file not found: {exist_model_use}")
        checkpoint = torch.load(exist_model_use, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get('model', None)
        if state_dict is None:
            raise ValueError(
                f"[RANK {rank}] Checkpoint has no 'model' key: {exist_model_use}")
        model.load_state_dict(state_dict)
        logger.info(f"[RANK {rank}] Model {exist_model_use} loaded")
    return model
def external_validation(rank, model, model_type, dataset, batch_size, chip_meta=None, cross_reconstruction=False):
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    all_x_pred, all_y_pred = [], []
    all_x_recon, all_y_recon = [], []
    all_u1, all_u2, all_y_gt = [], [], []
    all_y_from_u1 = []
    all_x_from_u2 = []
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            if chip_meta is not None:
                x, pos_batch = batch_data
                x = x.to(rank)
                y = extract_chip_from_batch(pos_batch, chip_meta, rank)
                y_gt = y                                       
            else:
                x, y = batch_data
                x, y = x.to(rank), y.to(rank)
                y_gt = None
            y_input = y.unsqueeze(0).permute(0, 2, 1) if y is not None and chip_meta is not None else y
            x_pred = y_pred = x_recon = y_recon = u1 = u2 = y_from_u1 = x_from_u2 = None
            if model_type == 'x_reconstruction':
                x_recon, u1 = model(x)
            elif model_type == 'y_prediction':
                y_pred, x_recon, u1 = model(x)
            elif model_type == 'x_prediction':
                x_pred, y_recon, u2 = model(y_input)
            elif model_type == 'y_reconstruction':
                y_recon, u2 = model(y_input)
            elif model_type == 'halfring':
                x_recon, y_recon, u1, u2 = model(x, y_input)
                if cross_reconstruction:
                    y_from_u1 = model.y_reconstruction.y_reconstructor(u1)
                    x_from_u2 = model.x_reconstruction.x_reconstructor(u2)
            elif model_type == 'ring':
                x_pred, y_pred, x_recon, y_recon, u1, u2 = model(x, y_input)
            else:
                raise ValueError(f"[TEST] Model type '{model_type}' is not supported")
            if x_pred is not None:
                all_x_pred.append(x_pred.cpu())
            if y_pred is not None:
                all_y_pred.append(y_pred.cpu())
            if x_recon is not None:
                all_x_recon.append(x_recon.cpu())
            if y_recon is not None:
                all_y_recon.append(y_recon.cpu())
            if u1 is not None:
                all_u1.append(u1.cpu())
            if u2 is not None:
                all_u2.append(u2.cpu())
            if y_gt is not None:
                all_y_gt.append(y_gt.cpu())
            if y_from_u1 is not None:
                all_y_from_u1.append(y_from_u1.cpu())
            if x_from_u2 is not None:
                all_x_from_u2.append(x_from_u2.cpu())
    def _cat(tensor_list):
        return torch.cat(tensor_list, dim=0) if tensor_list else None
    return (_cat(all_x_pred), _cat(all_y_pred),
            _cat(all_x_recon), _cat(all_y_recon),
            _cat(all_u1), _cat(all_u2), _cat(all_y_gt),
            _cat(all_y_from_u1) if cross_reconstruction else None,
            _cat(all_x_from_u2) if cross_reconstruction else None)
def run_forward(rank, model_type, config, dataset, batch_size,
                x_p=0, y_p=0, load_path=None, chip_meta=None, preset="SMALL", cross_reconstruction=False):
    y_length = None
    if chip_meta is not None:
        y_length = chip_meta['patch_bins'] * chip_meta['x_resolution'] // chip_meta['y_resolution']
    model = build_model(model_type, config, x_p, y_p, rank, y_length=y_length, preset=preset)
    model = specify_use_model(rank, load_path, model)
    (x_pred, y_pred, x_recon, y_recon, u1, u2, y_gt, y_from_u1, x_from_u2) = \
        external_validation(rank, model, model_type,
                            dataset, batch_size, chip_meta, cross_reconstruction=cross_reconstruction)
    return x_pred, y_pred, x_recon, y_recon, u1, u2, y_gt, y_from_u1, x_from_u2

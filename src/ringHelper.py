import torch
import json
import sys
import torch.distributed  as dist
import numpy as np
import psutil
import pandas as pd
import datetime
from torch.utils.data import Sampler
class BlockDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas, rank, block_size, drop_last=True):
        if not dist.is_available():
            raise RuntimeError("Requires torch.distributed package")
        self.rank = rank
        self.num_replicas = num_replicas
        self.block_size = block_size
        n_items = len(dataset)
        n_blocks = n_items // block_size
        if drop_last:
            n_blocks = n_blocks // num_replicas * num_replicas
        my_blocks = list(range(rank, n_blocks, num_replicas))
        self.indices = []
        for b in my_blocks:
            start = b * block_size
            self.indices.extend(range(start, start + block_size))
    def __iter__(self):
        return iter(self.indices)
    def __len__(self):
        return len(self.indices)
def setup(rank, ws):
    if sys.platform.startswith('win'):
        dist.init_process_group("gloo", rank=rank, world_size=ws)                
    else:
        dist.init_process_group("nccl", rank=rank, world_size=ws)
def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config
def grouped_mean(x, n_group):
    grouped = x.view(int(x.shape[0]), n_group, int(x.shape[1] / n_group))
    mask = ~torch.isnan(grouped)            
    sum_val = torch.where(mask, grouped, torch.zeros_like(grouped)).sum(dim=2)
    count_val = mask.sum(dim=2)
    result = torch.where(
        count_val > 0,             
        sum_val / count_val,           
        torch.zeros_like(sum_val)            
    )
    return result.float()
def format_params(num_params):
    if num_params > 1e9:
        return f"{num_params/1e9:.2f}B"
    return f"{num_params/1e6:.2f}M"
def log_reduced_losses(loss_dict):
    rank = dist.get_rank()
    with torch.no_grad():
        reduced_dict = {}
        for name, value in loss_dict.items():
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value, device=torch.cuda.current_device(),
                                     requires_grad=False)
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
            reduced_value = value
            reduced_dict[name] = reduced_value.item()
    if rank == 0:
        total_loss = reduced_dict.get('total',
                                      sum(v for k, v in reduced_dict.items() if k != 'total'))
        components = ' + '.join(f"{name}={val:.6f}"
                                for name, val in reduced_dict.items() if name != 'total')
        return (f"{total_loss:.6f} = {components}", total_loss, reduced_dict.get('topo'),
                reduced_dict.get('mse'), reduced_dict.get('pcc'))
    return "", 0, 0, 0, 0
def log_reduced_losses_ring(loss_dict, world_size):
    rank = dist.get_rank()
    with torch.no_grad():
        reduced_dict = {}
        for name, value in loss_dict.items():
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value, device=torch.cuda.current_device(),
                                     requires_grad=False)
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
            reduced_value = value / world_size
            reduced_dict[name] = reduced_value.item()
    if rank == 0:
        return reduced_dict
    return 0
def generate_position_encoding(rank, pos, d_model):
    n = pos.shape[0]
    PE = torch.zeros([n, d_model])
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    div_term = torch.tensor(div_term, dtype=torch.float32).to(rank)
    for i in range(n):
        p = pos[i]
        PE[i, 0::2] = torch.sin(p * div_term)          
        PE[i, 1::2] = torch.cos(p * div_term)          
    return PE
def monitor_resources():
    mem = psutil.virtual_memory()
    mem_used = (mem.total  - mem.available)  / 1024**2
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        alloc = torch.cuda.memory_allocated(device)  / 1024**2          
        max_alloc = torch.cuda.max_memory_allocated(device)  / 1024**2        
        torch.cuda.reset_peak_memory_stats(device)           
    else:
        alloc = max_alloc = 0
    return f"Memory Usage {mem_used:.0f}MB, GPU Usage {alloc:.0f}MB, GPU Peak {max_alloc:.0f}MB"
def get_tensor_memory(tensor):
    num_elements = tensor.nelement()
    bytes_per_element = tensor.element_size()
    total_bytes = num_elements * bytes_per_element
    return total_bytes / (1024**2)
def count_zeros(tensor):
    zero_mask = (tensor == 0)
    num_zeros = zero_mask.sum().item()
    total_elements = tensor.numel()
    zero_ratio = num_zeros / total_elements
    return num_zeros, zero_ratio
def summarize_parameters_size(rank, model, logger):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        logger.info(f"[RANK {rank}] Model parameters: {format_params(total_params)}")
def summarize_parameters_size_all_ranks(rank, model, logger):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[RANK {rank}] Model parameters: {format_params(total_params)}")
def symmetry_correction(x):
    if x.shape[0] != x.shape[1]:
        raise ValueError("Error in SymmetryCorrection: Input matrix must be a square matrix.")
    sym_matrix = (x + x.T) / 2.0
    return sym_matrix
class ReversibleZScoreScaler:
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
        self.mean = None
        self.std = None
        self.feature_names = None
        self.is_fitted = False
        self.zero_var_mask = None            
    def fit(self, x, feature_names=None):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be torch.Tensor")
        if x.dim() != 2:
            raise ValueError("x must be 2D (samples x features)")
        self.mean = torch.mean(x, dim=0, keepdim=True)
        self.std = torch.std(x, dim=0, keepdim=True, unbiased=False)
        self.zero_var_mask = self.std < self.epsilon
        self.std = torch.where(self.zero_var_mask, torch.ones_like(self.std), self.std)
        if self.zero_var_mask.any():
            zero_idx = torch.where(self.zero_var_mask.squeeze())[0].tolist()
            print(f":  {len(zero_idx)} : {zero_idx}")
        self.feature_names = feature_names
        self.is_fitted = True
        return self
    def transform(self, x):
        if not self.is_fitted:
            raise RuntimeError("Call fit first")
        if x.dim() != 2 or x.size(1) != self.mean.size(1):
            raise ValueError("Input shape mismatch")
        device = x.device
        mean = self.mean.to(device)
        std = self.std.to(device)
        with torch.no_grad():
            return (x - mean) / std            
    def inverse_transform(self, x_norm):
        if not self.is_fitted:
            raise RuntimeError("Call fit first")
        if x_norm.dim() != 2 or x_norm.size(1) != self.mean.size(1):
            raise ValueError("Input shape mismatch")
        device = x_norm.device
        mean = self.mean.to(device)
        std = self.std.to(device)
        with torch.no_grad():
            result = x_norm * std + mean
        if self.zero_var_mask is not None:
            zero_mask = self.zero_var_mask.to(device)
            result = torch.where(zero_mask, mean, result)
        return result
    def save_state(self, path):
        state = {
            'mean': self.mean.cpu() if self.mean is not None else None,
            'std': self.std.cpu() if self.std is not None else None,
            'feature_names': self.feature_names,
            'epsilon': self.epsilon,
            'zero_var_mask': self.zero_var_mask.cpu() if self.zero_var_mask is not None else None,
            'is_fitted': self.is_fitted
        }
        torch.save(state, path)
    @classmethod
    def load_state(cls, path, map_location=None):
        state = torch.load(path, map_location=map_location)
        scaler = cls(epsilon=state.get('epsilon', 1e-8))
        scaler.mean = state['mean']
        scaler.std = state['std']
        scaler.feature_names = state.get('feature_names')
        scaler.zero_var_mask = state.get('zero_var_mask')
        scaler.is_fitted = state.get('is_fitted', True)
        return scaler
class Log1pMinMaxScaler:
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
        self.max_log = None                            
        self.feature_names = None
        self.is_fitted = False
    def fit(self, x, feature_names=None):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be torch.Tensor")
        if x.dim() != 2:
            raise ValueError("x must be 2D (samples x features)")
        x_log = torch.log1p(x)
        self.max_log = x_log.max(dim=0, keepdim=True).values                    
        zero_mask = self.max_log < self.epsilon
        if zero_mask.any():
            self.max_log = torch.where(zero_mask, torch.ones_like(self.max_log), self.max_log)
            zero_idx = torch.where(zero_mask.squeeze())[0].tolist()
            print(f":  {len(zero_idx)} max_log=01: {zero_idx}")
        self.feature_names = feature_names
        self.is_fitted = True
        return self
    def transform(self, x):
        if not self.is_fitted:
            raise RuntimeError("Call fit first")
        if x.dim() != 2 or x.size(1) != self.max_log.size(1):
            raise ValueError("Input shape mismatch")
        device = x.device
        max_log = self.max_log.to(device)
        with torch.no_grad():
            x_log = torch.log1p(x)
            return x_log / max_log
    def inverse_transform(self, x_norm):
        if not self.is_fitted:
            raise RuntimeError("Call fit first")
        if x_norm.dim() != 2 or x_norm.size(1) != self.max_log.size(1):
            raise ValueError("Input shape mismatch")
        device = x_norm.device
        max_log = self.max_log.to(device)
        with torch.no_grad():
            x_log = x_norm * max_log
            return torch.expm1(x_log)                   
    def save_state(self, path):
        state = {
            'max_log': self.max_log.cpu() if self.max_log is not None else None,
            'feature_names': self.feature_names,
            'epsilon': self.epsilon,
            'is_fitted': self.is_fitted
        }
        torch.save(state, path)
    @classmethod
    def load_state(cls, path, map_location=None):
        state = torch.load(path, map_location=map_location)
        scaler = cls(epsilon=state.get('epsilon', 1e-8))
        scaler.max_log = state['max_log']
        scaler.feature_names = state.get('feature_names')
        scaler.is_fitted = state.get('is_fitted', True)
        return scaler
def knockout_protein(chip_data_norm, protein_indices, scaler=None):
    result = chip_data_norm.clone()
    if isinstance(protein_indices, int):
        protein_indices = [protein_indices]
    if scaler is not None:
        result_raw = scaler.inverse_transform(result)
        for idx in protein_indices:
            result_raw[:, idx] = 0.0
        result = scaler.transform(result_raw)
    else:
        for idx in protein_indices:
            result[:, idx] = 0.0
    return result
def tensor2dataframe(t, col_names, save_name=None):
    df = pd.DataFrame(
        t.numpy(),
        columns=col_names
    )
    if save_name is not None:
        df.to_csv(save_name, index=False, float_format='%.6f')
    return df
def compute_complete_time(rank, seconds, logger):
    current_time = datetime.datetime.now()
    delta = datetime.timedelta(seconds=seconds)
    completion_time = current_time + delta
    formatted_time = completion_time.strftime("%Y-%m-%d %H:%M")
    logger.info(f"[RANK {rank}] Training will be completed in {formatted_time}")
def save_tensor_txt(tensor, path):
    np_array = tensor.numpy()
    np.savetxt(path, np_array, delimiter='\t', fmt='%s')
def compute_gradient_cosine_similarity(model, loss_recon, loss_pred):
    shared_params = []
    for param in model.parameters():
        if param.requires_grad:
            shared_params.append(param)
    if not shared_params:
        print("")
        return None, None
    grads_recon = torch.autograd.grad(
        loss_recon, shared_params, retain_graph=True,                              
        allow_unused=True
    )
    grads_pred = torch.autograd.grad(
        loss_pred, shared_params, retain_graph=True,                    
        allow_unused=True
    )
    grad_recon_list = []
    grad_pred_list = []
    for gr, gp in zip(grads_recon, grads_pred):
        if gr is not None and gp is not None:
            grad_recon_list.append(gr.flatten())
            grad_pred_list.append(gp.flatten())
    if not grad_recon_list:
        print("")
        return None, None
    grad_recon = torch.cat(grad_recon_list)
    grad_pred = torch.cat(grad_pred_list)
    cos_sim = torch.dot(grad_recon, grad_pred) / (grad_recon.norm() * grad_pred.norm() + 1e-8)
    conflict_mask = (grad_recon * grad_pred) < 0
    conflict_ratio = conflict_mask.float().mean()
    return cos_sim.item(), conflict_ratio.item()
def analyze_gradients_by_module(model):
    print(f"{'Module':<30} {'Params':>8} {'WithGrad':>8} {'MeanGradNorm':>12}")
    print("-" * 65)
    for name, module in model.named_modules():
        total_params = 0
        with_grad = 0
        total_grad_norm = 0.0
        for p_name, p in module.named_parameters(recurse=False):
            total_params += p.numel()
            if p.grad is not None:
                with_grad += p.numel()
                total_grad_norm += p.grad.norm().item()
        if total_params > 0:
            mean_norm = total_grad_norm / with_grad if with_grad > 0 else 0.0
            flag = "⚠️  NO GRAD" if with_grad == 0 else ""
            print(f"{name:<30} {total_params:>8} {with_grad:>8} {mean_norm:>12.6f}  {flag}")

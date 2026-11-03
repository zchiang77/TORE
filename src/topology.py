import torch
import numpy as np
import threading
from dataclasses import dataclass
from scipy.stats  import wasserstein_distance
import gudhi as gd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.spatial.distance import cdist
import logging
try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except ImportError:
    go = None
    _HAS_PLOTLY = False
logger = logging.getLogger(__name__)
@dataclass
class TopoConfig:
    threshold: float = np.inf                                     
    h0_weight: float = 0.1                                
    smooth_weight: float = 0.1                        
    gamma: float = 3.0                                   
    topo_use_bins: float = 1.0
    degenerate_bandwidth: int = 5                          
    degenerate_threshold: float = 0.05                        
def get_topo_config(train_cfg=None):
    if train_cfg is None:
        return TopoConfig()
    thr_raw = train_cfg.get("topology_threshold", None)
    thr = np.inf if thr_raw is None else float(thr_raw)
    return TopoConfig(
        threshold=thr,
        h0_weight=train_cfg.get("topo_h0_weight", 0.1),
        smooth_weight=train_cfg.get("topo_smooth_weight", 0.1),
        gamma=train_cfg.get("gamma", 3.0),
        degenerate_bandwidth=train_cfg.get("topo_degenerate_bandwidth", 5),
        degenerate_threshold=train_cfg.get("topo_degenerate_threshold", 0.05),
    )
def gpu_distance_matrix(points):
    points_gpu = torch.tensor(points, device='cuda')
    dist_matrix = torch.cdist(points_gpu, points_gpu).cpu().numpy()
    return dist_matrix
def cpu_distance_matrix_parallel(points, num_threads=8):
    n = points.shape[0]
    dist_matrix = np.zeros((n, n))
    chunk_size = n // num_threads
    def compute_chunk(start, end):
        for i in range(start, end):
            for j in range(i + 1, n):
                dist = np.linalg.norm(points[i] - points[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
    threads = []
    for i in range(num_threads):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_threads - 1 else n
        thread = threading.Thread(target=compute_chunk, args=(start, end))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    return dist_matrix
def compute_persistence_diagram(point_cloud, device='gpu', threads=8):
    if device != 'cpu' and torch.cuda.is_available():
        dist_matrix = gpu_distance_matrix(point_cloud)
    else:
        dist_matrix = cpu_distance_matrix_parallel(point_cloud, num_threads=threads)
    rips_complex = gd.RipsComplex(distance_matrix=dist_matrix, max_edge_length=1e10)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    diag = simplex_tree.persistence()
    pers_diag = []
    for interval in diag:
        if interval[0] == 1:
            birth, death = interval[1]
            if death == np.inf:
                death = np.max(point_cloud) * 2
            pers_diag.append([birth, death])
    return np.array(pers_diag)
def normalize_distance_matrix(D, eps=1e-8):
    mask = D > 0                 
    if mask.any():
        d_min = D[mask].min()
        d_max = D[mask].max()
        D_norm = (D - d_min) / (d_max - d_min + eps)
    else:
        D_norm = D.clone()
    D_norm.fill_diagonal_(0.0)
    return D_norm
def is_degenerate_distance_matrix(D, bandwidth=5, rel_range_threshold=0.05, eps=1e-8):
    if hasattr(D, "detach"):
        D_np = D.detach().cpu().numpy()
    else:
        D_np = np.asarray(D)
    N = D_np.shape[0]
    bw = min(int(bandwidth), N - 1)
    if bw < 1:
        return True, 0.0
    mask = np.zeros((N, N), dtype=bool)
    for d in range(1, bw + 1):
        mask |= np.eye(N, k=d, dtype=bool)
        mask |= np.eye(N, k=-d, dtype=bool)
    band = D_np[mask]
    if band.size == 0:
        return True, 0.0
    b_max = float(band.max())
    b_min = float(band.min())
    if b_max < eps:
        return True, 0.0
    rel_range = (b_max - b_min) / (b_max + eps)
    return bool(rel_range < rel_range_threshold), float(rel_range)
class HomologyLoss(torch.nn.Module):
    def __init__(self, max_dim=2, h0_weight=0.1, smooth_weight=0.1, eps=1e-8):
        super(HomologyLoss, self).__init__()
        self.max_dim = max_dim
        self.h0_weight = h0_weight
        self.smooth_weight = smooth_weight
        self.eps = eps
    def persistence_computation(self, points, threshold):
        points = points.detach().cpu().numpy()
        rips_complex = gd.RipsComplex(points=points, max_edge_length=threshold)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=self.max_dim)
        diag = simplex_tree.persistence()
        pairs = simplex_tree.persistence_pairs()
        return diag, pairs
    def extract_h1_generator(self, pairs):
        boundary_edges = []
        for birth_simplex, death_simplex in pairs:
            if len(birth_simplex) == 2:
                edge = tuple(sorted(birth_simplex))
                boundary_edges.append(edge)
        return boundary_edges
    def extract_h1_with_persistence(self, points, threshold):
        points_np = points.detach().cpu().numpy()
        dist_matrix = cdist(points_np, points_np)
        rips_complex = gd.RipsComplex(points=points_np,
                                      max_edge_length=threshold)
        simplex_tree = rips_complex.create_simplex_tree(
            max_dimension=self.max_dim)
        simplex_tree.compute_persistence()
        pairs = simplex_tree.persistence_pairs()
        h1_info = {}                                
        for birth_simplex, death_simplex in pairs:
            if len(birth_simplex) != 2:
                continue
            if len(death_simplex) != 3:
                continue                                                 
            i, j = birth_simplex
            birth = dist_matrix[i, j]
            a, b, c = death_simplex
            death = max(dist_matrix[a, b],
                        dist_matrix[a, c],
                        dist_matrix[b, c])
            edge = tuple(sorted(birth_simplex))
            if edge in h1_info:
                old_pers = h1_info[edge][2] - h1_info[edge][1]
                new_pers = death - birth
                if new_pers > old_pers:
                    h1_info[edge] = (edge, birth, death)
            else:
                h1_info[edge] = (edge, birth, death)
        return list(h1_info.values())
    def persistence_from_dist_matrix(self, dist_matrix, threshold):
        rips_complex = gd.RipsComplex(distance_matrix=dist_matrix,
                                       max_edge_length=threshold)
        simplex_tree = rips_complex.create_simplex_tree(
            max_dimension=self.max_dim)
        diag = simplex_tree.persistence()
        pairs = simplex_tree.persistence_pairs()
        return diag, pairs
    def extract_h1_from_dist_matrix(self, dist_matrix, threshold):
        rips_complex = gd.RipsComplex(distance_matrix=dist_matrix,
                                       max_edge_length=threshold)
        simplex_tree = rips_complex.create_simplex_tree(
            max_dimension=self.max_dim)
        simplex_tree.compute_persistence()
        pairs = simplex_tree.persistence_pairs()
        h1_info = {}
        for birth_simplex, death_simplex in pairs:
            if len(birth_simplex) != 2:
                continue
            if len(death_simplex) != 3:
                continue
            i, j = birth_simplex
            birth = dist_matrix[i, j]
            a, b, c = death_simplex
            death = max(dist_matrix[a, b],
                        dist_matrix[a, c],
                        dist_matrix[b, c])
            edge = tuple(sorted(birth_simplex))
            if edge in h1_info:
                old_pers = h1_info[edge][2] - h1_info[edge][1]
                new_pers = death - birth
                if new_pers > old_pers:
                    h1_info[edge] = (edge, birth, death)
            else:
                h1_info[edge] = (edge, birth, death)
        return list(h1_info.values())
    def _extract_homology_from_dist(self, dist_matrix, threshold):
        rips_complex = gd.RipsComplex(distance_matrix=dist_matrix,
                                       max_edge_length=threshold)
        simplex_tree = rips_complex.create_simplex_tree(
            max_dimension=self.max_dim)
        simplex_tree.compute_persistence()
        pairs = simplex_tree.persistence_pairs()
        h0_info = []                                               
        h1_info = {}                                                
        h1_death_simplex = {}                                  
        for birth_simplex, death_simplex in pairs:
            if len(birth_simplex) == 1 and len(death_simplex) == 2:
                i, j = death_simplex
                edge = tuple(sorted((i, j)))
                death = float(dist_matrix[i, j])
                birth = 0.0                          
                h0_info.append((edge, birth, death))
            elif len(birth_simplex) == 2 and len(death_simplex) == 3:
                i, j = birth_simplex
                birth = float(dist_matrix[i, j])
                a, b, c = death_simplex
                death = max(float(dist_matrix[a, b]),
                            float(dist_matrix[a, c]),
                            float(dist_matrix[b, c]))
                edge = tuple(sorted(birth_simplex))
                if edge in h1_info:
                    old_pers = h1_info[edge][2] - h1_info[edge][1]
                    new_pers = death - birth
                    if new_pers > old_pers:
                        h1_info[edge] = (edge, birth, death)
                        h1_death_simplex[edge] = tuple(sorted((a, b, c)))
                else:
                    h1_info[edge] = (edge, birth, death)
                    h1_death_simplex[edge] = tuple(sorted((a, b, c)))
        return h0_info, list(h1_info.values()), h1_death_simplex
    def extract_homology_from_points(self, points, threshold):
        points_np = points.detach().cpu().numpy()
        dist_matrix = cdist(points_np, points_np)
        return self._extract_homology_from_dist(dist_matrix, threshold)
    def extract_homology_from_dist_matrix(self, dist_matrix, threshold):
        return self._extract_homology_from_dist(dist_matrix, threshold)
    def _edge_loss(self, D_x_norm, z_dists_norm,
                   x_edges, z_edges, x_weights, z_weights, device):
        if len(x_edges) == 0:
            LXZ = torch.tensor(0.0, device=device)
        else:
            dists_x = torch.stack([D_x_norm[i, j] for (i, j) in x_edges])
            dists_z = torch.stack([z_dists_norm[i, j] for (i, j) in x_edges])
            sq_diff = (dists_x - dists_z) ** 2
            if x_weights is not None:
                w = torch.tensor(x_weights, device=device, dtype=D_x_norm.dtype)
                LXZ = 0.5 * torch.mean(w * sq_diff)
            else:
                LXZ = 0.5 * torch.mean(sq_diff)
        if len(z_edges) == 0:
            LZX = torch.tensor(0.0, device=device)
        else:
            dists_x = torch.stack([D_x_norm[i, j] for (i, j) in z_edges])
            dists_z = torch.stack([z_dists_norm[i, j] for (i, j) in z_edges])
            sq_diff = (dists_x - dists_z) ** 2
            if z_weights is not None:
                w = torch.tensor(z_weights, device=device, dtype=D_x_norm.dtype)
                LZX = 0.5 * torch.mean(w * sq_diff)
            else:
                LZX = 0.5 * torch.mean(sq_diff)
        return LXZ + LZX
    def forward(self, D_x, z, threshold):
        device = D_x.device
        z_dists = torch.cdist(z, z)
        D_x_for_loss = normalize_distance_matrix(D_x)
        z_dists_for_loss = normalize_distance_matrix(z_dists)
        z_is_3d = (z.dim() == 2 and z.shape[1] == 3)
        if z_is_3d:
            logger.debug("HomologyLoss: z is 3D, using Rips(points=z)")
            z_h0_info, z_h1_info, z_h1_death = \
                self.extract_homology_from_points(z, threshold)
        else:
            logger.debug(f"HomologyLoss: z is {z.shape[1]}D, "
                         f"using Rips(distance_matrix=)")
            z_dists_np = z_dists.detach().cpu().numpy()
            z_h0_info, z_h1_info, z_h1_death = \
                self.extract_homology_from_dist_matrix(z_dists_np, threshold)
        D_x_np = D_x.detach().cpu().numpy()
        x_h0_info, x_h1_info, x_h1_death = \
            self.extract_homology_from_dist_matrix(D_x_np, threshold)
        x_h1_edges = [info[0] for info in x_h1_info]
        z_h1_edges = [info[0] for info in z_h1_info]
        x_h0_edges = [info[0] for info in x_h0_info]
        z_h0_edges = [info[0] for info in z_h0_info]
        x_h1_w = z_h1_w = x_h0_w = z_h0_w = None
        H1_loss = self._edge_loss(D_x_for_loss, z_dists_for_loss,
                                   x_h1_edges, z_h1_edges,
                                   x_h1_w, z_h1_w, device)
        x_death_edges = self._death_edges_from_simplex(x_h1_death, x_h1_edges)
        z_death_edges = self._death_edges_from_simplex(z_h1_death, z_h1_edges)
        H1_death_loss = self._edge_loss(D_x_for_loss, z_dists_for_loss,
                                        x_death_edges, z_death_edges,
                                        None, None, device)
        H0_loss = self._edge_loss(D_x_for_loss, z_dists_for_loss,
                                   x_h0_edges, z_h0_edges,
                                   x_h0_w, z_h0_w, device)
        homology_part = H1_loss + H1_death_loss + self.h0_weight * H0_loss
        smooth_part = distance_matrix_smooth_loss(D_x_for_loss, z_dists_for_loss,
                                                   eps=self.eps)
        return homology_part + self.smooth_weight * smooth_part
    @staticmethod
    def _death_edges_from_simplex(h1_death_simplex, birth_edges):
        birth_set = set(birth_edges)
        seen = set()
        death_edges = []
        for (a, b, c) in h1_death_simplex.values():
            for (i, j) in ((a, b), (a, c), (b, c)):
                edge = tuple(sorted((i, j)))
                if edge in birth_set or edge in seen:
                    continue
                seen.add(edge)
                death_edges.append(edge)
        return death_edges
def distance_matrix_loss(D_x, z, reduction='mean', use_triu=True):
    D_z = torch.cdist(z, z)
    D_x_norm = normalize_distance_matrix(D_x)
    D_z_norm = normalize_distance_matrix(D_z)
    diff_matrix = (D_x_norm - D_z_norm) ** 2
    if use_triu:
        mask = torch.triu(torch.ones_like(diff_matrix), diagonal=1).bool()
        diff_matrix = diff_matrix[mask]
    if reduction == 'mean':
        return diff_matrix.mean()
    elif reduction == 'sum':
        return diff_matrix.sum()
    return diff_matrix
def distance_matrix_smooth_loss(D_x_norm, z_dists_norm, eps=1e-8):
    mask = torch.triu(torch.ones_like(D_x_norm), diagonal=1).bool()
    dx = D_x_norm[mask]
    dz = z_dists_norm[mask]
    dx_c = dx - dx.mean()
    dz_c = dz - dz.mean()
    denom = torch.sqrt((dx_c * dx_c).sum() * (dz_c * dz_c).sum() + eps)
    r = (dx_c * dz_c).sum() / denom
    pearson_loss = 1.0 - r
    mse_loss = ((dx - dz) ** 2).mean()
    return pearson_loss + 0.1 * mse_loss
def count_h1_from_dist_matrix(dist_matrix, threshold):
    rips_complex = gd.RipsComplex(distance_matrix=dist_matrix,
                                   max_edge_length=threshold)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    simplex_tree.compute_persistence()
    pairs = simplex_tree.persistence_pairs()
    h1_count = sum(1 for b, d in pairs if len(b) == 2 and len(d) == 3)
    return h1_count
def count_h0_h1_from_dist_matrix(dist_matrix, threshold):
    rips_complex = gd.RipsComplex(distance_matrix=dist_matrix,
                                   max_edge_length=threshold)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    simplex_tree.compute_persistence()
    pairs = simplex_tree.persistence_pairs()
    h0_count = sum(1 for b, d in pairs if len(b) == 1 and len(d) == 2)
    h1_count = sum(1 for b, d in pairs if len(b) == 2 and len(d) == 3)
    return h0_count, h1_count
def _persistence_from_dist_matrix_helper(dist_matrix, threshold):
    rips_complex = gd.RipsComplex(distance_matrix=dist_matrix,
                                   max_edge_length=threshold)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    diag = simplex_tree.persistence()
    pairs = simplex_tree.persistence_pairs()
    return diag, pairs
def plot_persistence_diagram(D_x, z, output_filename, topo_cfg=None,
                             axis_multiplier=1.5):
    if topo_cfg is None:
        topo_cfg = TopoConfig()
    threshold = topo_cfg.threshold
    x_diag, _ = _persistence_from_dist_matrix_helper(D_x, threshold)
    z_dist = cdist(z, z)
    z_diag, _ = _persistence_from_dist_matrix_helper(z_dist, threshold)
    def distance_variance(A, B):
        diff = (A - B) ** 2
        n = diff.shape[0]
        mask = np.triu_indices(n, k=1)
        return np.mean(diff[mask])
    D_x_norm = D_x.copy()
    mask_x = D_x_norm > 0
    if mask_x.any():
        d_min, d_max = D_x_norm[mask_x].min(), D_x_norm[mask_x].max()
        if d_max > d_min:
            D_x_norm = (D_x_norm - d_min) / (d_max - d_min)
    z_dist_norm = z_dist.copy()
    mask_z = z_dist_norm > 0
    if mask_z.any():
        d_min, d_max = z_dist_norm[mask_z].min(), z_dist_norm[mask_z].max()
        if d_max > d_min:
            z_dist_norm = (z_dist_norm - d_min) / (d_max - d_min)
    dist_loss = distance_variance(D_x_norm, z_dist_norm)
    def choose_axis_limit(persistence):
        pairs = [p[1][1] for p in persistence if p[0] == 1]
        return max(pairs) if len(pairs) > 0 else 0
    plt_limit = choose_axis_limit(x_diag)
    def plot_diagram(ax, persistence, title, max_finite, loss_txt):
        dims = {}
        for dim in range(2):
            dims[dim] = [p[1] for p in persistence if p[0] == dim]
        colors = {0: 'red', 1: 'blue'}
        markers = {0: 'o', 1: 's'}
        labels = {0: 'H0 (Components)', 1: 'H1 (Loops)'}
        for dim in range(2):
            pairs = dims[dim]
            births = [b for b, d in pairs]
            deaths = [d if d != float('inf') else max_finite * 1.2 for b, d in pairs]
            ax.scatter(
                births, deaths,
                c=colors[dim], marker=markers[dim],
                s=50, alpha=0.7, label=labels[dim]
            )
        max_val = max_finite * axis_multiplier
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
        ax.plot([0, max_val], [max_finite * 1.2, max_finite * 1.2], 'k:', alpha=0.3)
        ax.set_xlabel('Birth Time', fontsize=12)
        ax.set_ylabel('Death Time', fontsize=12)
        ax.set_title(f'{title}\n(DistLoss={loss_txt:.4f})', fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(alpha=0.2)
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val * 1.1)
    plt.figure(figsize=(16, 8))
    gs = GridSpec(1, 2, width_ratios=[1, 1])
    ax1 = plt.subplot(gs[0])
    plot_diagram(ax1, x_diag, f"True (spatial distance)", plt_limit, dist_loss)
    ax2 = plt.subplot(gs[1])
    plot_diagram(ax2, z_diag, f"Embedding", plt_limit, dist_loss)
    plt.suptitle('Persistence Diagrams Comparison', fontsize=18, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
def plot_latent_space(z, output_filename, dim=None, title_prefix="Latent space"):
    z = np.asarray(z)
    if z.ndim != 2:
        logger.warning(f"plot_latent_space: expected 2D array, got shape {z.shape}; skip.")
        return
    d = dim if dim is not None else z.shape[1]
    if d not in (2, 3):
        logger.debug(f"plot_latent_space: dim={d} > 3, skip plotting.")
        return
    N = z.shape[0]
    colors = np.arange(N)
    title = f'{title_prefix} latent space (d={d}, N={N})'
    if _HAS_PLOTLY:
        if output_filename.lower().endswith('.png'):
            html_path = output_filename[:-4] + '.html'
        else:
            html_path = output_filename
        if d == 2:
            fig = go.Figure(go.Scatter(
                x=z[:, 0], y=z[:, 1],
                mode='markers',
                marker=dict(
                    size=5,
                    color=colors,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Genomic bin index'),
                ),
            ))
            fig.update_layout(
                title=title,
                xaxis_title='Latent dim 1',
                yaxis_title='Latent dim 2',
                template='plotly_white',
            )
        else:          
            fig = go.Figure(go.Scatter3d(
                x=z[:, 0], y=z[:, 1], z=z[:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=colors,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Genomic bin index'),
                    opacity=0.8,
                ),
            ))
            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title='Latent dim 1',
                    yaxis_title='Latent dim 2',
                    zaxis_title='Latent dim 3',
                ),
                template='plotly_white',
            )
        fig.write_html(html_path, include_plotlyjs='cdn')
        logger.info(f"plot_latent_space: interactive HTML saved to {html_path} "
                    f"(dim={d})")
        return
    logger.warning("plotly not installed; falling back to static PNG. "
                   "Install via `pip install plotly` for interactive HTML.")
    if d == 2:
        fig, ax = plt.subplots(figsize=(8, 7))
        sc = ax.scatter(z[:, 0], z[:, 1], c=colors, cmap='viridis',
                        s=30, alpha=0.8, edgecolors='none')
        ax.set_xlabel('Latent dim 1', fontsize=12)
        ax.set_ylabel('Latent dim 2', fontsize=12)
    else:          
        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter3D(z[:, 0], z[:, 1], z[:, 2], c=colors,
                          cmap='viridis', s=30, alpha=0.8, edgecolors='none')
        ax.set_xlabel('Latent dim 1', fontsize=11)
        ax.set_ylabel('Latent dim 2', fontsize=11)
        ax.set_zlabel('Latent dim 3', fontsize=11)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.1)
    cbar.set_label('Genomic bin index', fontsize=11)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
def compute_topology_diagnostics(D_x, z, topo_cfg=None, output_prefix=None,
                                 noise_alpha=0.1, logger=None):
    if topo_cfg is None:
        topo_cfg = TopoConfig()
    threshold = topo_cfg.threshold
    def _diag_and_h1(D):
        rips = gd.RipsComplex(distance_matrix=D, max_edge_length=threshold)
        st = rips.create_simplex_tree(max_dimension=2)
        st.compute_persistence()
        diag = st.persistence()
        pairs = st.persistence_pairs()
        h0, h1 = [], []
        for b_s, d_s in pairs:
            if len(b_s) == 1 and len(d_s) == 2:
                i, j = d_s
                h0.append((tuple(sorted((i, j))), 0.0, float(D[i, j])))
            elif len(b_s) == 2 and len(d_s) == 3:
                i, j = b_s
                birth = float(D[i, j])
                a, b, c = d_s
                death = max(float(D[a, b]), float(D[a, c]), float(D[b, c]))
                h1.append((tuple(sorted((i, j))), birth, death))
        return diag, h0, h1
    x_diag, x_h0, x_h1 = _diag_and_h1(np.asarray(D_x))
    z_dist = cdist(np.asarray(z), np.asarray(z))
    z_diag, z_h0, z_h1 = _diag_and_h1(z_dist)
    def _summarize(h1):
        if not h1:
            return {"n": 0, "min": 0.0, "mean": 0.0, "median": 0.0, "max": 0.0,
                    "std": 0.0, "n_significant": 0, "n_noise": 0}
        pers = np.array([d - b for _, b, d in h1])
        mx = float(pers.max())
        floor = max(noise_alpha * mx, 1e-9)
        return {
            "n": int(len(pers)),
            "min": float(pers.min()), "mean": float(pers.mean()),
            "median": float(np.median(pers)), "max": mx,
            "std": float(pers.std()),
            "n_significant": int((pers > floor).sum()),
            "n_noise": int((pers <= floor).sum()),
        }
    x_s = _summarize(x_h1)
    z_s = _summarize(z_h1)
    def _h1_pairs(diag):
        return [(b, d) for dim, (b, d) in diag if dim == 1 and d != float('inf')]
    try:
        bd = float(gd.bottleneck_distance(_h1_pairs(x_diag), _h1_pairs(z_diag)))
    except Exception as e:
        bd = float('nan')
        if logger is not None:
            logger.warning(f"[TOPO DIAG] bottleneck_distance failed: {e}")
    summary = {
        "x_h0": len(x_h0), "x_h1": x_s,
        "z_h0": len(z_h0), "z_h1": z_s,
        "bottleneck_H1": bd,
    }
    if output_prefix:
        with open(f"{output_prefix}_topology-diagnostics.txt", "w") as f:
            f.write(f"threshold={threshold}\n")
            f.write(f"x side: H0={summary['x_h0']}, H1={x_s['n']} "
                    f"(significant={x_s['n_significant']}, noise={x_s['n_noise']})\n")
            f.write(f"  H1 persistence: min={x_s['min']:.4g} mean={x_s['mean']:.4g} "
                    f"median={x_s['median']:.4g} max={x_s['max']:.4g} std={x_s['std']:.4g}\n")
            f.write(f"z side: H0={summary['z_h0']}, H1={z_s['n']} "
                    f"(significant={z_s['n_significant']}, noise={z_s['n_noise']})\n")
            f.write(f"  H1 persistence: min={z_s['min']:.4g} mean={z_s['mean']:.4g} "
                    f"median={z_s['median']:.4g} max={z_s['max']:.4g} std={z_s['std']:.4g}\n")
            f.write(f"H1 bottleneck distance (x vs z) = {bd:.4g}\n")
            f.write(f"noise criterion: persistence <= {noise_alpha} * max_persistence\n")
        mx_x = x_s['max'] if x_s['max'] > 0 else 1.0
        mx_z = z_s['max'] if z_s['max'] > 0 else 1.0
        with open(f"{output_prefix}_h1-generators.txt", "w") as f:
            f.write("side\tdim\tbirth\tdeath\tpersistence\tvertex_i\tvertex_j\tis_significant\n")
            for (edge, b, d) in x_h1:
                p = d - b
                f.write(f"x\t1\t{b:.6g}\t{d:.6g}\t{p:.6g}\t"
                        f"{edge[0]}\t{edge[1]}\t{int(p > noise_alpha * mx_x)}\n")
            for (edge, b, d) in z_h1:
                p = d - b
                f.write(f"z\t1\t{b:.6g}\t{d:.6g}\t{p:.6g}\t"
                        f"{edge[0]}\t{edge[1]}\t{int(p > noise_alpha * mx_z)}\n")
        _plot_persistence_barcode(x_diag, z_diag,
                                  f"{output_prefix}_persistence-barcode.png")
    if logger is not None:
        logger.info(f"[TOPO DIAG] x: H1={x_s['n']} (sig={x_s['n_significant']}, "
                    f"noise={x_s['n_noise']}); z: H1={z_s['n']} "
                    f"(sig={z_s['n_significant']}, noise={z_s['n_noise']}); "
                    f"bottleneck={bd:.4g}")
    return summary
def _plot_persistence_barcode(x_diag, z_diag, filename):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, diag, title in [(axes[0], x_diag, "True (D_x)"),
                            (axes[1], z_diag, "Embedding (z)")]:
        h0 = sorted([(b, d) for dim, (b, d) in diag
                     if dim == 0 and d != float('inf')],
                    key=lambda p: p[1] - p[0])
        h1 = sorted([(b, d) for dim, (b, d) in diag
                     if dim == 1 and d != float('inf')],
                    key=lambda p: p[1] - p[0])
        y = 0
        for b, d in h0:
            ax.hlines(y, b, d, colors='red', linewidth=1.5)
            y += 1
        for b, d in h1:
            ax.hlines(y, b, d, colors='blue', linewidth=1.5)
            y += 1
        ax.set_title(f"{title}\nH0(red)={len(h0)} H1(blue)={len(h1)}")
        ax.set_xlabel("filtration value")
        ax.set_ylabel("feature (sorted by persistence)")
        ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()

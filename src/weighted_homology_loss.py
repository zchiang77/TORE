import torch
import numpy as np
import gudhi as gd
from scipy.spatial.distance import cdist
class PersistenceWeightedHomologyLoss(torch.nn.Module):
    def __init__(self, max_dim=2, persistence_power=1.0, birth_decay=1.0, eps=1e-8):
        super(PersistenceWeightedHomologyLoss, self).__init__()
        self.max_dim = max_dim
        self.persistence_power = persistence_power
        self.birth_decay = birth_decay
        self.eps = eps
    def extract_h1_with_persistence(self, points, threshold):
        points_np = points.detach().cpu().numpy()
        dist_matrix = cdist(points_np, points_np)
        rips_complex = gd.RipsComplex(points=points_np, max_edge_length=threshold)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=self.max_dim)
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
            death = max(dist_matrix[a, b], dist_matrix[a, c], dist_matrix[b, c])
            edge = tuple(sorted(birth_simplex))
            if edge in h1_info:
                old_persistence = h1_info[edge][2] - h1_info[edge][1]
                new_persistence = death - birth
                if new_persistence > old_persistence:
                    h1_info[edge] = (edge, birth, death)
            else:
                h1_info[edge] = (edge, birth, death)
        return list(h1_info.values())
    def compute_weights(self, h1_info):
        if len(h1_info) == 0:
            return []
        births = np.array([b for _, b, _ in h1_info], dtype=np.float64)
        deaths = np.array([d for _, _, d in h1_info], dtype=np.float64)
        persistences = deaths - births
        b_max = births.max() if births.max() > self.eps else 1.0
        w_pers = np.power(persistences + self.eps, self.persistence_power)
        w_birth = np.exp(-self.birth_decay * births / b_max)
        weights = w_pers * w_birth
        w_mean = weights.mean()
        if w_mean > self.eps:
            weights = weights / w_mean
        return weights.tolist()
    def forward(self, x, z, threshold):
        x_dists = torch.cdist(x, x)
        z_dists = torch.cdist(z, z)
        x_h1_info = self.extract_h1_with_persistence(x, threshold)
        z_h1_info = self.extract_h1_with_persistence(z, threshold)
        x_weights = self.compute_weights(x_h1_info)
        z_weights = self.compute_weights(z_h1_info)
        x_edges = [info[0] for info in x_h1_info]
        z_edges = [info[0] for info in z_h1_info]
        if len(x_edges) == 0:
            LXZ = torch.tensor(0.0, device=x.device)
        else:
            dists_x_on_x = torch.stack([x_dists[i, j] for (i, j) in x_edges])
            dists_z_on_x = torch.stack([z_dists[i, j] for (i, j) in x_edges])
            x_w = torch.tensor(x_weights, device=x.device, dtype=x.dtype)
            LXZ = 0.5 * torch.sum(x_w * (dists_x_on_x - dists_z_on_x) ** 2)
        if len(z_edges) == 0:
            LZX = torch.tensor(0.0, device=x.device)
        else:
            dists_x_on_z = torch.stack([x_dists[i, j] for (i, j) in z_edges])
            dists_z_on_z = torch.stack([z_dists[i, j] for (i, j) in z_edges])
            z_w = torch.tensor(z_weights, device=z.device, dtype=z.dtype)
            LZX = 0.5 * torch.sum(z_w * (dists_x_on_z - dists_z_on_z) ** 2)
        return LXZ + LZX
def compute_weighted_topological_loss(x, u1, u2, topo_thresh,
                                       persistence_power=1.0, birth_decay=1.0):
    topo_loss_fn = PersistenceWeightedHomologyLoss(
        persistence_power=persistence_power,
        birth_decay=birth_decay
    )
    topo_x_latent = topo_loss_fn(x, u1, topo_thresh)
    if u2 is not None:
        topo_sim = topo_loss_fn(u1, u2, topo_thresh)
    else:
        topo_sim = 0
    topo_total = topo_sim + topo_x_latent
    return topo_total

import math
import glob
import re
import os
import random
from src.ringHelper import *
def generate_position_matrix(N):
    vector = torch.arange(1, N + 1)
    return vector
def bins_to_coords(cooler, bin):
    chrom = str(cooler.bins()[bin]['chrom'].iloc[0])
    start = int(cooler.bins()[bin]['start'].iloc[0])
    end = int(cooler.bins()[bin]['end'].iloc[0])
    return [chrom, start, end]
def extract_diagonal_blocks(cooler_mat, cooler_end, window_size,
                            overlap=0, sample_size=1, random_seed=42,
                            bins_table=None, zero_threshold=0.9):
    random.seed(random_seed)
    N = cooler_end
    step = window_size - overlap
    blocks = []
    skipped = 0
    for start_idx in range(0, N - window_size + 1, step):
        if bins_table is not None:
            start_chrom = str(bins_table.iloc[start_idx]['chrom'])
            end_chrom = str(bins_table.iloc[start_idx + window_size - 1]['chrom'])
            if start_chrom != end_chrom:
                skipped += 1
                continue
        sub_matrix = cooler_mat[start_idx:start_idx + window_size, start_idx:start_idx + window_size]
        sub_matrix[np.isnan(sub_matrix)] = 0
        if np.mean(sub_matrix == 0) > zero_threshold:
            skipped += 1
            continue
        position = (start_idx, start_idx)
        blocks.append((sub_matrix, position))
    if skipped > 0:
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"[DATA] Skipped {skipped} blocks (cross-chromosome or near-zero)")
    num_samples = int(len(blocks) * sample_size)
    sampled_blocks = random.sample(blocks, num_samples) if num_samples < len(blocks) else blocks
    return sampled_blocks
def split_blocks_to_datasets(blocks, proc, train_ratio=0.9, random_seed=42):
    np.random.seed(random_seed)
    n_blocks = len(blocks)
    indices = np.random.permutation(n_blocks)
    train_size = int(n_blocks * train_ratio)
    remainder = train_size % proc
    if remainder != 0:
        train_size -= remainder
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    train_blocks = []
    train_positions = []
    for idx in train_indices:
        sub_mat, pos = blocks[idx]
        train_blocks.append(sub_mat)
        train_positions.append(pos)
    test_blocks = []
    test_positions = []
    for idx in test_indices:
        sub_mat, pos = blocks[idx]
        test_blocks.append(sub_mat)
        test_positions.append(pos)
    return train_blocks, train_positions, test_blocks, test_positions
def extract_blocks_by_indices(cooler_mat, indices, window_size):
    blocks = []
    for idxs in indices:
        idx = np.array(idxs[0])
        if idx + window_size > cooler_mat.shape[0]:
            raise ValueError(f":  {idx} ")
        sub_matrix = cooler_mat[idx:idx + window_size, idx:idx + window_size]
        sub_matrix[np.isnan(sub_matrix)] = 0
        blocks.append(sub_matrix)
    return torch.tensor(np.array(blocks), dtype=torch.float32)
def coord_to_bin(df, length, resolution, bin_start, bin_end):
    coord_start_list = list(range(0, length, resolution))
    coord_end_list = list(range(resolution, length, resolution)) + [length]
    bins_list = list(range(bin_start, bin_end))
    bins_df = pd.DataFrame({"bins": bins_list, "start": coord_start_list, "end": coord_end_list})
    start_to_bin = bins_df.set_index('start')['bins'].astype(int).to_dict()
    df['bin1'] = df['x1'].map(start_to_bin)
    df['bin2'] = df['y1'].map(start_to_bin)
    df['length'] = (df['bin2'] - df['bin1']).abs()
    return df
def transform_interaction_data(df, bin_start, bin_end, N=3, seed=123):
    if seed is not None:
        np.random.seed(seed)
    full_bins = pd.Series(
        np.arange(bin_start, bin_end, dtype=np.int32),
        name='bin1'
    )
    filtered = df.copy()                   
    filtered.sort_values(['bin1', 'bin2'], inplace=True)
    filtered['shifted'] = filtered.groupby('bin1')['bin2'].shift(1)
    deduced = filtered[
        (filtered['bin2'] != filtered['shifted']) |
        filtered['shifted'].isna()
        ].drop(columns='shifted')
    grouped = (
        deduced.groupby('bin1')['bin2']
        .agg(lambda x: list(x))
        .reindex(full_bins, fill_value=[])
        .reset_index()
    )
    def select_random_n(row, n=N):
        bin1_val = row['bin1']
        bin2_list = row['bin2']
        relative_bins = [bin2 - bin1_val for bin2 in bin2_list]
        if len(relative_bins) <= n:
            return relative_bins + [np.nan] * (n - len(relative_bins))
        else:
            selected = np.random.choice(relative_bins, size=n, replace=False).tolist()
            selected.sort()
            return selected
    padded = grouped.apply(select_random_n, axis=1)
    result_df = pd.DataFrame(
        padded.tolist(),
        index=grouped['bin1'],
        columns=[f'interact_bin{i + 1}' for i in range(N)]
    )
    return result_df
def find_thresh(clr, n_dist=1, perc=75):
    matrix_region = clr.matrix(balance=True, as_pixels=True)[:, :]
    pixels = matrix_region[:]
    adjacent_mask = (pixels['bin2_id'] - pixels['bin1_id'] == n_dist)
    adjacent_pixels = pixels[adjacent_mask]['balanced'].values
    adjacent_pixels_dists = np.reciprocal(adjacent_pixels)
    return np.nanpercentile(adjacent_pixels_dists, perc)
def compute_spatial_distance_matrix(contact_matrix, gamma=3.0, eps=1e-6):
    D = np.power(contact_matrix.astype(np.float64) + eps, -1.0 / gamma)
    np.fill_diagonal(D, 0.0)
    return D.astype(np.float32)
def compute_oe_matrix(matrix):
    N = matrix.shape[0]
    oe = np.zeros_like(matrix, dtype=np.float64)
    for d in range(N):
        diag_vals = np.diagonal(matrix, offset=d)
        if d > 0:
            diag_vals_full = np.concatenate([diag_vals, np.diagonal(matrix, offset=-d)])
        else:
            diag_vals_full = diag_vals
        nonzero = diag_vals_full[diag_vals_full > 0]
        if len(nonzero) > 0:
            expected = np.mean(nonzero)
        else:
            expected = 0.0
        if expected > 0:
            rows = np.arange(N - d)
            oe[rows, rows + d] = matrix[rows, rows + d] / expected
            if d > 0:
                oe[rows + d, rows] = matrix[rows + d, rows] / expected
    return oe
def inverse_oe_matrix(oe_matrix, raw_reference):
    N = oe_matrix.shape[0]
    raw = np.zeros_like(oe_matrix, dtype=np.float64)
    for d in range(N):
        diag_vals = np.diagonal(raw_reference, offset=d)
        if d > 0:
            diag_vals_full = np.concatenate([diag_vals, np.diagonal(raw_reference, offset=-d)])
        else:
            diag_vals_full = diag_vals
        nonzero = diag_vals_full[diag_vals_full > 0]
        if len(nonzero) > 0:
            expected = np.mean(nonzero)
        else:
            expected = 0.0
        if expected > 0:
            rows = np.arange(N - d)
            raw[rows, rows + d] = oe_matrix[rows, rows + d] * expected
            if d > 0:
                raw[rows + d, rows] = oe_matrix[rows + d, rows] * expected
    return raw
def training_hic_data(proc, cooler_file, patch_bins, overlap_bins, use_bins=1,
                      normalization="log1p", zero_threshold=0.9, gamma=3.0,
                      topology_threshold=None):
    cooler_mat = cooler_file.matrix()
    cooler_end = cooler_file.extent('21')[1]
    if topology_threshold is None:
        topology_threshold = np.inf
    bins_table = cooler_file.bins()[:]
    data_blocks = extract_diagonal_blocks(cooler_mat, cooler_end, patch_bins, overlap_bins,
                                          bins_table=bins_table, sample_size=use_bins,
                                          zero_threshold=zero_threshold)
    train_blocks, train_positions, test_blocks, test_positions = (
        split_blocks_to_datasets(data_blocks, proc))
    train_blocks = np.array(train_blocks)
    test_blocks = np.array(test_blocks)
    train_dist = np.stack([
        compute_spatial_distance_matrix(train_blocks[i], gamma=gamma)
        for i in range(train_blocks.shape[0])
    ])
    test_dist = np.stack([
        compute_spatial_distance_matrix(test_blocks[i], gamma=gamma)
        for i in range(test_blocks.shape[0])
    ])
    if normalization == "log1p":
        train_blocks = np.log1p(train_blocks)
        test_blocks = np.log1p(test_blocks)
        if topology_threshold != np.inf:
            topology_threshold = np.log1p(topology_threshold)
    elif normalization == "oe":
        train_blocks_oe = np.zeros_like(train_blocks, dtype=np.float32)
        for i in range(train_blocks.shape[0]):
            train_blocks_oe[i] = compute_oe_matrix(train_blocks[i]).astype(np.float32)
        train_blocks = train_blocks_oe
        test_blocks_oe = np.zeros_like(test_blocks, dtype=np.float32)
        for i in range(test_blocks.shape[0]):
            test_blocks_oe[i] = compute_oe_matrix(test_blocks[i]).astype(np.float32)
        test_blocks = test_blocks_oe
    elif normalization == "none":
        if topology_threshold is None or topology_threshold == np.inf:
            topology_threshold = 10.0
    else:
        raise ValueError(f"Unknown normalization: '{normalization}'. "
                         f"Use 'log1p', 'oe', or 'none'.")
    return (torch.tensor(train_blocks, dtype=torch.float32),
            torch.tensor(test_blocks, dtype=torch.float32),
            torch.tensor(np.array(train_positions), dtype=torch.int32),
            torch.tensor(np.array(test_positions), dtype=torch.int32),
            topology_threshold,
            torch.tensor(train_dist, dtype=torch.float32),
            torch.tensor(test_dist, dtype=torch.float32))
def _resolve_chip_block(hic_bin, chip_meta):
    bins_table = chip_meta['bins_table']
    chrom_offsets = chip_meta['chrom_offsets']
    chrom_nbins = chip_meta['chrom_nbins']
    patch_bins_hic = chip_meta['patch_bins']
    y_res = chip_meta['y_resolution']
    start_chrom = str(bins_table.iloc[hic_bin]['chrom'])
    g_start = int(bins_table.iloc[hic_bin]['start'])
    g_end = int(bins_table.iloc[hic_bin + patch_bins_hic - 1]['end'])
    if start_chrom not in chrom_offsets:
        if 'chr' + start_chrom in chrom_offsets:
            start_chrom = 'chr' + start_chrom
        elif start_chrom.startswith('chr') and start_chrom[3:] in chrom_offsets:
            start_chrom = start_chrom[3:]
    if start_chrom not in chrom_offsets:
        return None
    chip_bin_start = g_start // y_res
    chip_bin_end = (g_end - 1) // y_res + 1
    max_bin = chrom_nbins.get(start_chrom, chip_bin_end)
    chip_bin_end = min(chip_bin_end, max_bin)
    row_start = chrom_offsets[start_chrom] + chip_bin_start
    row_end = chrom_offsets[start_chrom] + chip_bin_end
    return start_chrom, row_start, row_end
def training_chip_data(path, return_col=False, return_scaler=False):
    file_list = sorted(
        glob.glob(path + "ChIP.*.csv"),
        key=lambda x: int(re.search(r'ChIP\.(\d+)\.csv', x).group(1))
    )
    processed_dfs = []
    for f in file_list:
        df = pd.read_csv(f, index_col=False)
        extracted_df = df.iloc[:, 2:]
        extracted_df = extracted_df.reset_index(drop=True)
        processed_dfs.append(extracted_df)
    chip_data_result = pd.concat(processed_dfs, axis=1, ignore_index=False)
    chip_data_columns = chip_data_result.columns.tolist()
    chip_data_raw = torch.tensor(chip_data_result.values, dtype=torch.float32)
    chip_scaler = Log1pMinMaxScaler()
    chip_scaler.fit(chip_data_raw)
    chip_data_norm = chip_scaler.transform(chip_data_raw)
    if return_col and return_scaler:
        return chip_data_norm, chip_data_columns, chip_scaler
    if return_col:
        return chip_data_norm, chip_data_columns
    if return_scaler:
        return chip_data_norm, chip_scaler
    return chip_data_norm
def build_chip_chrom_offsets(chip_file_path):
    file_list = sorted(
        glob.glob(chip_file_path + "ChIP.*.csv"),
        key=lambda x: int(re.search(r'ChIP\.(\d+)\.csv', x).group(1))
    )
    first_file = file_list[0]
    import pandas as pd
    sample_df = pd.read_csv(first_file, index_col=False, nrows=1)
    chrom_candidates = [col for col in sample_df.columns
                        if col.lower() in ('chrom', 'chr')]
    if len(chrom_candidates) == 0:
        raise KeyError(
            f"No column named 'chrom' or 'chr' found in {first_file}. "
            f"Available columns: {list(sample_df.columns)}"
        )
    chrom_col = chrom_candidates[0]
    chrom_series = pd.read_csv(
        first_file, index_col=False, usecols=[chrom_col]
    ).iloc[:, 0].astype(str)
    chrom_offsets = {}
    chrom_nbins = {}
    current_row = 0
    for chrom, group in chrom_series.groupby(chrom_series, sort=False):
        n_bins = len(group)
        chrom_offsets[chrom] = current_row
        chrom_nbins[chrom] = n_bins
        current_row += n_bins
    return chrom_offsets, chrom_nbins, current_row
def extract_chip_from_batch(pos_batch, chip_meta, device):
    entire_chip = chip_meta['entire_chip'].to(device)
    chip_rows = []
    pos_cache = {}                                           
    for pos in pos_batch:
        hic_bin = int(pos[0].item()) if torch.is_tensor(pos) else int(pos[0])
        key = hic_bin
        if key not in pos_cache:
            block = _resolve_chip_block(hic_bin, chip_meta)
            if block is None:
                fallback_rows = (
                    next(iter(pos_cache.values())).shape[0]
                    if pos_cache else
                    chip_meta['patch_bins'] * chip_meta['x_resolution']
                    // chip_meta['y_resolution']
                )
                zero_block = torch.zeros(
                    (fallback_rows, entire_chip.shape[1]),
                    dtype=entire_chip.dtype, device=device
                )
                pos_cache[key] = zero_block
                chip_rows.append(zero_block)
                continue
            _, row_start, row_end = block
            pos_cache[key] = entire_chip[row_start:row_end]
            chip_rows.append(pos_cache[key])
    return torch.cat(chip_rows, dim=0)
def inverse_transform_chip_batch(y_norm, pos_batch, chip_meta, device):
    scaler = chip_meta['chip_scaler']
    return scaler.inverse_transform(y_norm.to(device))
def extract_chip_data(entire_chip, indices, batch_size):
    start_indices = [idx[0] for idx in indices]
    shuffled_labels_list = []
    for start in start_indices:
        batch = entire_chip[start:start + batch_size]
        shuffled_labels_list.append(batch)
    shuffled_labels = torch.cat(shuffled_labels_list, dim=0)
    return shuffled_labels
def get_chip_labels(chip_file_path):
    file_list = sorted(
        glob.glob(chip_file_path + "ChIP.*.csv"),
        key=lambda x: int(re.search(r'ChIP\.(\d+)\.csv', x).group(1))
    )
    all_labels = []
    for f in file_list:
        df = pd.read_csv(f, index_col=False, nrows=1)
        feature_cols = df.columns[2:].tolist()
        all_labels.extend(feature_cols)
    return all_labels

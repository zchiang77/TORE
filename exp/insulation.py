import numpy as np
import matplotlib.pyplot as plt
parent_path = ["../test/05072/Validation_Experiment=HIC01_Topological_true-x",
               "../test/05072/Validation_Experiment=HIC01_Topological_reconstructed-x"]
save_plot_path = "../test/05072/Validation_Experiment=HIC01_Topological_IS-comparison.png"
matrix_path = [p + ".txt" for p in parent_path]
save_path = [p + "_IS.txt" for p in parent_path]
def generate_plot(scores, filename):
    plt.figure(figsize=(12, 8))
    plt.plot(list(range(0, 400)), scores,
             linestyle='-', color='#d62728', linewidth=2)
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    return 0
def generate_plot_comparison(score1, score2, filename):
    plt.figure(figsize=(12, 8))
    plt.plot(list(range(0, 400)), score1,
             linestyle='-', color='#1f77b4', linewidth=2,
             label='Score1')
    plt.plot(list(range(0, 400)), score2,
             linestyle='-', color='#d62728', linewidth=2,
             label='Score2')
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    return 0
def insulation_score_numpy(mat, window=10, ignore_diags=2, norm_by_median=True):
    N = mat.shape[0]
    sum_counts = np.zeros(N)
    n_pixels = np.zeros(N, dtype=np.int64)
    max_d = (window - 1) * 2                                        
    for d in range(ignore_diags, max_d + 1):
        if d >= N:                               
            continue
        diag = np.diagonal(mat, offset=d)           
        i_start = max(0, d - (window - 1))
        i_end = min(d, window - 1)
        for i_shift in range(i_start, i_end + 1):
            target = np.arange(len(diag)) + i_shift
            sum_counts[target] += diag
            n_pixels[target] += 1
    with np.errstate(divide='ignore', invalid='ignore'):           
        score = sum_counts.astype(float) / n_pixels
    if norm_by_median:
        score = score / np.nanmedian(score)
    ins_track = np.log2(score)
    ins_track[~np.isfinite(ins_track)] = np.nan
    return ins_track, n_pixels, sum_counts
def load_matrix_from_txt(filepath, delimiter=None, skip_comments=False, dtype=float):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    if skip_comments:
        lines = [line for line in lines if not line.strip().startswith('#')]
    if delimiter is None:
        import re
        parts = re.split(r'[,\s]+', lines[0].strip())
        num_columns = len([p for p in parts if p != ''])
        if ',' in lines[0]:
            delimiter = ','
        elif '\t' in lines[0]:
            delimiter = '\t'
        else:
            delimiter = None
    data = []
    for line in lines:
        if delimiter:
            values = line.strip().split(delimiter)
        else:
            values = line.split()
        data.append([dtype(v) for v in values if v != ''])
    mat = np.array(data, dtype=dtype)
    return mat
if __name__ == '__main__':
    if len(parent_path) == 1:
        mat = load_matrix_from_txt(matrix_path)
        ins_obj = insulation_score_numpy(mat, window=10, ignore_diags=2)
        ins_scores = ins_obj[0]
        generate_plot(ins_scores, save_plot_path)
        np.savetxt(save_path, ins_scores, fmt='%.6f')
    elif len(parent_path) == 2:
        mat1 = load_matrix_from_txt(matrix_path[0])
        mat2 = load_matrix_from_txt(matrix_path[1])
        ins_obj_1 = insulation_score_numpy(mat1, window=10, ignore_diags=2)
        ins_scores_1 = ins_obj_1[0]
        ins_obj_2 = insulation_score_numpy(mat2, window=10, ignore_diags=2)
        ins_scores_2 = ins_obj_2[0]
        generate_plot_comparison(ins_scores_1, ins_scores_2, save_plot_path)
        np.savetxt(save_path[0], ins_scores_1, fmt='%.6f')
        np.savetxt(save_path[1], ins_scores_2, fmt='%.6f')
        print(f"Difference: {np.mean((ins_scores_1 - ins_scores_2) ** 2)}")
    else:
        raise NotImplementedError

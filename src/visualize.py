import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import os
import pandas as pd
from src.lossCompute import pearson_corr_np
def save_training_batch_for_visualize(input_data, trained_data, save_path):
    with torch.no_grad():
        input_data_save = input_data.cpu().numpy()
        input_data_path = save_path + '_input.txt'
        trained_data_save = trained_data.cpu().numpy()
        trained_data_path = save_path + '_trained.txt'
        np.savetxt(input_data_path, input_data_save, fmt='%.4f', delimiter=',')
        np.savetxt(trained_data_path, trained_data_save, fmt='%.4f', delimiter=',')
    return 0
def save_testing_for_visualize(save_data, save_path):
    with torch.no_grad():
        data = save_data.cpu().numpy()
        path = save_path + '.txt'
        np.savetxt(path, data, fmt='%.4f', delimiter=',')
    return 0
def log_custom(mat, base=10):
    mat = np.array(mat, dtype=float)
    mat[mat <= 0] = np.nan
    return np.log10(mat) if base == 10 else np.log(mat)/np.log(base)
def plot_custom(ax, mat, title, minval, maxval):
    cmap = LinearSegmentedColormap.from_list('blue_red', ['darkblue', 'white', 'darkred'])
    extent = [-0.5, mat.shape[1] - 0.5, -0.5, mat.shape[0] - 0.5]
    im = ax.imshow(mat, cmap=cmap, vmin=minval, vmax=maxval,
                   origin='upper', aspect='equal', interpolation='none',
                   extent=extent)
    ax.set_title(title, fontsize=11, pad=8)
    ax.tick_params(axis='both', which='both', length=2, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)
    return im
def plot_custom_chip(ax, mat, title, minval, maxval, xtick_labels, ytick_labels=None):
    cmap = LinearSegmentedColormap.from_list('blue_red', ['blue', 'white', 'darkred'])
    im = ax.imshow(mat, cmap=cmap, vmin=minval, vmax=maxval,
                   origin='upper', aspect='auto',
                   interpolation='none')
    ax.set_title(title, fontsize=8, pad=4)
    ax.set_xticks(np.arange(len(xtick_labels)))
    ax.set_xticklabels(xtick_labels,
                       rotation=90,
                       ha='right',
                       rotation_mode='anchor',
                       fontsize=5)
    if ytick_labels is not None:
        ax.set_yticks(np.arange(len(ytick_labels)))
        ax.set_yticklabels(ytick_labels, fontsize=8)
    ax.tick_params(axis='both', which='both',
                   length=2,
                   bottom=True, top=False,
                   labelbottom=True)
    ax.set_xlim(-0.5, mat.shape[1] - 0.5)
    ax.set_ylim(-0.5, mat.shape[0] - 0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)
    return im
def model_testing_plot(x, y, range_1, range_2,
                       caption_model, caption_data,
                       save_name, log_op=True):
    plot_range = slice(range_1, range_2)
    x_sub = x[plot_range, plot_range]
    y_sub = y[plot_range, plot_range]
    if log_op:
        x_log = log_custom(x_sub, 10)
        y_log = log_custom(y_sub, 10)
    else:
        x_log = x_sub
        y_log = y_sub
    combined = np.full_like(x_log, np.nan)
    n = x_log.shape[0]
    for i in range(n):
        for j in range(i, n):
            combined[i, j] = x_log[i, j]
    for i in range(1, n):
        for j in range(i):
            combined[i, j] = y_log[i, j]
    min_val = np.nanmin(y_log)
    max_val = np.nanmax(y_log)
    base_inches = 6.0
    scale = min(1.0, 400.0 / max(n, 1))
    heatmap_inches = max(base_inches, n * 0.025 * scale)
    fig_w = heatmap_inches + 0.8                       
    fig_h = heatmap_inches + 1.0                              
    plt.figure(figsize=(fig_w, fig_h), dpi=150, facecolor='white')
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 0.08],
                           width_ratios=[1, 0.035],
                           hspace=0.05, wspace=0.05)
    ax = plt.subplot(gs[0, 0])
    im = plot_custom(ax, combined, "True / Reconstructed", min_val, max_val)
    label_fs = max(6, min(10, 500 / max(n, 1)))
    ax.text(0.95, 0.95, "True",
            transform=ax.transAxes, fontsize=label_fs, fontweight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(0.05, 0.05, "Reconstruct",
            transform=ax.transAxes, fontsize=label_fs, fontweight='bold',
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    cax = plt.subplot(gs[0, 1])
    cbar = plt.colorbar(im, cax=cax)
    cbar_label = 'log10(value)' if log_op else 'value'
    cbar.set_label(cbar_label, fontsize=7, labelpad=4)
    cbar.ax.tick_params(labelsize=6)
    ax_caption = plt.subplot(gs[1, :])
    ax_caption.axis('off')
    caption_mse = np.mean((x - y) ** 2)
    caption_pcc = pearson_corr_np(x, y)
    caption_text = (
        f"Model: {caption_model}  |  "
        f"Data: {caption_data}  |  "
        f"MSE: {caption_mse:.4e}  |  "
        f"PCC: {caption_pcc:.4f}  |  "
        f"{n}x{n}"
    )
    ax_caption.text(0.5, 0.5, caption_text,
                    ha='center', va='center',
                    fontsize=6, color='#555555',
                    linespacing=1.2,
                    bbox=dict(boxstyle='round,pad=0.4',
                              facecolor='#f8f8f8',
                              edgecolor='#e0e0e0',
                              alpha=0.85))
    plt.suptitle("Hi-C Reconstruction", fontsize=10, fontweight='bold', y=0.99)
    plt.savefig(save_name, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
def model_testing_plot_chip(x, y, range_1, range_2, xticks,
                            caption_model, caption_data,
                            save_name, log_op=True,
                            caption_mse=None, caption_pcc=None):
    x_sub = x[range_1:range_2, :]
    y_sub = y[range_1:range_2, :]
    if log_op:
        x_log = log_custom(x_sub, 10)
        y_log = log_custom(y_sub, 10)
    else:
        x_log = x_sub
        y_log = y_sub
    valid_vals = y_log[np.isfinite(y_log)]
    min_val = np.nanmin(valid_vals)
    max_val = np.nanmax(valid_vals)
    if caption_mse is None:
        caption_mse = np.mean((x_sub.ravel() - y_sub.ravel()) ** 2)
    if caption_pcc is None:
        caption_pcc = pearson_corr_np(x_sub, y_sub)
    n_rows = x_log.shape[0]
    n_cols = x_log.shape[1]
    heat_h = max(4, n_rows * 0.025)
    heat_w = max(3, n_cols * 0.5)
    fig_w = heat_w * 2 + 0.5                            
    fig_h = heat_h + 0.8                                   
    plt.figure(figsize=(fig_w, fig_h), dpi=200, facecolor='white')
    gs = gridspec.GridSpec(2, 3, height_ratios=[0.94, 0.06], width_ratios=[1, 1, 0.04],
                           hspace=0.08, wspace=0.06)
    ax1 = plt.subplot(gs[0, 0])
    im1 = plot_custom_chip(ax1, x_log, "True", min_val, max_val, xticks)
    ax2 = plt.subplot(gs[0, 1])
    im2 = plot_custom_chip(ax2, y_log, "Prediction", min_val, max_val, xticks)
    cax = plt.subplot(gs[0, 2])
    cbar = plt.colorbar(im2, cax=cax)
    if log_op:
        cbar.set_label('log10(value)', fontsize=6, labelpad=4)
    else:
        cbar.set_label('value', fontsize=6, labelpad=4)
    cbar.ax.tick_params(labelsize=5)
    ax_caption1 = plt.subplot(gs[1, :])
    ax_caption1.axis('off')
    caption_text1 = (
        f"Model: {caption_model}  |  "
        f"Data: {caption_data}  |  "
        f"MSE: {caption_mse:.4e}  |  "
        f"PCC: {caption_pcc:.3f}"
    )
    ax_caption1.text(0.5, 0.5, caption_text1,
                     ha='center', va='center',
                     fontsize=5, color='#333333',
                     bbox=dict(boxstyle='round,pad=0.3',
                               facecolor='#f8f8f8',
                               edgecolor='#e0e0e0',
                               alpha=0.85))
    plt.suptitle("ChIP-seq Prediction Test",
                 fontsize=9, fontweight='bold', y=0.99)
    plt.savefig(save_name, dpi=200, bbox_inches='tight', facecolor='white')
class TrainingMetricsVisualizer:
    def __init__(self, file_path, output_dir='./', figsize=(10, 6), dpi=100):
        self.file_path = file_path
        self.output_dir = output_dir
        self.figsize = figsize
        self.dpi = dpi
        self.df = None
        os.makedirs(output_dir, exist_ok=True)            
    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path, sep='\t', encoding='utf-8')
            required_cols = {'Epoch', 'LR', 'Loss'}
            missing = required_cols - set(self.df.columns)
            if missing:
                raise ValueError(f": {', '.join(missing)}")
        except Exception as e:
            raise RuntimeError(f": {str(e)}")
    def _generate_plot(self, y_col, y_label, color, filename):
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        plt.plot(self.df['Epoch'], self.df[y_col],
                 marker='o', linestyle='-', color=color, linewidth=2)
        plt.title(f'Epoch  vs {y_label}', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        return save_path
    def generate_plots(self):
        if self.df is None:
            self.load_data()
        lr_path = self._generate_plot(
            y_col='LR',
            y_label='Learning Rate',
            color='#1f77b4',       
            filename='epoch_lr_plot.png'
        )
        loss_path = self._generate_plot(
            y_col='Loss',
            y_label='Loss',
            color='#d62728',       
            filename='epoch_loss_plot.png'
        )
        return lr_path, loss_path

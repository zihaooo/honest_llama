# -*- coding: utf-8 -*-
import pickle
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

title_name_by_dataset = {
    'mmlu_mc2': 'MMLU',
    'openbookqa_mc2': 'OpenBookQA',
    'tqa_mc2': 'TruthfulQA'
}

def plot_head_acc(model_name, dataset_name):
    save_dir = Path('../features/head_acc')
    file_name = f'{save_dir}/{model_name}_{dataset_name}_all_heads_acc_mat.pkl'
    with open(file_name, 'rb') as f:
        mat = pickle.load(f)
    mat = np.flip(np.sort(mat, axis=1), axis=1) * 100
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(mat, cmap='viridis', interpolation='none', aspect='equal', origin='lower',
                   vmin=np.min(mat), vmax=np.max(mat))
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    num_layers = mat.shape[0]
    ax.set_ylabel('Layer')
    ax.set_yticks(np.arange(0, num_layers, step=2))
    ax.set_yticklabels(np.arange(0, num_layers, step=2))
    num_heads = mat.shape[1]
    ax.set_xlabel('Head (sorted)')
    ax.set_xticks(np.arange(0, num_heads, step=2))
    # ax.set_xticklabels(np.arange(0, num_heads, step=2))
    ax.set_xticklabels([])
    ax.set_title(f'min: {np.min(mat):.1f}, mid: {np.median(mat):.1f}, max: {np.max(mat):.1f}, mean: {np.mean(mat):.1f}, std: {np.std(mat):.1f}')
    fig.suptitle(f'Head-wise accuracy for {title_name_by_dataset.get(dataset_name)} dataset')
    fig.tight_layout()
    fig_path = Path(f'{save_dir}/{model_name}_{dataset_name}_head_acc.png')
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=300)


if __name__ == '__main__':
    for model_name in ['llama_7B', 'llama2_chat_7B', 'llama3_8B_instruct']:
        for dataset_name in ['mmlu_mc2', 'openbookqa_mc2', 'tqa_mc2']:
            plot_head_acc(model_name, dataset_name)

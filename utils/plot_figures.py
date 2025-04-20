# -*- coding: utf-8 -*-
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
import seaborn as sns


def plot_kde_in_top_2_direction(train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels, top_heads, seed,
                                model_name, dataset_name):
    layer, head = top_heads[0]

    all_X_train = np.concatenate([separated_head_wise_activations[i] for i in train_set_idxs], axis=0)
    y_train = np.concatenate([separated_labels[i] for i in train_set_idxs], axis=0)

    X_train = all_X_train[:, layer, head, :]

    clf = LogisticRegression(random_state=seed, max_iter=1000).fit(X_train, y_train)
    top_1_direction = clf.coef_[0]
    # normalize the directions
    top_1_direction = top_1_direction / np.linalg.norm(top_1_direction)

    X_train = X_train - np.outer(X_train.dot(top_1_direction), top_1_direction)
    clf = LogisticRegression(random_state=seed, max_iter=1000).fit(X_train, y_train)
    top_2_direction = clf.coef_[0]
    top_2_direction = top_2_direction / np.linalg.norm(top_2_direction)

    # Find activations
    usable_idxs = np.concatenate([train_set_idxs, val_set_idxs], axis=0)
    usable_head_wise_activations = np.concatenate([separated_head_wise_activations[i][:, layer, head, :] for i in usable_idxs], axis=0)
    usable_labels = np.concatenate([separated_labels[i] for i in usable_idxs], axis=0)
    true_acts = usable_head_wise_activations[usable_labels == 1]
    false_acts = usable_head_wise_activations[usable_labels == 0]

    # project activations onto the top 2 directions
    true_points = true_acts.dot(np.array([top_1_direction, top_2_direction]).T)  #  shape (n_samples, 2)
    false_points = false_acts.dot(np.array([top_1_direction, top_2_direction]).T) #  shape (n_samples, 2)

    x_true, y_true = true_points[:, 0], true_points[:, 1]
    x_false, y_false = false_points[:, 0], false_points[:, 1]

    # set up the JointGrid
    g = sns.JointGrid(height=6, ratio=4)

    # 2D KDE contours in the joint axes
    sns.kdeplot(
        x=x_true, y=y_true,
        levels=10,  # number of contour levels
        color="C0",  # default seaborn color palette
        linewidths=1.2,
        ax=g.ax_joint
    )
    sns.kdeplot(
        x=x_false, y=y_false,
        levels=10,
        color="C1",
        linewidths=1.2,
        ax=g.ax_joint
    )

    # marginal (1D) KDEs on the top and right
    sns.kdeplot(x=x_true, ax=g.ax_marg_x, color="C0", fill=True)
    sns.kdeplot(x=x_false, ax=g.ax_marg_x, color="C1", fill=True)
    sns.kdeplot(y=y_true, ax=g.ax_marg_y, color="C0", fill=True)
    sns.kdeplot(y=y_false, ax=g.ax_marg_y, color="C1", fill=True)

    # labels, legend, styling
    g.ax_joint.set_xlabel("Projection on the 1st Truthful Direction")
    g.ax_joint.set_ylabel("Projection on the 2nd Truthful Direction")
    g.ax_marg_x.legend(["Truthful", "False"], frameon=False, loc="upper right")

    # tidy up
    fig_dir = Path('../figures/kde')
    fig_dir.mkdir(parents=True, exist_ok=True)
    g.fig.suptitle(f"{model_name} {dataset_name} ({layer}, {head})")
    g.fig.tight_layout()
    g.fig.savefig(f"{fig_dir}/{model_name}_{dataset_name}_kde_top_2_direction_{layer}_{head}_seed_{seed}.png", dpi=300)

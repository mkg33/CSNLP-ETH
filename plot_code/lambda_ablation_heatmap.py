#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lambda_ablation_heatmap_pdf.py
"""

import matplotlib as mpl
mpl.use("pgf")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "cm",
    "pgf.preamble": r"\usepackage{newtxtext}\usepackage{newtxmath}",
    "axes.labelsize": 5,
    "font.size": 6,
    "legend.fontsize": 6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
})


models = {
    r"\textbf{Sliced}": np.array([[0.9118, 0.9118, 0.9116, 0.9117],
                                  [0.8260, 0.8263, 0.8258, 0.8260],
                                  [0.7680, 0.7680, 0.7679, 0.7675]]),
    r"\textbf{Maximum Projection}": np.array([[0.9121, 0.9118, 0.9215, 0.9120],
                                   [0.8268, 0.8267, 0.8268, 0.8269],
                                   [0.7666, 0.7667, 0.7662, 0.7760]]),
    r"\textbf{Multi OT}": np.array([[0.9140, 0.9140, 0.9140, 0.9140],
                                   [0.8262, 0.8263, 0.8262, 0.8262],
                                   [0.7715, 0.7708, 0.7712, 0.7708]]),
}

"""
models = {
    r"\textbf{Sliced}": np.array([[0.9122, 0.9118, 0.9108, 0.9143],
                                  [0.8261, 0.8263, 0.8258, 0.8261],
                                  [0.7725, 0.7680, 0.7676, 0.7645]]),
    r"\textbf{Maximum Projection}": np.array([[0.9132, 0.9118, 0.9107, 0.9138],
                                   [0.8273, 0.8267, 0.8263, 0.8251],
                                   [0.7731, 0.7667, 0.7703, 0.7637]]),
    r"\textbf{Multi OT}": np.array([[0.9154, 0.9140, 0.9105, 0.9106],
                                   [0.8235, 0.8263, 0.8273, 0.8223],
                                   [0.7650, 0.7708, 0.7715, 0.7677]]),
}
"""

#Projection dimension ablation:
"""
models = {
    r"\textbf{Sliced}": np.array([[0.9118, 0.9118, 0.9115, 0.9092],
                                  [0.8263, 0.8263, 0.8262, 0.8262],
                                  [0.7678, 0.7680, 0.7680, 0.7715]]),
    r"\textbf{Maximum Projection}": np.array([[0.9123, 0.9118, 0.9118, 0.9088],
                                   [0.8274, 0.8267, 0.8277, 0.8254],
                                   [0.7690, 0.7667, 0.7692, 0.7696]])
}
"""


tasks   = [r"\textbf{Easy}", r"\textbf{Medium}", r"\textbf{Hard}"]
lambdas = [0, 0.001, 0.4, 0.7]

dim = [64, 128, 256, 512]


cell_w = 0.4
cell_h = 0.25

n_rows   = len(tasks)
n_cols   = len(lambdas)

#n_cols = len(dim)
n_models = len(models)

fig_width  = n_models * n_cols * cell_w   + 0.6*(n_models-1)
fig_height = n_rows  * cell_h            + 0.8

fig, axes = plt.subplots(
    1, n_models,
    sharey=True,
    figsize=(fig_width, fig_height),
    constrained_layout=True
)
if n_models == 1:
    axes = [axes]


vmin, vmax = 0.70, 0.92
norm = Normalize(vmin, vmax)
cmap = plt.get_cmap("cividis")


for ax, (model_name, arr) in zip(axes, models.items()):
    im = ax.imshow(arr, aspect="auto", cmap=cmap, norm=norm)


    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            txt_colour = "white" if norm(arr[i, j]) < 0.55 else "black"
            ax.text(j, i, f"{arr[i, j]:.4f}",
                    ha="center", va="center",
                    fontsize=6, color=txt_colour)


    ax.set_xticks(np.arange(len(lambdas)))
    ax.set_xticklabels(lambdas)

    #ax.set_xticks(np.arange(len(dim)))
    #ax.set_xticklabels(dim)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_title(model_name, pad=6)
    ax.grid(False)


axes[0].set_yticks(np.arange(len(tasks)))
axes[0].set_yticklabels(tasks)


sm = ScalarMappable(norm=norm, cmap=cmap)
cbar = fig.colorbar(sm, ax=axes, orientation="vertical",
                    fraction=0.05, pad=0.04,
                    label=r"Macro F$_1$ score")


#fig.supxlabel(r'Projection Dimension', fontsize=8, y=-0.04)
fig.supxlabel(r'$\lambda$ (Orthogonality Loss)', fontsize=8, y=-0.04)
#fig.supxlabel(r'Style Dimension', fontsize=8, y=-0.04)
fig.savefig("lambda_ablation_heatmap.pdf", bbox_inches="tight")
plt.close(fig)
print("Wrote ablation_heatmap.pdf")

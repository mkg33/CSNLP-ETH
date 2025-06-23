#!/usr/bin/env python3
import matplotlib as mpl
mpl.use("pgf")

import matplotlib.pyplot as plt
import numpy as np

def add_brush_ring(ax, x0, y0, bar_h,
                   n_layers=12,
                   base_r=0.9,
                   edge_jitter=0.025,
                   layer_jitter=0.01,
                   lw_base=2.2, lw_var=0.4,
                   alpha_base=0.08, alpha_var=0.02,
                   color="#d62828",
                   seed=42):

    rng   = np.random.default_rng(seed)
    r0    = base_r * bar_h / 2
    theta = np.linspace(0, 2*np.pi, 600, endpoint=False)

    for _ in range(n_layers):
        theta_shift = rng.uniform(0, 2*np.pi)
        t           = theta + theta_shift
        r_layer     = r0 * (1 + layer_jitter*rng.uniform(-1, 1))
        rr          = r_layer * (1 + edge_jitter*rng.uniform(-1, 1, t.size))
        xs          = x0 + rr * np.cos(t)
        ys          = y0 + rr * np.sin(t)
        ax.plot(xs, ys,
                color=color,
                linewidth=lw_base + lw_var*rng.uniform(-1, 1),
                alpha=max(0, alpha_base + alpha_var*rng.uniform(-1, 1)),
                solid_capstyle="round",
                clip_on=False)


mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "font.family": "serif",
    "mathtext.fontset": "custom",
    "pgf.preamble": r"\usepackage{mathpazo}",
    "axes.labelsize": 10,  "font.size": 10,
    "legend.fontsize":  9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.figsize": (6, 4),
    "lines.linewidth": 1.0,
    "figure.facecolor": "none", "axes.facecolor": "none",
})


models = [
    (r"\shortstack{CLS\\+ Orthogonality\\+ Style Features\\(\textbf{Full Model})}",
     [0.9034, 0.8208, 0.7904]),
    (r"\shortstack{CLS\\+ Style Features\\+ Content}",
     [0.9179, 0.8015, 0.7551]),
    (r"\shortstack{CLS\\+ Orthogonality}",
     [0.9025, 0.8258, 0.7852]),
    (r"CLS-only",
     [0.9167, 0.8216, 0.7669]),
]

difficulty = ["Easy", "Medium", "Hard"]
colors   = {"Easy": "#004488", "Medium": "#c47e00", "Hard": "#BB5566"}
hatches  = ["", "///", "\\\\"]

means = [np.mean(s) for _, s in models]
max_mean = max(means)
mean_tol = 1e-6

n, ypos, bar_h = len(models), np.arange(len(models)), 0.55
fig, ax = plt.subplots()

for i, (label, scores) in enumerate(models):
    left = 0.0
    for diff, score in zip(difficulty, scores):
        ax.barh(ypos[i], score, height=bar_h, left=left,
                color=colors[diff], edgecolor='black',
                hatch=hatches[difficulty.index(diff)],
                label=diff if i == 0 else "")
        ax.text(left + score/2, ypos[i], f"{score:.4f}",
                ha='center', va='center', fontsize=11,
                fontweight='bold', color='white')
        left += score


    mean_val = np.mean(scores)
    x_brace = 1.02 * max(sum(s) for _, s in models)
    max_total = max(sum(s) for _, s in models)
    gap = 0.05 * max_total

    ax.text(x_brace, ypos[i],
            r"$\Big\}$", fontsize=20, va='center',
            clip_on=False)

    weight = 'bold' if abs(mean_val - max_mean) < mean_tol else 'normal'

    ax.text(x_brace + gap, ypos[i],
            f"{mean_val:.4f}",
            va='center', ha='left',
            fontsize=11, fontweight=weight,
            clip_on=False)


ax.set_yticks(ypos)
ax.set_yticklabels([lbl for lbl, _ in models])
ax.set_xlabel(r"Sum of F$_1$ scores")
max_total = max(sum(s) for _, s in models)
ax.set_xlim(0, max_total * 1.05)
ax.legend(title="Difficulty", ncol=3, frameon=False,
          loc="lower center", bbox_to_anchor=(0.5, 1.02))

for side in ("right", "top", "left"):
    ax.spines[side].set_visible(False)

x_brace = 1.02 * max_total
ax.text(x_brace + gap,
        ypos[-1] + bar_h,
        r"\textbf{Mean}",
        ha="left", va="bottom",
        fontsize=11, clip_on=False)


targets = {0.918:  ("Easy",   "#ff0000"),
           0.826:  ("Medium", "#0011ed"),
           0.790:  ("Hard",   "#ffd600")}

tol = 0.0006

for row_idx, (_, scores) in enumerate(models):
    left = 0.0
    for diff, score in zip(difficulty, scores):
        for val, (diff_name, ring_colour) in targets.items():
            if diff == diff_name and abs(score - val) < tol:
                x_c = left + score/2
                y_c = ypos[row_idx]
                add_brush_ring(ax, x_c, y_c, bar_h,
                               color=ring_colour)
        left += score

plt.tight_layout()

fig.savefig("pyramid_f1_plot.pdf", transparent=True)

print("wrote the pyramid plot")

#!/usr/bin/env python3
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.use('pgf')

mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "font.family": "serif",
    "mathtext.fontset": "custom",
    "pgf.preamble": r"\usepackage{mathpazo}",

    "figure.facecolor": "none",
    "axes.facecolor":   "#E0F7F8",

    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,

    "figure.figsize": (5, 3),
    "lines.linewidth": 1.0,
    "grid.linestyle": '--',
    "grid.linewidth": 0.5,
})


f1_scores = [
    [0.9034, 0.8208, 0.7904],
    [0.9058, 0.7981, 0.7274],
    [0.9113, 0.8184, 0.7580],
]
methods = ['Feature Set 1', 'Feature Set 2', 'Feature Set 3']
difficulty = ['Easy', 'Medium', 'Hard']
contrast = {'easy'  : '#004488',
        'medium': '#DDAA33',
        'hard'  : '#BB5566'}

hatches = ['', '///', '\\\\']


y = np.arange(len(methods))
width = 0.25

fig, ax = plt.subplots()
for i, (level, hatch) in enumerate(zip(difficulty, hatches)):
    values = [scores[i] for scores in f1_scores]

    ax.bar(y + (i - 1)*width,
           values,
           width=width,
           color=contrast[level.lower()],
           edgecolor='black',
           hatch=hatch,
           label=rf'\textbf{{{level}}}')


ax.set_ylabel(r'Macro F$_1$ score', fontsize=10)
ax.set_title(r'Macro F$_1$ score for Different Feature Extraction Methods', fontsize=11)
ax.set_xticks(y)
ax.set_xticklabels(methods)
ax.set_ylim(0.7, 0.92)
#ax.legend(title=r'Validation set', loc='best')

leg = plt.legend(
        loc="center",
        bbox_to_anchor=(0.583, 0.7),
        frameon=True,
        fancybox=True)

leg.get_frame().set_facecolor("#fff7fc")
leg.get_frame().set_edgecolor("#BB5566")
leg.get_frame().set_alpha(1.0)

ax.grid(axis='y')
plt.tight_layout()

#plt.savefig('feature_extraction_f1_plot.pgf')
plt.savefig('feature_extraction_f1_plot.pdf')

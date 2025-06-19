#!/usr/bin/env python3
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.use('pgf')

mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],

    "pgf.preamble": "\\usepackage{newtxtext}\\usepackage{newtxmath}",

    "axes.labelsize": 8,
    "font.size": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,

    "figure.figsize": (4.5, 4),
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
ax.set_title(r'Macro F$_1$ score for Different Feature Extraction Methods', fontsize=12)
ax.set_xticks(y)
ax.set_xticklabels(methods)
ax.set_ylim(0.7, 1)
ax.legend(title=r'Validation set', loc='best')
ax.grid(axis='y')
plt.tight_layout()

plt.savefig('feature_extraction_f1_plot.pgf')
plt.savefig('feature_extraction_f1_plot.pdf')

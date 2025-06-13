#!/usr/bin/env python3
import matplotlib as mpl
import matplotlib.pyplot as plt


gmpl_backend = 'pgf'
mpl.use(gmpl_backend)


mpl.rcParams.update({

    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],

    "pgf.preamble": "\\usepackage{newtxtext}\\usepackage{newtxmath}",

    "axes.labelsize": 8,
    "font.size": 8,
    "legend.fontsize": 6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,

    "figure.figsize": (3.39, 2.53),
    "lines.linewidth": 1.0,
    "grid.linestyle": '--',
    "grid.linewidth": 0.5,
})


lambdas = [0.01, 0.1, 1]
f1_easy   = [0.9224, 0.9034, 0.9341]
f1_medium = [0.8134, 0.8208, 0.8069]
f1_hard   = [0.7808, 0.7904, 0.7719]


ticks = [0.01, 0.1, 1]
plt.plot(lambdas, f1_easy,   marker='o', label=r'\textbf{Easy}')
plt.plot(lambdas, f1_medium, marker='o', label=r'\textbf{Medium}')
plt.plot(lambdas, f1_hard,   marker='o', label=r'\textbf{Hard}')


plt.xscale('log')
plt.xticks(ticks, labels=["0.01", "0.1", "1"])


plt.xlabel(r'$\lambda$ (Orthogonality Loss)')
plt.ylabel(r'Macro F$_1$-score')
plt.title(r'Effect of Orthogonality Weight on Macro F$_1$')


plt.legend(loc='best')
plt.grid(True, which='both')
plt.tight_layout()


plt.savefig('orthogonality_f1_plot.pgf')
plt.savefig('orthogonality_f1_plot.pdf')

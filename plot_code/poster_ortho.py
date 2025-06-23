#!/usr/bin/env python3
import matplotlib as mpl
import matplotlib.pyplot as plt


gmpl_backend = 'pgf'
mpl.use(gmpl_backend)


mpl.rcParams.update({

    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "font.family": "serif",
    "mathtext.fontset": "custom",
    "pgf.preamble": r"\usepackage{mathpazo}",
    "figure.facecolor": "none",
    "axes.facecolor":   "#E0F7F8",

    "axes.labelsize": 8,
    "font.size": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,

    "figure.figsize": (3.39, 2.53),
    "lines.linewidth": 1.0,
    "grid.linestyle": '--',
    "grid.linewidth": 0.5,
})


lambdas = [0.01, 0.1, 1]

f1_easy   = [0.9224, 0.9034, 0.9341]
f1_medium = [0.8134, 0.8208, 0.8069]
f1_hard   = [0.7808, 0.7904, 0.7719]

contrast = {'easy'  : '#004488',
        'medium': '#DDAA33',
        'hard'  : '#BB5566'}

ticks = [0.01, 0.1, 1]

plt.plot(lambdas, f1_easy,   marker='o', color=contrast['easy'], label=r'\textbf{Easy}')
plt.plot(lambdas, f1_medium, marker='s', color=contrast['medium'],  label=r'\textbf{Medium}')
plt.plot(lambdas, f1_hard,   marker='^', color=contrast['hard'], label=r'\textbf{Hard}')


plt.xscale('log')
plt.xticks(ticks, labels=["0.01", "0.1", "1"])


plt.xlabel(r'$\lambda$ (Orthogonality Loss)', fontsize=10)
plt.ylabel(r'Macro F$_1$ score')
plt.title(r'Effect of Orthogonality Weight on Macro F$_1$')

leg = plt.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 0.45),
        frameon=True,
        fancybox=True)

leg.get_frame().set_facecolor("#fff7fc")
leg.get_frame().set_edgecolor("#BB5566")
leg.get_frame().set_alpha(1.0)

plt.grid(True, which='both')
plt.tight_layout()


plt.savefig('orthogonality_f1_plot.pgf')
plt.savefig('orthogonality_f1_plot.pdf')

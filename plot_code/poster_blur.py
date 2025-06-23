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

    "axes.labelsize": 9,
    "font.size": 10,
    "legend.fontsize": 5.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,

    "figure.figsize": (3.39, 2.53),
    "lines.linewidth": 1.0,
    "grid.linestyle": '--',
    "grid.linewidth": 0.5,
})


blur = [0.01, 0.03, 0.05]
f1_easy   = [0.8873, 0.9118, 0.8729]
f1_medium = [0.8188, 0.8252, 0.8138]
f1_hard   = [0.7105, 0.7671, 0.6913]

contrast = {'easy'  : '#004488',
        'medium': '#DDAA33',
        'hard'  : '#BB5566'}

ticks = [0.01, 0.03, 0.05]
plt.plot(blur, f1_easy,   marker='o', color=contrast['easy'], label=r'\textbf{Easy}')
plt.plot(blur, f1_medium, marker='s', color=contrast['medium'],  label=r'\textbf{Medium}')
plt.plot(blur, f1_hard,   marker='^', color=contrast['hard'], label=r'\textbf{Hard}')


#plt.xscale('log')
plt.xticks(ticks, labels=["0.01", "0.03", "0.05"])


plt.xlabel(r'Blur', fontsize=10)
plt.ylabel(r'Macro F$_1$ score')
plt.title(r'Effect of Blur on Macro F$_1$')

leg = plt.legend(
        loc="best",
        frameon=True,
        fancybox=True)

leg.get_frame().set_facecolor("#fff7fc")
leg.get_frame().set_edgecolor("#BB5566")
leg.get_frame().set_alpha(1.0)

ax = plt.gca()
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
ax.spines["right"].set_visible(True)
ax.spines["left"].set_visible(True)

plt.grid(True, which='both')
plt.tight_layout()


#plt.savefig('poster_blur_f1_plot.pgf')
plt.savefig('poster_blur_f1_plot.pdf')

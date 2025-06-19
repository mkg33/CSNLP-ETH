import matplotlib as mpl
import matplotlib.pyplot as plt


mpl.use('pgf')


mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "cm",

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


contrast = {'easy'  : '#004488',
        'medium': '#DDAA33',
        'hard'  : '#BB5566'}


dims      = [64, 128, 256, 512]
f1_easy   = [0.9365, 0.9034, 0.9242, 0.9295]
f1_medium = [0.7954, 0.8208, 0.8276, 0.8084]
f1_hard   = [0.7595, 0.7904, 0.7841, 0.7589]


plt.plot(dims, f1_easy,   marker='o', color=contrast['easy'], label=r'\textbf{Easy}')
plt.plot(dims, f1_medium, marker='s', color=contrast['medium'], label=r'\textbf{Medium}')
plt.plot(dims, f1_hard,   marker='^', color=contrast['hard'], label=r'\textbf{Hard}')

plt.xticks(dims)
plt.xlabel(r'Dimension of Style Embedding')
plt.ylabel(r'Macro F$_1$ score')
plt.title(r'Effect of Style Dimension on Macro F$_1$')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()

plt.savefig('style_dim_f1_plot.pgf')
plt.savefig('style_dim_f1_plot.pdf')

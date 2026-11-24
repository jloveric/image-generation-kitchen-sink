"""Draw the two network schematics in the manuscript."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "figures"
INK, WIRE, TEAL, GOLD = "#223444", "#81919a", "#006e74", "#98531d"
COOL, WARM, LIGHT = "#e2f3f0", "#f9e9d9", "#f0f4f6"
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "mathtext.fontset": "dejavusans"})


def canvas(width, height):
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, width), ylim=(0, height), aspect="equal")
    ax.axis("off")
    return fig, ax


def label(ax, x, y, s, size=11, color=INK, weight="normal", **kwargs):
    ax.text(x, y, s, fontsize=size, color=color, ha="center", va="center",
            fontweight=weight, **kwargs)


def wire(ax, points, color=WIRE, width=1.3, style="-"):
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=color, lw=width, ls=style, zorder=0, solid_capstyle="round")


def arrow(ax, a, b, color=INK, style="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, lw=1.4,
                                 linestyle=style, color=color, shrinkA=0, shrinkB=0))


def box(ax, x, y, w, h, title, sub=None, color=INK, fill=LIGHT, size=12):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                ec=color, fc=fill, lw=1.1))
    label(ax, x, y + (.12 if sub else 0), title, size, color, "bold")
    if sub:
        label(ax, x, y-.14, sub, 10, color)


def population(ax):
    ys = [1.82, 2.25, 2.68, 3.11]
    for y in ys:
        wire(ax, [(.81, 2.46), (2.03, y)])
        for z in ys:
            wire(ax, [(2.03, y), (4.31, z)], width=.85)
    for x in (2.03, 4.31):
        for i, y in enumerate(ys):
            color, fill, mark = (TEAL, COOL, "+") if i >= 2 else (GOLD, WARM, "−")
            ax.add_patch(Circle((x, y), .163, ec=color, fc=fill, lw=1.15))
            label(ax, x, y-.003, mark, 13, color, "bold")
    ax.add_patch(Circle((.81, 2.46), .18, ec=INK, fc="white", lw=1.1))
    label(ax, .81, 2.46, "$x$", 13)


def regressor():
    fig, ax = canvas(9.45, 3.9)
    for x, title in ((.81, "Input"), (2.03, "Layer 1"), (4.31, "Layer 2")):
        label(ax, x, 3.6, title, 13, weight="bold")
    for x in (2.03, 4.31):
        ax.add_patch(FancyBboxPatch((x-.36, 1.58), .72, 1.76,
                                    boxstyle="round,pad=0,rounding_size=.12", zorder=-2,
                                    fc="#f7f9fa", ec="#c5cfd3", lw=.95))
    population(ax)
    for x, i in ((2.03, 1), (4.31, 2)):
        arrow(ax, (x, 1.59), (x, 1.35), TEAL)
        box(ax, x, 1.11, 1.88, .46, rf"$r_{i}=\sum_{{+}}|h|-\sum_{{-}}|h|$", color=TEAL, fill=COOL)
        wire(ax, [(x, .88), (x, .46), (5.71, .46)], TEAL)
    arrow(ax, (5.71, .46), (5.71, .76), TEAL)
    box(ax, 6.37, 1.11, 1.92, .70, "Prediction", r"$\hat y=(r_1+r_2)/2$", TEAL, COOL)
    arrow(ax, (6.37, 1.46), (6.37, 2.12))
    box(ax, 6.93, 2.49, 2.90, .70, "Compare with target", r"$e=\mathrm{sign}(y-\hat y)$")
    arrow(ax, (7.59, 3.12), (7.59, 2.84))
    label(ax, 7.59, 3.38, "Training target  $y$", 12)
    wire(ax, [(5.67, 2.84), (5.67, 3.45), (2.32, 3.45)], GOLD, style="--")
    for x in (2.32, 4.60):
        arrow(ax, (x, 3.45), (x, 3.30), GOLD)
    label(ax, 5.81, 3.69, "Teaching sign", 11, GOLD)
    fig.savefig(OUT / "class-network.pdf")
    plt.close(fig)


def autoencoder():
    fig, ax = canvas(10, 2.85)
    xs = [.93, 2.87, 4.74, 6.77, 9.02]
    widths = [1.56, 1.64, 1.46, 2.08, 1.79]
    layers = [("Portrait", "12,288 RGB values"), ("Encoder", "3,072 neurons"),
              ("Code", "2,048 values"), ("Dense decoder", "24,576 neurons"),
              ("Subtract pairs", "12,288 RGB values")]
    for i, (x, w, (title, detail)) in enumerate(zip(xs, widths, layers)):
        accent = i in (2, 4)
        box(ax, x, 1.28, w, .83, title, detail, TEAL if accent else INK, COOL if accent else LIGHT)
        if i:
            arrow(ax, (xs[i-1] + widths[i-1]/2, 1.28), (x-w/2, 1.28))
    label(ax, 4.74, 2.60, "Sampled code for generation", 12, TEAL)
    arrow(ax, (4.74, 2.40), (4.74, 1.70), TEAL, "--")
    label(ax, 7.84, .39, "One increase/decrease pair per RGB value", 11)
    fig.savefig(OUT / "autoencoder.pdf")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    regressor()
    autoencoder()

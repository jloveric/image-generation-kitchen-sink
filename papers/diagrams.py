"""Draw the network and training diagrams in the manuscript."""
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
        for y in ys:
            ax.add_patch(Circle((x, y), .163, ec=TEAL, fc=COOL, lw=1.15))
            label(ax, x, y-.003, "$h$", 12, TEAL)
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
        box(ax, x, 1.11, 1.88, .46, rf"$r_{i}=\sum_j h_j$", color=TEAL, fill=COOL)
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
    label(ax, 5.81, 3.69, r"Error sign $e$", 11, GOLD)
    fig.savefig(OUT / "class-network.pdf")
    plt.close(fig)


def implicit_image():
    fig, ax = canvas(11.2, 5.27)
    ax.set_ylim(-.52, 4.75)
    inputs = ((.62, 2.84), (.62, 2.14))
    layers = ((2.18, 1), (4.06, 2), (6.44, 12))
    pools = (("R", 3.25, "#f8e9e6"), ("G", 2.49, "#e4f1e8"),
             ("B", 1.73, "#e7eff8"))
    heights = [center for _, center, _ in pools]
    label(ax, .62, 3.89, "Position", 12, weight="bold")
    for (x, y), name in zip(inputs, ("$x$", "$y$"), strict=True):
        for height in heights:
            wire(ax, [(x, y), (layers[0][0], height)], width=.65)
        ax.add_patch(Circle((x, y), .18, ec=INK, fc="white"))
        label(ax, x, y, name, 13)
    for source in heights:
        for dest in heights:
            wire(ax, [(layers[0][0], source), (layers[1][0], dest)], width=.55)
    for x, depth in layers:
        label(ax, x, 3.89, f"Layer {depth}", 12, weight="bold")
        ax.add_patch(FancyBboxPatch((x-.43, 1.28), .86, 2.42,
                                    boxstyle="round,pad=0,rounding_size=.11",
                                    ec=WIRE, fc=LIGHT, lw=.8, zorder=.8))
        for color, y, fill in pools:
            ax.add_patch(FancyBboxPatch((x-.35, y-.34), .70, .68,
                                        boxstyle="round,pad=0,rounding_size=.09",
                                        ec=WIRE, fc=fill, lw=.7))
            label(ax, x, y, color, 12, weight="bold")
        arrow(ax, (x, 1.27), (x, 1.05), TEAL)
        box(ax, x, .81, 1.10, .46, rf"$\mathbf{{r}}_{{{depth}}}$", color=TEAL, fill=COOL)
        wire(ax, [(x, .58), (x, .28)], TEAL)
    label(ax, 5.25, 2.92, "Layers 3–11", 11)
    arrow(ax, (4.47, 2.49), (4.92, 2.49))
    label(ax, 5.25, 2.53, "⋯", 22)
    arrow(ax, (5.56, 2.49), (6.04, 2.49))
    label(ax, 5.25, .82, "⋯", 18, TEAL)

    wire(ax, [(2.18, .28), (9.10, .28)], TEAL)
    arrow(ax, (9.10, .28), (9.10, .54), TEAL)
    box(ax, 9.10, 1.06, 3.62, 1.00, "Predicted RGB at $(x,y)$",
        "Mean of all 12 RGB estimates", TEAL, COOL, 13)
    label(ax, 9.10, .73, "Clip to [−1, 1] and round to 256 levels", 10, TEAL)
    arrow(ax, (9.10, 1.57), (9.10, 2.47))
    box(ax, 9.10, 2.91, 3.62, .86, "Compare each color",
        r"$e_c=\mathrm{sign}(\mathrm{target}_c-\mathrm{prediction}_c)$")
    label(ax, 10.00, 3.91, "Target RGB", 12, weight="bold")
    arrow(ax, (10.00, 3.70), (10.00, 3.35))
    wire(ax, [(8.17, 3.35), (8.17, 4.25), (2.18, 4.25)], GOLD, style="--")
    for x, _ in layers:
        arrow(ax, (x, 4.25), (x, 4.02), GOLD, "--")
    label(ax, 5.14, 4.52, r"At each layer: $e_R$ to red, $e_G$ to green, $e_B$ to blue", 12, GOLD)
    label(ax, 2.82, -.19, "Each color pool has 144 neurons.", 11)
    label(ax, 8.26, -.19, r"Each color estimate sums the signed activities in its pool.", 11)
    fig.savefig(OUT / "implicit-image.pdf")
    fig.savefig(OUT / "implicit-image.png", dpi=160)
    plt.close(fig)


def pixel_votes():
    fig, ax = canvas(10.4, 2.85)
    columns = (1.02, 2.72, 4.22, 5.40)
    for x, title in zip(columns, ("Pixel channel", r"Error $e_k$", r"Multiplier $v_k$", r"Vote $v_k e_k$"), strict=True):
        label(ax, x, 2.55, title, 12, weight="bold")
    rows = (
        ("Blue (9, 45)", "too low (+1)", r"$+1$", "+1"),
        ("Red (60, 43)", "too high (−1)", r"$-1$", "+1"),
        ("Blue (48, 59)", "too high (−1)", r"$+1$", "−1"),
        ("Red (31, 63)", "correct (0)", r"$-1$", "0"),
    )
    for y, row in zip((2.03, 1.58, 1.13, .68), rows, strict=True):
        for x, entry in zip(columns, row, strict=True):
            label(ax, x, y, entry, 11)
    wire(ax, [(.15, 2.30), (5.72, 2.30)], width=.8)
    wire(ax, [(5.76, 2.15), (5.87, 2.15), (5.87, .57), (5.76, .57)])
    arrow(ax, (5.87, 1.36), (6.16, 1.36))
    box(ax, 6.82, 1.36, 1.26, .70, "Sum", "+1")
    arrow(ax, (7.45, 1.36), (7.94, 1.36), GOLD)
    box(ax, 9.12, 1.36, 2.30, .76, "One hidden neuron", r"$u_i h_j\cdot(+1)$", GOLD, WARM)
    fig.savefig(OUT / "pixel-votes.pdf")
    plt.close(fig)


def autoencoder():
    from hebbian_fit.autoencoder import vote_assignment

    fig, ax = canvas(12.5, 5.78)
    sheet = plt.imread(OUT / "celeba-reconstruction.png")
    original, reconstruction = sheet[:, :sheet.shape[0]], sheet[:, sheet.shape[0]:]

    def heading(x, y, title):
        ax.text(x, y, title, fontsize=15, va="center", color=INK, weight="bold")

    heading(.15, 5.52, "Forward pass")
    y = 4.60
    for x, pixels, title, detail in (
        (.68, original, "Input image", "Center the RGB values"),
        (11.80, reconstruction, "Output image", r"Round $\min(1,|h_k|)$ to 8 bits"),
    ):
        ax.imshow(pixels, extent=(x-.41, x+.41, y-.41, y+.41), interpolation="nearest")
        label(ax, x, 5.17, title, 12, weight="bold")
        label(ax, max(1.03, min(11.39, x)), 3.95, detail, 11)

    previous = 1.09
    layers = (
        (2.37, "Encoder 1", 3072), (4.23, "Encoder 2", 1536), (6.09, "Bottleneck", 256),
        (7.95, "Decoder hidden", 2040), (9.81, "Decoder output", 12288),
    )
    for x, title, width in layers:
        box(ax, x, y, 1.60, .84, title, f"{width:,} neurons", TEAL, COOL)
        arrow(ax, (previous+.03, y), (x-.82, y))
        previous = x+.80
    arrow(ax, (previous+.03, y), (11.37, y))
    wire(ax, [(.15, 3.64), (12.35, 3.64)], width=.75)

    heading(.15, 3.34, "Connections between two hidden layers")
    heading(6.10, 3.34, "Class assignments used during training")
    wire(ax, [(5.86, .23), (5.86, 3.46)], width=.75)
    left, right = 1.49, 4.18
    rows = (2.21, 1.16)
    label(ax, left, 2.82, "Bottleneck (B)", 12, weight="bold")
    label(ax, right, 2.82, "Decoder hidden (D)", 12, weight="bold")
    for source in rows:
        for dest in rows:
            arrow(ax, (left+.31, source), (right-.31, dest))
    for x, layer, dx in ((left, "B", -1), (right, "D", 1)):
        mid = sum(rows)/2
        label(ax, x + dx*1.11, mid, r"$\cdots$", 18)
        for j, row_y in enumerate(rows, start=1):
            edge = (x + dx*.88, mid)
            node = (x + dx*.27, row_y + (.11 if row_y < mid else -.11))
            start, end = (edge, node) if dx < 0 else (node, edge)
            arrow(ax, start, end, WIRE)
            ax.add_patch(Circle((x, row_y), .29, ec=TEAL, fc=COOL, lw=1.2))
            label(ax, x, row_y, rf"$h^{{\mathrm{{{layer}}}}}_{{{j}}}$", 14, TEAL)
        label(ax, x, .58, r"$\vdots$", 16)

    columns = (6.70, 8.23, 10.63)
    for x, title in zip(columns, ("Neuron", "Class count", "Teaching signal"), strict=True):
        label(ax, x, 2.82, title, 12, weight="bold")
    wire(ax, [(6.10, 2.63), (12.35, 2.63)], width=.8)
    row_y = 2.36
    for layer, width, assignment_seed in (("B", 256, 0), ("D", 2040, 2)):
        groups, polarity = vote_assignment(width // 2, assignment_seed, True, independent=True)
        for j in range(2):
            k, m = groups[j, :2] + 1
            first, second = ("+" if v > 0 else "-" for v in polarity[:2])
            label(ax, columns[0], row_y, rf"$h^{{\mathrm{{{layer}}}}}_{{{j+1}}}$", 14, TEAL)
            label(ax, columns[1], row_y, str(groups.shape[1]), 13)
            label(ax, columns[2], row_y, rf"${first}e_{{{k}}}{second}e_{{{m}}}+\cdots$", 14)
            row_y -= .48
    label(ax, 9.20, .37, r"$e_k$ is the sign of the error at output neuron $k$.", 12)
    fig.savefig(OUT / "autoencoder.pdf")
    fig.savefig(OUT / "autoencoder.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    regressor()
    implicit_image()
    pixel_votes()
    autoencoder()

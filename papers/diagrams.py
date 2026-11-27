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

    fig, ax = canvas(12.5, 7.44)
    ax.set_ylim(2.44, 9.88)
    # The reconstruction command saves two square tiles side by side.
    sheet = plt.imread(OUT / "celeba-reconstruction.png")
    original = sheet[:, :sheet.shape[0]]
    reconstruction = sheet[:, sheet.shape[0]:]

    def draw_image(pixels, x, y):
        ax.imshow(pixels, extent=(x-.45, x+.45, y-.45, y+.45), interpolation="nearest")

    def heading(y, title, color):
        ax.text(.12, y, title, fontsize=15, va="center", color=color, weight="bold")

    heading(9.59, "How 12,288 output classes teach the hidden layers", INK)
    left, right = 4.68, 7.82
    rows = (8.27, 7.15)
    label(ax, left, 9.02, "Bottleneck", 13, weight="bold")
    label(ax, right, 9.02, "Decoder hidden", 13, weight="bold")
    for source in rows:
        for dest in rows:
            arrow(ax, (left+.30, source), (right-.30, dest), INK)
    # The ellipses flank the two expanded layers on the forward path.
    for x, dx in ((left, -1), (right, 1)):
        label(ax, x + dx*.93, 7.71, r"$\cdots$", 17)
        for y in rows:
            earlier = (x + dx*.73, 7.71)
            node = (x + dx*.21, y + (.20 if y < 7.71 else -.20))
            endpoints = (earlier, node) if dx < 0 else (node, earlier)
            arrow(ax, *endpoints, WIRE)

    # Use the same two assignments as the manuscript's trained autoencoder.
    for x, group_x, width, assignment_seed in ((left, 1.76, 256, 0), (right, 10.74, 2040, 2)):
        groups, polarity = vote_assignment(width // 2, assignment_seed, True, independent=True)
        label(ax, group_x, 9.02, f"{groups.shape[1]} classes per neuron", 13, GOLD, "bold")
        for j, y in enumerate(rows):
            ax.add_patch(Circle((x, y), .28, ec=TEAL, fc=COOL, lw=1.3))
            label(ax, x, y, rf"$h_{{{j+1}}}$", 15, TEAL)
            k, m = groups[j, :2] + 1  # Number the output neurons from one.
            first, second = ("+" if v > 0 else "-" for v in polarity[:2])
            box(ax, group_x, y, 3.22, .79, f"Output neurons {k:,}, {m:,}, …",
                rf"${first}e_{{{k}}}{second}e_{{{m}}}+\cdots$", GOLD, WARM, size=12)
            direction = 1 if group_x < x else -1
            arrow(ax, (group_x+direction*1.62, y), (x-direction*.30, y), GOLD, "--")
        label(ax, x, 6.66, r"$\vdots$", 15)
    arrow(ax, (.26, 6.32), (.96, 6.32))
    label(ax, 2.20, 6.32, "Forward connection", 12)
    arrow(ax, (4.24, 6.32), (4.94, 6.32), GOLD, "--")
    label(ax, 8.19, 6.32, r"Weighted sum of the assigned class error signs", 12, GOLD)
    wire(ax, [(.12, 6.02), (12.38, 6.02)], width=.7)

    heading(5.80, "Reconstruction", INK)
    label(ax, 8.35, 5.80, "The bottleneck passes 256 signed activities to the decoder.", 12)
    layers = (
        (2.37, "Encoder 1", 3072, 8),
        (4.21, "Encoder 2", 1536, 16),
        (6.05, "Bottleneck", 256, 96),
        (7.89, "Decoder hidden", 2040, 12),
        (9.73, "Decoder output", 12288, 1),
    )
    y = 4.74
    draw_image(original, .69, y)
    draw_image(reconstruction, 11.76, y)
    label(ax, .69, 5.44, "Input image", 12, weight="bold")
    label(ax, 11.76, 5.44, "Reconstruction", 12, weight="bold")
    label(ax, .69, 4.04, "Center the RGB\ninput values", 11)
    label(ax, 11.76, 4.04, r"$\min(1,|h_k|)$" + "\nround to 8 bits", 11)
    previous = 1.16
    for x, name, width, group_size in layers:
        label(ax, x, 5.44, name, 12, weight="bold")
        ax.add_patch(FancyBboxPatch((x-.73, y-.48), 1.46, .96,
                                   boxstyle="round,pad=0,rounding_size=.08",
                                   ec=TEAL if name.startswith("Decoder") or name == "Bottleneck" else WIRE,
                                   fc=COOL if name == "Bottleneck" else LIGHT))
        single = name == "Decoder output"
        ax.add_patch(Circle((x, y+.18), .18, ec=TEAL, fc=COOL, lw=1.1))
        label(ax, x, y+.18, "$h_k$" if single else "$h_j$", 13, TEAL)
        label(ax, x, y-.22, f"{width:,} neurons", 10.5, weight="bold")
        arrow(ax, (previous, y), (x-.75, y))
        previous = x+.75
        # Feedback travels from the final pixel comparison directly to each layer.
        arrow(ax, (x, 3.86), (x, y-.51), GOLD, "--")
        label(ax, x, 3.59, f"{group_size} sign" + ("s" if group_size > 1 else ""),
              12, GOLD, "bold")
        wire(ax, [(x, 3.40), (x, 3.19)], GOLD, style="--")
    arrow(ax, (previous, y), (11.26, y))
    heading(3.59, "Training", GOLD)
    wire(ax, [(layers[0][0], 3.19), (10.75, 3.19)], GOLD, style="--")
    box(ax, 11.55, 3.18, 1.72, .74, "Error signs", "sign(target − output)", GOLD, WARM, 12)
    arrow(ax, (11.76, 3.79), (11.76, 3.58), GOLD, "--")
    label(ax, 6.23, 2.72,
          "Signs per neuron. Each output neuron receives its own channel’s sign.",
          12, GOLD)
    fig.savefig(OUT / "autoencoder.pdf")
    fig.savefig(OUT / "autoencoder.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    regressor()
    implicit_image()
    pixel_votes()
    autoencoder()

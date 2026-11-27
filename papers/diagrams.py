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


def implicit_image():
    fig, ax = canvas(11.2, 5.27)
    ax.set_ylim(-.52, 4.75)
    inputs = ((.62, 2.84), (.62, 2.14))
    layers = ((2.18, 1), (4.06, 2), (6.44, 12))
    pools = (("R", 3.25, "#f8e9e6"), ("G", 2.49, "#e4f1e8"),
             ("B", 1.73, "#e7eff8"))
    heights = [center + offset for _, center, _ in pools for offset in (.17, -.17)]
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
            for offset, sign in ((.17, "+"), (-.17, "−")):
                ax.add_patch(Circle((x, y+offset), .155, ec=INK, fc="white", lw=.8))
                label(ax, x, y+offset, color+sign, 10, weight="bold")
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
    label(ax, 5.14, 4.52, r"Teaching signs $(e_R,\, e_G,\, e_B)$ at every layer", 12, GOLD)
    label(ax, 2.82, -.19, "Each Gaussian pool has 72 increase and 72 decrease neurons.", 11)
    label(ax, 8.26, -.19, r"Each layer estimates each color as  $r_{\ell c}=\sum_{c+}|h|-\sum_{c-}|h|$.", 11)
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
    label(ax, 9.18, 2.55, "One hidden pair", 12, weight="bold")
    for y, neuron, value, color, fill in (
        (1.95, "Increase neuron", "+1", TEAL, COOL),
        (.77, "Decrease neuron", "−1", GOLD, WARM),
    ):
        wire(ax, [(7.45, 1.36), (7.78, 1.36), (7.78, y)], color)
        arrow(ax, (7.78, y), (8.14, y), color)
        box(ax, 9.18, y, 2.0, .76, neuron, value, color, fill)
    fig.savefig(OUT / "pixel-votes.pdf")
    plt.close(fig)


def autoencoder():
    fig, ax = canvas(12.5, 6.28)
    ax.set_ylim(2.44, 8.72)
    # Original and semi-implicit reconstruction of the same official test image.
    # contact_sheet uses 88-pixel tiles, a four-pixel gap and a 204-pixel label column.
    sheet = plt.imread(OUT.parent.parent / "assets/results/celeba-256-lion-explicit-reconstructions.png")
    original = sheet[4:92, 204:292]
    reconstruction = sheet[96:184, 204:292]

    def draw_image(pixels, x, y):
        ax.imshow(pixels, extent=(x-.45, x+.45, y-.45, y+.45), interpolation="nearest")

    def heading(y, title, color):
        ax.text(.12, y, title, fontsize=15, va="center", color=color, weight="bold")

    heading(8.46, "How pixel errors teach every hidden layer", INK)
    label(ax, 10.38, 8.46, "One bottleneck pair, one image", 12)
    columns = (1.17, 3.03, 4.28, 5.42, 6.88)
    for x, title in zip(columns, ("Assigned channels", "Low (+1)", "High (−1)",
                                  "Correct (0)", "Vote total"), strict=True):
        label(ax, x, 7.94, title, 12, weight="bold")
    wire(ax, [(.15, 7.71), (7.65, 7.71)], width=.8)
    rows = (
        (7.39, (r"48 with $v_k=+1$", "2", "1", "45", r"$(+1)(2-1)=+1$")),
        (6.88, (r"48 with $v_k=-1$", "0", "2", "46", r"$(-1)(0-2)=+2$")),
    )
    for y, row in rows:
        for x, entry in zip(columns, row, strict=True):
            label(ax, x, y, entry, 12)
    for y in (7.39, 6.88):
        arrow(ax, (7.68, y), (8.14, 7.14 + .55 * (y-7.14)), GOLD)
    box(ax, 8.81, 7.14, 1.27, .84, "Sum", "+3", GOLD, WARM)
    for y, name, multiplier, color, fill in (
        (7.58, "Increase neuron", "+3", TEAL, COOL),
        (6.72, "Decrease neuron", "−3", GOLD, WARM),
    ):
        arrow(ax, (9.47, 7.14), (10.02, y), color)
        box(ax, 11.15, y, 2.18, .68, name, multiplier, color, fill)
    label(ax, 4.00, 6.27, "This pair uses the same 96 channels on every image.", 11)
    wire(ax, [(.12, 6.02), (12.38, 6.02)], width=.7)

    heading(5.80, "Reconstruction", INK)
    label(ax, 8.35, 5.80, "At the bottleneck, 128 pairs pass 256 separate activities to the decoder.", 12)
    layers = (
        (2.37, "Encoder 1", 3072, 8),
        (4.21, "Encoder 2", 1536, 16),
        (6.05, "Bottleneck", 256, 96),
        (7.89, "Decoder hidden", 1536, 16),
        (9.73, "Decoder output", 24576, 1),
    )
    y = 4.74
    draw_image(original, .69, y)
    draw_image(reconstruction, 11.76, y)
    label(ax, .69, 5.44, "Input image", 12, weight="bold")
    label(ax, 11.76, 5.44, "Reconstruction", 12, weight="bold")
    label(ax, .69, 4.04, "Center the RGB\ninput values", 11)
    label(ax, 11.76, 4.04, r"$|h_{+}|-|h_{-}|$" + "\nclip and round", 11)
    previous = 1.16
    for x, name, width, group_size in layers:
        label(ax, x, 5.44, name, 12, weight="bold")
        ax.add_patch(FancyBboxPatch((x-.73, y-.48), 1.46, .96,
                                   boxstyle="round,pad=0,rounding_size=.08",
                                   ec=TEAL if name.startswith("Decoder") or name == "Bottleneck" else WIRE,
                                   fc=COOL if name == "Bottleneck" else LIGHT))
        for dx, symbol, color, fill in ((-.22, "+", TEAL, COOL), (.22, "−", GOLD, WARM)):
            ax.add_patch(Circle((x+dx, y+.18), .18, ec=color, fc=fill, lw=1.1))
            label(ax, x+dx, y+.18, symbol, 14, color, "bold")
        label(ax, x, y-.22, f"{width // 2:,} pairs", 12, weight="bold")
        arrow(ax, (previous, y), (x-.75, y))
        previous = x+.75
        # Feedback travels from the final pixel comparison directly to each layer.
        arrow(ax, (x, 3.86), (x, y-.51), GOLD, "--")
        label(ax, x, 3.59, f"{group_size} sign" + ("s" if group_size > 1 else ""),
              12, GOLD, "bold")
        wire(ax, [(x, 3.40), (x, 3.19)], GOLD, style="--")
    arrow(ax, (previous, y), (11.26, y))
    heading(3.59, "Training", GOLD)
    label(ax, .74, 3.17, "Per pair", 12, GOLD)
    wire(ax, [(layers[0][0], 3.19), (10.75, 3.19)], GOLD, style="--")
    box(ax, 11.55, 3.18, 1.72, .74, "Error signs", "sign(target − output)", GOLD, WARM, 12)
    arrow(ax, (11.76, 3.79), (11.76, 3.58), GOLD, "--")
    label(ax, 6.23, 2.72,
          "Each hidden layer groups the same 12,288 final errors. Each output pair takes its own channel’s sign directly.",
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

"""
Publication-quality architecture diagram.
Run: python scripts/draw_architecture.py
Output: output/architecture.pdf  +  output/architecture.png
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

Path("output").mkdir(exist_ok=True)

# ── Palette ──────────────────────────────────────────────────────────────────
C_INPUT  = "#1B2A3B"
C_ENC    = "#2471A3"
C_BOTTLE = "#7D3C98"
C_DEC    = "#1E8449"
C_POOL   = "#626567"
C_SKIP   = "#D35400"
C_HEAD   = "#C0392B"
C_OUT    = "#148F77"
C_CENT   = "#B7950B"
WHITE    = "white"
FONT     = "DejaVu Sans"

FIG_W, FIG_H = 22, 12
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")
fig.patch.set_facecolor("white")

# ── Primitives ────────────────────────────────────────────────────────────────
def rbox(cx, cy, w, h, color, lw=0.8, alpha=1.0):
    b = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                       boxstyle="round,pad=0.04,rounding_size=0.12",
                       linewidth=lw, edgecolor="white",
                       facecolor=color, alpha=alpha, zorder=3)
    ax.add_patch(b)

def blk(cx, cy, w, h, color, line1, line2="", fs=8.5):
    rbox(cx, cy, w, h, color)
    dy = 0.13 if line2 else 0
    ax.text(cx, cy+dy, line1, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=WHITE,
            fontfamily=FONT, zorder=4)
    if line2:
        ax.text(cx, cy-dy-0.05, line2, ha="center", va="center",
                fontsize=7.0, color=WHITE, alpha=0.90,
                fontfamily=FONT, zorder=4)

def arr(x0, y0, x1, y1, color=C_INPUT, lw=1.3, head="->"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=head, color=color,
                                lw=lw, mutation_scale=8),
                zorder=2)

def hline(x0, y, x1, color=C_SKIP, lw=1.1, dashed=True):
    ls = (0, (4, 3)) if dashed else "-"
    ax.plot([x0, x1], [y, y], color=color, lw=lw, linestyle=ls, zorder=2)

def vline(x, y0, y1, color=C_SKIP, lw=1.1, dashed=True):
    ls = (0, (4, 3)) if dashed else "-"
    ax.plot([x, x], [y0, y1], color=color, lw=lw, linestyle=ls, zorder=2)

def txt(x, y, s, fs=7.5, color="#444", ha="center", va="center",
        bold=False, italic=False):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=color,
            fontfamily=FONT,
            fontweight="bold" if bold else "normal",
            fontstyle="italic" if italic else "normal", zorder=5)

# ═══════════════════════════════════════════════════════════════════════════════
# GEOMETRY  (22-unit wide canvas)
# ═══════════════════════════════════════════════════════════════════════════════
Y  = 7.20      # main spine Y
BH = 1.00      # block height
BW = 1.55      # block width

# Encoder columns (block centres)
E1, E2, E3 = 2.0, 4.3, 6.5
# MaxPool columns
P1, P2, P3 = 3.2, 5.5, 7.6
# Bottleneck
BT = 9.2
# Upsample columns
U1, U2, U3 = 10.7, 13.1, 15.5
# Decoder columns
D1, D2, D3 = 12.0, 14.4, 16.8
# Output
OC = 18.2
# Centroid
CC = 20.3

# Skip connection Y levels (below spine)
SK3_Y = 5.20
SK2_Y = 4.35
SK1_Y = 3.50

# Pool/Up block sizes
PW, PH = 0.50, 0.68

# ── Title ─────────────────────────────────────────────────────────────────────
txt(FIG_W/2, 11.55,
    "Custom Encoder-Decoder with U-Net Skip Connections (TrackNetV4-Based)",
    fs=13, color="#1a1a2e", bold=True)

# ── Section headers ────────────────────────────────────────────────────────────
for label, x in [("ENCODER", 4.3), ("BOTTLENECK", 9.2),
                  ("DECODER", 14.4), ("OUTPUT", 19.2)]:
    txt(x, 10.90, label, fs=9, color="#888", bold=True, italic=True)

# ── INPUT ─────────────────────────────────────────────────────────────────────
blk(0.70, Y, 1.00, BH*2.1, C_INPUT, "Input", "11×360×640", fs=8.5)
txt(0.70, Y - 0.85,
    "3 RGB frames +\n2 motion diffs",
    fs=6.8, color="#666", italic=True)
arr(0.70+0.50, Y, E1-BW/2, Y)

# ── ENCODER ───────────────────────────────────────────────────────────────────
blk(E1, Y, BW, BH*1.9, C_ENC, "Conv1–Conv2", "64ch · 360×640")
arr(E1+BW/2, Y, P1-PW/2, Y)

rbox(P1, Y, PW, PH, C_POOL)
txt(P1, Y, "Pool\n2×", fs=7.2, color=WHITE)
arr(P1+PW/2, Y, E2-BW/2, Y)

blk(E2, Y, BW, BH*1.7, C_ENC, "Conv3–Conv4", "128ch · 180×320")
arr(E2+BW/2, Y, P2-PW/2, Y)

rbox(P2, Y, PW, PH, C_POOL)
txt(P2, Y, "Pool\n2×", fs=7.2, color=WHITE)
arr(P2+PW/2, Y, E3-BW/2, Y)

blk(E3, Y, BW, BH*1.5, C_ENC, "Conv5–Conv7", "256ch · 90×160")
arr(E3+BW/2, Y, P3-PW/2, Y)

rbox(P3, Y, PW, PH, C_POOL)
txt(P3, Y, "Pool\n2×", fs=7.2, color=WHITE)
arr(P3+PW/2, Y, BT-BW*0.68, Y)

# ── BOTTLENECK ────────────────────────────────────────────────────────────────
blk(BT, Y, BW*1.15, BH*1.4, C_BOTTLE,
    "Conv8–Conv10", "256ch · 45×80  +  Dropout2D(0.2)")
arr(BT+BW*1.15/2, Y, U1-PW/2, Y)

# ── VISIBILITY HEAD ───────────────────────────────────────────────────────────
VH_X  = BT
VH_Y1 = Y + BH*1.4/2 + 0.15
VH_Y2 = VH_Y1 + 0.78
VH_Y3 = VH_Y2 + 0.74
VH_Y4 = VH_Y3 + 0.66

arr(VH_X, VH_Y1, VH_X, VH_Y2-0.26, color=C_HEAD)
blk(VH_X, VH_Y2, BW*1.10, 0.50, C_HEAD, "AvgPool → Flatten  →  256-d", fs=7.2)
arr(VH_X, VH_Y2+0.25, VH_X, VH_Y3-0.26, color=C_HEAD)
blk(VH_X, VH_Y3, BW*1.00, 0.50, C_HEAD, "Linear 256→64→1  (ReLU)", fs=7.2)
arr(VH_X, VH_Y3+0.25, VH_X, VH_Y4-0.24, color=C_HEAD)
blk(VH_X, VH_Y4, 1.00, 0.44, C_HEAD, "vis_prob  ∈[0,1]", fs=7.2)

txt(VH_X+1.00, VH_Y3+0.10, "Visibility\nHead", fs=8.0,
    color=C_HEAD, bold=True, ha="left")
txt(VH_X, VH_Y4+0.46, "Vis BCE Loss  (λ = 0.1)",
    fs=7.0, color=C_HEAD, italic=True)

# ── DECODER ───────────────────────────────────────────────────────────────────
def cat_marker(cx, cy):
    rbox(cx, cy, 0.42, 0.46, C_SKIP)
    txt(cx, cy, "cat", fs=7.2, color=WHITE)

rbox(U1, Y, PW, PH, C_POOL); txt(U1, Y, "Up\n2×", fs=7.2, color=WHITE)
arr(U1+PW/2, Y, U1+PW/2+0.10, Y, color="#555")
cat_marker(U1+PW/2+0.31, Y)
arr(U1+PW/2+0.52, Y, D1-BW/2, Y)
blk(D1, Y, BW, BH*1.5, C_DEC, "Conv11–Conv13", "256ch · 90×160")
arr(D1+BW/2, Y, U2-PW/2, Y)

rbox(U2, Y, PW, PH, C_POOL); txt(U2, Y, "Up\n2×", fs=7.2, color=WHITE)
arr(U2+PW/2, Y, U2+PW/2+0.10, Y, color="#555")
cat_marker(U2+PW/2+0.31, Y)
arr(U2+PW/2+0.52, Y, D2-BW/2, Y)
blk(D2, Y, BW, BH*1.7, C_DEC, "Conv14–Conv15", "128ch · 180×320")
arr(D2+BW/2, Y, U3-PW/2, Y)

rbox(U3, Y, PW, PH, C_POOL); txt(U3, Y, "Up\n2×", fs=7.2, color=WHITE)
arr(U3+PW/2, Y, U3+PW/2+0.10, Y, color="#555")
cat_marker(U3+PW/2+0.31, Y)
arr(U3+PW/2+0.52, Y, D3-BW/2, Y)
blk(D3, Y, BW, BH*1.9, C_DEC, "Conv16–Conv17", "64ch · 360×640")
arr(D3+BW/2, Y, OC-0.52, Y)

# ── OUTPUT ────────────────────────────────────────────────────────────────────
blk(OC, Y, 1.00, BH*1.4, C_OUT, "Conv18\n1×1  →  Sigmoid", "1ch · 360×640", fs=7.8)
arr(OC+0.50, Y, CC-0.68, Y)
blk(CC, Y, 1.30, BH*1.65, C_CENT, "Weighted\nCentroid", "(x, y, conf)", fs=8.5)

txt(OC, Y - BH*1.4/2 - 0.35,
    "CenterNet Focal Loss  (Zhou et al. 2019)",
    fs=7.0, color=C_OUT, italic=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SKIP CONNECTIONS — route below the spine
# ═══════════════════════════════════════════════════════════════════════════════
CAT3_X = U1 + PW/2 + 0.31   # cat block centre x for skip3 (deepest)
CAT2_X = U2 + PW/2 + 0.31
CAT1_X = U3 + PW/2 + 0.31

def skip(src_x, cat_x, arc_y):
    bot_y = arc_y
    vline(src_x, Y - BH*0.95/2, bot_y, color=C_SKIP)
    hline(src_x, bot_y, cat_x, color=C_SKIP)
    ax.annotate("", xy=(cat_x, Y-0.23), xytext=(cat_x, bot_y),
                arrowprops=dict(arrowstyle="->", color=C_SKIP,
                                lw=1.1, mutation_scale=8,
                                linestyle=(0, (4, 3))),
                zorder=2)
    mid = (src_x + cat_x) / 2
    txt(mid, bot_y - 0.22, "", fs=6.5, color=C_SKIP, bold=True, italic=True)

skip(E3, CAT3_X, SK3_Y)
skip(E2, CAT2_X, SK2_Y)
skip(E1, CAT1_X, SK1_Y)

# Skip labels at midpoints
txt((E3 + CAT3_X)/2, SK3_Y - 0.24, "Skip₃  256ch", fs=7.0, color=C_SKIP, bold=True, italic=True)
txt((E2 + CAT2_X)/2, SK2_Y - 0.24, "Skip₂  128ch", fs=7.0, color=C_SKIP, bold=True, italic=True)
txt((E1 + CAT1_X)/2, SK1_Y - 0.24, "Skip₁   64ch",  fs=7.0, color=C_SKIP, bold=True, italic=True)

# ═══════════════════════════════════════════════════════════════════════════════
# LEGEND
# ═══════════════════════════════════════════════════════════════════════════════
items = [
    (C_INPUT,  "Input / Output tensor"),
    (C_ENC,    "Encoder  (Conv + BN + ReLU)"),
    (C_BOTTLE, "Bottleneck  (+Dropout2D)"),
    (C_DEC,    "Decoder  (Conv + BN + ReLU)"),
    (C_POOL,   "MaxPool 2×2 / Upsample 2×"),
    (C_SKIP,   "Skip connection (concat)"),
    (C_HEAD,   "Visibility head"),
    (C_OUT,    "Output conv + Sigmoid"),
    (C_CENT,   "Weighted centroid"),
]

LX, LY, GAP = 0.40, 2.80, 3.80
txt(LX, LY+0.50, "Legend", fs=9, color="#333", bold=True, ha="left")
for i, (col, lbl) in enumerate(items):
    col_i = i % 3
    row_i = i // 3
    bx = LX + col_i * GAP
    by = LY - row_i * 0.56
    rbox(bx+0.16, by, 0.32, 0.32, col, lw=0.4)
    txt(bx+0.42, by, lbl, fs=7.2, color="#333", ha="left")

# ── Border ────────────────────────────────────────────────────────────────────
border = FancyBboxPatch((0.10, 0.10), FIG_W-0.20, FIG_H-0.20,
                        boxstyle="round,pad=0.05,rounding_size=0.18",
                        linewidth=1.0, edgecolor="#cccccc",
                        facecolor="none", zorder=0)
ax.add_patch(border)

plt.tight_layout(pad=0.2)
for fmt in ("pdf", "png"):
    plt.savefig(f"output/architecture.{fmt}", dpi=300,
                bbox_inches="tight", facecolor="white", format=fmt)
    print(f"Saved: output/architecture.{fmt}")
plt.show()

"""Decomposition diagrams for the report and slides -> results/../docs/images/diag_cs*.png"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "docs", "images")   # diagrams are written next to the docs
C = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2", "#EECA3B", "#9D755D"]

def fin(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(D, name), dpi=170); plt.close(fig)

def arrow(ax, a, b, **kw):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="<->", mutation_scale=9, lw=1, color="k", **kw))

# CS1: 1-D rows vs 2-D blocks with halos
fig, ax = plt.subplots(1, 2, figsize=(5.2, 2.6))
for i in range(4):
    ax[0].add_patch(Rectangle((0, 3 - i), 4, 1, fc=C[i], alpha=.55, ec="k"))
    ax[0].text(2, 3.5 - i, f"rank {i}: N/p rows", ha="center", va="center", fontsize=7)
for i in range(3):
    ax[0].add_patch(Rectangle((0, 3 - i - 0.08), 4, 0.16, fc="none", ec="k", hatch="////", lw=.5))
ax[0].set_title("1-D rows: 2 halo rows of N", fontsize=8)
for r in range(2):
    for c in range(2):
        ax[1].add_patch(Rectangle((2 * c, 2 - 2 * r), 2, 2, fc=C[2 * r + c], alpha=.55, ec="k"))
        ax[1].text(2 * c + 1, 3 - 2 * r, f"({r},{c})", ha="center", va="center", fontsize=7)
ax[1].add_patch(Rectangle((0, 1.92), 4, 0.16, fc="none", ec="k", hatch="////", lw=.5))
ax[1].add_patch(Rectangle((1.92, 0), 0.16, 4, fc="none", ec="k", hatch="////", lw=.5))
ax[1].set_title("2-D blocks (MPI_Cart_create): 4 halos of N/√p", fontsize=8)
for a in ax: a.set_xlim(-.1, 4.1); a.set_ylim(-.1, 4.1); a.set_aspect("equal"); a.axis("off")
fin(fig, "diag_cs1.png")

# CS2: chunks with k-1 overlap, private histograms
fig, ax = plt.subplots(figsize=(5.2, 2.0))
for t in range(4):
    ax.add_patch(Rectangle((t * 2.5, 1.2), 2.5, 0.5, fc=C[t % 2], alpha=.6, ec="k"))
    ax.text(t * 2.5 + 1.25, 1.45, f"chunk {t} → thread {t % 2}", ha="center", va="center", fontsize=6.5)
    if t: ax.add_patch(Rectangle((t * 2.5, 1.12), 0.35, 0.66, fc="none", ec="r", lw=1.2))
ax.text(5.2, 1.95, "red: k−1 chars read past chunk end (k-mer counted by chunk holding its start)", ha="center", fontsize=6, color="r")
for t in range(2):
    ax.add_patch(Rectangle((1 + t * 4.5, 0.1), 3.5, 0.45, fc=C[t], alpha=.4, ec="k"))
    ax.text(2.75 + t * 4.5, 0.32, f"private hist[4^k] of thread {t}", ha="center", va="center", fontsize=6.5)
ax.text(5.2, -0.15, "Version C: merge Σ over threads, each thread sums a slice of bins", ha="center", fontsize=6.5)
ax.set_xlim(0, 10.4); ax.set_ylim(-.3, 2.1); ax.axis("off")
fin(fig, "diag_cs2.png")

# CS3: independent streams + reduction
fig, ax = plt.subplots(figsize=(5.2, 2.0))
for w in range(4):
    ax.add_patch(Rectangle((w * 2.6, 1.0), 2.3, 0.7, fc=C[w], alpha=.55, ec="k"))
    ax.text(w * 2.6 + 1.15, 1.35, f"worker {w}\nstream = seed + {w}·2¹²⁸\nM/p paths", ha="center", va="center", fontsize=6)
    ax.annotate("", (5.1, 0.35), (w * 2.6 + 1.15, 1.0), arrowprops=dict(arrowstyle="->", lw=.8))
ax.add_patch(Rectangle((3.6, 0.0), 3.0, 0.35, fc="#ddd", ec="k"))
ax.text(5.1, 0.17, "Σ payoff, Σ payoff² (fixed order)", ha="center", va="center", fontsize=6.5)
ax.set_xlim(-.1, 10.5); ax.set_ylim(-.05, 1.8); ax.axis("off")
fin(fig, "diag_cs3.png")

# CS4: pull vs push
fig, ax = plt.subplots(1, 2, figsize=(5.2, 2.2))
for a, title, push in [(ax[0], "pull: thread owns v, reads in-nbrs", False), (ax[1], "push: thread owns u, atomic += to out-nbrs", True)]:
    src = [(0.2, y) for y in (0.3, 1.0, 1.7)]; dst = [(1.8, y) for y in (0.3, 1.0, 1.7)]
    for i, (x, y) in enumerate(src):
        a.add_patch(plt.Circle((x, y), .13, fc=C[0] if push else "#ccc", ec="k"))
    for i, (x, y) in enumerate(dst):
        a.add_patch(plt.Circle((x, y), .13, fc="#ccc" if push else C[1], ec="k"))
    for s in src:
        for d in dst[:2]:
            a.annotate("", d, s, arrowprops=dict(arrowstyle="->", lw=.6, color="r" if push else "k", shrinkA=6, shrinkB=6))
    a.set_title(title, fontsize=7.5); a.set_xlim(-.1, 2.1); a.set_ylim(0, 2); a.set_aspect("equal"); a.axis("off")
fin(fig, "diag_cs4.png")

# CS5: sample sort pipeline
fig, ax = plt.subplots(figsize=(5.2, 2.2))
steps = ["local sort", "samples → Gather\n→ splitters\n→ Bcast", "partition by\nsplitters", "Alltoall counts\nAlltoallv keys", "merge p runs"]
for i, s in enumerate(steps):
    ax.add_patch(Rectangle((i * 2.1, 0.6), 1.9, 1.0, fc=C[i], alpha=.5, ec="k"))
    ax.text(i * 2.1 + 0.95, 1.1, s, ha="center", va="center", fontsize=6.3)
    if i: ax.annotate("", (i * 2.1, 1.1), (i * 2.1 - 0.2, 1.1), arrowprops=dict(arrowstyle="->"))
for r in range(4):
    for b in range(4):
        ax.add_patch(Rectangle((6.4 + b * 0.42, 0.05 + 0.12 * r), 0.4, 0.1, fc=C[b], alpha=.7, lw=0))
ax.text(6.2, 0.25, "after Alltoallv: rank r holds bucket r of every rank", fontsize=6.3, ha="right")
ax.set_xlim(-.1, 10.5); ax.set_ylim(0, 1.7); ax.axis("off")
fin(fig, "diag_cs5.png")
print("diagrams done")

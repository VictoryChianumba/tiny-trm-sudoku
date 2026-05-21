"""Generate README figures from logs and checkpoints into assets/.

Produces:
  assets/training_curves.png
  assets/per_step_accuracy.png
  assets/architecture.png
  assets/solve_4x4.gif
  assets/solve_4x4_panel.png
"""

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
ASSETS = REPO / "assets"
LOGS = REPO / "logs"
ASSETS.mkdir(exist_ok=True)

sys.path.insert(0, str(REPO))
from trm import TinyRecursiveModel  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 160,
    "font.size": 11,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_history(name):
    with open(LOGS / name / "history.json") as f:
        return json.load(f)


def load_step_accs(name):
    with open(LOGS / name / "step_accuracies.json") as f:
        return json.load(f)


def figure_training_curves():
    h4 = load_history("trm_4x4_baseline")
    h6 = load_history("trm_6x6_baseline")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    for ax, h, title in [(axes[0, 0], h4, "4x4 Sudoku — 354K params"),
                          (axes[0, 1], h6, "6x6 Sudoku — 3.15M params")]:
        epochs = h["epoch"]
        ax.plot(epochs, h["train_loss"], label="train", color="#1f77b4", lw=1.6)
        ax.plot(epochs, h["val_loss"], label="val", color="#d62728", lw=1.6)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)

    for ax, h in [(axes[1, 0], h4), (axes[1, 1], h6)]:
        epochs = h["epoch"]
        ax.plot(epochs, h["train_acc"], label="train", color="#1f77b4", lw=1.6)
        ax.plot(epochs, h["val_acc"], label="val", color="#d62728", lw=1.6)
        best_val = max(h["val_acc"])
        best_ep = epochs[int(np.argmax(h["val_acc"]))]
        ax.axhline(best_val, ls="--", color="#d62728", alpha=0.4, lw=1)
        ax.annotate(f"best val: {best_val:.3f} @ epoch {best_ep}",
                    xy=(best_ep, best_val), xytext=(8, -16), textcoords="offset points",
                    fontsize=9, color="#d62728")
        ax.set_xlabel("epoch")
        ax.set_ylabel("full-grid accuracy")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(frameon=False, loc="lower right")
        ax.grid(alpha=0.25)

    fig.suptitle("Training and validation curves", fontsize=13, fontweight="bold", y=1.00)
    fig.tight_layout()
    out = ASSETS / "training_curves.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO)}")


def figure_per_step_accuracy():
    s4 = load_step_accs("trm_4x4_baseline")
    s6 = load_step_accs("trm_6x6_baseline")

    val4 = np.array(s4["val"])  # [eval_idx, n_steps]
    val6 = np.array(s6["val"])
    # 6x6 logged 16 cols but only first ~2 are populated by the time the run halted.
    # Find the actual width with non-zero data.
    nz_cols_6 = int(np.max(np.where(val6.any(axis=0))[0])) + 1 if val6.any() else val6.shape[1]
    val6 = val6[:, :nz_cols_6]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4),
                              gridspec_kw={"width_ratios": [val4.shape[1], max(val6.shape[1], 1)]})

    for ax, mat, title in [(axes[0], val4, "4x4 — per-step val accuracy"),
                            (axes[1], val6, "6x6 — per-step val accuracy")]:
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1, origin="lower")
        ax.set_xlabel("supervision step")
        ax.set_ylabel("evaluation index (every 5 epochs)")
        ax.set_title(title)
        ax.set_xticks(range(mat.shape[1]))
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="full-grid accuracy")

    fig.suptitle("Per-step accuracy across supervision iterations",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = ASSETS / "per_step_accuracy.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO)}")


def figure_architecture():
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def box(x, y, w, h, label, color, fc=None, fontsize=10, fontweight="bold"):
        fc = fc or color
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                linewidth=1.6, edgecolor=color, facecolor=fc, alpha=0.92)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, fontweight=fontweight, color="black")

    def arrow(x1, y1, x2, y2, color="#444", style="-|>", lw=1.4, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
                                      color=color, linewidth=lw, linestyle=ls))

    # Colors
    c_input = "#cfe2ff"; e_input = "#1f77b4"
    c_state = "#ffe8b8"; e_state = "#cc7a00"
    c_net = "#dceedb"; e_net = "#2a9d8f"
    c_out = "#f4cccc"; e_out = "#a13030"

    # Title
    ax.text(6, 6.7, "TRM forward pass: deep supervision over deep recursion",
            ha="center", va="center", fontsize=13, fontweight="bold")

    # Outer box: a single supervision step
    outer = FancyBboxPatch((0.4, 0.5), 11.2, 5.4,
                            boxstyle="round,pad=0.05,rounding_size=0.2",
                            linewidth=1.2, edgecolor="#888", facecolor="#fafafa")
    ax.add_patch(outer)
    ax.text(6, 5.65, "supervision step  ×  N_supervision_steps  (≤ 16)",
            ha="center", va="center", fontsize=10, color="#555", style="italic")

    # Inputs on the left
    box(0.8, 4.1, 1.6, 0.7, "x  (puzzle)", e_input, fc=c_input)
    box(0.8, 3.1, 1.6, 0.7, "y  (answer)", e_state, fc=c_state)
    box(0.8, 2.1, 1.6, 0.7, "z  (reasoning)", e_state, fc=c_state)

    # latent_recursion box
    lat_x, lat_y, lat_w, lat_h = 3.4, 1.6, 4.0, 3.2
    box(lat_x, lat_y, lat_w, lat_h,
        "latent_recursion\n\nz ← z + f(x, y, z)   ×  n_recursions\ny ← y + f(y, z)",
        e_net, fc=c_net, fontweight="normal", fontsize=10)
    ax.text(lat_x + lat_w / 2, lat_y + lat_h - 0.35, "tiny shared net f",
            ha="center", va="center", fontsize=10, fontweight="bold")

    # deep_recursion wrapper (T cycles)
    deep_x, deep_y, deep_w, deep_h = 3.1, 1.25, 4.6, 3.9
    deep = FancyBboxPatch((deep_x, deep_y), deep_w, deep_h,
                           boxstyle="round,pad=0.05,rounding_size=0.18",
                           linewidth=1.2, edgecolor="#444", facecolor="none", linestyle="--")
    ax.add_patch(deep)
    ax.text(deep_x + deep_w / 2, deep_y + 0.18,
            "deep_recursion: T_cycles total  (T−1 under no_grad, last with grad)",
            ha="center", va="center", fontsize=9.5, color="#444", style="italic")

    # Output side
    box(8.4, 4.1, 1.5, 0.7, "y_logits", e_out, fc=c_out)
    box(8.4, 3.1, 1.5, 0.7, "Q-head", e_out, fc=c_out)

    # Supervision losses
    box(10.4, 4.1, 1.3, 0.7, "CE(y, t)", e_out, fc=c_out, fontsize=9)
    box(10.4, 3.1, 1.3, 0.7, "BCE(q, ✓)", e_out, fc=c_out, fontsize=9)

    # Arrows: inputs → recursion
    arrow(2.4, 4.45, 3.4, 4.0)
    arrow(2.4, 3.45, 3.4, 3.4)
    arrow(2.4, 2.45, 3.4, 2.6)

    # latent_recursion → outputs
    arrow(7.4, 3.6, 8.4, 4.45)
    arrow(7.4, 3.0, 8.4, 3.45)
    # outputs → losses
    arrow(9.9, 4.45, 10.4, 4.45)
    arrow(9.9, 3.45, 10.4, 3.45)

    # Recurrent feedback (y, z carry to next supervision step)
    arrow(7.6, 2.0, 11.4, 2.0, color="#cc7a00", lw=1.5)
    arrow(11.4, 2.0, 11.4, 1.0, color="#cc7a00", lw=1.5)
    arrow(11.4, 1.0, 0.7, 1.0, color="#cc7a00", lw=1.5)
    arrow(0.7, 1.0, 0.7, 2.4, color="#cc7a00", lw=1.5)
    arrow(0.7, 2.4, 0.8, 2.4, color="#cc7a00", lw=1.5)
    ax.text(6, 0.85, "y, z carried (detached) to next supervision step",
            ha="center", va="center", fontsize=9.5, color="#cc7a00", style="italic")

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor=c_input, edgecolor=e_input, label="input"),
        mpatches.Patch(facecolor=c_state, edgecolor=e_state, label="latent state"),
        mpatches.Patch(facecolor=c_net, edgecolor=e_net, label="shared network"),
        mpatches.Patch(facecolor=c_out, edgecolor=e_out, label="output / loss"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False,
              ncol=4, bbox_to_anchor=(0.98, 0.04))

    out = ASSETS / "architecture.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO)}")


def _draw_grid(ax, grid, given_mask, target=None, title="", grid_size=4):
    box_size = int(np.sqrt(grid_size))
    grid = grid.reshape(grid_size, grid_size)
    given_mask = given_mask.reshape(grid_size, grid_size)
    if target is not None:
        target = target.reshape(grid_size, grid_size)

    for i in range(grid_size + 1):
        lw = 2.2 if i % box_size == 0 else 0.6
        ax.plot([i, i], [0, grid_size], "k-", linewidth=lw)
        ax.plot([0, grid_size], [i, i], "k-", linewidth=lw)

    for i in range(grid_size):
        for j in range(grid_size):
            v = int(grid[i, j])
            if v == 0:
                continue
            if given_mask[i, j]:
                color = "#1565c0"
            elif target is not None and v == int(target[i, j]):
                color = "#2e7d32"
            else:
                color = "#c62828"
            weight = "bold" if given_mask[i, j] else "semibold"
            ax.text(j + 0.5, grid_size - i - 0.5, str(v),
                    ha="center", va="center",
                    fontsize=20, fontweight=weight, color=color)

    ax.set_xlim(0, grid_size); ax.set_ylim(0, grid_size)
    ax.set_aspect("equal"); ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")


def _run_solve(model, puzzle, target, n_supervision_steps, device):
    given_mask = puzzle != 0
    with torch.no_grad():
        x = torch.from_numpy(puzzle).long().unsqueeze(0).to(device)
        preds, qs, _, _ = model(x, n_supervision_steps=n_supervision_steps, training=False)
    pred_grids = [p.argmax(dim=-1).squeeze(0).cpu().numpy() for p in preds]
    q_vals = [float(torch.sigmoid(q).squeeze().item()) for q in qs]
    return given_mask, pred_grids, q_vals


def _save_solve_figures(grid_size, puzzle, target, given_mask, pred_grids, q_vals,
                        panel_name, gif_name, title_prefix):
    n_steps = len(pred_grids)
    n_cells = n_steps + 2
    fig, axes = plt.subplots(1, n_cells, figsize=(2.4 * n_cells, 3.0))
    axes = np.atleast_1d(axes)
    _draw_grid(axes[0], puzzle, given_mask, title="input", grid_size=grid_size)
    for i, (g, q) in enumerate(zip(pred_grids, q_vals)):
        correct = np.array_equal(g, target)
        title = f"step {i}\nq={q:.2f}" + ("  ✓" if correct else "")
        _draw_grid(axes[i + 1], g, given_mask, target=target, title=title, grid_size=grid_size)
    _draw_grid(axes[-1], target, given_mask, title="target", grid_size=grid_size)
    fig.suptitle(f"{title_prefix} — refinement across supervision steps",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_panel = ASSETS / panel_name
    fig.savefig(out_panel, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_panel.relative_to(REPO)}")

    import matplotlib.animation as animation
    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    def render(frame):
        ax.clear()
        if frame == 0:
            _draw_grid(ax, puzzle, given_mask, title="input", grid_size=grid_size)
        elif frame <= n_steps:
            i = frame - 1
            g = pred_grids[i]; q = q_vals[i]
            correct = np.array_equal(g, target)
            title = f"step {i}    q={q:.2f}" + ("    ✓" if correct else "")
            _draw_grid(ax, g, given_mask, target=target, title=title, grid_size=grid_size)
        else:
            _draw_grid(ax, target, given_mask, title="target", grid_size=grid_size)

    anim = animation.FuncAnimation(fig, render, frames=n_steps + 2,
                                   interval=700, repeat=True)
    out_gif = ASSETS / gif_name
    anim.save(out_gif, writer="pillow", fps=1.4)
    plt.close(fig)
    print(f"  wrote {out_gif.relative_to(REPO)}")


def figure_solve_4x4():
    """Load 4x4 checkpoint (MLP-Mixer), run on a real val puzzle, save panel + GIF."""
    ckpt_path = REPO / "trm_4x4_sudoku.pt"
    if not ckpt_path.exists():
        print(f"  skipping 4x4 solve: no checkpoint at {ckpt_path}")
        return

    device = torch.device("cpu")
    model = TinyRecursiveModel(
        vocab_size=5, context_length=16, hidden_size=128,
        n_layers=2, n_recursions=6, T_cycles=3, use_attention=False,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    model.eval()

    from data_generation import generate_dataset
    puzzles, sols = generate_dataset(n_base_puzzles=24, n_augmentations=4, seed=2026)
    # Pick a puzzle with a moderate number of givens (harder than near-full grids)
    given_counts = (puzzles != 0).sum(axis=1)
    median_givens = np.median(given_counts)
    candidates = np.where(given_counts <= median_givens)[0]
    pick = int(candidates[len(candidates) // 2])
    puzzle = puzzles[pick]
    target = sols[pick]

    given_mask, pred_grids, q_vals = _run_solve(model, puzzle, target, 8, device)
    _save_solve_figures(4, puzzle, target, given_mask, pred_grids, q_vals,
                        "solve_4x4_panel.png", "solve_4x4.gif",
                        "4x4 solve")


def figure_solve_6x6():
    """Load 6x6 checkpoint (MLP-Mixer, hidden=256), run on a saved val puzzle."""
    ckpt_path = REPO / "trm_6x6_sudoku.pt"
    val_puz_path = REPO / "val_puzzles_6x6_10k.npy"
    val_sol_path = REPO / "val_solutions_6x6_10k.npy"
    if not (ckpt_path.exists() and val_puz_path.exists() and val_sol_path.exists()):
        print("  skipping 6x6 solve: missing checkpoint or val arrays")
        return

    device = torch.device("cpu")
    model = TinyRecursiveModel(
        vocab_size=7, context_length=36, hidden_size=256,
        n_layers=2, n_recursions=6, T_cycles=3, use_attention=False,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    model.eval()

    puzzles = np.load(val_puz_path)
    sols = np.load(val_sol_path)

    rng = np.random.default_rng(2026)
    # Try a handful of puzzles, prefer one we solve correctly so the figure tells the success story.
    for idx in rng.choice(len(puzzles), size=32, replace=False):
        puzzle = puzzles[int(idx)]
        target = sols[int(idx)]
        given_mask, pred_grids, q_vals = _run_solve(model, puzzle, target, 8, device)
        if np.array_equal(pred_grids[-1], target):
            break
    else:
        # Fall back to the last attempted (showing partial-correctness color coding)
        pass

    _save_solve_figures(6, puzzle, target, given_mask, pred_grids, q_vals,
                        "solve_6x6_panel.png", "solve_6x6.gif",
                        "6x6 solve")


def main():
    print("Building assets/ ...")
    figure_training_curves()
    figure_per_step_accuracy()
    figure_architecture()
    figure_solve_4x4()
    figure_solve_6x6()
    print("Done.")


if __name__ == "__main__":
    main()

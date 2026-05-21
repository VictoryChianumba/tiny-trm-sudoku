"""Generate results/small_sudoku_success.png — a 4x4 puzzle the model solves exactly.

Reuses the model loading and rendering helpers from build_assets.py. Searches a
batch of held-out 4x4 puzzles for one the trained model gets fully correct, and
saves a refinement panel (input -> per-step predictions -> target).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
RESULTS.mkdir(exist_ok=True)

sys.path.insert(0, str(REPO))
from trm import TinyRecursiveModel  # noqa: E402
from build_assets import _draw_grid, _run_solve, plt as _plt  # noqa: E402,F401
from data_generation import generate_dataset  # noqa: E402


def main():
    ckpt_path = REPO / "trm_4x4_sudoku.pt"
    device = torch.device("cpu")
    model = TinyRecursiveModel(
        vocab_size=5, context_length=16, hidden_size=128,
        n_layers=2, n_recursions=6, T_cycles=3, use_attention=False,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    model.eval()

    puzzles, sols = generate_dataset(n_base_puzzles=64, n_augmentations=4, seed=7)
    given_counts = (puzzles != 0).sum(axis=1)
    # Prefer harder puzzles (fewer givens) that the model still solves exactly.
    order = np.argsort(given_counts)
    chosen = None
    for idx in order:
        puzzle, target = puzzles[idx], sols[idx]
        given_mask, pred_grids, q_vals = _run_solve(model, puzzle, target, 8, device)
        if np.array_equal(pred_grids[-1], target):
            chosen = (puzzle, target, given_mask, pred_grids, q_vals, int(given_counts[idx]))
            break
    if chosen is None:
        raise SystemExit("No exactly-solved 4x4 puzzle found in sample.")

    puzzle, target, given_mask, pred_grids, q_vals, n_givens = chosen
    n_steps = len(pred_grids)
    n_cells = n_steps + 2
    fig, axes = plt.subplots(1, n_cells, figsize=(2.4 * n_cells, 3.0))
    axes = np.atleast_1d(axes)
    _draw_grid(axes[0], puzzle, given_mask, title=f"input ({n_givens} givens)", grid_size=4)
    for i, (g, q) in enumerate(zip(pred_grids, q_vals)):
        correct = np.array_equal(g, target)
        title = f"step {i}\nq={q:.2f}" + ("  ✓" if correct else "")
        _draw_grid(axes[i + 1], g, given_mask, target=target, title=title, grid_size=4)
    _draw_grid(axes[-1], target, given_mask, title="target", grid_size=4)
    fig.suptitle("4x4 success — model solves a held-out puzzle exactly (354K params, M1)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = RESULTS / "small_sudoku_success.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO)}  (givens={n_givens}, final q={q_vals[-1]:.2f})")


if __name__ == "__main__":
    main()

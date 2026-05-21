# 9×9 Sudoku: what failed, and why

**Short version:** full 9×9 Sudoku training did not complete on this machine. The
run ran out of memory on the M1's 8 GB unified memory before producing a single
epoch of history. Only `logs/trm_9x9_baseline/config.json` exists — there is no
`history.json`, no checkpoint, no figures.

## The configuration that was attempted

From `logs/trm_9x9_baseline/config.json`:

| field | value |
| ----- | ----- |
| model params | 1,955,083 |
| hidden size | 256 |
| layers | 2 |
| n_recursions | 6 |
| T_cycles | 3 |
| n_supervision_steps | 16 |
| batch size | 16 |
| train / val size | 10,000 / 1,000 |

## Why it OOM'd even though it has *fewer* params than 6×6

The 9×9 model (1.96M params) is actually smaller than the 6×6 model (3.15M),
so parameter count is not the problem. The problem is **activation memory during
the recursion unroll**, which scales with sequence length, not parameter count:

- A 9×9 grid is **81 tokens** vs **36** for 6×6 — a 2.25× longer sequence.
- The MLP-Mixer backbone (`use_attention=False`) mixes tokens with an MLP that
  takes the *whole sequence* as input, so its activations grow with sequence
  length on top of the per-token hidden width (256).
- Each `latent_recursion` applies the shared network `n_recursions = 6` times,
  and the last `deep_recursion` cycle retains the graph for backprop. With
  `batch_size = 16` and a 16-step supervision schedule, the peak activation
  footprint for 81 tokens exceeded the 8 GB budget.

The step-by-step trick in `train_step_by_step.py` (backprop + detach every
supervision step) is what makes 6×6 fit; it was not enough to bring 81-token
9×9 under the 8 GB ceiling at this batch size.

## What would make 9×9 tractable

In rough order of effort:

1. **More memory / a real GPU.** The honest fix: 16–24 GB lets this exact config
   run. This is the "would require remote GPU compute" boundary referenced in the
   README.
2. **Shrink the activation footprint locally** — `batch_size` down to 4–8,
   `n_recursions` down, or gradient checkpointing across the recursion unroll.
   These trade wall-clock and likely final accuracy for fitting in 8 GB.
3. **A bigger dataset.** Separate from memory: `train_9x9.py` only generates a
   small dataset by default (generation is slow), which is too small for a
   ~2M-param model regardless of whether it fits. See `docs/reproduction_notes.md`.

This is the intended takeaway of the repo: 4×4 and 6×6 are reproducible on an
8 GB M1; 9×9 is where local hardware stops being enough.

# Hardware notes: running TRM on an 8 GB M1

All experiments in this repo were run on a single machine:

- **Apple M1**, MacBook Pro (13-inch, 2020) — `MacBookPro17,1`
- **8 GB unified memory** (CPU and GPU share the same pool)
- PyTorch 2.7, **MPS** backend (`torch.device('mps' if torch.backends.mps.is_available() else 'cpu')`, see `build.py`)

The 8 GB unified-memory ceiling — not raw compute — is the constraint that
shapes what trains here.

## What fits, what doesn't

| size | tokens | params | status on 8 GB M1 |
| ---- | ------ | ------ | ----------------- |
| 4×4  | 16 | 354K  | Trains comfortably. ~91.8% val acc in tens of epochs. |
| 6×6  | 36 | 3.15M | Trains **only with the step-by-step trick** (below). Plateaus ~43.6%. |
| 9×9  | 81 | 1.96M | **OOM** — never completed an epoch. See `results/9x9_failure_notes.md`. |

## Why the limit is activation memory, not parameters

Parameter count is small everywhere (under 4M). What actually grows is the
**activation memory of the recursion unroll**:

- The forward pass applies the shared net `n_recursions` times per
  `latent_recursion`, wrapped in `T_cycles` of `deep_recursion`, all inside an
  outer loop of up to 16 supervision steps.
- With an MLP-Mixer backbone, token-mixing acts over the whole sequence, so
  activation size scales with sequence length: 16 → 36 → 81 tokens.
- Optimizer state (AdamW keeps two moments per parameter) and the EMA shadow
  copy (`build.py`) add fixed overhead on top.

So the jump from 6×6 to 9×9 (36 → 81 tokens, 2.25×) is what pushes peak memory
over 8 GB, even though 9×9 has fewer parameters than 6×6.

## The trick that buys headroom

`train_step_by_step.py` makes 6×6 fit by never holding the full supervision graph
in memory at once:

```python
for step_idx in range(n_supervision_steps):
    y, z, y_logits, q = model.forward_single_step(x_embedded, y, z)
    step_loss = (CE(y_logits, target) + BCE(q, is_correct)) / n_supervision_steps
    step_loss.backward(retain_graph=(step_idx < n_supervision_steps - 1))
    y, z = y.detach(), z.detach()   # carry state forward, drop its history
optimizer.step()                    # one update after all 16 steps
```

Each supervision step backprops and frees its graph immediately; only the
detached latent states `y, z` carry forward. Peak memory is therefore set by a
single step, not by all 16. This is enough for 36-token 6×6 but not for
81-token 9×9 at `batch_size=16`.

## Practical knobs if you're also on 8 GB

- Lower `batch_size` (16 → 8 → 4) — first thing to try for OOM.
- Lower `n_recursions` / `T_cycles` — fewer unrolled activations.
- Use the step-by-step loop (`train_step_by_step.py`), not the vanilla loop in
  `build.py`, for anything past 4×4.
- Expect MPS to be slower than a discrete GPU and to occasionally fall back to
  CPU for unsupported ops; watch for `PYTORCH_ENABLE_MPS_FALLBACK` warnings.

For 9×9 the realistic answer is more memory — see the "would require remote GPU"
note in `results/9x9_failure_notes.md`.

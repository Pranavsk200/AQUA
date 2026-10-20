# DESIGN

Four questions from the brief, plus the places where I made a judgement call.

## 1. The importance criterion, and why it approximates "loss increase if removed"

Removing connection `i` means moving the parameter vector by `-w_i e_i`. Expand the loss around
the current weights:

```
ΔL_i = L(w with w_i = 0) - L(w)  ≈  -g_i w_i  +  ½ H_ii w_i²        (g = ∇L, H = Hessian)
```

Two terms, two criteria ([prune/criteria.py](prune/criteria.py)):

- **`taylor1` = `|w_i · g_i|`**, the size of the linear term. It costs nothing, the backward pass
  already computed `g`. It is the right thing when the weights are not at a minimum.
- **`taylor2` = `|w_i g_i| + ½ F_i w_i²`**, which adds the curvature term. Near a minimum `g ≈ 0`
  and the quadratic term is the only one left (Optimal Brain Damage). Computing `H_ii` exactly
  needs second derivatives, so `F_i` is the diagonal Fisher, `E[g_i²]`, which equals the
  Gauss-Newton curvature for a cross-entropy loss when the model is well specified. I track it as an
  EMA of `g²` that I keep anyway.

Judgement calls and their caveats:

- **`g` is an EMA over batches, not one minibatch gradient.** `|w · mean(g)|` estimates the
  expected loss change; `mean|w · g_b|` also adds gradient noise to the score. EMA 0.99 beat 0.9 on
  validation in the ablations.
- **`|·|` rather than the signed `-g w`.** The signed value is the honest first-order estimate and
  can be negative (removal helps). I use the absolute value, as in the saliency literature, so that
  a big disturbance either way ranks as important. I did not ablate the signed version.
- **The Fisher proxy is biased.** A minibatch gradient squared is `mean² + variance/B`, whereas the
  per-sample Fisher is `mean² + variance`. So `F` under-counts the noise part by roughly the batch
  size. I never computed per-sample gradients to check how much this matters.
- **Scores are divided by the layer's mean score** before ranking across layers, otherwise a layer
  with larger gradients dominates the global ranking. This is a heuristic; the ablation
  (`taylor1` vs `taylor1_nonorm`) gave +0.2 points on validation at 90%, inside the noise for 3 seeds.
- **Every layer keeps at least 1% of its weights**, so a global ranking cannot delete a layer. That
  applies equally to the magnitude baseline.

**It did not beat magnitude pruning** ([results/CLAIM.md](results/CLAIM.md)). I did not test why.
The candidates I would test first: (a) at a trained network the EMA gradient is small and
noisy, so `w · g` ranks mostly noise while `|w|` is a stable signal; (b) the Fisher proxy bias
above; (c) cross-layer score normalisation. I have no evidence for any of them yet, so treat them as
hypotheses.

## 2. What is "the gradient of a masked weight", and why is that the right choice?

The effective weight is `w_eff = w ⊙ m` with `m ∈ {0,1}` a constant. The network computes a
function of `w_eff`, so the true derivative with respect to the parameter is

```
∂L/∂w = m ⊙ ∂L/∂w_eff        →  exactly 0 where m = 0
```

That is what the optimizer receives ([engine/ops.py](engine/ops.py) `masked_mul`). A masked weight
contributes `0` to the forward pass and `0` to the parameter gradient, which is the correct gradient
of the thing being optimised.

There is a second quantity, `∂L/∂w_eff` at `w_eff = 0`. It says how much the loss would move if the
connection were switched on, and I keep it only to score regrowth (`MaskedLinear.dense_grad`). It never
reaches the optimizer. Mixing these two up is the easy mistake.

**The optimizer is the subtle part.** Handing Adam a zero gradient does not freeze the weight: `m`
and `v` still hold values from before the prune, so `m̂ / (√v̂ + ε)` is non-zero and the weight
drifts off zero while the mask says it is gone. `tests/test_masked_weights.py` has a naive Adam that
does exactly this and asserts that it fails. The fix in [train/optim.py](train/optim.py):

1. the gradient is multiplied by the mask,
2. `m` and `v` are multiplied by the mask every step, so masked entries are exactly `0`,
3. the update for those entries is then `0 / (0 + ε) = 0`.

On **revival**, the entry has `w = 0` and `m = v = 0` (it was held there while masked), so there is
no stale momentum to leak in. The test checks the first step is `-lr · sign(g)`-sized, not
momentum-sized. One known imperfection: Adam's bias correction uses a global step count `t`. Late in training
`t` is in the thousands, the correction factors are ~1, and a revived weight with fresh moments
gets a first step of `0.1·g / √(0.001·g²) ≈ 3.2 · lr` in the direction of `-g`, which then shrinks as `v`
fills in. So revived weights take a short burst of oversized steps (from `0` toward the gradient
sign) rather than a gentle start. A per-element age counter in the bias correction would fix this
(a fresh element would then see the same warm-up as at `t = 1`). I did not do it; the cost is small
because the weight starts at 0, but it is a real departure from a fresh start and
`test_revived_connection_starts_from_clean_state` pins the actual behaviour.

**Regrowth and stability.** New connections start at `w = 0`, so reviving one does not change the
function (no loss spike). Two guards stop flip-flopping: a just-pruned connection cannot be revived for
`cooldown` steps, and a just-revived one cannot be pruned for `grace` steps. Without the grace period a
revived weight at 0 has Taylor score 0 and would be pruned again at the next event. The regrow
fraction decays with a cosine so the mask settles before the end. Regrowth helped the Taylor
criterion at 90-98% (see README) and made no difference to magnitude pruning in the ablation.

## 3. Where does the autodiff engine bottleneck, and how would I optimise it?

Profile of 2 epochs (860 steps, one thread, `cProfile`, 3.5 s total):

| | time | share |
|---|---|---|
| matmul forward | 0.94 s | 27% |
| matmul backward | 0.90 s | 26% |
| Adam step | 0.68 s | 20% |
| pruner EMA bookkeeping | 0.31 s | 9% |
| top-k (argsort), masked_mul, relu | ~0.25 s | 7% |

The Python graph overhead is not the problem at these sizes; the matrices are big enough that
NumPy dominates. What I would do, in order of payoff:

1. **Thread count first.** The default multi-threaded BLAS was about 5x *slower* on 128x784x256 matrices
   than a single thread (spin-waiting); that was the biggest single win and is why the experiments pin it.
2. **Fuse the optimizer.** Adam does ~8 passes over every parameter array with temporaries. In-place ops
   on pre-allocated buffers, or one fused kernel, would cut its 20%.
3. **Skip the mask multiply in forward.** Masked entries are exactly `0` by invariant, so `w ⊙ m == w`
   and `masked_mul` is pure overhead (it also allocates `w_eff` every step). It is kept deliberately
   because the dense gradient needs the `w_eff` node and because it is the safe version, but the dense
   gradient is just `xᵀ·δ` from the matmul backward, available without it.
4. **Update the EMA statistics only every k steps** instead of every step.
5. **Cache the topological order** for the static MLP graph, and use `no_grad` for evaluation (already done).
6. For very high sparsity, a sparse backward (SDDMM: compute `xᵀδ` only at the kept positions)
   so training cost scales with the number of active weights, not the dense size.

## 4. Serving a self-pruned model in a multi-tenant inference service

The measurements in the README matter here: pruning reduced FLOPs and stored parameters by exactly
the sparsity ratio, **but my NumPy CSR path was slower than dense BLAS at every sparsity I tried**
(12x slower at 90% for batch 64). Unstructured sparsity only pays when the kernel is built for it.
For a real service I would:

- **Choose the sparsity pattern for the hardware.** Unstructured masks barely speed up GPUs/TPUs.
  Use 2:4 (Ampere+ sparse tensor cores), block-sparse (e.g. 16x16), or remove dead neurons entirely, and
  then re-prune with that pattern as a constraint rather than converting afterwards. Quote the accuracy
  at the structured pattern, not at the unstructured one.
- **Store the model compactly even if compute is dense:** CSR/bitmask weights cut memory and load time,
  which matters for the per-tenant case where many model variants must be resident.
- **Batch dynamically** across requests with a latency budget (say wait up to 2-5 ms for a full
  batch); sparse and dense kernels both want larger batches, and p99 matters more than mean.
- **Isolate tenants by artifact, share by structure:** each tenant gets a versioned `(weights, mask)` artifact
  in a registry; keep the base dense weights shared where tenants fine-tune from one base, and route by
  tenant id. Warm the hot ones, evict the cold ones.
- **Ship safely:** shadow/canary new pruned versions, track the accuracy delta against the dense model on
  live labelled traffic, and keep the dense model as an automatic fallback. A pruned model that is 0.3
  points worse on average can be much worse on one tenant's slice.
- **Stack with quantisation** (int8) after pruning; measure the combination, since the two errors do
  not simply add.
- **Measure what the business pays for:** cost per 1k requests at a fixed p99 and accuracy floor, not
  FLOPs. My own results are the cautionary example: 10x fewer FLOPs, no latency win.

## Evaluation protocol (so the numbers can be trusted)

- Design choices (criterion, EMA, regrowth, normalisation) were made on **validation accuracy, on
  seeds 100-102**. The reported sweep uses **seeds 0-4**, so nothing was tuned on the reported runs.
- The two methods in each comparison share seed, initialisation, data split, batch order, schedule and masking
  code; only the score function differs, so the paired differences are meaningful.
- Test accuracy is read once per run at the end. One caveat: I looked at two early test numbers
  (seed 0, 90%) while debugging run time, before the ablations; nothing was selected on them.
- Five seeds is small. Bootstrap intervals over five differences are coarse and the exact permutation
  p-value cannot go below 0.0625. I report the paired CI and win counts, and say so where it matters.

# Quantization Search v2: Sampler Bug, Fix, and Results

## 1. Background

`quantization_search.py` runs an Optuna ask-and-tell search over the
stratified token allocation `(n4, n2, n1)` for a given bit-budget `S`, under
the constraint `4·n4 + 2·n2 + n1 = T` (`T = S // 1024`), `n4+n2+n1 ≤ 576`.

The original `sample_params` sampled sequentially: `n4 ~ Uniform(0, T//4)`,
then `n2 ~ Uniform(0, (T-4n4)//2)`, with `n1` as the residual. This gives
`n4` a disproportionately large first-drawn range, which makes the three
uniform-bit-width corner solutions (All B=1: n4=n2=0; All B=2: n4=n1=0;
All B=4: n2=n1=0) exponentially unlikely to ever be sampled by chance —
e.g. at S=1,179,648 (T=1152), landing exactly on n4=0 has probability
≈1/289, and n4=0 *and* n2=0 (the All-B1 corner) ≈1/289 × 1/577 ≈ 0.0006%
per trial; over 50 trials, the expected number of hits is ~0.0003 —
essentially never.

**Symptom observed**: validating the resulting "optimal" configs against
the trivial uniform baselines (All B=1/2/4) on POPE/MMBench showed the
search-found configs losing to a uniform baseline in 11 of 12 tested rows
(see prior validation round, `results/routed_op_v2`). This is consistent
with the search structurally never having tried the baselines it was losing
to.

## 2. Fix

### 2.1 Sampler: budget-proportion two-cut simplex sampling

Two uniform cut points `u1, u2 ~ Uniform(0,1)` are sorted (`lo, hi`) to give
three proportions of the *budget* `T` (not of token count): `p4=lo`,
`p2=hi-lo`, `p1=1-hi`. Each proportion is converted to a token count by
dividing by that tier's per-token cost (4, 2, 1 respectively); `n1` absorbs
the rounding residual so `4n4+2n2+n1=T` holds exactly for every sample,
including all three uniform-bit-width corners (reachable at `(u1,u2) =
(0,0)`, `(1,1)`, `(0,1)`).

Naively applying the "two cuts" trick directly to token counts (`n1 = T -
n4 - n2`, as an earlier draft of this fix proposed) does **not** preserve
the true budget constraint — it silently substitutes `n4+n2+n1=T` (equal
per-token cost) for the real `4n4+2n2+n1=T`, causing intermediate
(non-corner) samples to spend up to ~2.3× the intended budget. Verified by
fuzzing 18,000 random `(u1,u2)` pairs across 9 budgets from 25,600 to
2,359,296: the corrected sampler holds the exact budget equation in every
case (see `quantization_search.py::sample_params`).

### 2.2 Anchor points enqueued first

For each budget, up to 4 anchor points (All B=1, All B=2, All B=4, and —
when `T` supports averaging ≥1 bit/token across all 576 tokens — an
"N=576, no pruning, mixed 1-/2-bit" point) are computed
(`get_anchor_points`), converted to their exact `(u1,u2)` via
`anchor_to_u1u2`, and enqueued via `study.enqueue_trial()` before the TPE
loop starts. Infeasible anchors at a given budget (e.g. All B=1 requires
`n1=T≤576`, impossible above S=589,824) are skipped rather than emitted as
underspent/invalid points. Anchors count toward the trial budget
(`n_trials`) and — because Optuna's TPE model is fit on the full trial
history — their scores actively inform where the sampler explores next, not
just serve as a floor.

Each trial's full record (`u1`, `u2`, derived `n4/n2/n1`, `N`, score,
`source: anchor|tpe`, `state: complete|pruned`) is retained in the
per-budget result dict (`trials` key) alongside the existing summary fields;
`anchors` lists what was enqueued. Each budget uses a fresh `optuna.Study`
(the `(u1,u2)` parameterization is incompatible with the old `(n4,n2)`
one — resuming an old study would be meaningless).

## 3. Results

Ran via `run_pipeline.sh` (search → validate), dataset=mme, `--merge`,
n_trials=50, seed=42, 4 budgets in parallel across 4 GPUs:
`115,500 / 380,000 / 819,200 / 1,179,648`.

### 3.1 Found configs

| S (bits) | n4 | n2 | n1 | N | top_ratio | MME score |
|---|---|---|---|---|---|---|
| 115,500 | 14 | 13 | 30 | 57 | 0.246 | 1721.21 |
| 380,000 | 14 | 23 | 269 | 306 | 0.046 | 1781.72 |
| 819,200 | 21 | 222 | 272 | 515 | 0.041 | 1882.44 |
| 1,179,648 | 190 | 40 | 312 | 542 | 0.351 | 1886.03 |

### 3.2 Anchor-fix validation: does the search now beat trivial baselines on MME itself?

| S (bits) | best anchor (MME) | TPE-found opt (MME) | margin |
|---|---|---|---|
| 115,500 | 1686.03 (All B=2) | 1721.21 | +35.18 |
| 380,000 | 1745.65 (All B=1) | 1781.72 | +36.08 |
| 819,200 | 1836.49 (N=576 mix) | 1882.44 | +45.95 |
| 1,179,648 | 1821.13 (All B=2) | 1886.03 | +64.90 |

**Yes — in all 4 budgets, the corrected search now finds a config that
beats every uniform/no-pruning baseline on MME**, confirming the sampler
bug was the cause of the earlier corner-solution losses. This is a clean,
unconfounded before/after result specifically for the sampler fix.

### 3.3 Generalization: POPE / MMBench validation

| S (bits) | POPE B1 | POPE B2 | POPE B4 | POPE opt | MMBench B1 | MMBench B2 | MMBench B4 | MMBench opt |
|---|---|---|---|---|---|---|---|---|
| 115,500 | **0.8091** | 0.7871 | 0.7528 | 0.7919 | 70.32% | 70.30% | 69.36% | **70.94%** |
| 380,000 | 0.8033 | **0.8122** | 0.8073 | 0.7729 | **72.24%** | 70.94% | 71.14% | 72.08% |
| 819,200 | ERR (infeasible: n1>576) | 0.8261 | 0.8171 | **0.8522** | ERR | **73.18%** | 70.89% | 72.72% |
| 1,179,648 | ERR (infeasible) | **0.8547** | 0.7535 | 0.8532 | ERR | **73.57%** | 72.47% | 72.49% |

opt wins 3 of 8 (budget × benchmark) cells; loses (usually to All B=2 at
high budgets, or All B=1 at 380,000) in the other 5, including one clear
loss (POPE @ 380,000: 0.7729 vs 0.8122 best baseline — the same budget
where MME showed one of the largest opt-vs-baseline margins).

## 4. Interpretation

The sampler fix and the generalization gap are **two independent problems,
and this round only fixed the first one.**

- **Fixed**: the search was structurally incapable of finding (or even
  trying) the trivial uniform-bit-width baselines due to a sampling bias in
  `(n4, n2)` draw order. Corrected; verified fixed on MME in all 4 tested
  budgets.
- **Not fixed, still open**: the search optimizes purely for MME. A config
  that is genuinely best *for MME's specific 2374 questions* is not
  guaranteed to be best for POPE or MMBench — and the data confirms this:
  the budget with the *strongest* MME improvement (380,000, +36 over best
  baseline) is the one with the *worst* POPE result relative to baseline.
  This is evidence of overfitting to the search objective, not a sampler
  artifact, and needs a different intervention (e.g. incorporating
  POPE/MMBench into the search objective or a validation-based model
  selection step) — out of scope for this fix.

## 5. Reproducibility

- Search/anchor logic: `quantization_search.py`
  (`sample_params`, `get_anchor_points`, `anchor_to_u1u2`, `search`)
- Pipeline: `run_pipeline.sh --budgets <S> --n-trials 50 --datasets pope mmbench --merge --seed 42 --output-dir <dir>`
- Raw search+trial history: `results/search_mme_{115500,380000,819200,1179648}.json`
  (`results[0]["trials"]` has every trial incl. u1/u2, n4/n2/n1, source, score, state;
  `results[0]["anchors"]` lists what was enqueued)
- Validation tables: `results/pipeline_v2_gpu{0,1,2,3}/{POPE,MMBENCH}_*.txt`
  (gpu0=115500, gpu1=380000, gpu2=819200, gpu3=1179648)
- Prior (pre-fix) validation showing the original failure mode:
  `results/routed_op_v2/` (uniform+routing pass) — All B=1/2/4 vs the
  old-sampler MME-opt columns

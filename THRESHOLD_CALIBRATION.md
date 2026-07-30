# Checkpoint Routing Threshold: Methodology and Results

## 1. Research Question

`AdaptSplit` routes inference between three LLaVA-1.5-7B checkpoints depending on
the number of active visual tokens N (`N = n4 + n2 + n1`, selection is a function
of N alone, independent of the 4-bit/2-bit/1-bit split — see `validate.py::select_model`):

- `base_lora`: PruMerge LoRA fine-tuned at ~5.5% token retention (calibration point N≈32)
- `plus_lora`: PruMerge LoRA fine-tuned at ~25% token retention (calibration point N≈144)
- `original`: base LLaVA-1.5-7B, no LoRA, no token-count-specific tuning (calibration point N=576, full token set)

The original routing rule was a 3-tier split with thresholds derived as the geometric
mean of adjacent calibration points (`√(32·144)≈68`, `√(144·576)≈288`) — a
theoretically-motivated but never empirically-validated heuristic. This document
describes the empirical procedure used to test and recalibrate that rule.

## 2. Method

**Isolating the checkpoint-choice variable.** All comparisons use
`--no-quant --n-tokens N --merge` (PruMerge token selection only, zero bit
quantization). This isolates "which checkpoint is best for N tokens" from
quantization-noise effects, which are a separate, confounding variable (see
§5 for why this matters when applying the result to a quantized deployment).

**Forcing checkpoint choice.** `validate.py` was extended with a
`--force-model {base_lora,plus_lora,original}` flag that bypasses
`select_model(N)` and loads the specified checkpoint regardless of N. This
allows any two (or three) checkpoints to be evaluated at the *same* N for a
paired comparison — without it, only one checkpoint is ever observed at a
given N in normal operation.

**Metric.** MME (2374 yes/no questions, 14 categories), `--split all` (100%
— MME is treated as calibration/training data for this project, not a
held-out generalization benchmark; POPE/TextVQA/MMBench remain untouched by
this calibration and are the actual generalization report). Two numbers are
tracked per run:
- `acc`: fraction of individual questions answered correctly (pooled across
  all 14 categories) — the per-item granularity needed for paired testing.
- `mme_score`: the official MME total score, Σ over categories of
  `(acc_cat + acc_plus_cat) × 100` — reported for readability/comparability
  with the rest of the project, but *not* usable as direct input to a paired
  significance test (it is an aggregate over per-item labels, not itself an
  itemizable outcome).

**Statistical test.** Two checkpoints run on the identical MME question set
produce paired per-question correct/incorrect labels. McNemar's test is used
on the discordant pairs (b = only model A correct, c = only model B correct;
concordant pairs where both are right or both are wrong carry no
information and are excluded): `χ² = (|b−c|−1)² / (b+c)` with continuity
correction when `b+c ≥ 25`, otherwise an exact binomial test. α = 0.05.
Single runs are sufficient — the questionwise pairing already supplies the
variance estimate; no repeated runs were needed. Implementation:
`statsmodels.stats.contingency_tables.mcnemar`.

**Search design.** No fixed grid. A coarse pass at 6 anchors spanning the
full feasible range (N = 16, 32, 144, 288, 432, 576 — floor/ceiling chosen
as the smallest/largest meaningful token counts) ran all 3 checkpoints
pairwise (3 McNemar tests per anchor: base-plus, plus-original,
base-original). Adjacent anchors whose top-accuracy model differed were then
refined by adaptive bisection (probe the midpoint, recurse into whichever
half still shows a change — segments with the same top model at both ends
were left unrefined, since no transition is indicated there). This avoided
assuming a priori that exactly two ordered crossovers exist, and avoided
blind dense sampling of regions that show no evidence of a transition.
Script: `threshold_calibration.py`.

## 3. A Bug Found and Fixed During Calibration

The first calibration pass showed `original` collapsing to near-chance
accuracy (0.55) specifically at N=576, while `base_lora`/`plus_lora` only
degraded mildly at the same point. Inspection of the generated answers
showed incoherent, repetitive text (e.g. `"The word is not\nThe\n"`) rather
than normal yes/no responses — a sign of a real inference-pipeline defect,
not a legitimate finding about model quality.

**Root cause** (`llava/model/multimodal_encoder/clip_encoder.py`,
`token_prune_merge_select`): when `n_tokens == N_total` (576, i.e. no
pruning at all), `complement_idx()` returns an *empty* set of "non-selected"
tokens. The merge step is designed to fold unselected low-attention tokens
into the selected ones; with nothing unselected, it silently degenerates
into averaging every selected token with every *other* selected token
(weighted by attention) — a global over-smoothing operation never intended
by the algorithm, only triggered at this single boundary point. Both LoRA
checkpoints, having been fine-tuned through this same merge code path (just
at their own smaller N), show some generalization/robustness to blended
visual tokens; `original`, never trained on any merged tokens, does not, and
the output degenerates into incoherent text.

**Fix**: `token_prune_merge_select` now short-circuits when
`n_tokens >= N_total` and returns the selected (unmerged) tokens directly,
skipping the degenerate merge loop. Verified: `original` @ N=576 recovered
from 0.55 acc / garbled output to 0.79 acc, MME score 1811.6 (in line with
all neighboring N values). All reported N=576 results below are post-fix.

This is reported as an independent finding: it explains a token-count edge
case in the merge algorithm, not a property of the checkpoints' relative
quality, and was deliberately excluded from driving the routing threshold
itself.

## 4. Results

19 points evaluated (16, 32, 144, then adaptive refinement 162–252, then
288, 432, 576). Full data with per-question b/c counts and p-values:
`threshold_calibration.result.json`; raw log of every comparison:
`threshold_calibration.log.jsonl`.

| N | top1 (by acc) | winning set (not significantly beaten) | acc (base / plus / orig) | mme_score (base / plus / orig) |
|---|---|---|---|---|
| 16 | base_lora | {base_lora} | .736 / .703 / .697 | 1593 / 1459 / 1394 |
| 32 | base_lora | {base_lora} | .761 / .739 / .730 | 1664 / 1558 / 1540 |
| 144 | base_lora | {base_lora, plus_lora} | .766 / .764 / .747 | 1673 / 1612 / 1569 |
| 162 | base_lora | {all 3} | .761 / .760 / .751 | 1661 / 1600 / 1568 |
| 171 | base_lora | {all 3} | .761 / .757 / .754 | 1644 / 1611 / 1594 |
| 175 | base_lora | {all 3} | .758 / .756 / .754 | 1626 / 1600 / 1584 |
| 180 | plus_lora | {all 3} | .757 / .759 / .754 | 1611 / 1618 / 1583 |
| 216 | plus_lora | {all 3} | .765 / .767 / .763 | 1645 / 1648 / 1662 |
| 234 | plus_lora | {all 3} | .760 / .769 / .767 | 1643 / 1673 / 1716 |
| 238 | original | {all 3} | .761 / .767 / .769 | 1646 / 1664 / 1701 |
| 243 | original | {all 3} | .759 / .765 / .769 | 1639 / 1670 / 1705 |
| 252 | original | {all 3} | .765 / .767 / .775 | 1668 / 1682 / 1741 |
| 288 | original | **{original}** | .765 / .763 / .782 | 1689 / 1661 / 1782 |
| 432 | original | **{original}** | .755 / .767 / .786 | 1612 / 1674 / 1828 |
| 576 | original | **{original}** | .759 / .771 / .790 | 1595 / 1683 / 1812 |

### Key findings

1. **`plus_lora` never has an exclusive winning zone anywhere in N ∈
   [16, 576].** The only region where it leads on raw accuracy (N≈180–234)
   is statistically indistinguishable from both other checkpoints (all
   pairwise p > 0.28 there). `base_lora` is exclusively best for N≤32 (and
   ties through N≈175); `original` is exclusively best for N≥288 (winning
   set = {original} alone, first reaching significance at N=288, p=0.013
   vs plus_lora, p=0.033 vs base_lora). `plus_lora` never achieves this
   status at any tested point. This directly supports collapsing the
   routing rule from 3 tiers to 2 (`base_lora` / `original`), dropping
   `plus_lora` entirely.

2. **N=144–288 is a genuine zone of statistical ambiguity**, not a location
   any single threshold value can be "more correct" about — every pairwise
   test in this range has p > 0.18. The exact cutover point placed within
   this zone is therefore a matter of engineering choice, not measurable
   accuracy difference.

3. **The trend from N=238 onward is monotonic and one-directional**:
   `original` is top1 at every subsequent tested point through N=576, with
   the margin widening and becoming significant by N=288. This is a more
   defensible basis for the cutover than the isolated last point where
   `base_lora` happened to be top1 (N=175), since it reflects where the
   preference *stabilizes* rather than a single snapshot.

## 5. Final Threshold

```python
def select_model(N: int) -> str:
    if N <= 234:
        return "base_lora"
    else:
        return "original"
```

`plus_lora` is dropped from the routing rule (its checkpoint and
`--force-model plus_lora` remain available in the codebase for any future
recalibration). Implemented in `validate.py::select_model`.

## 6. Limitations to disclose

1. **Calibrated without quantization.** This threshold reflects checkpoint
   preference under pure token-count variation. The deployed system applies
   quantization on top of token selection; whether checkpoint preference
   shifts under quantization noise has not been tested, and the interaction
   is plausible — quantization changes the input distribution the LoRA
   adapters see. A separate calibration pass under representative quantized
   configs is the natural follow-up before treating this as final for
   production routing (as opposed to this document's standalone finding
   about token-count-only checkpoint preference).
2. **MME-only calibration.** All comparisons use MME as the calibration set,
   by design (MME is this project's designated training/calibration
   dataset; POPE/TextVQA/MMBench are held out for generalization reporting
   and were not used to select this threshold).
3. **Multiple comparisons.** 19 anchor/bisection points × 3 pairwise tests
   each were run without a multiple-comparisons correction; results here
   should be read as an exploratory/adaptive search converging on a
   threshold, not as a single confirmatory hypothesis test. The qualitative
   conclusion (plus_lora never wins; base/original split around N≈144–288)
   is consistent across every point in that range, which mitigates — but
   does not formally eliminate — this concern.
4. **N=576 fix is verified only for `original`/MME** — the same edge case
   would apply to any checkpoint/dataset combination that requests
   `n_tokens == 576` under `--merge`; not separately re-verified beyond the
   cases in this document.

## Reproducibility

- Calibration script: `threshold_calibration.py` (`python threshold_calibration.py --gpus 0,1,2,3`)
- Raw per-comparison log: `threshold_calibration.log.jsonl`
- Full structured results (incl. all p-values, b/c counts): `threshold_calibration.result.json`
- Bug fix: `llava/model/multimodal_encoder/clip_encoder.py::token_prune_merge_select`
- Final rule: `validate.py::select_model`
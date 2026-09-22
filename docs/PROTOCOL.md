# Experimental protocol

## Class sessions

- Houston2013: base classes 1–9, then classes 10–12 and 13–15.
- PaviaU: base classes 1–5, then classes 6–7 and 8–9.

The reported main comparison uses three seeds (`42`, `123`, `3407`) and a
fixed total persistent-image budget. The ER and full configurations within a
dataset/seed pair share sampled images, initialization, epoch order, and random
number generator seeds. The audit code records hashes for these objects.

## Main configurations

- `persistent_er`: replay control without prototype scoring, IPOC, or old-logit
  distillation.
- `persistent_full`: multi-prototype scoring with adaptive capacity, IPOC, and
  old-logit distillation.
- `persistent_pool_lse` / `persistent_pool_lme`: a paired intervention changing
  only the prototype pooling normalization under the main memory protocol.

Additional component configurations are declared in
`run_persistent_ablation.py`; their scientific arguments are serialized before
training and checked when existing results are resumed.

## Metrics

Reports contain final-session overall accuracy (OA), average accuracy (AA),
base-class OA, current-class OA, and base forgetting. Aggregate OA should be
interpreted together with base/current accuracy because prototype fusion can
redistribute performance between retained and arriving classes.

## Spatial controls

Patch-centered HSI classification can leak scene pixels if neighboring train
and test patches overlap. The split generators in `datasets/` exclude test
centers whose patch support intersects a training patch. The strict Houston
experiment should use a separate generated data directory and must not be
numerically mixed with results from the official train/test labels.

## Output integrity

Formal queue drivers:

1. record effective arguments, runtime versions, code hashes, and data hashes;
2. reject duplicate jobs and incompatible output directories;
3. update a machine-readable queue status;
4. audit paired controls before producing summaries.

Keep raw per-seed directories outside Git. Only lightweight audited reports
should be committed to `results/`.

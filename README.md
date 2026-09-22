# Auditing Prototype Pooling Bias in Class-Incremental Hyperspectral Classification

Official research code and audited experiment drivers for the manuscript
**“Auditing Prototype Pooling Bias in Class-Incremental Hyperspectral Classification.”**

The repository studies two effects of multi-prototype prediction in
class-incremental hyperspectral image (HSI) classification:

1. prototype aggregation can change competition between retained and arriving
   classes even when every class uses the same number of prototypes; and
2. with unequal active prototype counts, log-sum-exp pooling adds the explicit
   class-dependent term `log(K_c)`. Log-mean-exp removes this term.

The code contains the Houston2013 and Pavia University protocols, matched
persistent-memory controls, component and pooling ablations, iCaRL-HSI, strict
spatial-split utilities, provenance checks, and result summarizers used in the
paper.

## Repository layout

```text
.
├── main_houston_cls.py              # core HSI model, training and evaluation
├── main_persistent_controls.py      # fixed-total-memory experiment entry point
├── run_persistent_controls.py       # matched ER / full configuration queue
├── run_persistent_ablation.py       # component and LSE/LME interventions
├── run_capacity_experiments.py      # prototype-capacity grid
├── run_icarl_hsi.py                 # protocol-aligned iCaRL-HSI baseline
├── persistent_control_audit.py      # paired protocol/provenance checks
├── summarize_*.py                   # audited report generation
├── datasets/                        # dataset loaders and spatial split tools
├── tests/                           # unit and protocol-integrity tests
├── docs/                            # protocol and audit notes
└── results/                         # compact reported metrics and result map
```

## Environment

The experiments reported in the paper were run with Python 3.10.21,
PyTorch 2.5.1+cu124, NumPy 2.2.6, and SciPy 1.15.3. A CUDA GPU is recommended;
formal queue drivers intentionally require explicit authorization for CPU-only
runs.

```bash
conda create -n prototype-pooling python=3.10 -y
conda activate prototype-pooling
pip install -r requirements.txt
```

For the exact CUDA 12.4 PyTorch build used in our runs:

```bash
pip install -r requirements-cu124.txt
```

Alternatively, create the Conda environment with:

```bash
conda env create -f environment.yml
conda activate prototype-pooling
```

## Data preparation

Dataset files are not redistributed. Download them from their official sources
and arrange them as follows:

```text
data/
├── Houston2013/
│   ├── HSI.mat
│   ├── TRLabel.mat
│   └── TSLabel.mat
└── PaviaU/
    ├── PaviaU.mat
    ├── PaviaU_train_spatial.mat
    └── PaviaU_test_spatial.mat
```

See [DATA.md](DATA.md) for dataset links, MATLAB keys, split generation, and
leakage-control details. Never commit downloaded `.mat` files to GitHub.

## Quick verification

Run the tests before launching training:

```bash
python -m pytest -q
```

Check the complete command queue without training:

```bash
python run_persistent_controls.py \
  --houston-data-dir /path/to/Houston2013 \
  --pavia-data-dir /path/to/PaviaU \
  --dry-run
```

Run a one-seed smoke test on one visible GPU:

```bash
python run_persistent_controls.py \
  --gpu 0 --smoke --seeds 42 \
  --houston-data-dir /path/to/Houston2013 \
  --pavia-data-dir /path/to/PaviaU
```

## Main experiments

Matched persistent-memory ER and full configurations (two datasets, three
seeds):

```bash
python -u run_persistent_controls.py \
  --gpu 0 \
  --houston-data-dir /path/to/Houston2013 \
  --pavia-data-dir /path/to/PaviaU

python -u summarize_persistent_controls.py \
  --houston-data-dir /path/to/Houston2013 \
  --pavia-data-dir /path/to/PaviaU
```

Paired component ablations:

```bash
python -u run_persistent_ablation.py \
  --gpu 0 \
  --houston-data-dir /path/to/Houston2013 \
  --pavia-data-dir /path/to/PaviaU

python -u summarize_persistent_ablation.py
```

Main-protocol LSE versus LME pooling intervention:

```bash
python -u run_persistent_ablation.py \
  --gpu 0 --variants persistent_pool_lse persistent_pool_lme \
  --output-root persistent_pooling_outputs \
  --houston-data-dir /path/to/Houston2013 \
  --pavia-data-dir /path/to/PaviaU

python -u summarize_persistent_pooling.py
```

Capacity grid and iCaRL-HSI baseline:

```bash
python -u run_capacity_experiments.py \
  --gpu 0 \
  --houston-data-dir /path/to/Houston2013 \
  --pavia-data-dir /path/to/PaviaU

python -u run_icarl_hsi.py \
  --gpu 0 \
  --houston-data-dir /path/to/Houston2013 \
  --pavia-data-dir /path/to/PaviaU
```

Each formal queue writes its effective arguments, code/data hashes, runtime,
status, per-run metrics, and audit results into its output directory. Existing
incompatible or unfinished output directories are rejected rather than silently
overwritten.

## Reproducibility notes

- Reported values use seeds `42`, `123`, and `3407`.
- Houston2013 sessions are classes `1–9`, `10–12`, and `13–15`.
- PaviaU sessions are classes `1–5`, `6–7`, and `8–9`.
- Main comparisons use a fixed total persistent memory and paired data/order
  audits. See `docs/PROTOCOL.md`.
- The strict Houston spatial protocol is generated by
  `datasets/prepare_houston_spatial_split.py`.
- Generated checkpoints and raw datasets are deliberately excluded from this
  repository.

## License

Released under the [MIT License](LICENSE). Dataset licenses and terms remain
those of the original dataset providers. Author and citation metadata are
intentionally omitted from this review-time repository and should be restored
after anonymous review.

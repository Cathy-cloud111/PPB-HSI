# Dataset preparation

The repository does not redistribute raw hyperspectral imagery or labels.

## Houston2013

Use the IEEE GRSS Data Fusion Contest 2013 Houston dataset. The classification
loader expects these files in one directory:

```text
HSI.mat       # hyperspectral cube; default key: HSI
TRLabel.mat   # training label raster
TSLabel.mat   # test label raster
```

The expected HSI shape is `349 x 1905 x 144`. If a MATLAB file contains more
than one candidate variable, pass the corresponding command-line key to the
entry point.

For the guarded spatial protocol, generate a separate directory rather than
overwriting the original labels:

```bash
python datasets/prepare_houston_spatial_split.py \
  --gt-file /path/to/Houston2013_gt.mat \
  --gt-key ground_truth \
  --output-dir /path/to/Houston2013_strict
```

Run `python datasets/prepare_houston_spatial_split.py --help` for all split and
patch-separation options.

## Pavia University

Download the Pavia University scene and ground truth from the University of the
Basque Country hyperspectral remote-sensing dataset page:

<https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes>

Generate deterministic, patch-disjoint labels:

```bash
python datasets/prepare_paviau_split.py \
  --gt_file /path/to/PaviaU_gt.mat \
  --gt_key paviaU_gt \
  --output_dir /path/to/PaviaU \
  --train_ratio 0.10 \
  --patch_size 15 \
  --split_seed 2027
```

The generated experiment directory must contain:

```text
PaviaU.mat
PaviaU_train_spatial.mat
PaviaU_test_spatial.mat
```

Always retain the generated split metadata and verify that train/test patch
supports do not overlap before reporting results.

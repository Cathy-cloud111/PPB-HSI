# Audited result artifacts

This directory contains compact CSV copies of the reported aggregate metrics
and the Houston2013 classification-map figure. Raw datasets, per-sample
predictions, server plans, and model checkpoints are excluded.

The CSV files are provided for convenient result checking; they are not inputs
to training. Values use percentages/percentage points and are copied from the
audited three-seed manuscript tables.

To regenerate full per-seed reports, run the appropriate `summarize_*.py`
script against a completed local output directory. Do not compare rows produced
under different memory or spatial protocols unless explicitly labeled.

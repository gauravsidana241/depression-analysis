# Depression Detection from Speech

Code for my MSc dissertation at the University of Glasgow.

A stacked RNN is compared against Logistic Regression and SVM baselines on IS09 acoustic features, across segment lengths from 32 to 1024 frames and under two audio conditions.

## Contents

- `feature_extraction.ipynb`: extracts frame-level features with openSMILE
- `train_rnn.ipynb`: trains and evaluates the stacked RNN
- `train_static.ipynb`: trains and evaluates the static baselines
- `results.ipynb`: produces the tables and figures reported in the dissertation
- `common.py`: shared helper functions, which must stay in the same directory as the notebooks
- `Opensmile_dd.conf`: openSMILE configuration file
- `androids-environment.yml`: conda environment with all required package versions

## Setup

1. Create the conda environment and activate it:

   ```
   conda env create -f androids-environment.yml
   ```

2. Create the `features/`, `features_clips/`, `results/` and `figures/` folders.

3. The dataset (Androids-corpus, 3.69 GB) is not included. Download it from https://github.com/androidscorpus/data and place it in the same directory as the notebooks, at `./Androids-corpus/`.

4. openSMILE is not included either. Download it and place it at `./opensmile-3.0.2-windows-x86_64/`.

## Running the experiments

1. Run `feature_extraction.ipynb`. Features are written to `features/` and `features_clips/`.
2. Run `train_rnn.ipynb` and `train_static.ipynb` twice each, once per audio condition (po and raw). Instructions for switching conditions are in the notebooks.
3. Run `results.ipynb` to produce all results and observations reported in the dissertation. Outputs are written to `results/` and `figures/`.

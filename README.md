
# A Comparative Study of Machine Learning Methods for Blood-Brain Barrier Permeability Prediction

## Authors

Sanvi Jagtap(sj5129), Yila Cao(yc8155), Yiyao Zhang(yz12490) — NYU Center for Data Science (DS-GA 1003)

## Overview

This repository contains code and results for our ML final project comparing six models for BBB permeability prediction under random and Bemis-Murcko scaffold splits.

## Models

- Lipinski's Rule of Five (rule-based baseline)

- Logistic Regression, SVM, MLP, Random Forest (fingerprint-based)

- Graph Convolutional Network (graph-based)

## Repository Structure

- `src/data/` — dataset loading, ECFP4 fingerprints, scaffold splits

- `src/models/` — all model implementations

- `src/evaluation/` — metrics and multi-seed runner

- `src/analysis/` — error analysis, t-SNE, representation analysis

- `results/` — all figures, CSVs, and result files

- `tests/` — unit tests for data, models, and metrics

## Setup

```bash

pip install -r requirements.txt

```

## Reproducibility

All experiments use seeds {42, 123, 7}. To reproduce main results:

```bash

python src/evaluation/runner.py

```

## Dataset

BBBP dataset from MoleculeNet — 2039 molecules, binary BBB permeability labels.


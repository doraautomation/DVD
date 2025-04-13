# Distributed PCA-KMeans Clustering with Thread-Based Blockchain Validation

This project shows how to combine PCA (for dimensionality reduction), KMeans clustering, and a simple two-phase commit simulation for validating data, similar to how blockchains work.

The code uses MPI for running tasks in parallel across processes and Python threads to simulate internal node validation.

---

## What's Included

| File | Description |
|------|-------------|
| `parallel_kmeans_threads.py` | Clusters data rows using KMeans, validates each block using threads |
| `pca_distributed_columns.py` | Splits data by columns, applies PCA, and runs commit validation with timing logs |
| `wdbc.csv` | Input file – based on the Breast Cancer dataset from UCI |

---

## Setup

### 1. Install Python Packages

You’ll need Python 3.8 or later. Install the required packages with:

```bash
pip install mpi4py numpy pandas scikit-learn

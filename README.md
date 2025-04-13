# Distributed PCA-KMeans Clustering with Thread-Based Blockchain Validation

This project demonstrates a lightweight distributed system that combines:

- **Feature reduction** using Principal Component Analysis (PCA)
- **Clustering** with KMeans
- **Feature-based sharding** — splitting the dataset across nodes by feature columns
- **Blockchain-style validation** using a two-phase commit simulated with Python threads

The system uses **MPI** to distribute tasks across processes, and **Python threads** to simulate internal node consensus (like in a sub-cluster or committee).

---

## What's Included

| File | Description |
|------|-------------|
| `parallel_kmeans_threads.py` | Performs row-wise KMeans clustering after PCA, with thread-based validation |
| `pca_distributed_columns.py` | Distributes dataset **by features**, runs PCA, and validates using commit-style consensus |
| `wdbc.csv` | Input data (UCI Breast Cancer dataset in CSV format) |

---

## Setup

### Install Python Packages

Make sure you’re using Python 3.8 or higher. Then install the required libraries:

```bash
pip install mpi4py numpy pandas scikit-learn

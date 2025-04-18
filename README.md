# DVD
Distributed Vector Data Management system with High-Performance Distributed Ledgers for Scientific Computing.

# Features:

- Feature reduction using Principal Component Analysis (PCA)
- Clustering with Parallel K-Means
- Feature-based sharding — splitting the dataset across nodes by feature columns
- Blockchain-style validation using a two-phase commit simulated with python code
- The system uses MPI to distribute tasks across processes, and simulate internal node consensus (like in a sub-cluster or committee).

## Development Setup

### Install Python Packages

Make sure you’re using Python 3.8 or higher. Then install the required libraries:

pip install mpi4py numpy pandas scikit-learn

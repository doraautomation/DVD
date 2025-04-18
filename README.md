# DVD
Distributed Vector Data Management system with High-Performance Distributed Ledgers for Scientific Computing.

# Features:

- Feature reduction using Principal Component Analysis (PCA)
- Clustering with Parallel K-Means
- Feature-based sharding — splitting the dataset across nodes by feature columns
- Blockchain-style validation using a two-phase commit simulated with python code
- The system uses MPI to distribute tasks across processes, and simulate internal node consensus (like in a sub-cluster or committee).

## Development Setup
DVD should be run using python.
First install **[python]( https://www.python.org/downloads/)** 

DVD is integrated with MPI
Then install **[mpi4py](https://github.com/mpi4py/mpi4py/)**

To clone the code to your target directory
```bash
git clone https://github.com/doraautomation/DVD
cd DVD

Install all required package.

pip install -r requirements.txt

Run the Project Locally

After installing the dependencies, you can run the project using `mpiexec`.
Here’s an example with 4 processes:

mpiexec -n 4 python DVD_with_features_based_sharding.py **or** DVD_with_k_means_clustering.py

Run on HPC with SLURM

If you're working in an HPC environment, you can use the provided SLURM script to run your job.

Submit the Job

sbatch run_job.slurm

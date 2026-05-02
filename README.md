# DVD
Distributed Vector Data Management system with High-Performance Distributed Ledgers for Scientific Computing.

# Features
- **Semantic Clustering via Parallel K-Means**  
  Partitions high-dimensional vectors into similarity-preserving shards, 
  ensuring balanced load distribution across distributed nodes.

- **Lightweight Block Construction & Tamper-Evident Validation**  
  Implements a two-phase LiteQuorum consensus protocol, leveraging MPI 
  to distribute tasks across processes and simulate sub-cluster 
  committee-based consensus.

- **Hybrid On-Chain/Off-Chain Ledger Management**  
  Raw vector shard data resides in off-chain distributed storage, while 
  only lightweight shard metadata blocks are committed on-chain, ensuring 
  tamper-evident provenance without excessive storage overhead.

## Development Setup
DVD should be run using python.
First install **[python]( https://www.python.org/downloads/)** 

DVD is integrated with MPI
Then install **[mpi4py](https://github.com/mpi4py/mpi4py/)**

To clone the code to your target directory
```bash
git clone https://github.com/doraautomation/DVD
cd DVD
```
Install all required package.
```bash
pip install -r requirements.txt
```
Run the Project Locally

After installing the dependencies, you can run the project using `mpiexec`.
Here’s an example with 4 processes:
```bash
mpiexec -n 4 python DVD.py 
```
Run on HPC with SLURM

If you're working in an HPC environment, you can use the provided SLURM script to run your job.

Submit the Job
```bash
sbatch run_job.slurm
```

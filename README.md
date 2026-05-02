# DVD
Distributed Vector Data Management system with High-Performance Distributed Ledgers for Scientific Computing.

# Features:

    \item Semantic clustering via Parallel K-Means to partition high-dimensional 
    vectors into similarity-preserving shards with balanced load distribution 
    across distributed nodes.
    
    \item Lightweight block construction and tamper-evident validation through 
    a two-phase LiteQuorum consensus protocol, where MPI distributes tasks 
    across processes to simulate sub-cluster committee-based consensus.
    
    \item Hybrid on-chain/off-chain ledger management, where raw vector shard 
    data resides in off-chain distributed storage and only lightweight shard 
    metadata blocks are committed on-chain, ensuring tamper-evident provenance 
    without excessive storage overhead.

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
mpiexec -n 4 python DVD_with.py 
```
Run on HPC with SLURM

If you're working in an HPC environment, you can use the provided SLURM script to run your job.

Submit the Job
```bash
sbatch run_job.slurm
```

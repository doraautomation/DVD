from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from concurrent.futures import ThreadPoolExecutor
import time
import hashlib
import json
import sys

def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    if 'ID' in data.columns: data = data.drop(columns='ID')
    if 'Diagnosis' in data.columns: data = data.drop(columns='Diagnosis')
    scaler = StandardScaler()
    return scaler.fit_transform(data)

def evaluate_pca_quality(data, n_components=17, mse_threshold=0.01):
    pca = PCA(n_components=n_components)
    pca.fit(data)
    reduced_data = pca.transform(data)
    reconstructed_data = pca.inverse_transform(reduced_data)
    mse = mean_squared_error(data, reconstructed_data)
    return mse < mse_threshold, reduced_data, pca, data, mse

def distribute_columns(data, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    n_cols = data.shape[1]
    cols_per_rank = n_cols // size
    remainder = n_cols % size
    start = rank * cols_per_rank + min(rank, remainder)
    end = start + cols_per_rank + (1 if rank < remainder else 0)
    return data[:, start:end]

def hash_data(data):
    data_string = json.dumps(data.tolist())
    return hashlib.sha256(data_string.encode()).hexdigest()

def simulate_blockchain_node(data, hash_value):
    return hash_data(data) == hash_value

def two_phase_commit(data, comm, sub_cluster_size=20):
    hash_value = hash_data(data)
    with ThreadPoolExecutor(max_workers=sub_cluster_size) as executor:
        results = list(executor.map(simulate_blockchain_node, [data] * sub_cluster_size, [hash_value] * sub_cluster_size))
    return sum(results) >= (sub_cluster_size * 0.51)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    start_time = MPI.Wtime()

    filepath = 'wdbc.csv'  
    data = load_and_preprocess_data(filepath)

    # PCA and quality check
    if rank == 0:
        passes_quality, reduced_data, pca_model, original_data, mse = evaluate_pca_quality(data)
        if not passes_quality:
            print(f"PCA quality check failed. MSE = {mse:.6f}. Must be < 0.01.")
            sys.exit(0)
    else:
        reduced_data = None

    # Broadcast reduced data to all ranks
    reduced_data = comm.bcast(reduced_data if rank == 0 else None, root=0)

    # Distribute features
    local_data = distribute_columns(reduced_data, comm)

    # === PUSH : Writing to shared ledger after commit approval ===
    push_start = MPI.Wtime()
    commit_success = two_phase_commit(local_data, comm)
    push_end = MPI.Wtime()
    push_time = push_end - push_start

    if commit_success:
        # Store committed data
        np.savetxt(f'committed_data_rank_{rank}.csv', local_data, delimiter=",")
    else:
        print(f"Commit aborted.")

    # === PULL : Sync local ledgers from shared ledger ===
    pull_start = MPI.Wtime()
    updated_ledger_block = comm.bcast("Block_XYZ" if rank == 0 else None, root=0)
    local_ledger = updated_ledger_block  # simulate storing locally
    pull_end = MPI.Wtime()
    pull_time = pull_end - pull_start
    
    # Gather timings 
    all_push_times = comm.gather(push_time, root=0)
    all_pull_times = comm.gather(pull_time, root=0)

    comm.Barrier()
    if rank == 0:
        end_time = MPI.Wtime()
        print("=== Timing Summary ===")
        print(f"Avg Push Time: {np.mean(all_push_times):.6f} sec")
        print(f"Avg Pull Time: {np.mean(all_pull_times):.6f} sec")
        print("Total Execution Time: {:.4f} seconds".format(end_time - start_time))

if __name__ == "__main__":
    main()

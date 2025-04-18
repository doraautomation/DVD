from mpi4py import MPI
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import hashlib
import json
import time
import sys

def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    id_col = data['ID'].values
    diag_col = data['Diagnosis'].values
    data_features = data.drop(['ID', 'Diagnosis'], axis=1)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_features)
    return data_scaled, id_col, diag_col

def initialize_centroids(data, k):
    indices = np.random.choice(data.shape[0], size=k, replace=False)
    return data[indices, :]

def assign_clusters(data, centroids):
    distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

def compute_centroids(data, labels, k):
    centroids = np.zeros((k, data.shape[1]))
    for i in range(k):
        points = data[labels == i]
        if len(points) > 0:
            centroids[i] = points.mean(axis=0)
    return centroids

def apply_pca_and_check_mse(data, n_components=10, mse_threshold=0.01):
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(data)
    reconstructed = pca.inverse_transform(reduced)
    mse = np.mean((data - reconstructed) ** 2)
    return mse < mse_threshold, mse, reduced, pca, data

def hash_data(data):
    data_string = json.dumps(data.tolist())
    return hashlib.sha256(data_string.encode()).hexdigest()

def simulate_blockchain_node(data, hash_value):
    return hash_data(data) == hash_value

def two_phase_commit_subcluster(data, sub_cluster_size=20):
    hash_value = hash_data(data)
    results = [simulate_blockchain_node(data, hash_value) for _ in range(sub_cluster_size)]
    return sum(results) >= (sub_cluster_size * 0.51)

def parallel_kmeans(data, id_col, diag_col, k, num_steps=100, n_components=None, local_steps=10, mse_threshold=0.01):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    start_time = MPI.Wtime()

    if rank == 0 and n_components is not None:
        passed, mse, reduced_data, pca_model, original_data = apply_pca_and_check_mse(data, n_components, mse_threshold)
        if not passed:
            sys.exit(0)
        else:
            data = reduced_data
    else:
        data = None

    # Broadcast reduced data
    data = comm.bcast(data, root=0)
    id_col = comm.bcast(id_col, root=0)
    diag_col = comm.bcast(diag_col, root=0)

    # Distribute data 
    data_split = np.array_split(data, size, axis=0)
    id_split = np.array_split(id_col, size)
    diag_split = np.array_split(diag_col, size)

    local_data = data_split[rank]

    # Initialize and broadcast centroids
    centroids = initialize_centroids(data, k) if rank == 0 else None
    centroids = comm.bcast(centroids, root=0)

    for i in range(num_steps):
        local_labels = assign_clusters(local_data, centroids)
        local_centroids = compute_centroids(local_data, local_labels, k)

        if (i + 1) % local_steps == 0 or i == num_steps - 1:
            all_centroids = np.zeros_like(local_centroids)
            comm.Allreduce(local_centroids, all_centroids, op=MPI.SUM)
            centroids = all_centroids / size

    # === PUSH : Storing to shared ledger after commit approval ===
    push_start = MPI.Wtime()
    if two_phase_commit_subcluster(local_data):
        np.savetxt(f'committed_data_rank_{rank}.csv', local_data, delimiter=",")
    else:
        print(f"Rank {rank}: Commit Aborted.")
    push_end = MPI.Wtime()
    push_time = push_end - push_start
    
    # === PULL : Sync local ledgers from shared ledger ===
    pull_start = MPI.Wtime()
    updated_ledger_block = comm.bcast("Block_XYZ" if rank == 0 else None, root=0)
    local_ledger = updated_ledger_block
    pull_end = MPI.Wtime()
    pull_time = pull_end - pull_start
    
   
    # Final clusters
    final_labels = assign_clusters(local_data, centroids)
    local_cluster_sizes = np.array([np.sum(final_labels == i) for i in range(k)], dtype=np.int32)
    total_cluster_sizes = np.zeros(k, dtype=np.int32)
    comm.Reduce(local_cluster_sizes, total_cluster_sizes, op=MPI.SUM, root=0)

     # Gather timings 
    all_push_times = comm.gather(push_time, root=0)
    all_pull_times = comm.gather(pull_time, root=0)
    comm.Barrier()
    if rank == 0:
        end_time = MPI.Wtime()
        print("=== Timing Summary ===")
        print(f"Avg Push Time: {np.mean(all_push_times):.6f} sec")
        print(f"Avg Pull Time: {np.mean(all_pull_times):.6f} sec")
        print("Total time taken: {:.4f} seconds".format(end_time - start_time))

if __name__ == "__main__":
    filepath =  'wdbc.csv'  
    data, ids, diags = load_and_preprocess_data(filepath)
    k = 5
    n_components = 17
    parallel_kmeans(data, ids, diags, k, n_components=n_components, local_steps=10)

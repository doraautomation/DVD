import hashlib
import datetime as date
import json
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from concurrent.futures import ThreadPoolExecutor
from mpi4py import MPI
import sys
import os 

os.makedirs('output', exist_ok=True) #output_directory

# Block structure
class Block:
    def __init__(self, index, timestamp, data, previous_hash):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        hash_string = str(self.index) + str(self.timestamp) + str(self.data) + str(self.previous_hash)
        return hashlib.sha256(hash_string.encode()).hexdigest()

# Blockchain structure
class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]

    def create_genesis_block(self):
        return Block(0, date.datetime.now(), "Genesis Block", "0")

    def get_latest_block(self):
        return self.chain[-1]

    def add_block(self, new_block):
        new_block.previous_hash = self.get_latest_block().hash
        new_block.hash = new_block.calculate_hash()
        self.chain.append(new_block)

    def is_valid(self):
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            if current_block.hash != current_block.calculate_hash():
                return False
            if current_block.previous_hash != previous_block.hash:
                return False
        return True

    def get_chain(self):
        chain_output = []
        for block in self.chain:
            chain_output.append("Block #{}".format(block.index))
            chain_output.append("Timestamp: {}".format(block.timestamp))
            chain_output.append("Data: {}".format(block.data))
            chain_output.append("Hash: {}".format(block.hash))
            chain_output.append("Previous Hash: {}".format(block.previous_hash))
            chain_output.append("")
        output_text = "\n".join(chain_output)
        with open('output/Blockchain.txt', 'w') as f:
            f.write(output_text)

    def consensus(self, block, sub_cluster_size=20):
        # Phase 1: Prepare
        prepare_votes = [data_validation(block) for _ in range(sub_cluster_size)]
        if sum(prepare_votes) >= (sub_cluster_size * 0.51):
        # Phase 2: Commit 
            commit_memory = [None] * sub_cluster_size
            
            def commit_phase(i):
                    commit_memory[i] = block 
                    
                    block_df = pd.DataFrame([{
                        'index': block.index,
                        'timestamp': block.timestamp,
                        'data': block.data,
                        'previous_hash': block.previous_hash,
                        'hash': block.hash
                    }])
                    block_df.to_csv(f'output/subcluster_node_{i}_coordinator_{MPI.COMM_WORLD.Get_rank()}.csv', index=False)
                    return True

            with ThreadPoolExecutor(max_workers=sub_cluster_size) as executor:
                commit_votes = list(executor.map(commit_phase, range(sub_cluster_size)))
                
            if sum(commit_votes) >= (sub_cluster_size * 0.51):
                self.add_block(block)
                return True
        return False

# Parallel KMeans Sharding
def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    id_col = data['ID'].values
    diag_col = data['Diagnosis'].values
    data_features = data.drop(['ID', 'Diagnosis'], axis=1)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_features)
    return data_scaled, id_col, diag_col

def evaluate_pca_quality(data, n_components=17, mse_threshold=0.01):
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    reconstructed = pca.inverse_transform(reduced_data)
    mse = mean_squared_error(data, reconstructed)
    return mse < mse_threshold, reduced_data, mse

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

def hash_data(data):
    data_string = json.dumps(data)
    return hashlib.sha256(data_string.encode()).hexdigest()

def data_validation(block):
    content = json.loads(block.data)
    return hash_data(content['data']) == content['hash']

def parallel_kmeans_with_blockchain(filepath, k=5, n_components=17, num_steps=100, local_steps=10, mse_threshold=0.01):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    start_time = MPI.Wtime()

    data, id_col, diag_col = load_and_preprocess_data(filepath)

    if rank == 0:
        passed, reduced_data, mse = evaluate_pca_quality(data)
        if not passed:
            print(f"PCA MSE too high: {mse:.6f}")
            sys.exit()
    else:
        reduced_data = None

    data = comm.bcast(reduced_data if rank == 0 else None, root=0)
    id_col = comm.bcast(id_col, root=0)
    diag_col = comm.bcast(diag_col, root=0)

    data_split = np.array_split(data, size, axis=0)
    local_data = data_split[rank]

    centroids = initialize_centroids(data, k) if rank == 0 else None
    centroids = comm.bcast(centroids, root=0)

    for i in range(num_steps):
        local_labels = assign_clusters(local_data, centroids)
        local_centroids = compute_centroids(local_data, local_labels, k)
        if (i + 1) % local_steps == 0 or i == num_steps - 1:
            all_centroids = np.zeros_like(local_centroids)
            comm.Allreduce(local_centroids, all_centroids, op=MPI.SUM)
            centroids = all_centroids / size

    blockchain = Blockchain()
    data_to_commit = local_data.tolist()
    hash_value = hash_data(data_to_commit)
    block_data = {
        'coordinator': rank,
        'data': data_to_commit,
        'hash': hash_value
    }
    new_block = Block(rank + 1, date.datetime.now(), json.dumps(block_data), "0")

    committed = blockchain.consensus(new_block)
    if committed:
        block_df = pd.DataFrame([{
            'index': new_block.index,
            'timestamp': new_block.timestamp,
            'data': new_block.data,
            'previous_hash': new_block.previous_hash,
            'hash': new_block.hash
        }])
        
        block_df.to_csv(f'output/block_coordinator_{rank}.csv', index=False)

    comm.Barrier()
    
    #Push
    all_blocks = comm.gather(new_block if committed else None, root=0)
    if rank == 0:
        for blk in all_blocks:
            if blk and blk.hash not in [b.hash for b in blockchain.chain]:
                blockchain.add_block(blk)
    
    #Pull           
    comm.bcast(blockchain.chain, root=0)
    if rank == 0:
        end_time = MPI.Wtime()
        print("Execution Time: {:.4f} sec".format(end_time - start_time))
        blockchain.get_chain()
        print("Blockchain is valid." if blockchain.is_valid() else "Blockchain is invalid!")
    
if __name__ == "__main__":
    filepath = 'wdbc.csv'
    parallel_kmeans_with_blockchain(filepath)

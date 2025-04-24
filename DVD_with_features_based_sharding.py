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

os.makedirs('output', exist_ok=True)

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
    
class Sharding:
    @staticmethod
    def distribute_columns(data, comm):
        rank = comm.Get_rank()
        size = comm.Get_size()
        n_cols = data.shape[1]
        cols_per_rank = n_cols // size
        remainder = n_cols % size
        start = rank * cols_per_rank + min(rank, remainder)
        end = start + cols_per_rank + (1 if rank < remainder else 0)
        return data[:, start:end]

def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    id_col = data['ID'].values if 'ID' in data.columns else None
    diag_col = data['Diagnosis'].values if 'Diagnosis' in data.columns else None
    data_features = data.drop(['ID', 'Diagnosis'], axis=1, errors='ignore')
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_features)
    return data_scaled, id_col, diag_col

def evaluate_pca_quality(data, n_components=17, mse_threshold=0.01):
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    reconstructed = pca.inverse_transform(reduced_data)
    mse = mean_squared_error(data, reconstructed)
    return mse < mse_threshold, reduced_data, mse

def hash_data(data):
    data_string = json.dumps(data)
    return hashlib.sha256(data_string.encode()).hexdigest()

def data_validation(block):
    content = json.loads(block.data)
    return hash_data(content['data']) == content['hash']

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    start_time = MPI.Wtime()

    filepath = 'wdbc.csv'
    data, id_col, diag_col = load_and_preprocess_data(filepath)

    if rank == 0:
        passed, reduced_data, mse = evaluate_pca_quality(data)
        if not passed:
            print(f"PCA MSE too high: {mse:.6f}")
            sys.exit()
    else:
        reduced_data = None

    reduced_data = comm.bcast(reduced_data if rank == 0 else None, root=0)
    local_data = Sharding.distribute_columns(reduced_data, comm)
    
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

if __name__ == '__main__':
    main()


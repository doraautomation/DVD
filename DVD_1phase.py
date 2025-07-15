import hashlib
import datetime as date
import json
import pandas as pd
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from mpi4py import MPI
import os
import time
import sys
import random
import math

os.makedirs('output', exist_ok=True)

class Preprocessor:
    def __init__(self, filepath, n_components=180, mse_threshold=0.01):
        self.filepath = filepath
        self.n_components = n_components
        self.mse_threshold = mse_threshold

    def load_and_scale(self):
        df = pd.read_csv(self.filepath)
        features = df.drop(['Activity'], axis=1, errors='ignore')
        return StandardScaler().fit_transform(features)

    def apply_pca_and_check(self, data):
        pca = PCA(n_components=self.n_components)
        reduced = pca.fit_transform(data)
        mse = mean_squared_error(data, pca.inverse_transform(reduced))
        return mse < self.mse_threshold, reduced, mse

class Network:
    def __init__(self, num_nodes_per_cluster=5, total_clusters=1):
        self.num_nodes_per_cluster = num_nodes_per_cluster
        self.total_clusters = total_clusters
        self.clusters = self._create_clusters()

    def _create_clusters(self):
        clusters = {}
        node_id = 0
        for cluster_id in range(self.total_clusters):
            clusters[cluster_id] = []
            for _ in range(self.num_nodes_per_cluster):
                clusters[cluster_id].append(node_id)
                node_id += 1
        return clusters

    def get_nodes_for_cluster(self, cluster_id):
        return self.clusters.get(cluster_id, [])

    def display(self):
        pass

class Block:
    def __init__(self, index, timestamp, data, previous_hash):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        return hashlib.sha256((str(self.index) + str(self.timestamp) + str(self.data) + str(self.previous_hash)).encode()).hexdigest()

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
            if self.chain[i].hash != self.chain[i].calculate_hash():
                return False
            if self.chain[i].previous_hash != self.chain[i - 1].hash:
                return False
        return True

    def get_chain(self):
        with open('output/Blockchain.txt', 'w') as f:
            for block in self.chain:
                f.write(f"Block #{block.index}\nTimestamp: {block.timestamp}\nHash: {block.hash}\nPrevious Hash: {block.previous_hash}\nData: {block.data}\n\n")

    def simulate_faulty_nodes(self, sub_cluster_size, fault_percentage):
        # Calculate the number of faulty nodes based on the fault percentage
        num_faulty_nodes = int(sub_cluster_size * fault_percentage)  
        faulty_nodes = set(random.sample(range(sub_cluster_size), num_faulty_nodes))
        
        print(f"Fault percentage: {fault_percentage * 100:.2f}%")
        print(f"Number of faulty nodes: {num_faulty_nodes} out of {sub_cluster_size}")
        return faulty_nodes

    def consensus_success_rate(self, n, f, t):
        h = n - f
        failure_probability = 0
        if f >= t:
            return 0.0  
        
        for x in range(t, f + 1):
                failure_probability += math.comb(f, x) * (0.5 ** x) * (0.5 ** (f - x))
        
        success_rate = 1 - failure_probability
        return success_rate

    def consensus(self, block, rank, fault_percentage, sub_cluster_size=10):
        parsed_data = json.loads(block.data)
        data_hash = hash_data(parsed_data['data'])
        blk_hash = parsed_data['hash']

        votes = np.zeros(sub_cluster_size, dtype=bool)
        temp_commit_data = [None] * sub_cluster_size
        commit_event = threading.Event()

        # Simulate faulty nodes, all ranks will have the same faulty nodes
        faulty_nodes = self.simulate_faulty_nodes(sub_cluster_size, fault_percentage)

        # ------------------ Push Phase ------------------
        push_start = time.time()
        
        def send_prepare(i):
            if i in faulty_nodes:
                print(f"Node {i} is faulty and provides a wrong vote.")
                votes[i] = False
            else:
                if data_hash == blk_hash:
                    temp = parsed_data.copy()
                    temp['meta'] = {
                        'validator_node': i,
                        'coordinator_rank': rank,
                        'prepare_time': str(date.datetime.now()),
                        'status': 'PREPARED'
                    }
                    temp_commit_data[i] = temp
                    votes[i] = True

        with ThreadPoolExecutor(max_workers=sub_cluster_size) as executor:
            executor.map(send_prepare, range(sub_cluster_size))

        committed = False
        if votes.sum() >= int(sub_cluster_size * 0.51):
            coordinator_block = parsed_data.copy()
            coordinator_block['meta'] = {
                'coordinator_rank': rank,
                'commit_time': str(date.datetime.now()),
                'status': 'COORDINATOR_COMMITTED'
            }
            pd.DataFrame([coordinator_block]).to_csv(f'output/coordinator_commit_rank_{rank}.csv', index=False)
            commit_event.set()
            committed = True
        
        push_end = time.time()
        push_duration = push_end - push_start

        # ------------------ Pull Phase ------------------
        pull_start = time.time()
        
        def receive_commit(i):
            if votes[i]:
                commit_event.wait(timeout=3)
                temp_commit_data[i]['meta']['commit_time'] = str(date.datetime.now())
                temp_commit_data[i]['meta']['status'] = 'COMMITTED'
            else:
                if commit_event.wait(timeout=3):
                    fallback = parsed_data.copy()
                    fallback['meta'] = {
                        'coordinator_rank': rank,
                        'node_id': i,
                        'commit_time': str(date.datetime.now()),
                        'status': 'COMMIT_READ_FROM_LEDGER'
                    }
                    temp_commit_data[i] = fallback

        with ThreadPoolExecutor(max_workers=sub_cluster_size) as executor:
            executor.map(receive_commit, range(sub_cluster_size))
        
        pull_end = time.time()
        pull_duration = pull_end - pull_start   

        pd.DataFrame(temp_commit_data).to_csv(f'output/subcluster_all_nodes_coordinator_{rank}.csv', index=False)

        if committed:
            self.add_block(block)

        # Calculate consensus success rate
        consensus_rate = self.consensus_success_rate(sub_cluster_size, len(faulty_nodes), int(sub_cluster_size * 0.51))
        print(f"Consensus Success Rate: {consensus_rate * 100:.2f}%")

        return committed, push_duration, pull_duration


def hash_data(data):
    return hashlib.sha256(json.dumps(data).encode()).hexdigest()

def data_validation(block):
    try:
        content = json.loads(block.data)
        return hash_data(content['data']) == content['hash']
    except:
        return False

class KMeansProcessor:
    def __init__(self, k=5, num_steps=100):
        self.k = k
        self.num_steps = num_steps

    def initialize_centroids(self, data):
        indices = np.random.choice(data.shape[0], size=self.k, replace=False)
        return data[indices]

    def assign_clusters(self, data, centroids):
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        return np.argmin(distances, axis=1)

    def compute_centroids(self, data, labels):
        centroids = np.zeros((self.k, data.shape[1]))
        for i in range(self.k):
            cluster = data[labels == i]
            if len(cluster) > 0:
                centroids[i] = np.mean(cluster, axis=0)
        return centroids

    def run(self, local_data, comm, global_data=None):
        rank = comm.Get_rank()
        size = comm.Get_size()
        centroids = self.initialize_centroids(global_data) if rank == 0 else None
        centroids = comm.bcast(centroids, root=0)

        for _ in range(self.num_steps):
            labels = self.assign_clusters(local_data, centroids)
            local_centroids = self.compute_centroids(local_data, labels)
            global_centroids = np.zeros_like(local_centroids)
            comm.Allreduce(local_centroids, global_centroids, op=MPI.SUM)
            centroids = global_centroids / size

        return local_data

class ColumnShardProcessor:
    @staticmethod
    def distribute_columns(data, comm):
        rank = comm.Get_rank()
        size = comm.Get_size()
        n_cols = data.shape[1]
        per = n_cols // size
        rem = n_cols % size
        start = rank * per + min(rank, rem)
        end = start + per + (1 if rank < rem else 0)
        return data[:, start:end]

class KMeansRunner:
    def __init__(self, filepath, k=5, num_steps=100):
        self.filepath = filepath
        self.processor = KMeansProcessor(k, num_steps)

    def execute(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        start = MPI.Wtime()
        
        pre = Preprocessor(self.filepath)
        data = pre.load_and_scale()
        fault_percentage = random.random() * 0.49
        
        if rank == 0:
            passed, reduced, mse = pre.apply_pca_and_check(data)
            if not passed:
                print(f"PCA MSE too high: {mse:.6f}")
                sys.exit()
        else:
            reduced = None

        fault_percentage = comm.bcast(fault_percentage, root=0)
        reduced = comm.bcast(reduced, root=0)
        local_data = ColumnShardProcessor.distribute_columns(reduced, comm)
        local_data = np.array_split(reduced, comm.Get_size())[rank]

        clustered_data = self.processor.run(local_data, comm, reduced)

        blockchain = Blockchain()
        blk_data = {
            'coordinator': rank,
            'data': clustered_data.tolist(),
            'hash': hash_data(clustered_data.tolist())
        }
        blk = Block(rank + 1, date.datetime.now(), json.dumps(blk_data), "0")
        committed, push_duration, pull_duration = blockchain.consensus(blk, rank, fault_percentage)
        
        end = MPI.Wtime()
        
        push_times = comm.gather(push_duration, root=0)
        pull_times = comm.gather(pull_duration, root=0)
        
        all_blocks = comm.gather(blk if committed else None, root=0)
        
        if rank == 0:
            for b in all_blocks:
                if b and b.hash not in [blk.hash for blk in blockchain.chain]:
                    blockchain.add_block(b)
            total_vectors = reduced.shape[0]        
            avg_push = sum(push_times) / len(push_times)
            avg_pull = sum(pull_times) / len(pull_times)
            
            print(f"[Average] Push phase: {avg_push:.6f} sec, Pull phase: {avg_pull:.6f} sec")
            print(f"[K-means mode] Execution Time: {end - start:.4f} sec")
            print(f"[K-means mode] Throughput: {total_vectors / (end - start):.2f} vectors/sec")
            blockchain.get_chain()
            print("Blockchain is valid." if blockchain.is_valid() else "Blockchain is invalid!")

class ColumnShardRunner:
    def __init__(self, filepath):
        self.filepath = filepath

    def execute(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        start = MPI.Wtime()

        pre = Preprocessor(self.filepath)
        data = pre.load_and_scale()
        fault_percentage = random.random() * 0.49
        
        if rank == 0:
            passed, reduced, mse = pre.apply_pca_and_check(data)
            if not passed:
                print(f"PCA MSE too high: {mse:.6f}")
                sys.exit()
        else:
            reduced = None
        fault_percentage = comm.bcast(fault_percentage, root=0)
        reduced = comm.bcast(reduced, root=0)
        local_data = ColumnShardProcessor.distribute_columns(reduced, comm)

        blockchain = Blockchain()
        blk_data = {
            'coordinator': rank,
            'data': local_data.tolist(),
            'hash': hash_data(local_data.tolist())
        }
        blk = Block(rank + 1001, date.datetime.now(), json.dumps(blk_data), "0")
        committed, push_duration, pull_duration = blockchain.consensus(blk, rank, fault_percentage)
        
        end = MPI.Wtime()
        
        push_times = comm.gather(push_duration, root=0)
        pull_times = comm.gather(pull_duration, root=0)
        
        all_blocks = comm.gather(blk if committed else None, root=0)
        if rank == 0:
            for b in all_blocks:
                if b and b.hash not in [blk.hash for blk in blockchain.chain]:
                    blockchain.add_block(b)
            total_vectors = reduced.shape[0]
            avg_push = sum(push_times) / len(push_times)
            avg_pull = sum(pull_times) / len(pull_times)
            
            print(f"[Average] Push phase: {avg_push:.6f} sec, Pull phase: {avg_pull:.6f} sec")
            print(f"[Column mode] Execution Time: {end - start:.4f} sec")
            print(f"Column mode] Throughput: {total_vectors / (end - start):.2f} vectors/sec")
            blockchain.get_chain()
            print("Blockchain is valid." if blockchain.is_valid() else "Blockchain is invalid!")

if __name__ == "__main__":
    filepath = 'data1.csv'
    KMeansRunner(filepath).execute()
    #ColumnShardRunner(filepath).execute()

import hashlib
import datetime as date
import json
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from mpi4py import MPI
from blspy import PrivateKey, AugSchemeMPL, G1Element, G2Element
import os
import threading
import sys

os.makedirs('output', exist_ok=True)

class Preprocessor:
    def __init__(self, filepath, n_components=17, mse_threshold=0.01):
        self.filepath = filepath
        self.n_components = n_components
        self.mse_threshold = mse_threshold

    def load_and_scale(self):
        df = pd.read_csv(self.filepath)
        features = df.drop(['ID', 'Diagnosis'], axis=1, errors='ignore')
        return StandardScaler().fit_transform(features)

    def apply_pca_and_check(self, data):
        pca = PCA(n_components=self.n_components)
        reduced = pca.fit_transform(data)
        mse = mean_squared_error(data, pca.inverse_transform(reduced))
        return mse < self.mse_threshold, reduced, mse

class Block:
    def __init__(self, index, timestamp, data, previous_hash, bls_signature=None):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.bls_signature = bls_signature  # Store aggregated BLS signature
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

    def consensus(self, block, rank, sub_cluster_size=20):
        block_hash = hashlib.sha256(block.data.encode()).digest()
        signatures = []
        pubkeys = []
        lock = threading.Lock()

        def validator_bls(i):
            seed = hashlib.sha256(f"node-{rank}-thread-{i}".encode()).digest()
            sk = AugSchemeMPL.key_gen(seed)
            pk = sk.get_g1()
            if data_validation(block):
                sig = AugSchemeMPL.sign(sk, block_hash)
                with lock:
                    signatures.append(sig)
                    pubkeys.append(pk)

        threads = [threading.Thread(target=validator_bls, args=(i,)) for i in range(sub_cluster_size)]
        for t in threads: t.start()
        for t in threads: t.join()

        if len(signatures) >= int(sub_cluster_size * 0.67):
            agg_sig = AugSchemeMPL.aggregate(signatures)
            if AugSchemeMPL.aggregate_verify(pubkeys, [block_hash] * len(pubkeys), agg_sig):
                # Store aggregated BLS signature
                block.bls_signature = bytes(agg_sig).hex()

                for i in range(sub_cluster_size):
                    block_df = pd.DataFrame([{
                        'index': block.index,
                        'timestamp': block.timestamp,
                        'data': block.data,
                        'previous_hash': block.previous_hash,
                        'hash': block.hash,
                        'bls_signature': block.bls_signature
                    }])
                    block_df.to_csv(f'output/subcluster_node_{i}_coordinator_{rank}.csv', index=False)

                self.add_block(block)
                return True

        # 2PQC
        prepare_results = [False] * sub_cluster_size

        def prepare_phase(i):
            prepare_results[i] = data_validation(block)

        prepare_threads = [threading.Thread(target=prepare_phase, args=(i,)) for i in range(sub_cluster_size)]
        for t in prepare_threads: t.start()
        for t in prepare_threads: t.join()

        if sum(prepare_results) >= int(sub_cluster_size * 0.51):
            def commit_phase(i):
                block_df = pd.DataFrame([{
                    'index': block.index,
                    'timestamp': block.timestamp,
                    'data': block.data,
                    'previous_hash': block.previous_hash,
                    'hash': block.hash,
                    'bls_signature': block.bls_signature  # might be None
                }])
                block_df.to_csv(f'output/subcluster_node_{i}_coordinator_{rank}.csv', index=False)

            commit_threads = [threading.Thread(target=commit_phase, args=(i,)) for i in range(sub_cluster_size)]
            for t in commit_threads: t.start()
            for t in commit_threads: t.join()

            self.add_block(block)
            return True

        return False

def hash_data(data):
    return hashlib.sha256(json.dumps(data).encode()).hexdigest()

def data_validation(block):
    try:
        content = json.loads(block.data)
        return hash_data(content['data']) == content['hash']
    except:
        return False

def get_chain(blockchain, filename):
    output = []
    for block in blockchain.chain:
        output.append(
            f"Block #{block.index}\n"
            f"Timestamp: {block.timestamp}\n"
            f"Data: {block.data}\n"
            f"Hash: {block.hash}\n"
            f"PrevHash: {block.previous_hash}\n"
            f"BLS Signature: {block.bls_signature}\n"
        )
    with open(filename, "w") as f:
        f.write("\n".join(output))

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

        for i in range(self.num_steps):
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
        reduced = pre.apply_pca_and_check(data)[1] if rank == 0 else None
        reduced = comm.bcast(reduced, root=0)
        local_data = np.array_split(reduced, comm.Get_size())[rank]

        clustered_data = self.processor.run(local_data, comm, reduced)

        blockchain = Blockchain()
        blk_data = {
            'coordinator': rank,
            'data': clustered_data.tolist(),
            'hash': hash_data(clustered_data.tolist())
        }
        blk = Block(rank + 1, date.datetime.now(), json.dumps(blk_data), "0")
        committed = blockchain.consensus(blk, rank)

        if committed:
            pd.DataFrame([{
                'index': blk.index,
                'timestamp': blk.timestamp,
                'data': blk.data,
                'previous_hash': blk.previous_hash,
                'hash': blk.hash
            }]).to_csv(f'output/kmeans_block_rank_{rank}.csv', index=False)

        end = MPI.Wtime()
        all_blocks = comm.gather(blk if committed else None, root=0)
        if rank == 0:
            for b in all_blocks:
                if b and b.hash not in [blk.hash for blk in blockchain.chain]:
                    blockchain.add_block(b)
            get_chain(blockchain, "output/kmeans_blockchain.txt")
            print(f"[KMEANS MODE] Execution Time: {end - start:.4f} sec")

class ColumnShardRunner:
    def __init__(self, filepath):
        self.filepath = filepath

    def execute(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        start = MPI.Wtime()

        pre = Preprocessor(self.filepath)
        data = pre.load_and_scale()
        reduced = pre.apply_pca_and_check(data)[1] if rank == 0 else None
        reduced = comm.bcast(reduced, root=0)
        local_data = ColumnShardProcessor.distribute_columns(reduced, comm)

        blockchain = Blockchain()
        blk_data = {
            'coordinator': rank,
            'data': local_data.tolist(),
            'hash': hash_data(local_data.tolist())
        }
        blk = Block(rank + 1001, date.datetime.now(), json.dumps(blk_data), "0")
        committed = blockchain.consensus(blk, rank)

        if committed:
            pd.DataFrame([{
                'index': blk.index,
                'timestamp': blk.timestamp,
                'data': blk.data,
                'previous_hash': blk.previous_hash,
                'hash': blk.hash
            }]).to_csv(f'output/column_block_rank_{rank}.csv', index=False)

        end = MPI.Wtime()
        all_blocks = comm.gather(blk if committed else None, root=0)
        if rank == 0:
            for b in all_blocks:
                if b and b.hash not in [blk.hash for blk in blockchain.chain]:
                    blockchain.add_block(b)
            get_chain(blockchain, "output/column_blockchain.txt")
            print(f"[COLUMN SHARD MODE] Execution Time: {end - start:.4f} sec")

if __name__ == "__main__":
    filepath = 'wdbc.csv'
    KMeansRunner(filepath).execute()
    ColumnShardRunner(filepath).execute()

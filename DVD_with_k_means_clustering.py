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
import hyperloglog
from typing import List, Dict, Optional

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

        # Encoding: fit + transform
        t_enc_start = time.perf_counter()
        reduced = pca.fit_transform(data)
        t_enc_end = time.perf_counter()
        encoding_time = t_enc_end - t_enc_start

        # Reconstruction: inverse transform
        t_rec_start = time.perf_counter()
        reconstructed = pca.inverse_transform(reduced)
        t_rec_end = time.perf_counter()
        reconstruction_time = t_rec_end - t_rec_start

        # Reconstruction error
        mse = mean_squared_error(data, reconstructed)

        passed = mse < self.mse_threshold

        return {
            "passed": passed,
            "reduced": reduced,
            "reconstructed": reconstructed,
            "mse": mse,
            "encoding_time": encoding_time,
            "reconstruction_time": reconstruction_time,
            "n_components": self.n_components,
            "original_dim": data.shape[1],
        }


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


class ExpanderOverlay:
    def __init__(self, n: int, d: int = 4, seed: int = 42, max_tries: int = 50):
        if d <= 0:
            raise ValueError("d must be > 0")
        if d >= n:
            raise ValueError("d must be < n")
        if (n * d) % 2 != 0:
            raise ValueError("n*d must be even")

        self.n = n
        self.d = d
        self.seed = seed
        self.neighbors: List[List[int]] = self._build(max_tries)

    def _build(self, max_tries: int) -> List[List[int]]:
        for attempt in range(max_tries):
            rng = random.Random(self.seed + attempt)
            stubs = []
            for i in range(self.n):
                stubs.extend([i] * self.d)
            rng.shuffle(stubs)

            adj = [set() for _ in range(self.n)]
            ok = True

            for k in range(0, len(stubs), 2):
                a, b = stubs[k], stubs[k + 1]
                if (
                    a == b
                    or b in adj[a]
                    or len(adj[a]) >= self.d
                    or len(adj[b]) >= self.d
                ):
                    ok = False
                    break
                adj[a].add(b)
                adj[b].add(a)

            if ok and all(len(adj[i]) == self.d for i in range(self.n)):
                return [sorted(list(s)) for s in adj]

        raise RuntimeError("Failed to build expander overlay")

    def rounds(self) -> int:
        return int(math.ceil(math.log2(self.n))) + 1


def expander_gossip_maps(
    initial_maps: List[Dict],
    neighbors: List[List[int]],
    rounds: int
) -> List[Dict]:

    known = [dict(m) for m in initial_maps]
    n = len(known)

    for _ in range(rounds):
        new_known = [dict(k) for k in known]
        for i in range(n):
            for nb in neighbors[i]:
                new_known[i].update(known[nb])
        known = new_known

    return known


class Block:
    def __init__(self, index, timestamp, data, previous_hash):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        return hashlib.sha256(
            (str(self.index) + str(self.timestamp) + str(self.data) + str(self.previous_hash)).encode()
        ).hexdigest()
        
def expander_gossip_or_int(
    initial_vals: List[int],
    neighbors: List[List[int]],
    rounds: int
) -> List[int]:
    known = initial_vals[:]
    n = len(known)

    for _ in range(rounds):
        new_known = known[:]  # copy
        for i in range(n):
            acc = known[i]
            for nb in neighbors[i]:
                acc |= known[nb]
            new_known[i] = acc
        known = new_known

    return known


def pack_decision(committed: bool, known: bool) -> int:
    return (1 if known else 0) | ((1 if committed else 0) << 1)


def unpack_decision(x: int) -> (bool, bool):
    known = bool(x & 1)
    committed = bool((x >> 1) & 1)
    return committed, known


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
                f.write(
                    f"Block #{block.index}\nTimestamp: {block.timestamp}\nHash: {block.hash}\n"
                    f"Previous Hash: {block.previous_hash}\nData: {block.data}\n\n"
                )

    def simulate_faulty_nodes(self, sub_cluster_size, fault_percentage):
        num_faulty_nodes = int(sub_cluster_size * fault_percentage)
        faulty_nodes = set(random.sample(range(sub_cluster_size), num_faulty_nodes))
        print(f"Fault percentage: {fault_percentage * 100:.2f}%")
        print(f"Number of faulty nodes: {num_faulty_nodes} out of {sub_cluster_size}")
        return faulty_nodes

    def consensus_success_rate(self, n, f, t):
        if f >= t:
            return 0.0
        failure_probability = 0
        for x in range(t, f + 1):
            failure_probability += math.comb(f, x) * (0.5 ** x) * (0.5 ** (f - x))
        return 1 - failure_probability

    def consensus(self, block, rank, fault_percentage, sub_cluster_size=10):
        parsed_data = json.loads(block.data)
        data_hash = hash_data(parsed_data['data'])
        blk_hash = parsed_data['hash']

        hll_ok = True
        if 'hll_estimate' in parsed_data:
            est_commit = float(parsed_data['hll_estimate'])

            hll_check = hyperloglog.HyperLogLog(parsed_data.get('error_rate', 0.01))
            for item in parsed_data['data']:
                hll_check.add(data_to_bytes(item))
            est = len(hll_check)

            rel_tol = 3.0 * parsed_data.get('error_rate', 0.01)
            hll_ok = abs(est - est_commit) <= rel_tol * max(est_commit, 1.0)

        expander_degree = 4
        overlay = ExpanderOverlay(sub_cluster_size, d=expander_degree, seed=123)
        R = overlay.rounds()

        votes = np.zeros(sub_cluster_size, dtype=bool)
        temp_commit_data = [None] * sub_cluster_size

        faulty_nodes = self.simulate_faulty_nodes(sub_cluster_size, fault_percentage)

        # ------------------ Push Phase (Prepare + Vote Gossip) ------------------
        push_start = time.time()

        def local_prepare(i):
            if i in faulty_nodes:
                votes[i] = False
                temp_commit_data[i] = None
                return

            if (data_hash == blk_hash) and hll_ok:
                temp = parsed_data.copy()
                temp['meta'] = temp.get('meta', {})
                temp['meta'].update({
                    'validator_node': i,
                    'coordinator_rank': rank,
                    'prepare_time': str(date.datetime.now()),
                    'status': 'PREPARED'
                })
                temp_commit_data[i] = temp
                votes[i] = True
            else:
                votes[i] = False
                temp_commit_data[i] = None

        with ThreadPoolExecutor(max_workers=sub_cluster_size) as executor:
            list(executor.map(local_prepare, range(sub_cluster_size)))
        
        initial_yes_masks = [
            (1 << i) if votes[i] else 0
            for i in range(sub_cluster_size)
        ]

        yes_masks = expander_gossip_or_int(initial_yes_masks, overlay.neighbors, rounds=R)
        union_yes_mask = yes_masks[0]
        yes_votes_seen = bin(union_yes_mask).count("1")


        majority_validators = (sub_cluster_size // 2) + 1
        committed = (yes_votes_seen >= majority_validators)

        #initial = [{i: bool(votes[i])} for i in range(sub_cluster_size)]
        #known_votes = expander_gossip_maps(initial, overlay.neighbors, rounds=R)

        #majority_maps = (sub_cluster_size // 2) + 1 
        #majority_validators = (sub_cluster_size // 2) + 1
        #global_vote = np.zeros(sub_cluster_size, dtype=bool)

        #for j in range(sub_cluster_size):
            #true_count = 0
            #for i in range(sub_cluster_size):
                # Missing keys are treated as False (conservative)
                #if known_votes[i].get(j, False):
                    #true_count += 1
            #global_vote[j] = (true_count >= majority_maps)

        #yes_votes_seen = int(np.sum(global_vote))

        #committed = (yes_votes_seen >= majority_validators)
       
        if committed:
            coordinator_block = parsed_data.copy()
            coordinator_block['meta'] = coordinator_block.get('meta', {})
            coordinator_block['meta'].update({
                'coordinator': rank,
                'commit_time': str(date.datetime.now()),
                'status': 'COORDINATOR_COMMITTED',
                'votes_seen': yes_votes_seen,
                'overlay_degree': expander_degree,
                'gossip_rounds': R,
                'decision_rule': 'bitmask_or_union',
                'majority_validators': majority_validators,
                'yes_mask_hex': hex(union_yes_mask) 
                #'decision_rule': 'aggregate_all_maps_majority',
                #'majority_maps': majority_maps,
                #'majority_validators': majority_validators
            })
            pd.DataFrame([coordinator_block]).to_csv(f'output/coordinator_commit_rank_{rank}.csv', index=False)
            committed = True

        push_end = time.time()
        push_duration = push_end - push_start

        # ------------------ Pull Phase (Commit Decision Gossip) ------------------
        pull_start = time.time()

        decision_key = "DECISION"
        decision_initial = [{} for _ in range(sub_cluster_size)]
        seed_node = 0
        #decision_initial = [pack_decision(committed=False, known=False) for _ in range(sub_cluster_size)]
        #decision_initial[seed_node] = pack_decision(committed=committed, known=True)

        #decision_states = expander_gossip_or_int(decision_initial, overlay.neighbors, rounds=R)
        
        decision_initial[seed_node] = {decision_key: committed}
        decision_known = expander_gossip_maps(decision_initial, overlay.neighbors, rounds=R)

        def apply_commit(i):
            #dec_value, dec_known = unpack_decision(decision_states[i])
            #decided = dec_known and dec_value
            decided = decision_known[i].get(decision_key, False)

            if decided:
                if votes[i] and temp_commit_data[i] is not None:
                    temp_commit_data[i]['meta'] = temp_commit_data[i].get('meta', {})
                    temp_commit_data[i]['meta']['commit_time'] = str(date.datetime.now())
                    temp_commit_data[i]['meta']['status'] = 'COMMITTED' 
                else:
                    fallback = parsed_data.copy()
                    fallback['meta'] = fallback.get('meta', {})
                    fallback['meta'].update({
                          'coordinator_rank': rank,
                          'node_id': i,
                          'commit_time': str(date.datetime.now()),
                          'status': 'COMMIT_READ_FROM_LEDGER'
                    })
                    temp_commit_data[i] = fallback
            
            else:
                    # Abort path
                if temp_commit_data[i] is None:
                    # ensure record exists for logging
                    temp_commit_data[i] = parsed_data.copy()
                    temp_commit_data[i]['meta'] = temp_commit_data[i].get('meta', {})
                temp_commit_data[i]['meta']['commit_time'] = str(date.datetime.now())
                temp_commit_data[i]['meta']['status'] = 'ABORTED'
                
        with ThreadPoolExecutor(max_workers=sub_cluster_size) as executor:
            list(executor.map(apply_commit, range(sub_cluster_size)))

        pull_end = time.time()
        pull_duration = pull_end - pull_start

        pd.DataFrame(temp_commit_data).to_csv(f'output/subcluster_all_nodes_coordinator_{rank}.csv', index=False)

        if committed:
            self.add_block(block)

        consensus_rate = self.consensus_success_rate(sub_cluster_size, len(faulty_nodes), int(sub_cluster_size * 0.51))
        print(f"Consensus Success Rate: {consensus_rate * 100:.2f}%")
        print(f"[Expander] degree={expander_degree}, rounds={R}, seed_node={seed_node}, yes_seen={yes_votes_seen}")

        return committed, push_duration, pull_duration


def hash_data(data):
    return hashlib.sha256(json.dumps(data).encode()).hexdigest()


def data_to_bytes(item, float_ndigits=6):
    def norm(x):
        if isinstance(x, float):
            return round(x, float_ndigits)
        if isinstance(x, (list, tuple)):
            return [norm(v) for v in x]
        return x

    canon = json.dumps(norm(item), separators=(',', ':'), ensure_ascii=False)
    return hashlib.blake2b(canon.encode('utf-8'), digest_size=8).digest()


def data_validation(block):
    try:
        content = json.loads(block.data)
        return hash_data(content['data']) == content['hash']
    except Exception:
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

        clustered_data = np.column_stack((local_data, labels))
        return clustered_data


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
        #fault_percentage = 1.0
        fault_percentage = random.random() * 0.49

        # ------------------ PCA (rank 0 only), then broadcast reduced ------------------
        if rank == 0:
            pca_result = pre.apply_pca_and_check(data)
            if not pca_result["passed"]:
                print(f"PCA MSE too high: {pca_result['mse']:.6f}")
                sys.exit()

            reduced = pca_result["reduced"]

            pd.DataFrame([{
                "timestamp": str(date.datetime.now()),
                "original_dim": pca_result["original_dim"],
                "reduced_dim": pca_result["n_components"],
                "mse": pca_result["mse"],
                "encoding_time_sec": pca_result["encoding_time"],
                "reconstruction_time_sec": pca_result["reconstruction_time"],
            }]).to_csv("output/pca_reconstruction_metrics.csv", index=False)

            print(
                f"[PCA] original_dim={pca_result['original_dim']} "
                f"reduced_dim={pca_result['n_components']} "
                f"MSE={pca_result['mse']:.6f} "
                f"encode_time={pca_result['encoding_time']:.6f}s "
                f"reconstruct_time={pca_result['reconstruction_time']:.6f}s"
            )
        else:
            reduced = None
            pca_result = None  # not used on other ranks

        fault_percentage = comm.bcast(fault_percentage, root=0)
        reduced = comm.bcast(reduced, root=0)

        # ------------------ Parallel  ------------------
        local_data = ColumnShardProcessor.distribute_columns(reduced, comm)
        local_data = np.array_split(reduced, comm.Get_size())[rank]

        clustered_data = self.processor.run(local_data, comm, reduced)

        blockchain = Blockchain()

        # HLL: build sketch for this block's data
        hll = hyperloglog.HyperLogLog(0.01)  # 1% error rate
        for item in clustered_data.tolist():
            hll.add(data_to_bytes(item))

        blk_data = {
            'coordinator': rank,
            'data': clustered_data.tolist(),
            'hash': hash_data(clustered_data.tolist()),
            'hll_estimate': len(hll),
            'error_rate': 0.01
        }

    
        pca_metrics = None
        if rank == 0:
            pca_metrics = {
                "original_dim": pca_result["original_dim"],
                "reduced_dim": pca_result["n_components"],
                "mse": float(pca_result["mse"]),
                "encoding_time_sec": float(pca_result["encoding_time"]),
                "reconstruction_time_sec": float(pca_result["reconstruction_time"]),
                "mse_threshold": float(pre.mse_threshold),
            }
        pca_metrics = comm.bcast(pca_metrics, root=0)
        blk_data["pca_metrics"] = pca_metrics

        blk = Block(rank + 1, date.datetime.now(), json.dumps(blk_data), "0")
        committed, push_duration, pull_duration = blockchain.consensus(blk, rank, fault_percentage)

        end = MPI.Wtime()

        push_times = comm.gather(push_duration, root=0)
        pull_times = comm.gather(pull_duration, root=0)
        all_blocks = comm.gather(blk if committed else None, root=0)

        if rank == 0:
            for b in all_blocks:
                if b and b.hash not in [x.hash for x in blockchain.chain]:
                    blockchain.add_block(b)

            total_vectors = reduced.shape[0]
            avg_push = sum(push_times) / len(push_times)
            avg_pull = sum(pull_times) / len(pull_times)

            print(f"[Average] Push phase: {avg_push:.6f} sec, Pull phase: {avg_pull:.6f} sec")
            print(f"[K-means mode] Execution Time: {end - start:.4f} sec")
            print(f"[K-means mode] Throughput: {total_vectors / (end - start):.2f} vectors/sec")
            blockchain.get_chain()
            print("Blockchain is valid." if blockchain.is_valid() else "Blockchain is invalid!")
            ledger_read(blockchain)

def ledger_read(blockchain):
    t0 = time.perf_counter()
    vectors_read = 0
    bytes_read = 0

    for block in blockchain.chain[1:]:  
        bytes_read += len(block.data.encode("utf-8"))
        try:
            content = json.loads(block.data)
            if "data" in content:
                vectors_read += len(content["data"])
        except Exception:
            pass

    t1 = time.perf_counter()
    read_time = t1 - t0
    throughput = vectors_read / read_time if read_time > 0 else 0
    mb_read = bytes_read / (1024 * 1024)

    print("\n========== Ledger Read After Insertion ==========")
    print(f"Vectors read            : {vectors_read}")
    print(f"Total read time (sec)   : {read_time:.6f}")
    print(f"Read throughput (v/s)   : {throughput:.2f}")
    print(f"Bytes read              : {bytes_read}")
    print(f"Data read (MB)          : {mb_read:.3f}")
    print("================================================\n")


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

        # ------------------ PCA (rank 0 only), then broadcast reduced ------------------
        if rank == 0:
            pca_result = pre.apply_pca_and_check(data)
            if not pca_result["passed"]:
                print(f"PCA MSE too high: {pca_result['mse']:.6f}")
                sys.exit()

            reduced = pca_result["reduced"]

            # (3) Log to file (rank 0 only)
            pd.DataFrame([{
                "timestamp": str(date.datetime.now()),
                "original_dim": pca_result["original_dim"],
                "reduced_dim": pca_result["n_components"],
                "mse": pca_result["mse"],
                "encoding_time_sec": pca_result["encoding_time"],
                "reconstruction_time_sec": pca_result["reconstruction_time"],
            }]).to_csv("output/pca_reconstruction_metrics.csv", index=False)

            print(
                f"[PCA] original_dim={pca_result['original_dim']} "
                f"reduced_dim={pca_result['n_components']} "
                f"MSE={pca_result['mse']:.6f} "
                f"encode_time={pca_result['encoding_time']:.6f}s "
                f"reconstruct_time={pca_result['reconstruction_time']:.6f}s"
            )
        else:
            reduced = None
            pca_result = None

        fault_percentage = comm.bcast(fault_percentage, root=0)
        reduced = comm.bcast(reduced, root=0)

        local_data = ColumnShardProcessor.distribute_columns(reduced, comm)

        blockchain = Blockchain()

        hll = hyperloglog.HyperLogLog(0.01)
        for item in local_data.tolist():
            hll.add(data_to_bytes(item))

        blk_data = {
            'coordinator': rank,
            'data': local_data.tolist(),
            'hash': hash_data(local_data.tolist()),
            'hll_estimate': len(hll),
            'error_rate': 0.01
        }

        # (4) Optional: attach PCA metrics to block metadata
        pca_metrics = None
        if rank == 0:
            pca_metrics = {
                "original_dim": pca_result["original_dim"],
                "reduced_dim": pca_result["n_components"],
                "mse": float(pca_result["mse"]),
                "encoding_time_sec": float(pca_result["encoding_time"]),
                "reconstruction_time_sec": float(pca_result["reconstruction_time"]),
                "mse_threshold": float(pre.mse_threshold),
            }
        pca_metrics = comm.bcast(pca_metrics, root=0)
        blk_data["pca_metrics"] = pca_metrics

        blk = Block(rank + 1001, date.datetime.now(), json.dumps(blk_data), "0")
        committed, push_duration, pull_duration = blockchain.consensus(blk, rank, fault_percentage)

        end = MPI.Wtime()

        push_times = comm.gather(push_duration, root=0)
        pull_times = comm.gather(pull_duration, root=0)

        all_blocks = comm.gather(blk if committed else None, root=0)
        if rank == 0:
            for b in all_blocks:
                if b and b.hash not in [x.hash for x in blockchain.chain]:
                    blockchain.add_block(b)

            total_vectors = reduced.shape[0]
            avg_push = sum(push_times) / len(push_times)
            avg_pull = sum(pull_times) / len(pull_times)

            print(f"[Average] Push phase: {avg_push:.6f} sec, Pull phase: {avg_pull:.6f} sec")
            print(f"[Column mode] Execution Time: {end - start:.4f} sec")
            print(f"[Column mode] Throughput: {total_vectors / (end - start):.2f} vectors/sec")
            blockchain.get_chain()
            print("Blockchain is valid." if blockchain.is_valid() else "Blockchain is invalid!")
            ledger_read(blockchain)


if __name__ == "__main__":
    filepath = 'data.csv'
    KMeansRunner(filepath).execute()
    #ColumnShardRunner(filepath).execute()

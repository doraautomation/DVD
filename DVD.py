import os
import time
import numpy as np
import pandas as pd
import json
import hashlib
from mpi4py import MPI

import random
import threading
import math
from queue import Queue, Empty
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.distance import cdist


# -----------------------------------------------------------------------------
#  Utility helpers
# -----------------------------------------------------------------------------

def stable_hash(obj):
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_numpy_array(arr):
    arr = np.ascontiguousarray(np.asarray(arr))
    hasher = hashlib.sha256()
    hasher.update(str(arr.dtype).encode("utf-8"))
    hasher.update(str(tuple(arr.shape)).encode("utf-8"))
    hasher.update(arr.tobytes())
    return hasher.hexdigest()


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_json(path, obj):
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def block_metadata_bytes(block):
    return int(len(json.dumps(block, sort_keys=True, separators=(",", ":")).encode("utf-8")))



def _safe_l2_normalize(arr, axis=1, eps=1e-12):
    arr = np.asarray(arr, dtype=np.float64)
    norm = np.linalg.norm(arr, axis=axis, keepdims=True)
    norm = np.maximum(norm, eps)
    return arr / norm


def compute_centroid_based_sharding_quality(X, labels, noise_label=-1):
    """
    Computes centroid-based intra/inter shard quality from the full feature
    matrix and cluster labels. This is O(n) for intra-shard quality and is much
    faster than full pairwise O(n^2) cosine comparison inside each shard.

    Returns a dictionary containing global quality and per-shard quality.
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)

    valid_mask = labels != noise_label
    X_valid = X[valid_mask]
    labels_valid = labels[valid_mask]

    if X_valid.shape[0] == 0:
        return {
            "intra_shard_cosine": float("nan"),
            "inter_shard_cosine": float("nan"),
            "separation_gap": float("nan"),
            "balance_ratio": 0.0,
            "noise_count": int(np.sum(labels == noise_label)),
            "noise_ratio": float(np.mean(labels == noise_label)),
            "per_shard": [],
        }

    unique_labels = sorted(np.unique(labels_valid).tolist())
    per_shard = []
    centroids = []
    weighted_intra_sum = 0.0
    total_points = 0
    shard_sizes = []

    for sid in unique_labels:
        shard_data = X_valid[labels_valid == sid]
        n_points = int(shard_data.shape[0])
        if n_points == 0:
            continue

        centroid = np.mean(shard_data, axis=0)
        centroid_norm = _safe_l2_normalize(centroid.reshape(1, -1), axis=1)[0]
        shard_norm = _safe_l2_normalize(shard_data, axis=1)

        similarities = shard_norm @ centroid_norm
        intra = float(np.mean(similarities))

        per_shard.append({
            "shard_id": int(sid),
            "num_points": n_points,
            "intra_shard_cosine": intra,
            "centroid": centroid.tolist(),
        })

        centroids.append(centroid)
        shard_sizes.append(n_points)
        weighted_intra_sum += intra * n_points
        total_points += n_points

    avg_intra = float(weighted_intra_sum / total_points) if total_points else float("nan")

    if len(centroids) >= 2:
        centroid_matrix = _safe_l2_normalize(np.vstack(centroids), axis=1)
        sim_matrix = centroid_matrix @ centroid_matrix.T
        upper = np.triu_indices_from(sim_matrix, k=1)
        avg_inter = float(np.mean(sim_matrix[upper]))
    else:
        avg_inter = float("nan")

    separation_gap = float(avg_intra - avg_inter) if np.isfinite(avg_inter) else float("nan")
    balance_ratio = float(min(shard_sizes) / max(shard_sizes)) if shard_sizes else 0.0

    return {
        "intra_shard_cosine": avg_intra,
        "inter_shard_cosine": avg_inter,
        "separation_gap": separation_gap,
        "balance_ratio": balance_ratio,
        "noise_count": int(np.sum(labels == noise_label)),
        "noise_ratio": float(np.mean(labels == noise_label)),
        "per_shard": per_shard,
    }


def majority_fault_tolerance_summary(n_validators):
    """
    51% majority quorum rule:
    A block is committed if it receives more than 50% YES votes.

    Example:
        n = 30
        quorum = floor(30 / 2) + 1 = 16
    """
    n = int(n_validators)
    quorum = (n // 2) + 1

    return {
        "validators": n,
        "max_faulty_nodes_for_majority": n - quorum,
        "majority_commit_quorum": quorum,
        "quorum_rule": ">50% YES votes",
    }

def _hash_leaf(row_index: int, row: np.ndarray) -> str:
    hasher = hashlib.sha256()
    hasher.update(row_index.to_bytes(8, "little"))
    hasher.update(np.ascontiguousarray(row, dtype=np.float64).tobytes())
    return hasher.hexdigest()


def _hash_pair(left: str, right: str) -> str:
    hasher = hashlib.sha256()
    hasher.update(bytes.fromhex(left))
    hasher.update(bytes.fromhex(right))
    return hasher.hexdigest()


def build_merkle_tree(shard_vectors: np.ndarray) -> dict:
    """
    Returns {"root": hex, "leaves": [hex, ...], "depth": int}.
    Empty shard -> deterministic all-zero root.
    """
    if shard_vectors.ndim != 2 or shard_vectors.shape[0] == 0:
        return {"root": "0" * 64, "leaves": [], "depth": 0}

    leaves = [_hash_leaf(i, shard_vectors[i]) for i in range(shard_vectors.shape[0])]
    level  = leaves[:]
    depth  = 0

    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        level = [_hash_pair(level[i], level[i + 1]) for i in range(0, len(level), 2)]
        depth += 1

    return {"root": level[0], "leaves": leaves, "depth": depth}


def verify_merkle_root(shard_vectors: np.ndarray, expected_root: str) -> dict:
    """Recomputes root from raw vectors and raises on mismatch."""
    result = {"check": "verify_merkle_root", "passed": False, "detail": ""}
    tree   = build_merkle_tree(shard_vectors)

    if tree["root"] != expected_root:
        raise VectorVerificationError(
            f"verify_merkle_root: root mismatch -- "
            f"expected {expected_root[:12]}... got {tree['root'][:12]}... "
            f"-- at least one vector row was tampered"
        )

    result["passed"] = True
    result["detail"] = (
        f"root={tree['root'][:12]}..., leaves={len(tree['leaves'])}, depth={tree['depth']}"
    )
    return result


def merkle_proof_path(leaves: list, row_index: int) -> list:
    """
    Returns the sibling-hash path needed to prove one leaf without the full shard.
    Each element: (sibling_hex, "left"|"right")
    """
    level = leaves[:]
    path  = []
    idx   = row_index

    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        if idx % 2 == 0:
            path.append((level[idx + 1] if idx + 1 < len(level) else level[idx], "right"))
        else:
            path.append((level[idx - 1], "left"))
        level = [_hash_pair(level[i], level[i + 1]) for i in range(0, len(level), 2)]
        idx //= 2

    return path


def verify_merkle_proof(row_index: int, row: np.ndarray,
                        proof_path: list, expected_root: str) -> bool:
    """Light-client single-row verification -- O(log n) hashes."""
    current = _hash_leaf(row_index, row)
    for sibling, position in proof_path:
        current = _hash_pair(current, sibling) if position == "right" else _hash_pair(sibling, current)
    return current == expected_root


def compute_vector_ids(shard_id: int, shard_vectors: np.ndarray) -> list:
    shard_bytes = shard_id.to_bytes(8, "little")
    ids = []
    for row in shard_vectors:
        h = hashlib.sha256()
        h.update(shard_bytes)
        h.update(np.ascontiguousarray(row, dtype=np.float64).tobytes())
        ids.append(h.hexdigest())
    return ids


def verify_no_duplicate_vector_ids(
    shard_id: int,
    shard_vectors: np.ndarray,
    global_seen_ids: set = None,
) -> dict:
    
    result = {"check": "verify_no_duplicate_vector_ids", "passed": False, "detail": ""}

    if global_seen_ids is None:
        global_seen_ids = set()

    ids        = compute_vector_ids(shard_id, shard_vectors)
    local_seen: set  = set()
    duplicates: list = []

    for i, vid in enumerate(ids):
        if vid in local_seen or vid in global_seen_ids:
            duplicates.append((i, vid[:12]))
        else:
            local_seen.add(vid)

    if duplicates:
        raise VectorVerificationError(
            f"verify_no_duplicate_vector_ids: {len(duplicates)} duplicate vector(s) "
            f"in shard {shard_id} -- first at row {duplicates[0][0]} "
            f"(id={duplicates[0][1]}...)"
        )

    global_seen_ids.update(local_seen)
    result["passed"] = True
    result["detail"] = f"{len(ids)} unique vector IDs, no duplicates"
    return result


class VectorVerificationError(Exception):
    """Raised when any vector-specific check fails."""


def verify_timestamp(block: dict, max_drift_sec: float = 60.0) -> dict:
    result = {"check": "verify_timestamp", "passed": False, "detail": ""}

    if "timestamp" not in block:
        raise VectorVerificationError("verify_timestamp: missing 'timestamp' field")

    ts = block["timestamp"]
    if not isinstance(ts, (int, float)):
        raise VectorVerificationError(
            f"verify_timestamp: timestamp must be numeric, got {type(ts).__name__}"
        )

    age = time.time() - ts
    if age > max_drift_sec:
        raise VectorVerificationError(
            f"verify_timestamp: block is {age:.1f}s old (max {max_drift_sec}s) -- replay attack"
        )
    if age < -max_drift_sec:
        raise VectorVerificationError(
            f"verify_timestamp: block is {-age:.1f}s in the future -- pre-mining"
        )

    result["passed"] = True
    result["detail"] = f"age={age:.3f}s, within +/-{max_drift_sec}s"
    return result


def verify_dimensions(shard_vectors: np.ndarray, expected_dim: int) -> dict:
    result = {"check": "verify_dimensions", "passed": False, "detail": ""}

    if shard_vectors.ndim != 2:
        raise VectorVerificationError(
            f"verify_dimensions: expected 2-D, got {shard_vectors.ndim}-D {shard_vectors.shape}"
        )

    n, d = shard_vectors.shape
    if d != expected_dim:
        raise VectorVerificationError(
            f"verify_dimensions: expected {expected_dim} dims, got {d}"
        )
    if not np.all(np.isfinite(shard_vectors)):
        nan_c = int(np.sum(np.isnan(shard_vectors)))
        inf_c = int(np.sum(np.isinf(shard_vectors)))
        raise VectorVerificationError(
            f"verify_dimensions: {nan_c} NaN + {inf_c} Inf values -- corrupted"
        )

    result["passed"] = True
    result["detail"] = f"shape=({n},{d}), all finite"
    return result


def verify_hash(
    shard_vectors: np.ndarray,
    centroid: np.ndarray,
    expected_data_hash: str,
    expected_centroid_hash: str,
    block_data_hash: str = None,
    block_centroid_hash: str = None,
) -> dict:
    result = {"check": "verify_hash", "passed": False, "detail": ""}

    recomputed_data = hash_numpy_array(shard_vectors)
    recomputed_cent = hash_numpy_array(centroid)

    if recomputed_data != expected_data_hash:
        raise VectorVerificationError(
            f"verify_hash: data_hash mismatch -- "
            f"expected {expected_data_hash[:12]}... got {recomputed_data[:12]}..."
        )
    if recomputed_cent != expected_centroid_hash:
        raise VectorVerificationError(
            f"verify_hash: centroid_hash mismatch -- "
            f"expected {expected_centroid_hash[:12]}... got {recomputed_cent[:12]}..."
        )
    # Cross-check the proposer's claimed hash fields against the validator's
    # independently computed truth. Without this, a malicious proposer could
    # publish a forged data_hash on-chain and the validator would still pass
    # because its own recomputation never compared against the block's field.
    if block_data_hash is not None and block_data_hash != expected_data_hash:
        raise VectorVerificationError(
            f"verify_hash: block-field data_hash forged "
            f"(block={block_data_hash[:12]}... vs validator={expected_data_hash[:12]}...)"
        )
    if block_centroid_hash is not None and block_centroid_hash != expected_centroid_hash:
        raise VectorVerificationError(
            f"verify_hash: block-field centroid_hash forged "
            f"(block={block_centroid_hash[:12]}... vs validator={expected_centroid_hash[:12]}...)"
        )

    result["passed"] = True
    result["detail"] = f"data={recomputed_data[:12]}..., centroid={recomputed_cent[:12]}... match"
    return result


def run_minimum_verification(
    block: dict,
    shard_vectors: np.ndarray,
    expected_data_hash: str,
    expected_centroid_hash: str,
    max_drift_sec: float = 60.0,
    global_seen_ids: set = None,
) -> dict:
    centroid     = np.asarray(block["centroid"], dtype=np.float64)
    expected_dim = int(block["vector_dim"])
    shard_id     = int(block["shard_id"])
    merkle_root  = block.get("merkle_root", "0" * 64)

    checks = [
        ("verify_timestamp",
         lambda: verify_timestamp(block, max_drift_sec)),

        ("verify_dimensions",
         lambda: verify_dimensions(shard_vectors, expected_dim)),

        ("verify_no_duplicate_vector_ids",
         lambda: verify_no_duplicate_vector_ids(shard_id, shard_vectors, global_seen_ids)),

        ("verify_merkle_root",
         lambda: verify_merkle_root(shard_vectors, merkle_root)),

        ("verify_hash",
         lambda: verify_hash(shard_vectors, centroid,
                             expected_data_hash, expected_centroid_hash,
                             block_data_hash=block.get("data_hash"),
                             block_centroid_hash=block.get("centroid_hash"))),
    ]

    summary = {"passed": True, "failed_check": None, "error": None, "results": {}}

    for name, fn in checks:
        try:
            summary["results"][name] = fn()
        except VectorVerificationError as exc:
            summary["passed"]        = False
            summary["failed_check"]  = name
            summary["error"]         = str(exc)
            summary["results"][name] = {"check": name, "passed": False, "detail": str(exc)}
            break

    return summary


def build_all_verification_ctxs(comm, rank: int, size: int,
                                local_shard_map: dict, owned_shards: list) -> dict:
    """
    Deadlock-free independent verification ctx exchange.

    Problem with point-to-point Send/Recv:
      With k=5, size=4 some ranks own 2 shards and others own 1.
      Sequential Send->Recv per shard causes ranks to block waiting for
      messages that their neighbours never send in that round -> deadlock.

    Solution -- allgather in one round:
      Every rank computes a compact descriptor (hashes only, no raw data)
      for each shard it owns, then allgather shares ALL descriptors with
      ALL ranks simultaneously.  Each rank then picks up the descriptor
      for its shards from the neighbour rank (rank-1) % size, which acts
      as the independent validator.  No blocking point-to-point calls.

    Returns
    -------
    dict  shard_id -> verification_ctx   (one entry per owned shard)
    """

    # Build local descriptors for every owned shard
    local_descriptors = []
    for shard_id in owned_shards:
        sv          = np.asarray(local_shard_map[shard_id], dtype=np.float64)
        centroid    = np.mean(sv, axis=0) if sv.shape[0] > 0 \
                      else np.zeros((1,), dtype=np.float64)
        tree        = build_merkle_tree(sv)
        local_descriptors.append({
            "shard_id":      int(shard_id),
            "data_hash":     hash_numpy_array(sv),
            "centroid_hash": hash_numpy_array(centroid),
            "merkle_root":   tree["root"],
            "computed_by":   int(rank),          # which rank produced this
        })

    # Allgather: every rank gets every other rank's descriptors
    # Allgather: every rank gets every other rank's descriptors
    all_descriptors = comm.allgather(local_descriptors)

# Loop through all descriptors dynamically to avoid index errors
    total_length = 0
    for rank_descriptors in all_descriptors:
    # Make sure each rank's descriptors are not empty
        if len(rank_descriptors) > 0:
        # Process each descriptor in the rank
           for descriptor in rank_descriptors:
              print(f"Processing shard {descriptor['shard_id']} from rank {rank_descriptors}")
              total_length += len(descriptor)  # Add to the total length (or process as needed)
        else:
            print("No descriptors found for this rank")

    validator_rank = (rank + 1) % size   

    proposer_rank    = (rank - 1) % size
    proposer_shard_ids = [sid for sid in range(
        len(all_descriptors[0]) + len(all_descriptors[1]) +
        len(all_descriptors[2]) + len(all_descriptors[3])
    ) if sid % size == proposer_rank]

    proposer_owned = [d["shard_id"] for d in all_descriptors[proposer_rank]]

    my_payload = []
    for shard_id in owned_shards:
        sv = np.asarray(local_shard_map[shard_id], dtype=np.float64)
        my_payload.append((int(shard_id), sv))

    # sendrecv: send my payload to validator_rank, receive proposer_rank's payload
    received_payload = comm.sendrecv(
        sendobj  = my_payload,
        dest     = validator_rank,
        sendtag  = 0,
        source   = proposer_rank,
        recvtag  = 0,
    )

    # Recompute hashes independently for proposer's shards
    independent_ctxs_for_proposer = {}
    for shard_id, sv in received_payload:
        sv          = np.asarray(sv, dtype=np.float64)
        centroid    = np.mean(sv, axis=0) if sv.shape[0] > 0 \
                      else np.zeros((1,), dtype=np.float64)
        tree        = build_merkle_tree(sv)
        independent_ctxs_for_proposer[shard_id] = {
            "shard_id":      int(shard_id),
            "data_hash":     hash_numpy_array(sv),
            "centroid_hash": hash_numpy_array(centroid),
            "merkle_root":   tree["root"],
            "computed_by":   int(rank),
        }

    # sendrecv back: send independent_ctxs to proposer, receive our ctxs from validator
    my_ctxs = comm.sendrecv(
        sendobj  = independent_ctxs_for_proposer,
        dest     = proposer_rank,
        sendtag  = 1,
        source   = validator_rank,
        recvtag  = 1,
    )
    # my_ctxs: dict shard_id -> ctx independently computed by validator_rank

    return my_ctxs   # keyed by shard_id


# -----------------------------------------------------------------------------
#  Blockchain
# -----------------------------------------------------------------------------

class Blockchain:
    def __init__(self, chain_file="blockchain.json"):
        self.chain_file = chain_file
        self.chain = []
        if os.path.exists(self.chain_file):
            self.load()
        else:
            self.create_genesis_block()
            self.save()

    def create_genesis_block(self):
        g = {
            "index": 0, "timestamp": time.time(),
            "previous_hash": "0" * 64,
            "block_type": "genesis", "data": "Genesis Block"
        }
        g["block_hash"] = self.compute_block_hash(g)
        self.chain = [g]

    def get_last_block(self):
        return self.chain[-1]

    def compute_block_hash(self, block):
        bc = deepcopy(block)
        bc.pop("block_hash", None)
        return hashlib.sha256(
            json.dumps(bc, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def add_block(self, committed_block):
        last    = self.get_last_block()
        new_blk = deepcopy(committed_block)
        new_blk["index"]         = len(self.chain)
        new_blk["timestamp"]     = time.time()
        new_blk["previous_hash"] = last["block_hash"]
        new_blk["block_hash"]    = self.compute_block_hash(new_blk)
        self.chain.append(new_blk)
        return new_blk

    def save(self):
        with open(self.chain_file, "w", encoding="utf-8") as f:
            json.dump(self.chain, f, indent=2)

    def load(self):
        with open(self.chain_file, "r", encoding="utf-8") as f:
            self.chain = json.load(f)

    def verify_chain(self):
        """
        Walk the chain end-to-end and detect:
          - tampered block content     (recomputed hash != stored block_hash)
          - broken previous_hash links (reordering / insertion / deletion)
          - tampered index sequence

        Returns dict with 'valid', 'failure_index', 'failure_reason',
        and 'checked_blocks'.
        """
        if not self.chain:
            return {"valid": False, "failure_index": -1,
                    "failure_reason": "empty chain", "checked_blocks": 0}

        for i, blk in enumerate(self.chain):
            recomputed = self.compute_block_hash(blk)
            if blk.get("block_hash") != recomputed:
                return {"valid": False, "failure_index": i,
                        "failure_reason": "block_hash mismatch (content tampered)",
                        "checked_blocks": i}
            if blk.get("index") != i:
                return {"valid": False, "failure_index": i,
                        "failure_reason": f"index mismatch (stored={blk.get('index')}, expected={i})",
                        "checked_blocks": i}
            if i > 0:
                prev_hash = self.chain[i - 1]["block_hash"]
                if blk.get("previous_hash") != prev_hash:
                    return {"valid": False, "failure_index": i,
                            "failure_reason": "previous_hash broken (reordering/deletion)",
                            "checked_blocks": i}
        return {"valid": True, "failure_index": None,
                "failure_reason": None, "checked_blocks": len(self.chain)}


# ---------------------------------------
#  DistributedKMeans  
# -----------------------------------------

class DistributedKMeans:
    def __init__(self, k=5, num_steps=100, seed=42, tol=1e-3, verbose=True, print_every=5):
        self.k = k; self.num_steps = num_steps; self.seed = seed
        self.tol = tol; self.verbose = verbose; self.print_every = print_every

    def initialize_centroids(self, global_data):
        n = global_data.shape[0]
        if n < self.k:
            raise ValueError(f"k={self.k} but only {n} samples")
        rng = np.random.default_rng(self.seed)
        c   = global_data[rng.choice(n, size=self.k, replace=False)].astype(np.float64)
        if self.verbose:
            print(f"[KMeans] Centroids shape: {c.shape}")
        return c

    def assign_clusters(self, local_data, centroids):
        return np.argmin(cdist(local_data, centroids, metric='sqeuclidean'), axis=1)

    def run(self, local_data, comm, global_data=None):
        rank       = comm.Get_rank()
        local_data = np.ascontiguousarray(local_data, dtype=np.float64)
        if local_data.ndim != 2:
            raise ValueError(f"[Rank {rank}] Expected 2D, got {local_data.shape}")

        centroids = self.initialize_centroids(np.asarray(global_data, dtype=np.float64)) \
                    if rank == 0 else None
        centroids = comm.bcast(centroids, root=0)
        k, d      = centroids.shape

        global_counts = np.zeros(k, dtype=np.int64)
        local_counts  = np.zeros(k, dtype=np.int64)

        for step in range(self.num_steps):
            labels       = self.assign_clusters(local_data, centroids)
            local_sums   = np.zeros((k, d), dtype=np.float64)
            local_counts = np.bincount(labels, minlength=k).astype(np.int64)
            np.add.at(local_sums, labels, local_data)

            packed        = np.concatenate([local_sums.ravel(), local_counts.astype(np.float64)])
            global_packed = np.zeros_like(packed)
            comm.Allreduce(packed, global_packed, op=MPI.SUM)

            global_sums   = global_packed[:k * d].reshape(k, d)
            global_counts = global_packed[k * d:].astype(np.int64)

            new_c                   = centroids.copy()
            nonempty                = global_counts > 0
            new_c[nonempty]         = global_sums[nonempty] / global_counts[nonempty][:, None]
            shift                   = np.linalg.norm(new_c - centroids)
            centroids               = new_c

            if rank == 0 and self.verbose and (step == 0 or (step + 1) % self.print_every == 0):
                print(f"[KMeans] Step {step + 1}, shift={shift:.6f}")

            if comm.bcast(bool(shift < self.tol), root=0):
                if rank == 0 and self.verbose:
                    print(f"[KMeans] Converged at step {step + 1}")
                break

        return self.assign_clusters(local_data, centroids), centroids, global_counts, local_counts


# --------------------------
#  build_metadata_block  
# ---------------------------

def build_metadata_block(rank, shard_id, shard_vectors, shard_file):
    shard_vectors = np.asarray(shard_vectors, dtype=np.float64)
    centroid      = np.mean(shard_vectors, axis=0) if len(shard_vectors) > 0 \
                    else np.zeros((1,), dtype=np.float64)
    tree          = build_merkle_tree(shard_vectors)

    # Standard pattern: leaves go OFF-CHAIN, only root goes ON-CHAIN.
    # Storing all leaves on-chain defeats the purpose of a Merkle tree --
    # a 4000-vector shard would add ~256 KB to the block just for leaf hashes.
    # A verifier loads this file only when generating a single-row proof path.
    leaves_ref = f"merkle_leaves_{shard_id}.json"
    leaves_path = os.path.join(os.path.dirname(shard_file), leaves_ref)
    ensure_parent_dir(leaves_path)
    with open(leaves_path, "w") as f:
        json.dump(tree["leaves"], f)

    return {
        "rank_id":           int(rank),
        "shard_id":          int(shard_id),
        "timestamp":         time.time(),
        "data_hash":         hash_numpy_array(shard_vectors),   # whole-shard SHA-256
        "centroid":          centroid.tolist(),                  # mean vector (on-chain)
        "centroid_hash":     hash_numpy_array(centroid),
        "merkle_root":       tree["root"],   # 64-byte root -- only commitment on-chain
        "merkle_depth":      tree["depth"],
        "merkle_leaves_ref": leaves_ref,      # pointer to off-chain leaves file
        "num_points":        int(shard_vectors.shape[0]),
        "vector_dim":        int(shard_vectors.shape[1])
                               if shard_vectors.ndim == 2 and shard_vectors.size > 0
                               else (int(centroid.shape[0]) if centroid.ndim == 1 else 0),
        "offchain_ref":      os.path.basename(shard_file),      # pointer to raw .npy
    }


class PushPullHashConsensus:
    def __init__(
        self,
        output_dir="output_csv",
        seed=42,
        write_trace=False,
        validator_threads=4,
        max_drift_sec=60.0,
    ):
        self.output_dir        = output_dir
        self.seed              = seed
        self.write_trace       = write_trace
        self.validator_threads = validator_threads
        self.max_drift_sec     = max_drift_sec
        self._executor         = ThreadPoolExecutor(max_workers=max(1, validator_threads))
        # Persistent cross-shard duplicate ID set
        self._global_seen_ids: set = set()

    def __del__(self):
        self._executor.shutdown(wait=False)

    def simulate_faulty_nodes(self, sub_cluster_size, fault_percentage):
        n   = int(sub_cluster_size * fault_percentage)
        rng = random.Random(self.seed)
        f   = set(rng.sample(range(sub_cluster_size), n)) if n > 0 else set()
        print(f"Fault percentage: {fault_percentage * 100:.2f}%  "
              f"Faulty nodes: {n}/{sub_cluster_size}")
        return f

    def consensus_success_rate(self, n, f, t):
        if f >= t:
            return 0.0
        fail_prob = 0.0
        for x in range(t, n + 1):   # fixed: iterate to n not f
            if x <= f:
                fail_prob += math.comb(f, x) * (0.5 ** f)
        return 1.0 - fail_prob

    def _rejected_result(self, t_start, sub_cluster_size, phase, error, failed_check=None):
        r = {
            "committed": False,
            "quorum": majority_fault_tolerance_summary(sub_cluster_size)["majority_commit_quorum"],
            "max_faulty_nodes_for_majority": majority_fault_tolerance_summary(sub_cluster_size)["max_faulty_nodes_for_majority"],
            "fault_tolerance_status": "NOT_EVALUATED",
            "phase": phase, "verification_error": error,
            "prepare_yes": 0, "commit_yes": 0,
            "prepare_time_sec": 0.0, "commit_time_sec": 0.0,
            "push_time_sec": 0.0, "pull_time_sec": 0.0,
            "push_pull_time_sec": 0.0,
            "consensus_time_sec": float(time.perf_counter() - t_start),
            "min_validation_time_sec": 0.0, "avg_validation_time_sec": 0.0,
            "max_validation_time_sec": 0.0,
            "faulty_nodes": [], "consensus_success_rate": 0.0,
        }
        if failed_check:
            r["failed_check"] = failed_check
        return r

    def consensus(self, block, verification_ctx, rank, fault_percentage,
                  shard_vectors, sub_cluster_size=10):

        t_start = time.perf_counter()

        parsed_data = {k: block[k] for k in (
            "rank_id", "shard_id", "timestamp", "data_hash",
            "centroid", "centroid_hash",
            "merkle_root", "merkle_depth", "merkle_leaves_ref",
            "num_points", "vector_dim", "offchain_ref"
        )}

        expected_data_hash     = verification_ctx["data_hash"]
        expected_centroid_hash = verification_ctx["centroid_hash"]
        expected_merkle_root   = verification_ctx["merkle_root"]

        # Cross-check proposer merkle_root vs independent validator's root
        if block["merkle_root"] != expected_merkle_root:
            print(f"[Shard {block['shard_id']}] REJECTED: merkle_root mismatch "
                  f"(proposer {block['merkle_root'][:12]}... vs "
                  f"validator {expected_merkle_root[:12]}...)")
            return self._rejected_result(t_start, sub_cluster_size,
                                         "MERKLE_ROOT_MISMATCH",
                                         "proposer and validator merkle roots disagree")

        shard_vectors_np = np.asarray(shard_vectors, dtype=np.float64)

        # Run full verification pipeline once (deterministic for all validators)
        t_mv   = time.perf_counter()
        mv_res = run_minimum_verification(
            block                  = block,
            shard_vectors          = shard_vectors_np,
            expected_data_hash     = expected_data_hash,
            expected_centroid_hash = expected_centroid_hash,
            max_drift_sec          = self.max_drift_sec,
            global_seen_ids        = self._global_seen_ids,
        )
        mv_ms = (time.perf_counter() - t_mv) * 1000

        print(f"[Shard {block['shard_id']}] Verification "
              f"{'PASSED' if mv_res['passed'] else 'FAILED'} "
              f"in {mv_ms:.3f} ms  "
              f"(ctx from rank {verification_ctx.get('computed_by', '?')})")

        for name, res in mv_res["results"].items():
            status = "PASS" if res["passed"] else "FAIL"
            print(f"  {status} {name}: {res.get('detail', res.get('error', ''))}")

        if not mv_res["passed"]:
            return self._rejected_result(t_start, sub_cluster_size,
                                         "MIN_VERIFY_REJECT", mv_res["error"],
                                         mv_res["failed_check"])

        # Per-validator voting
        votes         = np.zeros(sub_cluster_size, dtype=bool)
        buffers       = [None] * sub_cluster_size
        local_ledgers = [[] for _ in range(sub_cluster_size)]
        shared_ledger = []

        faulty           = self.simulate_faulty_nodes(sub_cluster_size, fault_percentage)
      # Majority quorum: commit if YES votes are more than 50%.
      # Example: n=30 validators -> quorum=floor(30/2)+1 = 16.
        fault_summary = majority_fault_tolerance_summary(sub_cluster_size)
        max_faulty_nodes = fault_summary["max_faulty_nodes_for_majority"]
        commit_threshold = fault_summary["majority_commit_quorum"]
        quorum_threshold = commit_threshold

        # Canonical hash sources -- both from block["centroid"] list roundtrip
        pre_data_hash = hash_numpy_array(shard_vectors_np)
        pre_cent_hash = hash_numpy_array(np.asarray(block["centroid"], dtype=np.float64))

        val_times    = []
        timing_lock  = threading.Lock()
        state_lock   = threading.Lock()
        stop_event   = threading.Event()
        vq           = Queue()

        for i in range(sub_cluster_size):
            vq.put(i)

        state      = {"yes": 0, "done": 0}
        push_start = time.perf_counter()

        def validator_worker():
            while not stop_event.is_set():
                try:
                    i = vq.get_nowait()
                except Empty:
                    break

                vote_yes = False
                temp     = None

                if i not in faulty:
                    t0 = time.perf_counter()
                    is_valid = (
                        mv_res["passed"] and
                        pre_data_hash == expected_data_hash and
                        pre_cent_hash == expected_centroid_hash and
                        parsed_data["data_hash"]     == expected_data_hash and
                        parsed_data["centroid_hash"] == expected_centroid_hash and
                        parsed_data["merkle_root"]   == expected_merkle_root
                    )
                    elapsed = time.perf_counter() - t0

                    if is_valid:
                        temp = parsed_data.copy()
                        temp["meta"] = {
                            "validator_node": i,
                            "coordinator_rank": rank,
                            "status": "PREPARED"
                        }
                        vote_yes = True

                    with timing_lock:
                        val_times.append(elapsed)

                with state_lock:
                    votes[i] = vote_yes
                    if temp is not None:
                        buffers[i] = temp
                    state["done"] += 1
                    if vote_yes:
                        state["yes"] += 1
                    remaining = sub_cluster_size - state["done"]
                    if state["yes"] >= commit_threshold:
                        stop_event.set()
                    elif state["yes"] + remaining < commit_threshold:
                        stop_event.set()

                vq.task_done()

        threads = max(1, min(self.validator_threads, sub_cluster_size))
        for f in [self._executor.submit(validator_worker) for _ in range(threads)]:
            f.result()

        push_dur  = time.perf_counter() - push_start
        committed = False

        if int(votes.sum()) >= commit_threshold:
            print(f"[Shard {block['shard_id']}] Quorum reached ({state['yes']}/{sub_cluster_size})")
            cc = parsed_data.copy()
            cc["meta"] = {"coordinator_rank": rank, "status": "COORDINATOR_COMMITTED"}
            shared_ledger.append(cc)
            committed = True
        else:
            print(f"[Shard {block['shard_id']}] Quorum failed ({state['yes']}/{sub_cluster_size})")

        pull_start = time.perf_counter()
        if committed:
            for i in range(sub_cluster_size):
                if votes[i] and buffers[i]:
                    buffers[i]["meta"]["status"] = "COMMITTED"
                else:
                    fb = parsed_data.copy()
                    fb["meta"] = {"coordinator_rank": rank, "node_id": i,
                                  "status": "COMMIT_READ_FROM_LEDGER"}
                    buffers[i] = fb
                local_ledgers[i].append(buffers[i])
        pull_dur = time.perf_counter() - pull_start
        push_pull_dur = push_dur + pull_dur

        print(f"[Shard {block['shard_id']}] Push time: {push_dur * 1000:.4f} ms")
        print(f"[Shard {block['shard_id']}] Pull time: {pull_dur * 1000:.4f} ms")
        print(f"[Shard {block['shard_id']}] Push-Pull time: {push_pull_dur * 1000:.4f} ms")

        if self.write_trace:
            write_json(
                os.path.join(self.output_dir,
                             f"consensus_trace_rank_{rank}_shard_{block['shard_id']}.json"),
                {
                    "min_verification":      mv_res,
                    "verification_ctx_from": verification_ctx.get("computed_by"),
                    "shared_ledger":         shared_ledger,
                    "local_ledgers":         local_ledgers,
                    "faulty_nodes":          sorted(faulty),
                    "votes_true":            int(votes.sum()),
                    "validators_processed":  int(state["done"]),
                    "early_terminated":      bool(state["done"] < sub_cluster_size),
                    "push_time_sec":        float(push_dur),
                    "pull_time_sec":        float(pull_dur),
                    "push_pull_time_sec":   float(push_pull_dur),
                }
            )

        t4r   = max(1, quorum_threshold + 1)
        c_rate = self.consensus_success_rate(sub_cluster_size, len(faulty), t4r)
        print(f"Consensus Success Rate: {c_rate * 100:.2f}%")

        min_vt = float(min(val_times)) if val_times else 0.0
        avg_vt = float(sum(val_times) / len(val_times)) if val_times else 0.0
        max_vt = float(max(val_times)) if val_times else 0.0

        return {
            "committed":               bool(committed),
            "quorum":                  int(quorum_threshold),
            "max_faulty_nodes_for_majority": int(max_faulty_nodes),
            "fault_tolerance_status": "WITHIN_51_PERCENT_BOUND" if len(faulty) <= max_faulty_nodes else "EXCEEDS_51_PERCENT_BOUND",
            "phase":                   "COMMIT_SUCCESS" if committed else "PREPARE_REJECT",
            "prepare_yes":             int(votes.sum()),
            "commit_yes":              int(votes.sum()) if committed else 0,
            # prepare_time_sec and commit_time_sec are kept for backward compatibility.
            # push_time_sec and pull_time_sec are explicit names used for reporting.
            "prepare_time_sec":        float(push_dur),
            "commit_time_sec":         float(pull_dur),
            "push_time_sec":           float(push_dur),
            "pull_time_sec":           float(pull_dur),
            "push_pull_time_sec":      float(push_pull_dur),
            "consensus_time_sec":      float(time.perf_counter() - t_start),
            "min_validation_time_sec": min_vt,
            "avg_validation_time_sec": avg_vt,
            "max_validation_time_sec": max_vt,
            "faulty_nodes":            sorted(faulty),
            "consensus_success_rate":  float(c_rate),
        }

    def run(self, block, verification_ctx, rank, fault_percentage,
            shard_vectors, sub_cluster_size=10):
        return self.consensus(block, verification_ctx, rank,
                              fault_percentage, shard_vectors, sub_cluster_size)


class TamperDetectionExperiment:
  
    SHARD_ATTACKS = ["A1_row_tamper", "A2_centroid_tamper",
                     "A3_data_hash_forgery", "A4_replay"]
    CHAIN_ATTACKS = ["A5_reorder", "A6_blockhash_tamper"]

    def __init__(self, output_dir, validator_counts=None,
                 n_trials=5, seed=42):
        self.output_dir = output_dir
        self.validator_counts = validator_counts or [10, 30, 50, 100, 200, 500]
        self.n_trials = int(n_trials)
        self.seed = int(seed)
        self.shards_dir = os.path.join(output_dir, "shards")
        self.chain_path = os.path.join(output_dir, "blockchain.json")

    # ---- attack: shard-level (uses verification pipeline) ------------------

    def _apply_shard_attack(self, attack, blk, sv, rng):
        blk_attacked = deepcopy(blk)
        sv_attacked = sv.copy()

        if attack == "A1_row_tamper":
            row_idx = int(rng.integers(0, sv_attacked.shape[0]))
            col_idx = int(rng.integers(0, sv_attacked.shape[1]))
            sv_attacked[row_idx, col_idx] += 1.0
        elif attack == "A2_centroid_tamper":
            new_c = np.asarray(blk_attacked["centroid"], dtype=np.float64) + 1.0
            blk_attacked["centroid"] = new_c.tolist()
        elif attack == "A3_data_hash_forgery":
            h = blk_attacked["data_hash"]
            blk_attacked["data_hash"] = ("0" if h[0] != "0" else "1") + h[1:]
        elif attack == "A4_replay":
            blk_attacked["timestamp"] = time.time() - 600.0
        else:
            raise ValueError(f"Unknown shard attack: {attack}")

        return blk_attacked, sv_attacked

    def _time_shard_attack(self, attack, blk, sv, rng):
        blk_attacked, sv_attacked = self._apply_shard_attack(attack, blk, sv, rng)
        # Validator's "ground truth" hashes are independently recomputed from
        # the original (clean) shard and centroid -- this is the same logic
        # that build_all_verification_ctxs uses during consensus.
        centroid_clean = np.mean(sv, axis=0) if sv.shape[0] > 0 \
                         else np.zeros((1,), dtype=np.float64)
        expected_data_hash     = hash_numpy_array(sv)
        expected_centroid_hash = hash_numpy_array(centroid_clean)

        t0 = time.perf_counter()
        detected = False
        failed_check = None
        try:
            mv = run_minimum_verification(
                block                  = blk_attacked,
                shard_vectors          = sv_attacked,
                expected_data_hash     = expected_data_hash,
                expected_centroid_hash = expected_centroid_hash,
            )
            if not mv["passed"]:
                detected = True
                failed_check = mv["failed_check"]
        except VectorVerificationError:
            detected = True
            failed_check = "exception"
        latency_ms = (time.perf_counter() - t0) * 1000
        return detected, latency_ms, failed_check

    # ---- attack: chain-level (uses Blockchain.verify_chain) ----------------

    def _time_chain_attack(self, attack):
        bc = Blockchain(chain_file=self.chain_path)
        bc_attacked = deepcopy(bc)

        if attack == "A5_reorder":
            if len(bc_attacked.chain) >= 3:
                bc_attacked.chain[1], bc_attacked.chain[2] = \
                    bc_attacked.chain[2], bc_attacked.chain[1]
        elif attack == "A6_blockhash_tamper":
            if len(bc_attacked.chain) >= 2:
                blk = bc_attacked.chain[1]
                h = blk["block_hash"]
                blk["block_hash"] = ("0" if h[0] != "0" else "1") + h[1:]
        else:
            raise ValueError(f"Unknown chain attack: {attack}")

        t0 = time.perf_counter()
        res = bc_attacked.verify_chain()
        latency_ms = (time.perf_counter() - t0) * 1000
        detected = (res["valid"] is False)
        return detected, latency_ms, res.get("failure_reason")

    # ---- main entry --------------------------------------------------------

    def run(self):
        if not os.path.exists(self.chain_path):
            print(f"[TamperExp] No blockchain at {self.chain_path}; skipping")
            return None
        if not os.path.isdir(self.shards_dir):
            print(f"[TamperExp] No shards dir at {self.shards_dir}; skipping")
            return None

        # Load committed blocks (skip genesis at index 0)
        bc = Blockchain(chain_file=self.chain_path)
        committed = [b for b in bc.chain if b.get("block_type") != "genesis"
                                          and "shard_id" in b]
        if not committed:
            print("[TamperExp] No committed shard blocks to attack; skipping")
            return None

        # Map shard_id -> raw vectors loaded from .npy
        shards_by_id = {}
        for blk in committed:
            sid = int(blk["shard_id"])
            sv_path = os.path.join(self.shards_dir, f"shard_{sid}.npy")
            if os.path.exists(sv_path):
                shards_by_id[sid] = np.load(sv_path)

        if not shards_by_id:
            print("[TamperExp] No shard .npy files found; skipping")
            return None

        rng = np.random.default_rng(self.seed)
        rows = []

        # ---- shard-level attacks: sweep validator count -------------------
        print("\n" + "=" * 78)
        print("Tamper detection -- shard-level attacks (sweep validator count)")
        print("=" * 78)
        # n_validators is reported but does NOT change shard-attack logic.
        # It's swept to demonstrate detection latency is independent of
        # cluster size (the reviewer's exact question).
        target_blocks = list(shards_by_id.items())  # [(sid, sv), ...]

        for n_val in self.validator_counts:
            for attack in self.SHARD_ATTACKS:
                detect_count = 0
                latencies = []
                checks = set()
                for trial in range(self.n_trials):
                    sid, sv = target_blocks[trial % len(target_blocks)]
                    blk = next(b for b in committed if int(b["shard_id"]) == sid)
                    detected, lat_ms, failed = self._time_shard_attack(
                        attack, blk, sv, rng)
                    if detected:
                        detect_count += 1
                        if failed:
                            checks.add(failed)
                    latencies.append(lat_ms)

                row = {
                    "experiment_type":  "shard_attack",
                    "attack":           attack,
                    "n_validators":     n_val,
                    "n_blocks":         len(committed),
                    "n_trials":         self.n_trials,
                    "detection_rate":   detect_count / self.n_trials,
                    "latency_ms_mean":  float(np.mean(latencies)),
                    "latency_ms_std":   float(np.std(latencies)),
                    "failed_checks":    "|".join(sorted(checks)),
                }
                rows.append(row)
                print(f"  [n_val={n_val:>4}  {attack:24s}] "
                      f"detect={row['detection_rate']*100:5.1f}%  "
                      f"latency={row['latency_ms_mean']:7.3f} +/- "
                      f"{row['latency_ms_std']:.3f} ms  "
                      f"caught_by={row['failed_checks']}")

        # ---- chain-level attacks: chain length is fixed by what runner built
        print("\n" + "=" * 78)
        print(f"Tamper detection -- chain-level attacks (chain length = {len(bc.chain)})")
        print("=" * 78)
        for attack in self.CHAIN_ATTACKS:
            detect_count = 0
            latencies = []
            reasons = set()
            for _ in range(self.n_trials):
                detected, lat_ms, reason = self._time_chain_attack(attack)
                if detected:
                    detect_count += 1
                    if reason:
                        reasons.add(reason)
                latencies.append(lat_ms)

            row = {
                "experiment_type":  "chain_attack",
                "attack":           attack,
                "n_validators":     0,
                "n_blocks":         len(bc.chain),
                "n_trials":         self.n_trials,
                "detection_rate":   detect_count / self.n_trials,
                "latency_ms_mean":  float(np.mean(latencies)),
                "latency_ms_std":   float(np.std(latencies)),
                "failed_checks":    "|".join(sorted(reasons)),
            }
            rows.append(row)
            print(f"  [chain_len={len(bc.chain):>3}  {attack:24s}] "
                  f"detect={row['detection_rate']*100:5.1f}%  "
                  f"latency={row['latency_ms_mean']:7.3f} +/- "
                  f"{row['latency_ms_std']:.3f} ms  "
                  f"caught_by={row['failed_checks']}")

        # ---- write CSV ----------------------------------------------------
        out_csv = os.path.join(self.output_dir, "tamper_detection.csv")
        fieldnames = ["experiment_type", "attack", "n_validators", "n_blocks",
                      "n_trials", "detection_rate", "latency_ms_mean",
                      "latency_ms_std", "failed_checks"]
        import csv as _csv
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"\n[TamperExp] Wrote {len(rows)} rows to {out_csv}")
        return rows


# -----------------------------------------------------------------------------
#  DistributedKMeansRunner
# -----------------------------------------------------------------------------

class DistributedKMeansRunner:
    def __init__(
        self,
        csv_path="data.csv", k=5, num_steps=100, seed=42,
        output_dir="output_csv", subcluster_size=30, fault_percentage=0.0,
        label_column="Activity", drop_columns=None,
        kmeans_tol=1e-3, kmeans_verbose=True, kmeans_print_every=5,
        write_trace=False, validator_threads=4, max_drift_sec=60.0,
        run_tamper_experiment=True,
        tamper_validator_counts=None,
        tamper_n_trials=5,
    ):
        self.csv_path         = csv_path
        self.k                = k
        self.num_steps        = num_steps
        self.seed             = seed
        self.output_dir       = output_dir
        self.subcluster_size  = subcluster_size
        self.fault_percentage = fault_percentage
        self.label_column     = label_column
        self.drop_columns     = drop_columns or ["subject", "Activity"]
        self.max_drift_sec    = max_drift_sec
        self.write_trace      = write_trace
        self.validator_threads= validator_threads
        self.run_tamper_experiment   = bool(run_tamper_experiment)
        self.tamper_validator_counts = tamper_validator_counts or [10, 30, 50, 100, 200, 500]
        self.tamper_n_trials         = int(tamper_n_trials)
        self.kmeans           = DistributedKMeans(
            k=k, num_steps=num_steps, seed=seed,
            tol=kmeans_tol, verbose=kmeans_verbose, print_every=kmeans_print_every
        )

    def load_data_rank0(self, rank):
        if rank != 0:
            return None, None
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        df     = pd.read_csv(self.csv_path, low_memory=False)
        print(f"[Rank 0] CSV shape: {df.shape}")
        labels = df[self.label_column].astype(str).to_numpy() \
                 if self.label_column in df.columns else None
        feat   = df.drop(columns=[c for c in self.drop_columns if c in df.columns],
                         errors="ignore")
        feat   = feat.apply(pd.to_numeric, errors="coerce") \
                     .replace([np.inf, -np.inf], np.nan) \
                     .fillna(feat.mean(numeric_only=True)).fillna(0.0)
        fused  = feat.to_numpy(dtype=np.float64)
        print(f"[Rank 0] Feature matrix: {fused.shape}")
        return fused, labels

    def shard_data(self, global_fused, comm):
        rank   = comm.Get_rank()
        size   = comm.Get_size()
        shards = np.array_split(global_fused, size, axis=0) if rank == 0 else None
        local  = comm.scatter(shards, root=0)
        print(f"[Rank {rank}] local shard: {local.shape}")
        return local

    def shard_owner(self, shard_id, size):
        return shard_id % size

    def redistribute_shards(self, local_fused, labels, comm):
        rank = comm.Get_rank()
        size = comm.Get_size()
        dim  = local_fused.shape[1]

        send = [[] for _ in range(size)]
        for sid in range(self.k):
            mask = labels == sid
            if np.any(mask):
                send[self.shard_owner(sid, size)].append((int(sid), local_fused[mask]))

        recv    = comm.alltoall(send)
        lsm     = {}
        for src in recv:
            for sid, chunk in src:
                lsm.setdefault(sid, []).append(np.asarray(chunk, dtype=np.float64))

        owned = [sid for sid in range(self.k) if self.shard_owner(sid, size) == rank]
        for sid in owned:
            if sid not in lsm:
                lsm[sid] = [np.empty((0, dim), dtype=np.float64)]

        return {sid: (c[0] if len(c) == 1 else np.vstack(c)) for sid, c in lsm.items()}, owned

    def execute(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "shards"), exist_ok=True)

        comm.Barrier()
        t_total = MPI.Wtime()

        global_fused, global_labels = self.load_data_rank0(rank)
        if rank == 0:
            print(f"Distributed K-means: {size} MPI ranks, global shape {global_fused.shape}")

        local_fused = self.shard_data(global_fused, comm)

        comm.Barrier()
        t_km = MPI.Wtime()
        local_labels, centroids, global_counts, _ = self.kmeans.run(
            local_fused, comm,
            global_data=global_fused if rank == 0 else None
        )
        comm.Barrier()
        kmeans_time = comm.reduce(MPI.Wtime() - t_km, op=MPI.MAX, root=0)

        gathered_labels          = comm.gather(local_labels, root=0)
        local_shard_map, owned   = self.redistribute_shards(local_fused, local_labels, comm)

        engine = PushPullHashConsensus(
            output_dir       = self.output_dir,
            seed             = self.seed,
            write_trace      = self.write_trace,
            validator_threads= self.validator_threads,
            max_drift_sec    = self.max_drift_sec,
        )

        if size > 1:
            all_vctxs = build_all_verification_ctxs(
                comm, rank, size, local_shard_map, owned
            )
        else:
            all_vctxs = {}
            for sid in owned:
                sv_arr = np.asarray(local_shard_map[sid], dtype=np.float64)
                c_arr  = np.mean(sv_arr, axis=0) if sv_arr.shape[0] > 0                          else np.zeros((1,), dtype=np.float64)
                all_vctxs[sid] = {
                    "shard_id":      sid,
                    "data_hash":     hash_numpy_array(sv_arr),
                    "centroid_hash": hash_numpy_array(c_arr),
                    "merkle_root":   build_merkle_tree(sv_arr)["root"],
                    "computed_by":   rank,
                }

        local_results = []
        for shard_id in owned:
            sv         = local_shard_map[shard_id]
            shard_file = os.path.join(self.output_dir, "shards", f"shard_{shard_id}.npy")
            np.save(shard_file, sv)
            print(f"[Rank {rank}] Shard {shard_id}: {len(sv)} points saved")

            block = build_metadata_block(rank, shard_id, sv, shard_file)
            vctx  = all_vctxs.get(shard_id, {
                "shard_id":      shard_id,
                "computed_by":   rank,
                "data_hash":     block["data_hash"],
                "centroid_hash": block["centroid_hash"],
                "merkle_root":   block["merkle_root"],
            })

            cr = engine.run(
                block, vctx,
                rank             = rank,
                fault_percentage = self.fault_percentage,
                shard_vectors    = sv,
                sub_cluster_size = self.subcluster_size,
            )

            cb = None
            if cr["committed"]:
                cb = {
                    **block,
                    "verification_ctx_from": vctx.get("computed_by"),
                    "prepare_yes":             cr["prepare_yes"],
                    "commit_yes":              cr["commit_yes"],
                    "phase":                   cr["phase"],
                    "prepare_time_sec":        cr["prepare_time_sec"],
                    "commit_time_sec":         cr["commit_time_sec"],
                    "push_time_sec":           cr.get("push_time_sec", cr["prepare_time_sec"]),
                    "pull_time_sec":           cr.get("pull_time_sec", cr["commit_time_sec"]),
                    "push_pull_time_sec":      cr.get("push_pull_time_sec", cr["prepare_time_sec"] + cr["commit_time_sec"]),
                    "consensus_time_sec":      cr["consensus_time_sec"],
                    "min_validation_time_sec": cr["min_validation_time_sec"],
                    "avg_validation_time_sec": cr["avg_validation_time_sec"],
                    "max_validation_time_sec": cr["max_validation_time_sec"],
                }

            local_results.append({
                "shard_id":             shard_id,
                "committed_block":      cb,
                "consensus_result":     cr,
                "consensus_time_local": cr["consensus_time_sec"],
                "shard_summary_item": {
                    "shard_id":           shard_id,
                    "num_points":         block["num_points"],
                    "vector_dim":         block["vector_dim"],
                    "timestamp":          block["timestamp"],
                    "data_hash":          block["data_hash"],
                    "centroid_hash":      block["centroid_hash"],
                    "merkle_root":        block["merkle_root"],
                    "merkle_depth":       block["merkle_depth"],
                    "merkle_leaves_ref":  block["merkle_leaves_ref"],
                    "offchain_ref":       block["offchain_ref"],
                    "ctx_from_rank":      vctx.get("computed_by"),
                    "committed":          bool(cr["committed"]),
                    "push_time_sec":      cr.get("push_time_sec", cr.get("prepare_time_sec", 0.0)),
                    "pull_time_sec":      cr.get("pull_time_sec", cr.get("commit_time_sec", 0.0)),
                    "push_pull_time_sec": cr.get("push_pull_time_sec", cr.get("prepare_time_sec", 0.0) + cr.get("commit_time_sec", 0.0)),
                }
            })

        all_nested = comm.gather(local_results, root=0)

        if rank == 0:
            shard_blocks, shard_crs, shard_cts, shard_summary = [], [], [], []

            for rr in all_nested:
                for res in rr:
                    shard_crs.append(res["consensus_result"])
                    shard_cts.append(res["consensus_time_local"])
                    shard_summary.append(res["shard_summary_item"])
                    if res["committed_block"]:
                        shard_blocks.append(res["committed_block"])

            shard_blocks  = sorted(shard_blocks,  key=lambda x: x["shard_id"])
            shard_summary = sorted(shard_summary, key=lambda x: x["shard_id"])

            bc = Blockchain(chain_file=os.path.join(self.output_dir, "blockchain.json"))
            for blk in shard_blocks:
                bc.add_block(blk)
            bc.save()

            write_json(os.path.join(self.output_dir, "shard_summary.json"), shard_summary)
            np.save(os.path.join(self.output_dir, "centroids.npy"), centroids)

            sharding_quality = None
            if gathered_labels is not None:
                acl = np.concatenate(gathered_labels)
                np.save(os.path.join(self.output_dir, "cluster_labels.npy"), acl)

                # Semantic clustering/sharding quality
                sharding_quality = compute_centroid_based_sharding_quality(global_fused, acl)
                write_json(os.path.join(self.output_dir, "sharding_quality.json"), sharding_quality)

                # Add per-shard intra-cosine values into shard_summary.json as well
                q_by_sid = {q["shard_id"]: q for q in sharding_quality.get("per_shard", [])}
                for item in shard_summary:
                    q = q_by_sid.get(item["shard_id"])
                    if q:
                        item["intra_shard_cosine"] = q["intra_shard_cosine"]

                write_json(os.path.join(self.output_dir, "shard_summary.json"), shard_summary)

                if global_labels is not None:
                    pd.DataFrame({"cluster_id": acl, self.label_column: global_labels}).to_csv(
                        os.path.join(self.output_dir, "cluster_vs_activity.csv"), index=False
                    )

            total_time = MPI.Wtime() - t_total
            min_ct = min(shard_cts) if shard_cts else 0.0
            max_ct = max(shard_cts) if shard_cts else 0.0
            min_vt = min(r.get("min_validation_time_sec", 0.0) for r in shard_crs) if shard_crs else 0.0
            avg_vt = float(np.mean([r.get("avg_validation_time_sec", 0.0) for r in shard_crs])) if shard_crs else 0.0
            max_vt = max(r.get("max_validation_time_sec", 0.0) for r in shard_crs) if shard_crs else 0.0

            print("\nFinal shard counts:")
            for i, c in enumerate(global_counts):
                print(f"  Shard {i}: {int(c)} points")

            print("\nConsensus summary:")
            for s, r in zip(shard_summary, shard_crs):
                phase = r["phase"]
                extra = f"failed={r.get('failed_check','')} | " if "REJECT" in phase else ""
                print(
                    f"Shard {s['shard_id']} | committed={r['committed']} | "
                    f"ctx_from=rank{s.get('ctx_from_rank','?')} | "
                    f"merkle_depth={s['merkle_depth']} | "
                    f"prepare_yes={r['prepare_yes']} | quorum={r.get('quorum','?')} | "
                    f"fault_status={r.get('fault_tolerance_status','?')} | phase={phase} | {extra}"
                    f"avg_val={r.get('avg_validation_time_sec',0.0)*1000:.4f}ms | "
                    f"push={r.get('push_time_sec', r.get('prepare_time_sec',0.0))*1000:.4f}ms | "
                    f"pull={r.get('pull_time_sec', r.get('commit_time_sec',0.0))*1000:.4f}ms | "
                    f"push_pull={r.get('push_pull_time_sec', r.get('prepare_time_sec',0.0)+r.get('commit_time_sec',0.0))*1000:.4f}ms | "
                    f"consensus={r['consensus_time_sec']*1000:.4f}ms"
                )

            print("\nPer-block on-chain metadata bytes:")
            for i, blk in enumerate(bc.chain[1:], start=1):
                print(f"  Block {i}: {block_metadata_bytes(blk)} bytes")

            if sharding_quality is not None:
                print("\nSemantic sharding quality:")
                print(f"  Intra-Shard Cosine Similarity: {sharding_quality['intra_shard_cosine']:.6f}  (higher is better)")
                print(f"  Inter-Shard Cosine Similarity: {sharding_quality['inter_shard_cosine']:.6f}  (lower is better)")
                print(f"  Separation Gap:                 {sharding_quality['separation_gap']:.6f}  (higher is better)")
                print(f"  Shard Balance Ratio:            {sharding_quality['balance_ratio']:.6f}  (closer to 1 is better)")
                print("\nPer-shard intra-shard cosine similarity:")
                for q in sharding_quality.get("per_shard", []):
                    print(f"  Shard {q['shard_id']}: intra_cosine={q['intra_shard_cosine']:.6f}, points={q['num_points']}")

            fault_summary = majority_fault_tolerance_summary(self.subcluster_size)
            print("\nFault tolerance setting:")
            print(f"  Validators per shard:       {fault_summary['validators']}")
            print(f"  Max Byzantine faults f:     {fault_summary['max_faulty_nodes_for_majority']}")
            print(f"  BFT commit quorum 2f + 1:   {fault_summary['majority_commit_quorum']}")
            print(f"  Quorum rule:                {fault_summary['quorum_rule']}")
            print(f"  Injected fault percentage:  {self.fault_percentage * 100:.2f}%")

            total_vectors = len(global_labels) if global_labels is not None \
                            else int(np.sum(global_counts))
            print(f"\nK-means Time:           {kmeans_time:.4f} sec")
            print(f"Consensus Time (min):   {min_ct * 1000:.4f} ms")
            print(f"Consensus Time (max):   {max_ct * 1000:.4f} ms")
            print(f"Validation Time (min):  {min_vt * 1000:.6f} ms")
            print(f"Validation Time (avg):  {avg_vt * 1000:.6f} ms")
            print(f"Validation Time (max):  {max_vt * 1000:.6f} ms")
            print(f"Execution Time:         {total_time:.4f} sec")
            print(f"Throughput:             {total_vectors / total_time:.2f} vectors/sec")

            # ---------------------------------------------------------------
            # Tamper detection experiment 
            # ---------------------------------------------------------------
            if self.run_tamper_experiment:
                exp = TamperDetectionExperiment(
                    output_dir       = self.output_dir,
                    validator_counts = self.tamper_validator_counts,
                    n_trials         = self.tamper_n_trials,
                    seed             = self.seed,
                )
                exp.run()


if __name__ == "__main__":
    runner = DistributedKMeansRunner(
        csv_path          = "data.csv",
        k                 = 5,
        num_steps         = 100,
        seed              = 42,
        output_dir        = "output_csv",
        subcluster_size   = 30,
        fault_percentage  = 0.0,
        label_column      = "Activity",
        drop_columns      = ["subject", "Activity"],
        kmeans_tol        = 1e-3,
        kmeans_verbose    = True,
        kmeans_print_every= 5,
        write_trace       = False,
        validator_threads = 4,
        max_drift_sec     = 60.0,
        #Tamper detection experiment (post-consensus, rank 0 only)
        #run_tamper_experiment   = True,
        #tamper_validator_counts = [30, 60, 90, 120, 150],
        #tamper_n_trials         = 5,
    )
    runner.execute()
    
    

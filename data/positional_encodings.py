import time
import dgl
import torch
import numpy as np
import cvxpy as cp
import random
import math
from scipy import sparse as sp
import networkx as nx


def _spe(A, solver, tol=1e-6, targetd=None, verbose=True):
    """
    SPE with nearest-neighbor constraints (using a working set method).
    Translated from the original MATLAB implementation.

    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix of size (N, N).
    tol : float
        Tolerance parameter (not used in this direct translation, but retained for consistency).
    targetd : int
        Target dimension for final embedding (not strictly used here; the code always computes full dimension).
    verbose : bool
        If True, periodically prints status information.

    Returns
    -------
    Y : np.ndarray
        The embedding coordinates, shape = (dims, N). (Later transposed to (N, dims)).
    K : np.ndarray
        The final PSD matrix found by the SDP solver, shape = (N, N).
    eigVals : np.ndarray
        Sorted eigenvalues of K, largest first.
    """
    N = A.shape[0]  # number of nodes
    K = A @ A.T  # initial guess for kernel matrix
    K0 = K.copy()

    Cs = []
    bs = []
    slackInd = []

    # Step 1: Centering constraint
    c = 1
    C1 = np.ones((N, N))
    Cs.append(C1)
    bs.append(0.0)
    slackInd.append(0)

    # Step 2: For each edge, ensure distance^2 <= 1
    idx, jdx = np.where(np.triu(A) == 1)
    for i in range(len(idx)):
        c += 1
        ii = idx[i]
        jj = jdx[i]
        C_temp = np.zeros((N, N))
        C_temp[ii, ii] = 1
        C_temp[jj, jj] = 1
        C_temp[ii, jj] = -1
        C_temp[jj, ii] = -1
        Cs.append(C_temp)
        bs.append(1.0)
        slackInd.append(1)

    # Step 3: For non-edges, enforce distance^2 >= 1 (working set)
    mask_non_edges = 1 - (A + np.eye(N, dtype=int))
    idx2, jdx2 = np.where(np.triu(mask_non_edges) == 1)
    AA2 = np.zeros((len(idx2), N * N))
    for i in range(len(idx2)):
        ii = idx2[i]
        jj = jdx2[i]
        C_temp = np.zeros((N, N))
        C_temp[ii, ii] = 1
        C_temp[jj, jj] = 1
        C_temp[ii, jj] = -1
        C_temp[jj, ii] = -1
        AA2[i, :] = C_temp.flatten()

    # Step 4: Main working set loop
    numViolations = 1
    iteration = 0
    while numViolations > 0:
        iteration += 1
        if verbose:
            print(f"===== SDP Iteration {iteration} =====")
        X = cp.Variable((N + c, N + c), PSD=True)
        obj = cp.Maximize(cp.trace(X[:N, :N]))
        constraints = []
        for i_constr in range(c):
            Cmat = Cs[i_constr]
            bval = bs[i_constr]
            si = slackInd[i_constr]
            lhs = cp.trace(Cmat @ X[:N, :N]) + si * X[N + i_constr, N + i_constr]
            constraints.append(lhs == bval)

        prob = cp.Problem(obj, constraints)
        try:
            if solver == "MOSEK":
                prob.solve(solver=cp.MOSEK, verbose=False)
            else:
                prob.solve(solver=cp.SCS, verbose=False, max_iters=10000)
        except Exception as e:
            print(f"[ERROR] Solver failed: {e}")
            return np.zeros((N, targetd if targetd is not None else 2)), None, None

        if X.value is None:
            print("[ERROR] Solver returned None — numerical issues?")
            return np.zeros((N, targetd if targetd is not None else 2)), None, None

        K_solved = X.value[:N, :N]
        distances_squared = AA2 @ K_solved.flatten()
        newIDX = np.where(distances_squared < 1.0)[0]
        numViolations = len(newIDX)
        if verbose:
            print(f"Number of new constraints to add: {numViolations}")

        for idx_violation in newIDX:
            c += 1
            ii = idx2[idx_violation]
            jj = jdx2[idx_violation]
            C_temp = np.zeros((N, N))
            C_temp[ii, ii] = 1
            C_temp[jj, jj] = 1
            C_temp[ii, jj] = -1
            C_temp[jj, ii] = -1
            Cs.append(C_temp)
            bs.append(1.0)
            slackInd.append(-1)

        # Optional: perform eigendecomposition for debugging
        Y_temp, eigVals_temp = _eigDecomp(K_solved)

    Y, eigVals = _eigDecomp(K_solved)
    return Y, K_solved, eigVals

def _eigDecomp(K):
    """
    Perform eigen-decomposition on K = V * D * V^T, sort eigenvalues in descending order,
    and compute Y = (V * sqrt(D))^T.
    """
    w, v = np.linalg.eig(K)
    w = np.real(w)
    v = np.real(v)
    idx_sorted = np.argsort(w)[::-1]
    w_sorted = w[idx_sorted]
    v_sorted = v[:, idx_sorted]
    Dsqrt = np.diag(np.sqrt(np.maximum(w_sorted, 0.0)))
    V_sqrtD = v_sorted @ Dsqrt
    Y = V_sqrtD.T
    return Y, w_sorted

def _get_spe_node_embeddings(adjacency_matrix, solver, target_dimension=2, plot=False):
    """
    Run SPE on a graph's adjacency matrix and return the node embeddings.
    """
    assert adjacency_matrix.shape[0] == adjacency_matrix.shape[1], "Adjacency matrix must be square"
    Y, K, eigVals = _spe(adjacency_matrix, solver, tol=0.99, targetd=target_dimension, verbose=False)
    embeddings = Y[:target_dimension, :].T
    return embeddings

def _sample_bourgain_anchor_sets(G, c=2, seed=None):
    """
    Sample log^2(n) anchor sets using Bourgain's sampling scheme.
    """
    n = G.number_of_nodes()
    logn = int(math.ceil(math.log2(n)))
    rng = random.Random(seed)
    anchor_sets = []
    for i in range(1, logn + 1):
        p = 1 / (2 ** i)
        for _ in range(c * logn):
            S = {v for v in G.nodes() if rng.random() < p}
            anchor_sets.append(S)
    return anchor_sets

def _compute_shortest_path_dict(G, cutoff=None):
    """
    Compute shortest path lengths between all node pairs in G with an optional cutoff.
    """
    return dict(nx.all_pairs_shortest_path_length(G, cutoff=cutoff))

def _anchor_distance_pe(G, c=2, q=3, seed=None, transform="inverse"):
    """
    Generate anchor distance-based positional encodings.
    Returns a numpy array of shape [num_nodes, num_anchors].
    """
    G = nx.convert_node_labels_to_integers(G)
    n = G.number_of_nodes()
    node_list = list(G.nodes())
    anchor_sets = _sample_bourgain_anchor_sets(G, c=c, seed=seed)
    sp_dict = _compute_shortest_path_dict(G, cutoff=q)
    
    def distance_transform(d):
        if d is None or d > q:
            return 0.0
        if transform == "inverse":
            return 1.0 / (d + 1)
        elif transform == "exp":
            return np.exp(-d)
        else:
            return float(d)
    
    pe_matrix = []
    for v in node_list:
        pe_row = []
        for S in anchor_sets:
            if not S:
                pe_row.append(0.0)
                continue
            dists = [sp_dict[v].get(u, float("inf")) for u in S]
            min_d = min(dists) if dists else float("inf")
            pe_row.append(distance_transform(min_d))
        pe_matrix.append(pe_row)
    return np.array(pe_matrix)

def lap_positional_encoding(g, pos_enc_dim):
    """
    Laplacian positional encoding using eigen-decomposition.
    """
    G_nx = g.to_networkx().to_undirected()
    A = nx.to_scipy_sparse_array(G_nx, format='csr').astype(float)

    degrees = np.array(A.sum(axis=1)).flatten()
    degrees = np.clip(degrees, 1, None)  
    D_inv_sqrt = sp.diags(degrees ** -0.5)

    L = sp.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt

    # Eigen-decomposition
    EigVal, EigVec = np.linalg.eigh(L.toarray())  
    idx = np.argsort(EigVal)
    EigVec = EigVec[:, idx]
    g.ndata['pos_enc'] = torch.from_numpy(EigVec[:, 1:pos_enc_dim+1]).float()
    return g

def spe_positional_encoding(g, pos_enc_dim, solver="SCS", plot=False):
    """
    Compute SPE (structure-preserving embeddings) and assign as positional encoding.
    If SPE fails, fallback to a simple degree-based positional encoding.
    """
    import warnings

    G_nx = g.to_networkx().to_undirected()
    A = nx.to_numpy_array(G_nx)
    n_nodes = A.shape[0]

    print(f"[DEBUG] SPE starting on graph with {n_nodes} nodes")
    
    try:
        node_embeddings = _get_spe_node_embeddings(
            A,
            solver=solver,
            target_dimension=pos_enc_dim,
            plot=plot
        )
        assert node_embeddings.shape[0] == n_nodes
        g.ndata['pos_enc'] = torch.from_numpy(node_embeddings).float()
        print(f"[DEBUG] SPE finished for graph of size {n_nodes}")
        
    except Exception as e:
        warnings.warn(f"[WARN] SPE failed on graph with {n_nodes} nodes. Using degree-based fallback. Error: {e}")
        
        degrees = np.array([deg for _, deg in G_nx.degree()], dtype=np.float32).reshape(-1, 1)
        degrees /= (degrees.max() + 1e-5)  # Normalize to [0, 1]
        
        # Pad or truncate to pos_enc_dim
        if degrees.shape[1] >= pos_enc_dim:
            pos_enc = degrees[:, :pos_enc_dim]
        else:
            pad = np.zeros((n_nodes, pos_enc_dim - degrees.shape[1]), dtype=np.float32)
            pos_enc = np.concatenate([degrees, pad], axis=1)
        
        g.ndata['pos_enc'] = torch.from_numpy(pos_enc).float()
        print(f"[DEBUG] Fallback PE shape: {g.ndata['pos_enc'].shape} | sum: {g.ndata['pos_enc'].sum().item()}")

    return g


def anchor_positional_encoding(g, pos_enc_dim, c=10, q=5, seed=None, transform="exp"):
    """
    Compute anchor distance-based positional encodings and assign them as 'pos_enc'.
    """
    G_nx = g.to_networkx().to_undirected()
    pe_matrix = _anchor_distance_pe(G_nx, c=c, q=q, seed=seed, transform=transform)
    n, d = pe_matrix.shape
    if d >= pos_enc_dim:
        pe_matrix = pe_matrix[:, :pos_enc_dim]
    else:
        pad = np.zeros((n, pos_enc_dim - d), dtype=np.float32)
        pe_matrix = np.concatenate([pe_matrix, pad], axis=1)
    g.ndata['pos_enc'] = torch.from_numpy(pe_matrix).float()
    print("[DEBUG] Anchor PE shape:", g.ndata['pos_enc'].shape, "| sum:", g.ndata['pos_enc'].sum().item())
    return g

def random_walk_positional_encoding(g, pos_enc_dim):
    """
    Compute Random Walk Positional Encoding (RWPE) using successive powers of the random walk matrix.
    """

    G_nx = g.to_networkx().to_undirected()
    A = nx.to_scipy_sparse_array(G_nx, format='csr').astype(float)

    degs = np.array(A.sum(axis=1)).flatten()
    degs = np.clip(degs, 1, None)  # prevent division by zero
    Dinv = sp.diags(1.0 / degs)

    RW = Dinv @ A  # A normalized by rows (random walk matrix)

    # Compute powers of RW and extract diagonals
    M_power = RW.copy()
    PE = [torch.from_numpy(M_power.diagonal()).float()]
    for _ in range(1, pos_enc_dim):
        M_power = M_power @ RW
        PE.append(torch.from_numpy(M_power.diagonal()).float())

    # Stack and store in graph
    g.ndata['pos_enc'] = torch.stack(PE, dim=-1)
    return g


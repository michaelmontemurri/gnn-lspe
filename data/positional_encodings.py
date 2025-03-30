import time
import dgl
import torch
import numpy as np
import cvxpy as cp
import random
import math
from scipy import sparse as sp
import numpy as np
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
        If True, periodically plots and prints status information.

    Returns
    -------
    Y : np.ndarray
        The embedding coordinates, shape = (N, N). The rows represent dimensions, columns the points.
    K : np.ndarray
        The final PSD matrix found by the SDP solver, shape = (N, N).
    eigVals : np.ndarray
        Sorted eigenvalues of K, largest first.
    """
    # Initialize variables
    N = A.shape[0] #num nodes
    K = A @ A.T  # just an initial guess for the kernel matrix K, not used directly except for dimension references
    K0 = K.copy()

    # We store sets of constraints in Cs, with corresponding right-hand sides bs
    # and slackInd indicating whether that constraint can have a positive or negative slack.
    Cs = []
    bs = []
    slackInd = []

    # Step 1 is the centering constraint, i.e. all entries sum to 1
    # trace(ones(N) * K) = 0, but we handle it with the slack block as well.

    c = 1
    C1 = np.ones((N, N))
    Cs.append(C1)
    bs.append(0.0)
    slackInd.append(0)  # no slack in the diagonal extension for this constraint

    # Step 2, for each edge, ensure the distance^2 is less than or equal to 1
    # K[ii,ii] + K[jj,jj] - 2*K[ii,jj] <= 1
    # We store it in a matrix form so that trace(C1*K) <= 1.

    idx, jdx = np.where(np.triu(A) == 1)
    for i in range(len(idx)):
        c += 1
        ii = idx[i]
        jj = jdx[i]
        C1 = np.zeros((N, N))
        C1[ii, ii] = 1
        C1[jj, jj] = 1
        C1[ii, jj] = -1
        C1[jj, ii] = -1

        Cs.append(C1)
        bs.append(1.0)   # connected => distance^2 must be <= 1
        slackInd.append(1)  # allow positive slack in one direction


    # Step 3,  For non-edges (except diagonal), we want distance^2 >= 1
    # detect violations of that and add constraints as needed.
    # this is the "working set" portion.

    mask_non_edges = (1 - (A + np.eye(N, dtype=int)))  # 1 where not connected and not self
    idx2, jdx2 = np.where(np.triu(mask_non_edges) == 1)
    AA2 = np.zeros((len(idx2), N*N))  # store each "C1" in row form for quick checks
    for i in range(len(idx2)):
        ii = idx2[i]
        jj = jdx2[i]
        C1 = np.zeros((N, N))
        C1[ii, ii] = 1
        C1[jj, jj] = 1
        C1[ii, jj] = -1
        C1[jj, ii] = -1
        AA2[i, :] = C1.flatten()


    # Step 4, the main working set loop:
    # Build the SDP, solve, then find which unconnected pairs
    # are violating the distance >= 1 condition. Add constraints.
    numViolations = 1
    iteration = 0
    while numViolations > 0:
        iteration += 1
        if verbose:
            print(f"===== SDP Iteration {iteration} =====")

        X = cp.Variable((N + c, N + c), PSD=True)

        # Objective: we are effectively maximizing trace(K) 
        # because the MATLAB code sets c^T x = -trace(K)
        # => Minimize( -trace(K) ) => Maximize( trace(K) ).
        # We'll define: K_block = X[:N,:N]
        obj = cp.Maximize(cp.trace(X[:N, :N]))

        # Now build the constraints from Cs, bs, slackInd
        constraints = []
        for i_constr in range(c):
            Cmat = Cs[i_constr]
            bval = bs[i_constr]
            si   = slackInd[i_constr]

            # Build the big (N+c)x(N+c) block that, when dotted with X, 
            # yields trace(Cmat*K) + slackInd[i_constr]*X[N+i_constr, N+i_constr].
            # Instead of forming that matrix explicitly, we can directly write
            # the linear expression in cvxpy
            lhs = cp.trace(Cmat @ X[:N, :N]) + si * X[N + i_constr, N + i_constr]
            constraints.append(lhs == bval)

        # Solve the SDP
        prob = cp.Problem(obj, constraints)
        # I got an academic licencse for MOSEK its faster than standard SCS
        try:
            if solver == "MOSEK":
                prob.solve(solver=cp.MOSEK, verbose=False)
            else:
                prob.solve(solver=cp.SCS, verbose=False, max_iters=10000)
        except Exception as e:
            print(f"[ERROR] Solver failed: {e}")
            return np.zeros((N, targetd)), None, None

        if X.value is None:
            print("[ERROR] Solver returned None — possibly infeasible or numerical issues.")
            return np.zeros((N, targetd)), None, None

        # Extract the top-left NxN block as K
        K_solved = X.value[:N, :N]

        # Check which non-edges are violating distance^2 >= 1
        distances_squared = AA2 @ K_solved.flatten()  # vector of values
        newIDX = np.where(distances_squared < 1.0)[0]  # indexes of violations

        numViolations = len(newIDX)
        if verbose:
            print(f"Number of new constraints to add: {numViolations}")

        # Update the working set by adding constraints for each new violation
        for idx_violation in newIDX:
            c += 1
            ii = idx2[idx_violation]
            jj = jdx2[idx_violation]

            # Build the matrix for distance^2
            C1 = np.zeros((N, N))
            C1[ii, ii] = 1
            C1[jj, jj] = 1
            C1[ii, jj] = -1
            C1[jj, ii] = -1

            # Because these are unconnected pairs, we want distance^2 >= 1,
            # which translates to the same "trace(C1*K) >= 1" in the primal,
            # but in the standard form we do 'trace(C1*K) + slack*(-1) = 1'.
            # We store negative slackInd.
            Cs.append(C1)
            bs.append(1.0)
            slackInd.append(-1)

        # do an eigdecomp to get Y
        Y_temp, eigVals_temp = _eigDecomp(K_solved)

    # Once there is no more violations, do the final spectral embedding
    Y, eigVals = _eigDecomp(K_solved)

    return Y, K_solved, eigVals

#standard eigen decomp helper function
def _eigDecomp(K):
    """
    Decompose K = V * D * V^T and sort by descending eigenvalues.
    Return Y = (V * sqrt(D))^T as in the original script, and 
    the sorted eigenvalues.
    """
    w, v = np.linalg.eig(K)

    # Ensure real parts if small imaginary parts appear
    w = np.real(w)
    v = np.real(v)

    # Sort eigenvalues in ascending order, then reverse
    idx_sorted = np.argsort(w)
    w_sorted = w[idx_sorted]
    w_sorted = w_sorted[::-1]  # largest first
    v_sorted = v[:, idx_sorted]
    v_sorted = v_sorted[:, ::-1]  # reorder columns to match w_sorted

    # Build Y = (V * sqrt(D))^T, preserving the sorting
    # sqrt(D) is just sqrt of w on diagonal
    Dsqrt = np.diag(np.sqrt(np.maximum(w_sorted, 0.0)))
    V_sqrtD = v_sorted @ Dsqrt
    Y = V_sqrtD.T  # shape will be (N, N)

    return Y, w_sorted


# This is not in the original code but will serve our purpose as a wrapper function to extract the node embeddings
def _get_spe_node_embeddings(adjacency_matrix, solver, target_dimension=2, plot=False):
    """
    Run SPE on a graph's adjacency matrix
    and return the resulting node embeddings.

    Parameters
    ----------
    adjacency_matrix : np.ndarray
        Binary adjacency matrix of shape (N, N)
    target_dimension : int
        Target dimensionality of the embedding (2 or 3)
    plot : bool
        If True, show the graph embedding as a plot

    Returns
    -------
    embeddings : np.ndarray
        Node embeddings of shape (N, target_dimension)
    """
    assert adjacency_matrix.shape[0] == adjacency_matrix.shape[1], "Adjacency matrix must be square"

    # Run SPE
    Y, K, eigVals = _spe(adjacency_matrix, solver, tol=0.99, targetd=target_dimension, verbose=False)

    # Y is returned as shape (dims, N); transpose to (N, dims)
    embeddings = Y[:target_dimension, :].T  # shape (N, target_dimension)

    return embeddings


def _sample_bourgain_anchor_sets(G, c=2, seed=None):
    """
    sample log^2(n) anchor sets using bourgains sampling scheme a in the PGNN paper
    """

    #initializing
    n = G.number_of_nodes()
    logn = int(math.ceil(math.log2(n)))
    rng = random.Random(seed)
    anchor_sets = []

    #we choose c*log^2n anchor sets each labeled S_{i,j}, where i=1,..,logn controls teh sampling probability, and j is the number of sets per size level
    # to sample an anchor set S_{i,j}, we sample each node in V independently with prop 1/2^i, this will create many smaller sets and fewer large sets
    for i in range(1, logn + 1):
        p = 1 / (2 ** i)
        # for each size scale i, we want to generate c*logn anchor sets
        for _ in range(c * logn):
            S = {v for v in G.nodes() if rng.random() < p}
            anchor_sets.append(S)

    return anchor_sets

def _compute_shortest_path_dict(G, cutoff=None):
    """
    Compute the shortest path distances between all pairs of nodes, implement a q-hop limit for computational effeciency
    """
    #we can just use the networkx function w param cutoff
    # the PGNNs code uses a parallelized version of this, but I think thats overkill considering the size of the graphs in ZINC
    return dict(nx.all_pairs_shortest_path_length(G, cutoff=cutoff))

def _anchor_distance_pe(G, c=2, q=3, seed=None, transform="inverse"):
    """
    Generate the static PEs based on anchor set distances.
    Returns: np.ndarray of shape [num_nodes, num_anchors]
    """
    #convert the node labels to integers for consistent ordering
    G = nx.convert_node_labels_to_integers(G)  
    n = G.number_of_nodes()
    node_list = list(G.nodes())

    #generate the anchor sets
    anchor_sets = _sample_bourgain_anchor_sets(G, c=c, seed=seed)

    #using cutoff=q for compute efficiency
    sp = _compute_shortest_path_dict(G, cutoff=q)

    #use a distance transform, PGNN paper uses "inverse", so we set that as the default
    def distance_transform(d):
        if d is None or d > q:
            return 0.0
        if transform == "inverse":
            return 1.0 / (d + 1)
        elif transform == "exp":
            return np.exp(-d)
        else:
            return float(d)

    # build the PE matrix
    pe_matrix = []
    #for each node, we want to compute the PE as a vector of the transformed distances from the closest node in each anchor set
    for v in node_list:
        pe_row = []
        # now looping through anchor sets
        for S in anchor_sets:
            if not S:
                pe_row.append(0.0)
                continue

            #retrieve the shortest path from the node to each u in the anchor set
            dists = [sp[v].get(u, float("inf")) for u in S]

            #take the min 
            min_d = min(dists) if dists else float("inf")
    
            pe_row.append(distance_transform(min_d))

        pe_matrix.append(pe_row)

    return np.array(pe_matrix)  # shape [n, num_anchors]


def lap_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort() # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    g.ndata['pos_enc'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 
    
    # # Eigenvectors with scipy
    # EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    # EigVec = EigVec[:, EigVal.argsort()] # increasing order
    # g.ndata['pos_enc'] = torch.from_numpy(np.abs(EigVec[:,1:pos_enc_dim+1])).float() 
    
    return g


def spe_positional_encoding(g, pos_enc_dim, solver="SCS", plot=False):

    G_nx = g.to_networkx().to_undirected()
    A = nx.to_numpy_array(G_nx)

    print(f"[DEBUG] SPE starting on graph with {A.shape[0]} nodes")

    node_embeddings = _get_spe_node_embeddings(A, solver=solver, target_dimension=pos_enc_dim, plot=plot)

    g.ndata['pos_enc'] = torch.from_numpy(node_embeddings).float()

    print(f"[DEBUG] SPE finished for graph of size {A.shape[0]}")
    
    return g




def anchor_positional_encoding(g, pos_enc_dim, c=2, q=3, seed=None, transform="inverse"):
    """
    Graph positional encoding using Bourgain-inspired anchor set distance features.
    """
    # Convert DGL graph to NetworkX
    G_nx = g.to_networkx().to_undirected()
    
    # Compute anchor-based PE
    pe_matrix = _anchor_distance_pe(G_nx, c=c, q=q, seed=seed, transform=transform)

    # Truncate or pad to pos_enc_dim
    n, d = pe_matrix.shape
    if d >= pos_enc_dim:
        pe_matrix = pe_matrix[:, :pos_enc_dim]
    else:
        pad = np.zeros((n, pos_enc_dim - d), dtype=np.float32)
        pe_matrix = np.concatenate([pe_matrix, pad], axis=1)

    g.ndata['pos_enc'] = torch.from_numpy(pe_matrix).float()
    print("[DEBUG] Anchor PE shape:", g.ndata['pos_enc'].shape, " | sum:", g.ndata['pos_enc'].sum().item())

    return g

def random_walk_positional_encoding(g, pos_enc_dim):
    """
    Compute Random Walk Positional Encoding (RWPE).
    """
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    Dinv = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -1.0, dtype=float)  # D^-1
    RW = A * Dinv
    M = RW

    # Compute RWPE through successive powers of M
    PE = [torch.from_numpy(M.diagonal()).float()]
    M_power = M
    for _ in range(1, pos_enc_dim):
        M_power = M_power * M
        PE.append(torch.from_numpy(M_power.diagonal()).float())

    g.ndata['pos_enc'] = torch.stack(PE, dim=-1)
    return g


def degree_positional_encoding(g, pos_enc_dim):
    """
    Compute node degree encoding.
    """
    degrees = g.in_degrees().float().unsqueeze(1)  
    degrees = torch.cat([degrees] * pos_enc_dim, dim=1)  # match pos_enc_dim

    g.ndata['pos_enc'] = degrees

    return g

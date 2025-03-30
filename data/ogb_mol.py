import time
import dgl
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import cvxpy as cp
import random
import csv
import math
from ogb.graphproppred import DglGraphPropPredDataset, Evaluator
from data.positional_encodings import *
from scipy import sparse as sp
import numpy as np
import networkx as nx
from tqdm import tqdm

class OGBMOLDGL(torch.utils.data.Dataset):
    def __init__(self, data, split):
        self.split = split
        self.data = [g for g in data[self.split]]
        self.graph_lists = []
        self.graph_labels = []
        for g in self.data:
            if g[0].number_of_nodes() > 5:
                self.graph_lists.append(g[0])
                self.graph_labels.append(g[1])
        self.n_samples = len(self.graph_lists)

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.graph_labels[idx]

def add_eig_vec(g, pos_enc_dim):
    """
     Graph positional encoding v/ Laplacian eigenvectors
     This func is for eigvec visualization, same code as positional_encoding() func,
     but stores value in a diff key 'eigvec'
    """

    G_nx = g.to_networkx().to_undirected()
    A = nx.to_scipy_sparse_array(G_nx, format='csr').astype(float)

    # normalized Laplacian
    degs = np.array(A.sum(axis=1)).flatten()
    degs = np.clip(degs, 1, None)  # Avoid division by zero
    D_inv_sqrt = sp.diags(degs ** -0.5)
    L = sp.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt

    # Eigen-decomposition
    EigVal, EigVec = np.linalg.eigh(L.toarray())  
    idx = EigVal.argsort()  # increasing order
    EigVec = np.real(EigVec[:, idx])

    # Take first non-trivial eigenvectors
    eigvec_torch = torch.from_numpy(EigVec[:, 1:pos_enc_dim+1]).float()
    g.ndata['eigvec'] = eigvec_torch

    # Zero-padding if number of nodes < pos_enc_dim
    n, d = eigvec_torch.shape
    if d < pos_enc_dim:
        pad_width = pos_enc_dim - d
        g.ndata['eigvec'] = F.pad(g.ndata['eigvec'], (0, pad_width), value=0.0)

    return g

def init_positional_encoding(g, pos_enc_dim, type_init):
    """
    Initialize graph positional encodings based on the specified type.
    """
    if type_init == 'rand_walk':
        return random_walk_positional_encoding(g, pos_enc_dim)
    elif type_init == 'lapPE':
        return lap_positional_encoding(g, pos_enc_dim)
    elif type_init == 'spe':
        return spe_positional_encoding(g, pos_enc_dim)
    elif type_init == 'anchor':
        return anchor_positional_encoding(g, pos_enc_dim)
    else:
        raise ValueError(f"Unsupported positional encoding type: {type_init}")


def make_full_graph(graph, adaptive_weighting=None):
    g, label = graph

    full_g = dgl.from_networkx(nx.complete_graph(g.number_of_nodes()))

    # Copy over the node feature data and laplace  eigvecs
    full_g.ndata['feat'] = g.ndata['feat']
    
    if 'pos_enc' in g.ndata:
        full_g.ndata['pos_enc'] = g.ndata['pos_enc']
    if 'eigvec' in g.ndata:
        full_g.ndata['eigvec'] = g.ndata['eigvec']

    # Initalize fake edge features w/ 0s
    full_g.edata['feat'] = torch.zeros(full_g.number_of_edges(), 3, dtype=torch.long)
    full_g.edata['real'] = torch.zeros(full_g.number_of_edges(), dtype=torch.long)

    # Copy real edge data over, and identify real edges!
    # Get edge endpoints
    u, v = g.edges()
    num_skipped = 0
    num_total = u.shape[0]

    for i in range(num_total):
        try:
            eid = full_g.edge_ids(u[i], v[i])
            full_g.edata['feat'][eid] = g.edata['feat'][i]
            full_g.edata['real'][eid] = 1
        except:
            try:
                # Try reversed order (important for undirected)
                eid = full_g.edge_ids(v[i], u[i])
                full_g.edata['feat'][eid] = g.edata['feat'][i]
                full_g.edata['real'][eid] = 1
            except:
                print(f"[Warning] Failed to match edge ({u[i].item()}, {v[i].item()}) in complete graph")
                num_skipped += 1
                continue

    if num_skipped > 0:
        print(f"[Info] Skipped {num_skipped}/{num_total} edges for graph with {g.number_of_nodes()} nodes")


    # This code section only apply for GraphiT --------------------------------------------
    if adaptive_weighting is not None:
        p_steps, gamma = adaptive_weighting
    
        n = g.number_of_nodes()
        G_nx = g.to_networkx().to_undirected()
        A = nx.to_scipy_sparse_array(G_nx, format='csr').astype(float)


        # Adaptive weighting k_ij for each edge
        if p_steps == "qtr_num_nodes":
            p_steps = int(0.25*n)
        elif p_steps == "half_num_nodes":
            p_steps = int(0.5*n)
        elif p_steps == "num_nodes":
            p_steps = int(n)
        elif p_steps == "twice_num_nodes":
            p_steps = int(2*n)

        N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
        I = sp.eye(n)
        L = I - N * A * N

        k_RW = I - gamma*L
        k_RW_power = k_RW
        for _ in range(p_steps - 1):
            k_RW_power = k_RW_power.dot(k_RW)

        k_RW_power = torch.from_numpy(k_RW_power.toarray())

        # Assigning edge features k_RW_eij for adaptive weighting during attention
        full_edge_u, full_edge_v = full_g.edges()
        num_edges = full_g.number_of_edges()

        k_RW_e_ij = []
        for edge in range(num_edges):
            k_RW_e_ij.append(k_RW_power[full_edge_u[edge], full_edge_v[edge]])

        full_g.edata['k_RW'] = torch.stack(k_RW_e_ij,dim=-1).unsqueeze(-1).float()
    # --------------------------------------------------------------------------------------
    
    return full_g, label

class OGBMOLDataset(Dataset):
    def __init__(self, name, features='full'):

        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name.lower()
        
        self.dataset = DglGraphPropPredDataset(name=self.name, root='dataset')

        
        if features == 'full':
            pass 
        elif features == 'simple':
            print("[I] Retaining only simple features...")
            # only retain the top two node/edge features
            for g in self.dataset.graphs:
                g.ndata['feat'] = g.ndata['feat'][:, :2]
                g.edata['feat'] = g.edata['feat'][:, :2]
        
        split_idx = self.dataset.get_idx_split()

        self.train = OGBMOLDGL(self.dataset, split_idx['train'])
        self.val = OGBMOLDGL(self.dataset, split_idx['valid'])
        self.test = OGBMOLDGL(self.dataset, split_idx['test'])
        
        self.evaluator = Evaluator(name=self.name)
        
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))

    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        labels = torch.stack(labels)
        tab_sizes_n = [ graphs[i].number_of_nodes() for i in range(len(graphs))]
        tab_snorm_n = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n ]
        snorm_n = torch.cat(tab_snorm_n).sqrt()
        
        return batched_graph, labels, snorm_n

    def _add_lap_positional_encodings(self, pos_enc_dim):

        # Graph positional encoding v/ Laplacian eigenvectors
        self.train = [(lap_positional_encoding(g, pos_enc_dim), label) for g, label in self.train]
        self.val = [(lap_positional_encoding(g, pos_enc_dim), label) for g, label in self.val]
        self.test = [(lap_positional_encoding(g, pos_enc_dim), label) for g, label in self.test]
        
    def _add_eig_vecs(self, pos_enc_dim):

        # Graph positional encoding v/ Laplacian eigenvectors
        self.train = [(add_eig_vec(g, pos_enc_dim), label) for g, label in self.train]
        self.val = [(add_eig_vec(g, pos_enc_dim), label) for g, label in self.val]
        self.test = [(add_eig_vec(g, pos_enc_dim), label) for g, label in self.test]
        
        
    def _init_positional_encodings(self, pos_enc_dim, type_init):

        # Initializing positional encoding randomly with l2-norm 1
        self.train = [(init_positional_encoding(g, pos_enc_dim, type_init), label) for g, label in self.train]
        self.val = [(init_positional_encoding(g, pos_enc_dim, type_init), label) for g, label in self.val]
        self.test = [(init_positional_encoding(g, pos_enc_dim, type_init), label) for g, label in self.test]
        
    def _make_full_graph(self, adaptive_weighting=None):
        self.train = [make_full_graph(graph, adaptive_weighting) for graph in self.train]
        self.val = [make_full_graph(graph, adaptive_weighting) for graph in self.val]
        self.test = [make_full_graph(graph, adaptive_weighting) for graph in self.test]
        

    
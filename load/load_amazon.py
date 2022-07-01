import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
import torch_geometric as pyg
from torch_geometric.data import Data


def sparse_mx_to_torch_sparse_tensor(sparse_mx, device=None):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    tensor = torch.sparse.FloatTensor(indices, values, shape)
    if device is not None:
        tensor = tensor.to(device)
    return tensor

def load_raw_graph(dataset_str = "amazon-all", self_loops=True, root='dataset/'):
    txt_file = root + dataset_str + '/adj_list.txt'
    graph = {}
    with open(txt_file, 'r') as f:
        cur_idx = 0
        for row in f:
            row = row.strip().split()
            adjs = []
            for j in range(1, len(row)):
                adjs.append(int(row[j]))
            graph[cur_idx] = adjs
            cur_idx += 1
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    G = Data(edge_index=adj.coalesce().indices())
    G = pyg.transforms.GCNNorm(add_self_loops=self_loops)(G)
    adj = torch.sparse.FloatTensor(G.edge_index, G.edge_weight, torch.Size(adj.shape))

    return adj

def load_txt_data(dataset_str="amazon-all", portion='0.06', self_loops=True, root='dataset/'):
    adj = load_raw_graph(dataset_str, self_loops, root)
    idx_train = list(np.loadtxt(root + dataset_str + '/train_idx-' + str(portion) + '.txt', dtype=int))
    idx_val = list(np.loadtxt(root + dataset_str + '/test_idx.txt', dtype=int))
    idx_test = list(np.loadtxt(root + dataset_str + '/test_idx.txt', dtype=int))
    labels = np.loadtxt(root + dataset_str + '/label.txt')
    with open(root + dataset_str + '/meta.txt', 'r') as f:
        num_nodes, num_class = [int(w) for w in f.readline().strip().split()]

    features = sp.identity(num_nodes)

    # porting to pytorch
    features = sparse_mx_to_torch_sparse_tensor(features).float()
    labels = torch.FloatTensor(labels)
    # labels = torch.max(labels, dim=1)[1]
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, num_nodes, num_class
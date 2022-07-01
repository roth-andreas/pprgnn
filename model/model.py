import torch_geometric as pyg
import torch.nn as nn

import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
import torch

from model.pprlayer import PPRLayer


class PPRGNN_Amazon(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, dropout, num_nodes=None, **kwargs):
        super(PPRGNN_Amazon, self).__init__()

        self.conv = PPRLayer(in_dim, h_dim, **kwargs)
        self.dropout = dropout

        self.X_0 = nn.Parameter(torch.zeros(h_dim, num_nodes), requires_grad=False)
        self.V = nn.Linear(h_dim, out_dim, bias=False)

    def forward(self, x, edge_index=None, edge_weight=None, adj=None):
        if adj is None:
            adj = torch.sparse.FloatTensor(edge_index, edge_weight, torch.Size([x.shape[0], x.shape[0]]))
            x = x.T

        x, converge_index = self.conv(x, adj, self.X_0)
        x = F.dropout(x.T, self.dropout, training=self.training)
        x = self.V(x)
        return x, converge_index


class PPRGNN_PPI(nn.Module):
    def __init__(self, in_dim, out_dim, h_dim, **kwargs):
        super(PPRGNN_PPI, self).__init__()
        self.conv1 = PPRLayer(in_dim, 4 * h_dim, **kwargs)
        self.conv2 = PPRLayer(4 * h_dim, 2 * h_dim, **kwargs)
        self.conv3 = PPRLayer(2 * h_dim, 2 * h_dim, **kwargs)
        self.conv4 = PPRLayer(2 * h_dim, h_dim, **kwargs)
        self.conv5 = PPRLayer(h_dim, out_dim, **kwargs)

        self.X_0 = None

        self.V = nn.Linear(h_dim, out_dim)
        self.V_0 = nn.Linear(in_dim, 4 * h_dim)
        self.V_1 = nn.Linear(4 * h_dim, 2 * h_dim)
        self.V_2 = nn.Linear(2 * h_dim, 2 * h_dim)
        self.V_3 = nn.Linear(2 * h_dim, h_dim)

    def forward(self, features, edge_index, edge_weight):
        features = features.T
        adj = torch.sparse.FloatTensor(edge_index, edge_weight, torch.Size([features.shape[1], features.shape[1]]))

        x = features
        x_new, layers1 = self.conv1(x, adj, self.X_0)
        x = F.elu(x_new.T + self.V_0(x.T)).T
        x_new, layers2 = self.conv2(x, adj, self.X_0)
        x = F.elu(x_new.T + self.V_1(x.T)).T
        x_new, layers3 = self.conv3(x, adj, self.X_0)
        x = F.elu(x_new.T + self.V_2(x.T)).T
        x_new, layers4 = self.conv4(x, adj, self.X_0)
        x = F.elu(x_new.T + self.V_3(x.T)).T
        x_new, layers5 = self.conv5(x, adj, self.X_0)
        x = x_new.T + self.V(x.T)
        return x, layers1 + layers2 + layers3 + layers4 + layers5


class PPRGNN_GC(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, dropout, eps, **kwargs):
        super(PPRGNN_GC, self).__init__()

        self.conv1 = PPRLayer(in_dim, h_dim, eps=eps)
        self.conv2 = PPRLayer(h_dim, h_dim, eps=eps)
        self.conv3 = PPRLayer(h_dim, h_dim, eps=eps)

        self.dropout = dropout
        self.X_0 = None
        self.V_0 = nn.Linear(h_dim, h_dim)
        self.V_1 = nn.Linear(h_dim, out_dim)

    def forward(self, x, adj, batch):
        x, layers1 = self.conv1(x, adj, self.X_0)
        x, layers2 = self.conv2(x, adj, self.X_0)
        x, layers3 = self.conv3(x, adj, self.X_0)
        x = x.T

        x = global_add_pool(x, batch)
        x = F.relu(self.V_0(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.V_1(x)
        return F.log_softmax(x, dim=1), layers1 + layers2 + layers3


class APPNP(nn.Module):
    def __init__(self, in_dim, out_dim, h_dim):
        super(APPNP, self).__init__()
        self.lin1 = pyg.nn.GCNConv(in_dim, h_dim, normalize=False)
        self.lin2 = pyg.nn.GCNConv(h_dim, out_dim, normalize=False)
        self.appnp = pyg.nn.APPNP(K=10, alpha=0.1, dropout=0, normalize=False)

    def forward(self, features, adj=None, edge_index=None, edge_weight=None):
        if adj is not None:
            edge_index, edge_weight = adj.coalesce().indices(), adj.coalesce().values()
        x = self.lin1(features, edge_index, edge_weight)
        x = self.lin2(x, edge_index, edge_weight)
        x = self.appnp(x, edge_index, edge_weight)
        return x, 10


class APPNP_GC(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, dropout, **kwargs):
        super(APPNP_GC, self).__init__()

        self.lin1 = nn.Linear(in_dim, h_dim)
        self.lin2 = nn.Linear(h_dim, h_dim)
        self.appnp = pyg.nn.APPNP(K=10, alpha=0.1, dropout=0, normalize=False)

        self.dropout = dropout
        self.V_0 = nn.Linear(h_dim, h_dim)
        self.V_1 = nn.Linear(h_dim, out_dim)

    def forward(self, features, adj, batch):
        x = features.T
        edge_index, edge_weight = adj.coalesce().indices(), adj.coalesce().values()

        x = self.lin1(x)
        x = self.lin2(x)
        x = self.appnp(x, edge_index, edge_weight)

        x = global_add_pool(x, batch)
        x = F.relu(self.V_0(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.V_1(x)
        return F.log_softmax(x, dim=1), 10

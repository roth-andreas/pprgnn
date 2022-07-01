import math
import torch
import torch.sparse
from torch.nn import Parameter
from torch.nn import Module

from model.ppr_function import PPRFunction


class PPRLayer(Module):
    def __init__(self, in_features, out_features, eps, **kwargs):
        super(PPRLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps

        self.W = Parameter(torch.FloatTensor(self.out_features, self.out_features))
        self.Omega_1 = Parameter(torch.FloatTensor(self.out_features, self.in_features))
        self.init()

    def init(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        self.Omega_1.data.uniform_(-stdv, stdv)

    def forward(self, x, A, X_0, *args):
        x = torch.spmm(torch.transpose(x, 0, 1), self.Omega_1.T).T
        x = torch.spmm(torch.transpose(A, 0, 1), x.T).T
        x = PPRFunction.apply(self.W, x, A, X_0, self.eps)
        return x
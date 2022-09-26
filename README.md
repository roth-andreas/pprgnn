# Personalized Page Rank Graph Neural Network

This is the official implementation of the Personalized Page Rank Graph Neural Network (PPRGNN) with pytorch geometric.
The preprint of is available at https://arxiv.org/abs/2207.00684 . The paper is accepted for publication at ECML-PKDD 2022.

## Plug & Play Version

In case you are just interested in running a simplified version of our model, start with this code:

```
import torch_geometric as pyg
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class PPRGNN(nn.Module):
    def __init__(self, features, eps=1, backward_n=5):
        super(PPRGNN, self).__init__()

        self.conv = pyg.nn.GCNConv(features, features, normalize=False, add_self_loops=False, bias=False)
        self.eps = eps
        self.backward_n = backward_n

    def forward(self, x, edge_index, edge_weight):
        X_ = x
        X_0 = x
        with torch.no_grad():
            # Determine the convergence of a random walk
            for converge_index in range(300):
                alpha = 1 / (1 + (converge_index * self.eps))
                X_ = self.conv(X_, edge_index, edge_weight)
                X_ = F.relu(alpha * X_)
                if torch.norm(X_, np.inf) < 3e-6:
                    break

            X = X_0
            for j in range(converge_index + self.backward_n, self.backward_n, -1):
                alpha = 1 / (1 + (j * self.eps))
                X = self.conv(X, edge_index, edge_weight)
                X = alpha * X + x
                X = F.relu(X)

        for j in range(self.backward_n, - 1, -1):
            alpha = 1 / (1 + (j * self.eps))
            X = self.conv(X, edge_index, edge_weight)
            X = alpha * X + x
            X = F.relu(X)

        return X
```
Alternatively you can use the finite version:
```
class Finite_PPRGNN(nn.Module):
    def __init__(self, layers, features, eps=1):
        super(Finite_PPRGNN, self).__init__()

        self.conv = pyg.nn.GCNConv(features, features, normalize=False, add_self_loops=False, bias=False)
        self.eps = eps
        self.layers = layers

    def forward(self, x, edge_index, edge_weight):
        X = x

        for j in range(self.layers, - 1, -1):
            alpha = 1 / (1 + (j * self.eps))
            X = self.conv(X, edge_index, edge_weight)
            X = alpha * X + x
            X = F.relu(X)

        return X
```

To execute the experiments run the corresponding main_<NAME>.py file.

### Required Libraries:

* pytorch
* pytorch_geometric


### Datasets

* PPI, MUTAG, PROTEINS, NCI1, COX2 will be downloaded automatically via pytorch_geometric
* Amazon and PTC. Please download the dataset/ folder from the dropbox link below and place the folder inside this project
  * https://www.dropbox.com/sh/rjf9nlr94c7zmuk/AAAcCRLmgBNyp4aAyKcKHYi2a?dl=0

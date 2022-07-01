import time

import torch_geometric as pyg
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch_geometric.datasets import TUDataset

from load import load_ptc
from model.model import PPRGNN_GC, APPNP_GC
from torch_geometric.loader import DataLoader

from train.utils import init_seeds


def train_model(lr, weight_decay, self_loops, epochs, clip, width, dataset, **kwargs):
    init_seeds()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    criterion = nn.NLLLoss()
    if dataset == 'PTC':
        all_data, num_classes = load_ptc.load_data(dataset, True)
        ys = [g.y for g in all_data]
    else:
        all_data = TUDataset('./data', dataset).shuffle()
        num_classes = all_data.num_classes
        ys = all_data.data.y
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    idx_list = []
    for idx in skf.split(np.zeros(len(ys)), ys):
        idx_list.append(idx)
    best_accs = []
    best_losses = []
    best_layers = []
    start_time = time.time()
    for fold_idx in range(10):
        train_idx, test_idx = idx_list[fold_idx]
        test_dataset = [all_data[i] for i in test_idx]
        train_dataset = [all_data[i] for i in train_idx]
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)

        if kwargs['type'] == 'appnp':
            model = APPNP_GC(max(1, all_data[0].num_node_features), h_dim=width, out_dim=num_classes, **kwargs).to(
                device)
        elif kwargs['type'] == 'pprgnn':
            model = PPRGNN_GC(max(1, all_data[0].num_node_features), h_dim=width, out_dim=num_classes, **kwargs).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=24, verbose=False, factor=0.9)

        start_epoch = 1
        best_val_acc = 0
        best_layer_count = 0
        min_val_loss = np.inf

        not_improved_for = 0
        for epoch in range(start_epoch, epochs):
            epoch_start = time.time()
            output_string = f'Epoch {epoch}:'
            train_losses = []
            correct = 0
            model.train()
            for data in train_loader:
                data = data.to(device)
                if data.edge_attr is None:
                    data.edge_weight = torch.ones((data.edge_index.size(1),), dtype=torch.float32,
                                                  device=data.edge_index.device)
                else:
                    if dataset == 'MUTAG':
                        data.edge_weight = data.edge_attr.argmax(1).float()
                data = pyg.transforms.GCNNorm(add_self_loops=self_loops)(data)
                adj = torch.sparse.FloatTensor(data.edge_index, data.edge_weight,
                                               torch.Size([data.num_nodes, data.num_nodes]))
                if data.x is None:
                    data.x = torch.sparse.sum(adj, [0]).to_dense().unsqueeze(1).to(device)
                # train
                output, _ = model(data.x.T, adj, data.batch)
                train_loss = criterion(output, data.y)
                optimizer.zero_grad()
                train_loss.backward()
                if clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                # calc metrics
                train_losses.append(train_loss.item())
                pred = output.max(dim=1)[1]
                correct += pred.eq(data.y).sum().item()
            train_loss_mean = np.mean(train_losses)
            train_acc = correct / len(train_loader.dataset)
            output_string += f" Train: Loss: {train_loss_mean:.4f}, Acc: {train_acc:.4f}"

            model.eval()
            test_losses = []
            layers = []
            correct = 0
            with torch.no_grad():
                for data in test_loader:
                    data = data.to(device)
                    if data.edge_attr is None:
                        data.edge_weight = torch.ones((data.edge_index.size(1),), dtype=torch.float32,
                                                      device=data.edge_index.device)
                    else:
                        if dataset == 'MUTAG':
                            data.edge_weight = data.edge_attr.argmax(1).float()
                    data = pyg.transforms.GCNNorm(add_self_loops=self_loops)(data)
                    adj = torch.sparse.FloatTensor(data.edge_index, data.edge_weight,
                                                   torch.Size([data.num_nodes, data.num_nodes]))
                    if data.x is None:
                        data.x = torch.sparse.sum(adj, [0]).to_dense().unsqueeze(1).to(device)
                    # train
                    output, layer_count = model(data.x.T, adj, data.batch)
                    test_loss = criterion(output, data.y)
                    test_losses.append(test_loss.item())
                    pred = output.max(dim=1)[1]
                    correct += pred.eq(data.y).sum().item()
                    layers.append(layer_count)
            test_acc = val_acc = correct / len(test_loader.dataset)
            test_loss_mean = val_loss = np.mean(test_losses)
            output_string += f" Test: Loss: {test_loss_mean:.4f}, Acc: {test_acc:.4f}, Layers: {np.mean(layers)}"

            output_string += f' - ({time.time() - epoch_start:.4f}s)'
            #print(output_string)
            if val_acc > best_val_acc:
                #print("------------ This was a new best! ----------")
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_layer_count = np.mean(layers)
                not_improved_for = 0
            else:
                if val_loss > min_val_loss:
                    not_improved_for += 1
                    if not_improved_for == 350:
                        break
                else:
                    not_improved_for = 0
                    min_val_loss = val_loss
            scheduler.step(train_loss_mean)
        print(f'Fold: {fold_idx}, acc: {best_val_acc}, loss: {best_val_loss}, layers: {best_layer_count}')
        best_losses.append(best_val_loss)
        best_accs.append(best_val_acc)
        best_layers.append(best_layer_count)
    return {
        'loss': np.mean(best_losses),
        'acc': np.mean(best_accs),
        'stddev': np.std(best_accs),
        'layers': np.mean(best_layers),
        'total_time': time.time() - start_time,
    }

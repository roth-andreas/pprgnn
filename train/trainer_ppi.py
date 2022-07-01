import copy
import time
import torch_geometric as pyg
import torch
import numpy as np
from model.model import APPNP, PPRGNN_PPI
from torch_geometric.data import DataLoader

from train.train import train, test
from train.utils import init_seeds


def train_model(lr, weight_decay, self_loops, epochs, clip, width, schedule, **kwargs):
    init_seeds()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    criterion = torch.nn.BCEWithLogitsLoss()
    train_dataset = pyg.datasets.PPI('./data', split='train',
                                     transform=pyg.transforms.GCNNorm(add_self_loops=self_loops))
    val_dataset = pyg.datasets.PPI('./data', split='val',
                                   transform=pyg.transforms.GCNNorm(add_self_loops=self_loops))
    test_dataset = pyg.datasets.PPI('./data', split='test',
                                    transform=pyg.transforms.GCNNorm(add_self_loops=self_loops))
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    if kwargs["type"] == 'appnp':
        model = APPNP(in_dim=train_dataset.num_features, out_dim=train_dataset.num_classes, h_dim=width)
    elif kwargs["type"] == "pprgnn":
        model = PPRGNN_PPI(train_dataset.num_features, train_dataset.num_classes, h_dim=width, **kwargs)

    model = model.to(device)
    best_state = None

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if schedule:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4, verbose=True, factor=0.9)

    start_epoch = 1
    best_val_f1 = 0
    best_val_loss = np.inf

    not_improved_for = 0
    start_time = time.time()
    time_to_best = 0
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        output_string = f'Epoch {epoch}:'
        # train

        train_f1, train_loss, train_layers = train(model, train_loader, optimizer, criterion, device, clip)
        output_string += f' Train - Loss: {train_loss:.4f}, F1: {train_f1:.4f}'
        # val
        if val_dataset is not None:
            val_f1, val_loss, val_layers = test(model, val_loader, criterion, device)
            output_string += f' Val - Loss: {val_loss:.4f}, F1: {val_f1:.4f}'
        else:
            val_f1 = train_f1

        output_string += f' - ({time.time() - epoch_start:.4f}s)'
        print(output_string)

        if val_f1 > best_val_f1:
            print("------------ This was a new best! ----------")
            best_val_f1 = val_f1
            not_improved_for = 0
            best_state = copy.deepcopy(model.state_dict())
            time_to_best = time.time() - start_time
            best_epoch = epoch
        else:
            if val_loss > best_val_loss:
                not_improved_for += 1
                if not_improved_for == 50:
                    break
            else:
                not_improved_for = 0
                best_val_loss = val_loss
        if schedule:
            scheduler.step(train_loss)

    # test
    model.load_state_dict(best_state)
    test_f1, test_loss, test_layers = test(model, test_loader, criterion, device)
    return {
        'loss': test_loss,
        'f1': test_f1,
        'total_time': time.time() - start_time,
        'time_to_best': time_to_best,
        'total_iterations': epoch,
        'best_epoch': best_epoch,
        'time_per_iteration': time_to_best / best_epoch,
        'best_layers': test_layers
    }

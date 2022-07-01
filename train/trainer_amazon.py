import time
import torch
import torch.nn as nn
import numpy as np

from load.load_amazon import load_txt_data
from model.model import PPRGNN_Amazon, APPNP
from train.train import Evaluation
from train.utils import init_seeds


def train_model(lr, weight_decay, self_loops, epochs, clip,
                width, portion, schedule, **kwargs):
    init_seeds()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    adj, features, labels, idx_train, idx_val, _, num_nodes, num_class = load_txt_data('amazon-all', portion,
                                                                                       self_loops)
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    criterion = nn.BCEWithLogitsLoss()
    if kwargs["type"] == 'appnp':
        model = APPNP(in_dim=features.shape[1], out_dim=num_class, h_dim=width)
    elif kwargs["type"] == "pprgnn":
        model = PPRGNN_Amazon(in_dim=features.shape[1], out_dim=num_class, num_nodes=num_nodes, h_dim=width,
                              **kwargs)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if schedule:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4, verbose=True, factor=0.9)
    else:
        scheduler = None

    start_epoch = 1
    best_f1_micro = 0
    min_test_loss = np.inf

    best_loss = 0
    best_f1_macro = 0

    not_improved_for = 0
    start_time = time.time()
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        output_string = f'Epoch {epoch}:'
        # train
        model.train()
        output, layers_train = model(features, adj=adj)
        train_loss = criterion(output[idx_train], labels[idx_train])
        optimizer.zero_grad()
        train_loss.backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        f1_train_micro, f1_train_macro = Evaluation(output[idx_train], labels[idx_train])
        optimizer.step()

        model.eval()
        with torch.no_grad():
            output, layers_test = model(features, adj=adj)

            test_loss = criterion(output[idx_val], labels[idx_val])
            test_f1_micro, f1_test_macro = Evaluation(output[idx_val], labels[idx_val])

        output_string += f' Train: Loss: {train_loss:.4f}, Micro_F1: {f1_train_micro:.4f}, Macro_F1: {f1_train_macro:.4f}, Layers: {layers_train},' \
                         f' Test: Loss: {test_loss:.4f}, Micro_F1: {test_f1_micro:.4f}, Macro_F1: {f1_test_macro:.4f}, Layers: {layers_test}'

        output_string += f' - ({time.time() - epoch_start:.4f}s)'
        print(output_string)
        if test_f1_micro > best_f1_micro:
            print("------------ This was a new best! ----------")
            best_f1_micro = test_f1_micro.item()
            best_f1_macro = f1_test_macro.item()
            best_loss = test_loss.item()
            not_improved_for = 0
            time_to_best = time.time() - start_time
            best_epoch = epoch
        else:
            if test_loss > min_test_loss:
                not_improved_for += 1
                if not_improved_for == 100:
                    break
            else:
                not_improved_for = 0
                min_test_loss = test_loss
        if schedule:
            scheduler.step(train_loss)
    return {
        'f1_micro': best_f1_micro,
        'f1_macro': best_f1_macro,
        'loss': best_loss,
        'total_time': time.time() - start_time,
        'time_to_best': time_to_best,
        'total_iterations': epoch,
        'best_epoch': best_epoch,
        'time_per_iteration': time_to_best / best_epoch
    }

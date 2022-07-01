import torch
import numpy as np
from sklearn.metrics import f1_score
from sklearn import metrics


def Evaluation(output, labels):
    preds = output.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    num_correct = 0
    binary_pred = np.zeros(preds.shape).astype('int')
    for i in range(preds.shape[0]):
        k = labels[i].sum().astype('int')
        topk_idx = preds[i].argsort()[-k:]
        binary_pred[i][topk_idx] = 1
        for pos in list(labels[i].nonzero()[0]):
            if labels[i][pos] and labels[i][pos] == binary_pred[i][pos]:
                num_correct += 1

    print('total number of correct is: {}'.format(num_correct))

    return metrics.f1_score(labels, binary_pred, average="micro"), metrics.f1_score(labels, binary_pred, average="macro")


def train(model, dataloader, optimizer, criterion, device, clip):
    model.train()
    loss_list = []
    layers = []
    f1_scores = []
    for batch in dataloader:
        batch = batch.to(device)
        output, layers_used = model(batch.x, batch.edge_index, batch.edge_weight)
        loss = criterion(output, batch.y)
        model.zero_grad()
        loss.backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        predictions = (output > 0.5)
        loss_list.append(loss.item())
        layers.append(layers_used)
        f1_scores.append(f1_score(batch.y.cpu().numpy(), predictions.cpu().numpy(), average='micro'))
    return np.mean(f1_scores), np.mean(loss_list), np.mean(layers)


def test(model, dataloader, criterion, device):
    model.eval()
    with torch.no_grad():
        loss_list = []
        layers = []
        f1_scores = []
        for batch in dataloader:
            batch = batch.to(device)
            output, layers_used = model(batch.x, batch.edge_index, batch.edge_weight)
            loss = criterion(output, batch.y)
            predictions = (output > 0.5)
            loss_list.append(loss.item())
            layers.append(layers_used)
            f1_scores.append(f1_score(batch.y.cpu().numpy(), predictions.cpu().numpy(), average='micro'))
        return np.mean(f1_scores), np.mean(loss_list), np.mean(layers)

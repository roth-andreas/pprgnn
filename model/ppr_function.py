import time

import torch
import numpy as np


class PPRFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, W, B, A, X_0, eps=1):
        X_0 = B if X_0 is None else X_0
        At = torch.transpose(A, 0, 1)
        epsilon = eps
        offset = 5
        X_new, X_prev, i, converge_index = PPRFunction.evaluate(B, W, At, X_0, 0, epsilon, offset)
        ctx.save_for_backward(W, B, A, X_prev)
        ctx.max_index = i
        ctx.epsilon = epsilon
        ctx.offset = offset
        return X_new, converge_index

    @staticmethod
    def evaluate_x_i(B, W, At, X, i, stop_index, epsilon):
        for j in range(i, stop_index - 1, -1):
            X = torch.matmul(W / (1 + (j * epsilon)), X)
            X = torch.spmm(At, X.T).T
            X += B
            X = torch.nn.functional.relu(X)
        return X

    @staticmethod
    def evaluate(B, W, At, X_0, stop_index=0, epsilon=10, offset=5):
        k = 300
        h_stop_index = stop_index + offset
        X_ = B
        for converge_index in range(k):
            X_ = torch.matmul(W / (1 + (converge_index * epsilon)), X_)
            X_ = torch.spmm(At, X_.T).T
            X_ = torch.nn.functional.relu(X_)
            if torch.norm(X_, np.inf) < 3e-6:
                break
        X_prev = PPRFunction.evaluate_x_i(B, W, At, X_0, converge_index + offset, h_stop_index, epsilon)
        X_new = PPRFunction.evaluate_x_i(B, W, At, X_prev, offset - 1, stop_index, epsilon)
        return X_new, X_prev, converge_index + offset, converge_index

    @staticmethod
    def backward(ctx, grad_output, _):
        W, B, A, X = ctx.saved_tensors
        At = torch.transpose(A, 0, 1)
        epsilon = ctx.epsilon
        offset = ctx.offset
        dL_dQ = grad_output
        dL_dW = torch.zeros_like(W)
        dL_dB = torch.zeros_like(B)
        finished = False
        old_offset = 0

        forward_W = W
        backward_W = W.T
        while not finished:
            for i in range(old_offset, offset):
                X_i = PPRFunction.evaluate_x_i(B, forward_W, At, X, i=offset - 1, stop_index=i + 1, epsilon=epsilon)
                Z_i = torch.spmm(At, torch.matmul(forward_W / (1 + i*epsilon), X_i).T).T + B
                dL_dQ_new = (Z_i > 0) * dL_dQ
                dL_dB += dL_dQ_new
                dL_dQ_new = torch.spmm(A / (1 + i*epsilon), dL_dQ_new.T).T
                dL_dW += torch.matmul(dL_dQ_new, X_i.T)
                dL_dQ_new = torch.matmul(backward_W, dL_dQ_new)
                err_q = torch.norm(dL_dQ - dL_dQ_new, np.inf)
                dL_dQ = dL_dQ_new
                if err_q < 1e-5 or i == 5:
                    finished = True
                    break
            if finished:
                break
            else:
                X = PPRFunction.evaluate_x_i(B, W, At, X, i=ctx.max_index + offset, stop_index=offset, epsilon=epsilon)
                old_offset = offset
                offset = offset + ctx.offset
        return dL_dW, dL_dB, None, None, None, None, None, None, None, None, None

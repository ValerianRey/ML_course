# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""

from gradient_descent import compute_gradient


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    return compute_gradient(y, tx, w)


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    w = initial_w
    losses = np.zeros(max_iters)
    ws = np.zeros((max_iters, w.shape[0]))
    for i in range(max_iters):
        loss = compute_loss(y, tx, w)
        losses[i] = loss
        for batch_y, batch_tx in batch_iter(y, tx, batch_size=16, num_batches=1):
            gradient = compute_stoch_gradient(batch_y, batch_tx, w)
            w = w - gamma * gradient
        ws[i] = w
    return losses, ws


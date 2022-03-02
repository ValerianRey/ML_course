# -*- coding: utf-8 -*-
"""Gradient Descent"""
import numpy as np

from costs import compute_loss_mse


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    return (-1/len(y)) * np.dot(tx.T, y - np.dot(tx, w))


def gradient_descent(y, tx, initial_w, max_iters, gamma, verbose=False):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss = compute_loss_mse(y, tx, w)
        gradient = compute_gradient(y, tx, w)
        w = w - gamma * gradient
        # store w and loss
        ws.append(w)
        losses.append(loss)
        if verbose:
        	print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
        	bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws

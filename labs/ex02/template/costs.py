# -*- coding: utf-8 -*-
"""Function used to compute the loss."""

import numpy as np


def compute_loss_mse(y, tx, w):
    y_hat = np.dot(tx, w)
    e = y - y_hat
    sq_e = e ** 2
    loss = sq_e.mean()
    return loss


def compute_loss_mae(y, tx, w):
    y_hat = np.dot(tx, w)
    e = y - y_hat
    sq_e = np.abs(e)
    loss = sq_e.mean()
    return loss


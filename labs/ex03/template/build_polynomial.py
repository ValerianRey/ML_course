# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    return np.stack([x ** i for i in range(degree+1)], axis=1)


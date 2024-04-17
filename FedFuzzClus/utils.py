#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr. 15 9:37 a.m. 2024

@author: AI&RD Research Group, Department of Information Engineering, University of Pisa
"""
from numba import jit
import numpy as np
from typing import Final, List


FROBENIUS_NORM: Final[str] = 'fro'


@jit(nopython=True)
def numba_norm(u: np.ndarray, v: np.ndarray):
    return np.linalg.norm(u - v)


def norm_fro(u: np.ndarray):
    return np.linalg.norm(u, ord=FROBENIUS_NORM)


def generate_client_centers(K: int, num_features: int, num_clients: int, features_per_client: List, seed: int) -> List:
    import numpy as np
    np.random.seed(seed)
    C = np.random.rand(K, num_features)
    centers_chunks = []
    last_index = 0
    for i in range(num_clients):
        cstart_index = last_index
        clast_index = cstart_index + features_per_client[i]
        centers_chunks.append(C[:, cstart_index:clast_index])
        last_index = clast_index
    return centers_chunks

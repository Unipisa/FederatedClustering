#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr. 15 9:37 a.m. 2024

@author: AI&RD Research Group, Department of Information Engineering, University of Pisa
"""
import numpy as np
from typing import Dict, List, Tuple
from numba import jit, njit
from functools import reduce

from FedFuzzClus.utils import generate_client_centers, norm_fro


@njit
def compute_distances_numba(n_features: int, centers: np.array, dataset: np.array):
    n_centers = len(centers)
    n_obj = len(dataset)
    all_distances = [[0.0] * n_centers for i in range(n_obj)]
    for idx in range(n_obj):
        instance = dataset[idx]
        for idx_center in range(n_centers):
            center = centers[idx_center]
            distance = 0
            for i in range(n_features):
                distance = distance + (instance[i] - center[i]) ** 2
            all_distances[idx][idx_center] = distance
    return all_distances


@njit
def update_local_centers_numba(centers: np.array, n_features: int, dataset: np.array, u_t: np.array,
                               lambda_factor: float):
    n_centers = len(centers)
    total_numerators = [[0.] * n_features for i in range(n_centers)]
    total_denominators = [0.] * n_centers
    for i in range(len(dataset)):
        obj = dataset[i]
        for cluster_id in range(n_centers):
            tmp_prod = [obj[j] * (u_t[i][cluster_id] ** lambda_factor) for j in range(n_features)]
            total_numerators[cluster_id] = [total_numerators[cluster_id][j] + tmp_prod[j] for j in range(n_features)]
            total_denominators[cluster_id] = total_denominators[cluster_id] + u_t[i][cluster_id] ** lambda_factor
    return [([total_numerators[i][j] / total_denominators[i] for j in range(n_features)]
             if total_denominators[i] != 0 else None)
            for i in range(n_centers)]


@njit
def compute_u_c_t_matrix_numba(distance_matrix, lambda_factor):
    mem_obj_cluster = lambda idx, C: reduce(lambda a, b: a + b,
                                            [((C[idx] / C[i]) if idx != i else 1.0) ** (2.0 / (lambda_factor - 1)) for i
                                             in range(len(C))]) ** -1
    mem_obj_clusters = lambda C_param: [mem_obj_cluster(idx, C_param) for idx in range(len(C_param))]
    return [mem_obj_clusters(obj) for obj in distance_matrix]


@njit
def compute_distance_matrix_numba(n_centers: int, n_objects: int, client_responses):
    object_cluster_distance_mtrx = [[0.0] * n_centers for i in range(n_objects)]
    n_clients = len(client_responses)
    for i in range(n_objects):
        for j in range(n_centers):
            distance = 0
            for k in range(n_clients):
                distance = distance + client_responses[k][i][j]
            object_cluster_distance_mtrx[i][j] = distance ** 0.5
    return object_cluster_distance_mtrx


class FederatedVerticalFCMClient:

    def __init__(self, **kwargs) -> None:
        self.__dataset = np.array(kwargs.get("dataset"))
        self.__n_features = len(self.__dataset[0])
        self.__centers = np.array(kwargs.get("centers"))
        self.__lambda_factor = kwargs.get("lambda_factor")

    def __compute_distances(self) -> List:
        return compute_distances_numba(self.__n_features, self.__centers, self.__dataset)

    def update_local_centers(self, round: int = 0, u_t: List = None) -> List:
        if round > 0:
            lambda_factor = self.__lambda_factor
            centers = self.__centers
            dataset = self.__dataset
            tmp_centers = update_local_centers_numba(centers, self.__n_features, dataset, np.array(u_t), lambda_factor)
            self.__centers = [tmp_centers[i] if tmp_centers[i] is not None else centers[i] for i in range(len(centers))]
            self.__centers = np.array(self.__centers)
        return self.__compute_distances()


class FederatedVerticalFCMServer:

    def __init__(self, **kwargs) -> None:
        self.__current_round = 0
        self.__epsilon = kwargs.get("epsilon")
        self.__norm_fn = norm_fro
        self.__lambda_factor = kwargs.get("lambda_factor")
        self.__max_number_rounds = kwargs.get("max_number_rounds")
        self.__last_D_matrix = None
        self.__fnorms = []
        self.__n_clusters = kwargs.get("num_clusters")

    def process_round(self, client_responses: List) -> Tuple:
        lambda_factor = self.__lambda_factor

        n_objects = len(client_responses[0])
        client_responses = np.array(client_responses)
        distance_matrix = np.array(compute_distance_matrix_numba(self.__n_clusters, n_objects, client_responses))

        next_round = True

        if self.__current_round > 0:
            d_t_1 = self.__last_D_matrix
            fnorm_value = self.__norm_fn(distance_matrix - d_t_1)
            self.__fnorms.append(fnorm_value)
            next_round = self.__current_round < self.__max_number_rounds
            # if fnorm_value < self.__epsilon:
            #     next_round = False
        u_c_t = compute_u_c_t_matrix_numba(distance_matrix, lambda_factor)

        self.__current_round = self.__current_round + 1
        self.__last_D_matrix = distance_matrix
        return next_round, u_c_t

    @property
    def current_round(self) -> int:
        return self.__current_round

    def finalize(self):
        return self.__fnorms


def run_ffcv_experiment(dataset_chunks: List,
                        num_clients: int,
                        features_per_client: List,
                        server_params: Dict,
                        client_params: Dict,
                        seed: int):

    num_clusters = server_params.get("num_clusters")
    num_features = sum(features_per_client)

    centers_chunks = generate_client_centers(num_clusters, num_features, num_clients, features_per_client, seed)

    clients = [FederatedVerticalFCMClient(dataset=dataset_chunks[i], centers=centers_chunks[i], **client_params)
               for i in range(num_clients)]

    server = FederatedVerticalFCMServer(**server_params)

    u_t = None
    next_round = True

    while next_round:
        current_round = server.current_round
        client_responses = []
        for client in clients:
            response = client.update_local_centers(current_round, u_t)
            client_responses.append(response)

        next_round, u_t = server.process_round(client_responses)

    fnorms = server.finalize()

    return u_t

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr. 15 9:37 a.m. 2024

@author: AI&RD Research Group, Department of Information Engineering, University of Pisa
"""
from typing import List, Dict
from functools import reduce
import numpy as np
from collections import Counter
from FedFuzzClus.utils import norm_fro, generate_client_centers


class FederatedVerticalCMClient:

    def __init__(self, **kwargs) -> None:
        self.__dataset = np.array(kwargs.get("dataset"))
        self.__centers = kwargs.get("centers")

    def __compute_distances(self) -> List:
        centers = self.__centers
        dist = lambda x, c: sum([(x_i - c[i])**2 for i, x_i in enumerate(x)])
        distance_2_centers = lambda x: [dist(x, center) for center in centers]
        return [distance_2_centers(instance) for instance in self.__dataset]

    def update_local_centers(self, round: int = 0, q_pts: List = None, pts_by_cluster: List = None) -> List:
        if round > 0:
            cluster_ids = set(q_pts)
            centers = self.__centers
            dataset = self.__dataset
            for cluster_id in cluster_ids:
                mask = list(map(lambda y: y == cluster_id, q_pts))
                centers[cluster_id] = sum(dataset[mask, :]) / pts_by_cluster[cluster_id] * 1.0
        return self.__compute_distances()


class FederatedVerticalCMServer:

    def __init__(self, **kwargs) -> None:
        self.__current_round = 0
        self.__epsilon = kwargs.get("epsilon")
        self.__norm_fn = norm_fro
        self.__max_number_rounds = kwargs.get("max_number_rounds")
        self.__D_matrix = []
        self.__fnorms = []
        self.__num_clusters = kwargs.get("num_clusters")

    def process_round(self, client_responses: List) -> (bool, List, List):
        D_matrix = self.__D_matrix
        distance_matrices = list(map(lambda ar: np.matrix(ar), client_responses))
        object_cluster_distance = reduce(lambda a, b: a + b, distance_matrices)
        distance_matrix = list(map(lambda d: [i**0.5 for i in d], object_cluster_distance.tolist()))
        D_matrix.append(distance_matrix)
        next_round = True

        if self.__current_round > 0:
            d_t = np.array(D_matrix[-1])
            d_t_1 = np.array(D_matrix[-2])
            fnorm_value = self.__norm_fn(d_t - d_t_1)
            self.__fnorms.append(fnorm_value)
            next_round = self.__current_round < self.__max_number_rounds or fnorm_value > self.__epsilon

        q_pts = [np.argmin(distances) for distances in distance_matrix]
        counter = Counter(q_pts)
        pts_by_cluster = []
        for cluster in range(self.__num_clusters):
            pts_by_cluster.append(counter.get(cluster, 0))
        self.__current_round = self.__current_round + 1
        return next_round, q_pts, pts_by_cluster

    @property
    def current_round(self) -> int:
        return self.__current_round

    def finalize(self):
        return self.__fnorms[-1]


def run_fcv_experiment(dataset_chunks: List,
                       num_clients: int,
                       features_per_client: List,
                       server_params: Dict,
                       client_params: Dict,
                       seed: int):

    num_clusters = server_params.get("num_clusters")
    num_features = sum(features_per_client)

    centers_chunks = generate_client_centers(num_clusters, num_features, num_clients, features_per_client, seed)

    clients = [FederatedVerticalCMClient(dataset=dataset_chunks[i], centers=centers_chunks[i], **client_params)
               for i in range(num_clients)]

    server = FederatedVerticalCMServer(**server_params)

    q_pts = None
    pts_by_cluster = None
    next_round = True

    while next_round:

        current_round = server.current_round
        client_responses = []

        for client in clients:
            response = client.update_local_centers(current_round, q_pts, pts_by_cluster)
            client_responses.append(response)

        next_round, q_pts, pts_by_cluster = server.process_round(client_responses)

    fnorms = server.finalize()

    return q_pts

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr. 15 9:37 a.m. 2024

@author: AI&RD Research Group, Department of Information Engineering, University of Pisa
"""
import random
import numpy as np
from typing import Dict, List, Tuple
from FedFuzzClus.utils import numba_norm, norm_fro


class FederatedHorizontalFCMClient:

    def __init__(self, **kwargs):
        self.__dataset = kwargs.get('dataset')
        self.__num_features = len(self.__dataset[0])
        self.__classes = [-1] * len(self.__dataset)
        self.__distance_fn = numba_norm
        self.__lambda_factor = kwargs.get('lambda_factor')
        self.y_true = kwargs.get('y_true')
        self.__num_clusters = kwargs.get("num_clusters")
        self.__centroid_seed = kwargs.get("centroid_seed")

    def evaluate_cluster_assignment(self, centers: List) -> Tuple:
        return self.__local_sums(centers)

    def get_dataset(self):
        return self.__dataset

    def init_step(self):
        seed = self.__centroid_seed
        c = self.__num_clusters
        np.random.seed(seed)
        centers = np.random.rand(c, self.__num_features)
        return centers

    def finalize(self, centers: List[np.array]) -> None:
        num_clusters = len(centers)
        dataset = self.__dataset
        num_objects = len(dataset)
        lambda_factor = self.__lambda_factor
        distance_fn = self.__distance_fn

        u = [[0] * num_objects for i in range(num_clusters)]

        for i in range(num_objects):

            denom = 0
            numer = [0] * num_clusters
            x = dataset[i]

            for c in range(num_clusters):
                vc = centers[c]

                numer[c] = (distance_fn(x, vc)) ** ((2) / (lambda_factor - 1))
                if numer[c] == 0:
                    numer[c] = np.finfo(np.float64).eps
                denom = denom + (1 / numer[c])

            for c in range(num_clusters):
                u_c_i = (numer[c] * denom) ** -1
                u[c][i] = u_c_i
        u = np.asarray(u).T
        y_pred = np.argmax(u, 1)
        return u, y_pred

    def __local_sums(self, centers: List[np.array]) -> Tuple:
        # (1) some initialization
        num_clusters = len(centers)
        dataset = self.__dataset
        num_objects = len(dataset)
        lambda_factor = self.__lambda_factor
        distance_fn = self.__distance_fn
        num_features = self.__num_features

        ws = [[0] * num_features for i in range(num_clusters)]
        u = [0] * num_clusters
        u_x_c = list()
        d_x_c = list()

        for i in range(num_objects):

            denom = 0
            numer = [0] * num_clusters
            x = dataset[i]

            membership_c = list()

            for c in range(num_clusters):
                vc = centers[c]
                numer[c] = (distance_fn(x, vc)) ** ((2) / (lambda_factor - 1))
                if numer[c] == 0:
                    numer[c] = np.finfo(np.float64).eps
                denom = denom + (1 / numer[c])

            d_x_c.append(numer)

            for c in range(num_clusters):
                u_c_i = (numer[c] * denom) ** -1
                ws[c] = ws[c] + (u_c_i ** lambda_factor) * x
                u[c] = u[c] + (u_c_i ** lambda_factor)
                membership_c.append(u_c_i)

            u_x_c.append(membership_c)

        return u, ws


class FederatedHorizontalFCMServer:

    def __init__(self, **kwargs):
        self.__current_round = 0
        self.__epsilon = kwargs.get('epsilon')
        self.__max_number_rounds = kwargs.get('max_number_rounds', 10)
        self.__num_clusters = kwargs.get('num_clusters')
        self.__norm_fm = norm_fro
        self.__fnorms = []
        self.__cluster_centers = []

    def init_centers(self, centers):
        self.__cluster_centers.append(centers)
        return centers

    def next_round(self) -> bool:
        num_clusters = self.__num_clusters
        cluster_centers = self.__cluster_centers

        num_centers = len(cluster_centers)
        tot_diff_sum = None
        if num_centers > 1:
            centers_r = np.array(cluster_centers[-1])
            centers_r_1 = np.array(cluster_centers[-2])
            tot_diff_sum = self.__norm_fm(centers_r - centers_r_1)
            #             if (tot_diff_sum < self.__epsilon):
            #                 return False
            self.__fnorms.append(tot_diff_sum)
        result = self.__current_round < self.__max_number_rounds
        self.__current_round = self.__current_round + 1 if result else self.__current_round
        return result

    def process_clustering_results(self, client_responses: List):
        num_clients = len(client_responses)
        num_clusters = self.__num_clusters
        num_features = len(client_responses[0][1][0])
        u_list = [0] * num_clusters
        ws_list = [[0] * num_features for i in range(num_clusters)]

        for client_idx in range(num_clients):
            response = client_responses[client_idx]
            for i in range(num_clusters):
                client_u = response[0][i]
                client_ws = response[1][i] if response[1][i] is np.array else np.array(response[1][i])
                u_list[i] = u_list[i] + client_u
                ws_list[i] = ws_list[i] + client_ws

        new_cluster_centers = []
        prev_cluster_centers = None

        if self.__cluster_centers:
            prev_cluster_centers = self.__cluster_centers[-1]

        for i in range(num_clusters):
            u = u_list[i]
            ws = ws_list[i]
            if u == 0 and prev_cluster_centers:
                center = prev_cluster_centers[i]
            else:
                center = ws / u
            new_cluster_centers.append(np.array(center))

        self.__cluster_centers.append(new_cluster_centers)
        return new_cluster_centers

    def get_centers(self) -> List:
        return self.__cluster_centers[-1]

    @property
    def current_round(self) -> int:
        return self.__current_round

    def finalize(self) -> None:
        centers = self.__cluster_centers[1:]
        fnorms = self.__fnorms
        return fnorms, centers


def run_ffch_experiment(num_clients: int,
                        min_num_clients: int,
                        dataset_chunks: List[np.array],
                        y_chunks: List,
                        client_params: Dict[str, bytes],
                        server_params: Dict[str, bytes],
                        random_clients: bool = True,
                        random_clients_seed: int = None):

    max_num_rounds = server_params.get("max_number_rounds")
    # (1) CLIENTS INITIALIZATION ==================================================================
    clients = [FederatedHorizontalFCMClient(dataset=dataset_chunks[cli],
                                            y_true=y_chunks[cli],
                                            **client_params) for cli in range(num_clients)]

    print(f'starting experiment, total number of clients {num_clients}, min number of clients {min_num_clients}, '
          f'random_clients = {random_clients}')

    active_clients_idx: List = None
    all_clients = num_clients == min_num_clients

    random.seed(random_clients_seed)

    if not random_clients and not all_clients:
        active_clients_idx = random.sample(range(num_clients), min_num_clients)

    # (2) SERVER INITIALIZATION ==================================================================
    server = FederatedHorizontalFCMServer(**server_params)

    _clients_idx = [i for i in range(num_clients)]
    selected_client_idx = random.choice(_clients_idx)

    selected_client = clients[selected_client_idx]

    initial_centers = selected_client.init_step()

    server.init_centers(initial_centers)

    # (4) FEDERATED TRAINING ====================================================================
    while server.next_round():
        all_selected_clients = []

        if random_clients and not all_clients:
            active_clients_idx = random.sample(range(num_clients), min_num_clients)

        centers = server.get_centers()
        client_responses = []
        client_responses_append = client_responses.append

        for cli, client in enumerate(clients):
            if not all_clients and cli not in active_clients_idx:
                continue
            response = client.evaluate_cluster_assignment(centers)
            client_responses_append(response)
            all_selected_clients.append(cli)

        server.process_clustering_results(client_responses)

    centers = server.get_centers()
    return server, clients, centers

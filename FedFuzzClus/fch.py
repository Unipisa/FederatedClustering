#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr. 15 9:37 a.m. 2024

@author: AI&RD Research Group, Department of Information Engineering, University of Pisa
"""
import random
import numpy as np
from typing import Dict, List, Tuple

from FedFuzzClus.ffcv import norm_fro
from FedFuzzClus.utils import numba_norm


class FederatedHorizontalCMClient:

    def __init__(self, dataset, y_true, centroid_seed, num_clusters) -> None:
        self.__dataset = dataset
        self.__y_true = y_true
        self.__centroid_seed = centroid_seed
        self.__num_clusters = num_clusters
        self.__num_features = len(self.__dataset[0])
        self.__classes = [-1] * len(self.__dataset)
        self.__distance_fn = numba_norm

    def init_step(self):
        seed = self.__centroid_seed
        c = self.__num_clusters
        np.random.seed(seed)
        centers = np.random.rand(c, self.__num_features)
        return centers

    def evaluate_cluster_assignment(self, centers: List) -> List:
        # (1). some initialization
        num_clusters = len(centers)
        dataset = self.__dataset
        num_features = self.__num_features
        classes = self.__classes
        num_objects = len(dataset)
        get_label = self.__get_label

        nc_list = [0] * num_clusters
        lsc_list = [np.array([0] * num_features) for i in range(num_clusters)]

        # (2) Updating the class value for each object in the dataset
        for i in range(num_objects):
            obj = dataset[i]
            label = get_label(obj, centers)
            classes[i] = label
            # updating stats for each cluster
            nc_list[label] = nc_list[label] + 1
            lsc_list[label] = lsc_list[label] + obj

        for i in range(num_clusters):
            if nc_list[i] == 1:
                nc_list[i] = 0
                lsc_list[i] = 0

        # (3) Preparing data to return
        to_return = [(lsc_list[i], nc_list[i]) for i in range(num_clusters)]
        return to_return

    def finalize(self, centers: List[np.array]) -> np.array:
        dataset = self.__dataset
        num_objects = len(dataset)
        get_label = self.__get_label
        return np.array([get_label(dataset[i], centers) for i in range(num_objects)])

    def __get_label(self, obj_data: np.array, centers: List[np.array]):

        max_value = 2 ** 64
        num_clusters = len(centers)
        distance_fn = self.__distance_fn
        label_idx = -1

        for i in range(num_clusters):
            center = centers[i]
            distance = distance_fn(obj_data, center)

            if distance < max_value:
                label_idx = i
                max_value = distance

        return label_idx


class FederatedHorizontalCMServer:

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
        cluster_centers = self.__cluster_centers

        num_centers = len(cluster_centers)

        if num_centers > 1:
            centers_r = np.array(cluster_centers[-1])
            centers_r_1 = np.array(cluster_centers[-2])
            fnorm_value = self.__norm_fm(centers_r - centers_r_1)
            self.__fnorms.append(fnorm_value)
            # if fnorm_value < self.__epsilon:
            #     return False

        result = self.__current_round < self.__max_number_rounds

        self.__current_round = self.__current_round + 1 if result else 0
        return result

    def process_clustering_results(self, client_responses: List):
        num_clients = len(client_responses)
        num_clusters = self.__num_clusters

        num_features = len(client_responses[0][1][0])
        nc_list = [0] * num_clusters
        lsc_list = [np.array([0] * num_features) for i in range(num_clusters)]

        for client_idx in range(num_clients):
            # remember the response is a list of tuples where each tuple represents the (LSC, NC) for each cluster
            response = client_responses[client_idx]
            for i in range(num_clusters):
                client_lsc = response[i][0] if response[i][0] is np.array else np.array(response[i][0])
                client_nc = response[i][1]
                lsc_list[i] = lsc_list[i] + client_lsc
                nc_list[i] = nc_list[i] + client_nc

        new_cluster_centers = []
        prev_cluster_centers = self.__cluster_centers[-1]

        for i in range(num_clusters):
            nc = nc_list[i]
            lsc = lsc_list[i]
            if nc == 0:
                center = prev_cluster_centers[i]
            else:
                center = lsc / (nc * 1.0)
            new_cluster_centers.append(center)

        self.__cluster_centers.append(new_cluster_centers)

    def get_centers(self) -> List:
        return self.__cluster_centers[-1]

    @property
    def current_round(self) -> int:
        return self.__current_round

    def finalize(self) -> Tuple:
        centers = self.__cluster_centers[1:]
        fnorms = self.__fnorms
        return fnorms, centers


def run_fch_experiment(num_clients: int,
                       min_num_clients: int,
                       dataset_chunks: List[np.array],
                       y_chunks: List,
                       client_params: Dict[str, bytes],
                       server_params: Dict[str, bytes],
                       random_clients: bool = True,
                       random_clients_seed: int = None):

    # (1) CLIENTS INITIALIZATION ==================================================================
    clients = [FederatedHorizontalCMClient(dataset=dataset_chunks[cli],
                                           y_true=y_chunks[cli],
                                           **client_params)
               for cli in range(num_clients)]

    print(f'starting server, total number of clients {num_clients}, min number of clients {min_num_clients}')

    active_clients_idx: List = None
    all_clients = num_clients == min_num_clients

    random.seed(random_clients_seed)

    if not random_clients and not all_clients:
        active_clients_idx = random.sample(range(num_clients), min_num_clients)

    # (2) SERVER INITIALIZATION ==================================================================
    server = FederatedHorizontalCMServer(**server_params)

    _clients_idx = [i for i in range(num_clients)]
    selected_client_idx = random.choice(_clients_idx)
    selected_client = clients[selected_client_idx]

    initial_centers = selected_client.init_step()
    server.init_centers(initial_centers)

    # (3) FEDERATED TRAINING ====================================================================
    while server.next_round():

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

        server.process_clustering_results(client_responses)

    fnorms, centers = server.finalize()

    return server, clients, centers[-1]

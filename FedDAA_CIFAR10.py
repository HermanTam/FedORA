
from collections import Counter, defaultdict
import copy
import gc
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.args import *
from utils.constants import *
from utils.drift_utils import (
    DEFAULT_THRESHOLDS,
    apply_objective_policy,
    assign_objectives,
    diagnose_drift_type,
    label_distribution_shift,
    prototype_shift,
    shared_label_accuracy_drop,
)
from utils.experiment_logging import ExperimentLogger
from utils.experiment_helpers import (
    parse_seed_argument,
    evaluate_iterator_accuracy,
    build_bar_plot,
    aggregate_summary,
)
from utils.metrics_utils import sample_stats
from utils.utils import *
from utils.adaptive_thresholds import AdaptiveThresholdManager
from utils.privacy import DifferentialPrivacyManager


def release_client_resources(clients):
    if not clients:
        return
    for client in clients:
        logger = getattr(client, "logger", None)
        if logger is not None:
            try:
                logger.flush()
            except Exception:
                pass
            try:
                logger.close()
            except Exception:
                pass
            client.logger = None
        for attr in (
            "train_iterator",
            "val_iterator",
            "test_iterator",
            "train_loader",
            "current_train_iterator",
            "current_val_iterator",
            "current_test_iterator",
            "last_train_iterator",
            "last_val_iterator",
            "last_test_iterator",
        ):
            if hasattr(client, attr):
                setattr(client, attr, None)

def init_clients(args_, root_path, logs_dir):
    """
    initialize clients from data folders

    :param args_:
    :param root_path: path to directory containing data folders
    :param logs_dir: path to logs root
    :return: List[Client]
    """
    print("===> Building data iterators..")
    class_number = CLASS_NUMBER[LOADER_TYPE[args_.experiment]]

    if LOADER_TYPE[args_.experiment] == 'cifar10-c':
        if 'test' in root_path: 
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/cifar10-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True,
                    test_num = 3
                )
        else: 
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/cifar10-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation 
                )
    elif LOADER_TYPE[args_.experiment] == 'tiny-imagenet-c':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_imagenetC_loaders(
                    root_path='./data/tiny-imagenet-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True,
                    test_num = 3
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/tiny-imagenet-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'cifar100-c':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/cifar100-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True,
                    test_num = 1
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/cifar100-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'fmnist-c':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_fmnistC_loaders(
                    root_path='./data/fmnist-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_fmnistC_loaders(
                    root_path='./data/fmnist-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'airline':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/airline/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/airline/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'elec':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/elec/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/elec/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'powersupply':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/powersupply/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/powersupply/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    else: 

        train_iterators, val_iterators, test_iterators, client_types, feature_types =\
            get_loaders(
                type_=LOADER_TYPE[args_.experiment],
                root_path=root_path,
                batch_size=args_.bz,
                is_validation=args_.validation
            )
        client_types = [0] * len(train_iterators)


    print("===> Initializing clients..")
    clients_ = []
    for task_id, (train_iterator, val_iterator, test_iterator) in \
            enumerate(tqdm(zip(train_iterators, val_iterators, test_iterators), total=len(train_iterators))):

        if train_iterator is None or test_iterator is None:
            continue

        if args_.split:

            learners_ensemble =\
            get_split_learners_ensemble(
                n_learners=args_.n_learners,
                client_type=CLIENT_TYPE[args_.method],
                name=args_.experiment,
                device=args_.device,
                optimizer_name=args_.optimizer,
                scheduler_name=args_.lr_scheduler,
                initial_lr=args_.lr,
                input_dim=args_.input_dimension,
                output_dim=args_.output_dimension,
                n_rounds=args_.n_rounds,
                seed=args_.seed,
                mu=args_.mu,
                embedding_dim=args_.embedding_dimension,
                n_gmm=args_.n_gmm,
                domain_disc=args_.domain_disc,
                hard_cluster=args_.hard_cluster,
                binary=args_.binary
            )
        else:

            learners_ensemble =\
                get_learners_ensemble(
                    n_learners=args_.n_learners,
                    client_type=CLIENT_TYPE[args_.method],
                    name=args_.experiment,
                    device=args_.device,
                    optimizer_name=args_.optimizer,
                    scheduler_name=args_.lr_scheduler,
                    initial_lr=args_.lr,
                    input_dim=args_.input_dimension,
                    output_dim=args_.output_dimension,
                    n_rounds=args_.n_rounds,
                    seed=args_.seed,
                    mu=args_.mu,
                    n_gmm=args_.n_gmm,
                    embedding_dim=args_.embedding_dimension,
                    hard_cluster=args_.hard_cluster,
                    binary=args_.binary,
                    phi_model=args.phi_model
                )

        logs_path = os.path.join(logs_dir, "task_{}".format(task_id))
        os.makedirs(logs_path, exist_ok=True)
        logger = SummaryWriter(logs_path)

        if CLIENT_TYPE[args_.method] == "conceptEM_tune" and "train" in logs_dir:

            client = get_client(
                client_type=CLIENT_TYPE[args_.method],
                learners_ensemble=learners_ensemble,
                q=args_.q,
                train_iterator=train_iterator,
                val_iterator=val_iterator,
                test_iterator=test_iterator,
                logger=logger,
                local_steps=args_.local_steps,
                tune_locally=True,
                data_type = client_types[task_id],
                feature_type = feature_types[task_id],
                class_number = class_number
            )
        else:
            
            client = get_client(
                client_type=CLIENT_TYPE[args_.method],
                learners_ensemble=learners_ensemble,
                q=args_.q,
                train_iterator=train_iterator,
                val_iterator=val_iterator,
                test_iterator=test_iterator,
                logger=logger,
                local_steps=args_.local_steps,
                tune_locally=args_.locally_tune_clients,
                data_type = client_types[task_id],
                feature_type = feature_types[task_id],
                class_number = class_number
            )

        clients_.append(client) 

    return clients_

def get_data_iterator(args_, root_path, logs_dir):
    """
    initialize clients from data folders

    :param args_:
    :param root_path: path to directory containing data folders
    :param logs_dir: path to logs root
    :return: List[Client]
    """
    print("===> Building data iterators..")
    class_number = CLASS_NUMBER[LOADER_TYPE[args_.experiment]]

    if LOADER_TYPE[args_.experiment] == 'cifar10-c':
        if 'test' in root_path: 
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders_for_concept_drift(
                    root_path='./data/cifar10-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True,
                    test_num = 3
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders_for_concept_drift(
                    root_path='./data/cifar10-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation 
                )
    elif LOADER_TYPE[args_.experiment] == 'tiny-imagenet-c':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_imagenetC_loaders(
                    root_path='./data/tiny-imagenet-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True,
                    test_num = 3
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/tiny-imagenet-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'cifar100-c':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/cifar100-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True,
                    test_num = 1
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/cifar100-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'fmnist-c':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_fmnistC_loaders(
                    root_path='./data/fmnist-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_fmnistC_loaders(
                    root_path='./data/fmnist-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'airline':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/airline/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/airline/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'elec':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/elec/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/elec/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'powersupply':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/powersupply/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/powersupply/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    else: 
       
        train_iterators, val_iterators, test_iterators, client_types, feature_types =\
            get_loaders(
                type_=LOADER_TYPE[args_.experiment],
                root_path=root_path,
                batch_size=args_.bz,
                is_validation=args_.validation
            )
        client_types = [0] * len(train_iterators)



    return train_iterators, val_iterators, test_iterators, client_types, feature_types

def get_data_iterator_for_store_data(args_,data_indexes,root_path, logs_dir):
    """

    initialize clients from data folders

    :param args_:
    :param root_path: path to directory containing data folders
    :param logs_dir: path to logs root
    :return: List[Client]
    """
    print("===> Building data iterators..")
    class_number = CLASS_NUMBER[LOADER_TYPE[args_.experiment]]

    if LOADER_TYPE[args_.experiment] == 'cifar10-c':
        if 'test' in root_path: 
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders_for_store_history(
                    root_path='./data/cifar10-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    data_indexes = data_indexes,
                    test = True,
                    test_num = 3
                )
        else: 
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders_for_store_history(
                    root_path='./data/cifar10-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation, 
                    data_indexes=data_indexes
                )
    elif LOADER_TYPE[args_.experiment] == 'tiny-imagenet-c':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_imagenetC_loaders(
                    root_path='./data/tiny-imagenet-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True,
                    test_num = 3
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/tiny-imagenet-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'cifar100-c':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/cifar100-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True,
                    test_num = 1
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/cifar100-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'fmnist-c':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_fmnistC_loaders(
                    root_path='./data/fmnist-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_fmnistC_loaders(
                    root_path='./data/fmnist-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'airline':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/airline/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/airline/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'elec':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/elec/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/elec/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'powersupply':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/powersupply/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/powersupply/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    else: 
       
        train_iterators, val_iterators, test_iterators, client_types, feature_types =\
            get_loaders(
                type_=LOADER_TYPE[args_.experiment],
                root_path=root_path,
                batch_size=args_.bz,
                is_validation=args_.validation
            )
        client_types = [0] * len(train_iterators)



    return train_iterators, val_iterators, test_iterators, client_types, feature_types

def get_data_iterator_for_store_data_rotate_images(args_,data_indexes,root_path,rotate_degrees, logs_dir):
    """

    initialize clients from data folders

    :param args_:
    :param root_path: path to directory containing data folders
    :param logs_dir: path to logs root
    :return: List[Client]
    """
    print("===> Building data iterators..")
    class_number = CLASS_NUMBER[LOADER_TYPE[args_.experiment]]

    if LOADER_TYPE[args_.experiment] == 'cifar10-c':
        if 'test' in root_path: 
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders_for_store_history_rotate_images(
                    root_path='./data/cifar10-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    data_indexes = data_indexes,
                    rotate_degrees=rotate_degrees,
                    test = True,
                    test_num = 3
                )
        else: 
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders_for_store_history_rotate_images(
                    root_path='./data/cifar10-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    data_indexes=data_indexes,
                    rotate_degrees=rotate_degrees
                )
    elif LOADER_TYPE[args_.experiment] == 'tiny-imagenet-c':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_imagenetC_loaders(
                    root_path='./data/tiny-imagenet-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True,
                    test_num = 3
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/tiny-imagenet-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'cifar100-c':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/cifar100-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True,
                    test_num = 1
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/cifar100-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'fmnist-c':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_fmnistC_loaders(
                    root_path='./data/fmnist-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_fmnistC_loaders(
                    root_path='./data/fmnist-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'airline':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/airline/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/airline/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'elec':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/elec/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/elec/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'powersupply':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/powersupply/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/powersupply/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    else: 

        train_iterators, val_iterators, test_iterators, client_types, feature_types =\
            get_loaders(
                type_=LOADER_TYPE[args_.experiment],
                root_path=root_path,
                batch_size=args_.bz,
                is_validation=args_.validation
            )
        client_types = [0] * len(train_iterators)



    return train_iterators, val_iterators, test_iterators, client_types, feature_types



def init_clients_for_store_history(args_, last_data_indexes, current_data_indexes,root_path, logs_dir):
    """
    initialize clients from data folders

    :param args_:
    :param root_path: path to directory containing data folders
    :param logs_dir: path to logs root
    :return: List[Client]
    """
    print("===> Building data iterators..")


    class_number = CLASS_NUMBER[LOADER_TYPE[args_.experiment]]

    if LOADER_TYPE[args_.experiment] == 'cifar10-c':
        if 'test' in root_path: 
       
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders_for_store_history(
                    root_path='./data/cifar10-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True,
                    test_num = 3
                )


            last_train_iterators = train_iterators
            last_val_iterators = val_iterators
            last_test_iterators = test_iterators

        else: 

            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders_for_store_history(
                    root_path='./data/cifar10-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation, 
                    data_indexes = current_data_indexes
                )
            last_train_iterators, last_val_iterators, last_test_iterators, last_client_types, last_feature_types =\
                get_cifar10C_loaders_for_store_history(
                    root_path='./data/cifar10-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation, 
                    data_indexes = last_data_indexes
                )
    elif LOADER_TYPE[args_.experiment] == 'tiny-imagenet-c':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_imagenetC_loaders(
                    root_path='./data/tiny-imagenet-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True,
                    test_num = 3
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/tiny-imagenet-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'cifar100-c':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/cifar100-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True,
                    test_num = 1
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/cifar100-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'fmnist-c':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_fmnistC_loaders(
                    root_path='./data/fmnist-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_fmnistC_loaders(
                    root_path='./data/fmnist-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'airline':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/airline/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/airline/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'elec':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/elec/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/elec/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'powersupply':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/powersupply/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/powersupply/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    else: 
        
        train_iterators, val_iterators, test_iterators, client_types, feature_types =\
            get_loaders(
                type_=LOADER_TYPE[args_.experiment],
                root_path=root_path,
                batch_size=args_.bz,
                is_validation=args_.validation
            )
        client_types = [0] * len(train_iterators)


    print("===> Initializing clients..")
    clients_ = []
    
    for task_id, (train_iterator, val_iterator, test_iterator,last_train_iterator, last_val_iterator, last_test_iterator) in \
            enumerate(tqdm(zip(train_iterators, val_iterators, test_iterators,last_train_iterators, last_val_iterators, last_test_iterators), total=len(train_iterators))):

        if train_iterator is None or test_iterator is None:
            continue

        if args_.split:

            learners_ensemble =\
            get_split_learners_ensemble(
                n_learners=args_.n_learners,
                client_type=CLIENT_TYPE[args_.method],
                name=args_.experiment,
                device=args_.device,
                optimizer_name=args_.optimizer,
                scheduler_name=args_.lr_scheduler,
                initial_lr=args_.lr,
                input_dim=args_.input_dimension,
                output_dim=args_.output_dimension,
                n_rounds=args_.n_rounds,
                seed=args_.seed,
                mu=args_.mu,
                embedding_dim=args_.embedding_dimension,
                n_gmm=args_.n_gmm,
                domain_disc=args_.domain_disc,
                hard_cluster=args_.hard_cluster,
                binary=args_.binary
            )
        else:

            learners_ensemble =\
                get_learners_ensemble(
                    n_learners=args_.n_learners,
                    client_type=CLIENT_TYPE[args_.method],
                    name=args_.experiment,
                    device=args_.device,
                    optimizer_name=args_.optimizer,
                    scheduler_name=args_.lr_scheduler,
                    initial_lr=args_.lr,
                    input_dim=args_.input_dimension,
                    output_dim=args_.output_dimension,
                    n_rounds=args_.n_rounds,
                    seed=args_.seed,
                    mu=args_.mu,
                    n_gmm=args_.n_gmm,
                    embedding_dim=args_.embedding_dimension,
                    hard_cluster=args_.hard_cluster,
                    binary=args_.binary,
                    phi_model=args.phi_model
                )

        logs_path = os.path.join(logs_dir, "task_{}".format(task_id))
        os.makedirs(logs_path, exist_ok=True)
        logger = SummaryWriter(logs_path)

        if CLIENT_TYPE[args_.method] == "conceptEM_tune" and "train" in logs_dir:
          
            client = get_client(
                client_type=CLIENT_TYPE[args_.method],
                learners_ensemble=learners_ensemble,
                q=args_.q,
                train_iterator=train_iterator,
                val_iterator=val_iterator,
                test_iterator=test_iterator,
                logger=logger,
                local_steps=args_.local_steps,
                tune_locally=True,
                data_type = client_types[task_id],
                feature_type = feature_types[task_id],
                class_number = class_number
            )
        else:

            client = get_client_for_store_history(
                client_type=CLIENT_TYPE[args_.method],
                learners_ensemble=learners_ensemble,
                q=args_.q,
                train_iterator=train_iterator, 
                val_iterator=val_iterator,
                test_iterator=test_iterator,
                last_train_iterator=last_train_iterator, 
                last_val_iterator=last_val_iterator,
                last_test_iterator=last_test_iterator,
                logger=logger,
                local_steps=args_.local_steps,
                tune_locally=args_.locally_tune_clients,
                data_type = client_types[task_id],
                feature_type = feature_types[task_id],
                class_number = class_number
            )

        clients_.append(client) 

    return clients_


def rot_120deg_init_50_client_store_history_simple2_iid_auto_clst_num\
                (args_, last_data_indexes, current_data_indexes,rotate_degrees,root_path,logs_dir,data_root_path,cluster_num):
    """

    initialize clients from data folders

    :param args_:
    :param root_path: path to directory containing data folders
    :param logs_dir: path to logs root
    :return: List[Client]
    """
    print("===> Building data iterators..")


    class_number = CLASS_NUMBER[LOADER_TYPE[args_.experiment]]

    if LOADER_TYPE[args_.experiment] == 'cifar10-c':
        if 'test' in root_path: 
            
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders_for_store_history_rotate_images(
                    root_path=data_root_path,
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True,
                    test_num = cluster_num,
                    rotate_degrees=rotate_degrees
                )

            
            last_train_iterators = train_iterators
            last_val_iterators = val_iterators
            last_test_iterators = test_iterators
            last_client_types = client_types
            last_feature_types = feature_types

        else: 

            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders_for_store_history_rotate_images(
                    root_path=data_root_path,
                    batch_size=args_.bz,
                    is_validation=args_.validation, 
                    data_indexes = current_data_indexes,
                    rotate_degrees = rotate_degrees
                )
            last_train_iterators, last_val_iterators, last_test_iterators, last_client_types, last_feature_types =\
                get_cifar10C_loaders_for_store_history_rotate_images(
                    root_path=data_root_path,
                    batch_size=args_.bz,
                    is_validation=args_.validation, 
                    data_indexes = last_data_indexes,
                    rotate_degrees= rotate_degrees-120
                )
    elif LOADER_TYPE[args_.experiment] == 'tiny-imagenet-c':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_imagenetC_loaders(
                    root_path='./data/tiny-imagenet-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True,
                    test_num = 3
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/tiny-imagenet-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'cifar100-c':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/cifar100-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True,
                    test_num = 1
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/cifar100-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'fmnist-c':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_fmnistC_loaders(
                    root_path='./data/fmnist-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_fmnistC_loaders(
                    root_path='./data/fmnist-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'airline':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/airline/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/airline/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'elec':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/elec/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/elec/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'powersupply':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/powersupply/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/powersupply/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    else: 
        
        train_iterators, val_iterators, test_iterators, client_types, feature_types =\
            get_loaders(
                type_=LOADER_TYPE[args_.experiment],
                root_path=root_path,
                batch_size=args_.bz,
                is_validation=args_.validation
            )
        client_types = [0] * len(train_iterators)


    print("===> Initializing clients..")
    clients_ = []

    for task_id, (train_iterator, val_iterator, test_iterator,last_train_iterator, last_val_iterator, last_test_iterator) in \
            enumerate(tqdm(zip(train_iterators, val_iterators, test_iterators,last_train_iterators, last_val_iterators, last_test_iterators), total=len(train_iterators))):

        if train_iterator is None or test_iterator is None:
            continue

        if args_.split:

            learners_ensemble =\
            get_split_learners_ensemble(
                n_learners=args_.n_learners,
                client_type=CLIENT_TYPE[args_.method],
                name=args_.experiment,
                device=args_.device,
                optimizer_name=args_.optimizer,
                scheduler_name=args_.lr_scheduler,
                initial_lr=args_.lr,
                input_dim=args_.input_dimension,
                output_dim=args_.output_dimension,
                n_rounds=args_.n_rounds,
                seed=args_.seed,
                mu=args_.mu,
                embedding_dim=args_.embedding_dimension,
                n_gmm=args_.n_gmm,
                domain_disc=args_.domain_disc,
                hard_cluster=args_.hard_cluster,
                binary=args_.binary
            )
        else:
 
            learners_ensemble =\
                get_learners_ensemble(
                    n_learners=cluster_num,
                    client_type=CLIENT_TYPE[args_.method],
                    name=args_.experiment,
                    device=args_.device,
                    optimizer_name=args_.optimizer,
                    scheduler_name=args_.lr_scheduler,
                    initial_lr=args_.lr,
                    input_dim=args_.input_dimension,
                    output_dim=args_.output_dimension,
                    n_rounds=args_.n_rounds,
                    seed=args_.seed,
                    mu=args_.mu,
                    n_gmm=args_.n_gmm,
                    embedding_dim=args_.embedding_dimension,
                    hard_cluster=args_.hard_cluster,
                    binary=args_.binary,
                    phi_model=args.phi_model
                )

        logs_path = os.path.join(logs_dir, "task_{}".format(task_id))
        os.makedirs(logs_path, exist_ok=True)
        logger = SummaryWriter(logs_path)

        if CLIENT_TYPE[args_.method] == "conceptEM_tune" and "train" in logs_dir:
           
            client = get_client(
                client_type=CLIENT_TYPE[args_.method],
                learners_ensemble=learners_ensemble,
                q=args_.q,
                train_iterator=train_iterator,
                val_iterator=val_iterator,
                test_iterator=test_iterator,
                logger=logger,
                local_steps=args_.local_steps,
                tune_locally=True,
                data_type = client_types[task_id],
                feature_type = feature_types[task_id],
                class_number = class_number
            )
        else:

            client = get_client_for_store_history_version_2(
                client_type=CLIENT_TYPE[args_.method],
                learners_ensemble=learners_ensemble,
                q=args_.q,
                train_iterator=train_iterator, 
                val_iterator=val_iterator,
                test_iterator=test_iterator,
                last_train_iterator=last_train_iterator, 
                last_val_iterator=last_val_iterator,
                last_test_iterator=last_test_iterator,
                logger=logger,
                local_steps=args_.local_steps,
                tune_locally=args_.locally_tune_clients,
                data_type = client_types[task_id],
                feature_type = feature_types[task_id],
                last_data_type = last_client_types[task_id],
                last_feature_type= last_feature_types[task_id],
                class_number = class_number
            )

        clients_.append(client) 

    return clients_


def update_client_data_loaders(clients, last_data_indexes, current_data_indexes, rotate_degrees, data_root_path, args_, is_test=False, test_num=4):
    """
    Update existing clients' data loaders without recreating client objects.
    This prevents memory issues and maintains client state.
    
    :param clients: List of existing Client objects to update
    :param last_data_indexes: Previous time slot data indices
    :param current_data_indexes: Current time slot data indices
    :param rotate_degrees: Rotation angle for feature shift
    :param data_root_path: Root path to data
    :param args_: Arguments object
    :param is_test: Whether these are test clients
    :param test_num: Test number for test clients
    """
    print(f"===> Updating client data loaders (rotation={rotate_degrees}°, test={is_test})..")
    
    if LOADER_TYPE[args_.experiment] == 'cifar10-c':
        if is_test:
            train_iterators, val_iterators, test_iterators, client_types, feature_types = \
                get_cifar10C_loaders_for_store_history_rotate_images(
                    root_path=data_root_path,
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test=True,
                    test_num=test_num,
                    rotate_degrees=rotate_degrees
                )
            last_train_iterators = train_iterators
            last_val_iterators = val_iterators
            last_test_iterators = test_iterators
            last_client_types = client_types
            last_feature_types = feature_types
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types = \
                get_cifar10C_loaders_for_store_history_rotate_images(
                    root_path=data_root_path,
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    data_indexes=current_data_indexes,
                    rotate_degrees=rotate_degrees
                )
            last_train_iterators, last_val_iterators, last_test_iterators, last_client_types, last_feature_types = \
                get_cifar10C_loaders_for_store_history_rotate_images(
                    root_path=data_root_path,
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    data_indexes=last_data_indexes,
                    rotate_degrees=rotate_degrees - 120
                )
    
    # Update each client's data iterators
    for idx, client in enumerate(clients):
        if idx < len(train_iterators):
            client.train_iterator = train_iterators[idx]
            client.val_iterator = val_iterators[idx]
            client.test_iterator = test_iterators[idx]
            client.train_loader = iter(train_iterators[idx])
            
            client.n_train_samples = len(train_iterators[idx].dataset)
            client.n_test_samples = len(test_iterators[idx].dataset)
            
            # Update data type and feature type
            client.data_type = client_types[idx]
            client.feature_types = feature_types[idx]
            
            # Update stored iterators for history
            client.current_train_iterator = train_iterators[idx]
            client.current_val_iterator = val_iterators[idx]
            client.current_test_iterator = test_iterators[idx]
            client.last_train_iterator = last_train_iterators[idx]
            client.last_val_iterator = last_val_iterators[idx]
            client.last_test_iterator = last_test_iterators[idx]
            client.last_data_type = last_client_types[idx]
            client.last_feature_type = last_feature_types[idx]
    
    print(f"===> Updated {len(clients)} clients with new data loaders")


def rot_120deg_init_60_client_store_history_simple2_iid_auto_clst_num_balance_concept\
                (args_, last_data_indexes, current_data_indexes,rotate_degrees,root_path,logs_dir,data_root_path,cluster_num,test_num):
    """

    initialize clients from data folders

    :param args_:
    :param root_path: path to directory containing data folders
    :param logs_dir: path to logs root
    :return: List[Client]
    """
    print("===> Building data iterators..")


    class_number = CLASS_NUMBER[LOADER_TYPE[args_.experiment]]

    if LOADER_TYPE[args_.experiment] == 'cifar10-c':
        if 'test' in root_path: 
            
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders_for_store_history_rotate_images(
                    root_path=data_root_path,
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True,
                    test_num = test_num,
                    rotate_degrees=rotate_degrees
                )


            last_train_iterators = train_iterators
            last_val_iterators = val_iterators
            last_test_iterators = test_iterators
            last_client_types = client_types
            last_feature_types = feature_types

        else: 



            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders_for_store_history_rotate_images(
                    root_path=data_root_path,
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    data_indexes = current_data_indexes,
                    rotate_degrees = rotate_degrees
                )
            last_train_iterators, last_val_iterators, last_test_iterators, last_client_types, last_feature_types =\
                get_cifar10C_loaders_for_store_history_rotate_images(
                    root_path=data_root_path,
                    batch_size=args_.bz,
                    is_validation=args_.validation, 
                    data_indexes = last_data_indexes,
                    rotate_degrees= rotate_degrees-120
                )
    elif LOADER_TYPE[args_.experiment] == 'tiny-imagenet-c':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_imagenetC_loaders(
                    root_path='./data/tiny-imagenet-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True,
                    test_num = 3
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/tiny-imagenet-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'cifar100-c':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/cifar100-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True,
                    test_num = 1
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/cifar100-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'fmnist-c':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_fmnistC_loaders(
                    root_path='./data/fmnist-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_fmnistC_loaders(
                    root_path='./data/fmnist-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'airline':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/airline/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/airline/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'elec':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/elec/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/elec/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'powersupply':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/powersupply/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/powersupply/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    else: 
        
        train_iterators, val_iterators, test_iterators, client_types, feature_types =\
            get_loaders(
                type_=LOADER_TYPE[args_.experiment],
                root_path=root_path,
                batch_size=args_.bz,
                is_validation=args_.validation
            )
        client_types = [0] * len(train_iterators)


    print("===> Initializing clients..")
    clients_ = []

    for task_id, (train_iterator, val_iterator, test_iterator,last_train_iterator, last_val_iterator, last_test_iterator) in \
            enumerate(tqdm(zip(train_iterators, val_iterators, test_iterators,last_train_iterators, last_val_iterators, last_test_iterators), total=len(train_iterators))):

        if train_iterator is None or test_iterator is None:
            continue

        if args_.split:

            learners_ensemble =\
            get_split_learners_ensemble(
                n_learners=args_.n_learners,
                client_type=CLIENT_TYPE[args_.method],
                name=args_.experiment,
                device=args_.device,
                optimizer_name=args_.optimizer,
                scheduler_name=args_.lr_scheduler,
                initial_lr=args_.lr,
                input_dim=args_.input_dimension,
                output_dim=args_.output_dimension,
                n_rounds=args_.n_rounds,
                seed=args_.seed,
                mu=args_.mu,
                embedding_dim=args_.embedding_dimension,
                n_gmm=args_.n_gmm,
                domain_disc=args_.domain_disc,
                hard_cluster=args_.hard_cluster,
                binary=args_.binary
            )
        else:

            learners_ensemble =\
                get_learners_ensemble(
                    n_learners=cluster_num,
                    client_type=CLIENT_TYPE[args_.method],
                    name=args_.experiment,
                    device=args_.device,
                    optimizer_name=args_.optimizer,
                    scheduler_name=args_.lr_scheduler,
                    initial_lr=args_.lr,
                    input_dim=args_.input_dimension,
                    output_dim=args_.output_dimension,
                    n_rounds=args_.n_rounds,
                    seed=args_.seed,
                    mu=args_.mu,
                    n_gmm=args_.n_gmm,
                    embedding_dim=args_.embedding_dimension,
                    hard_cluster=args_.hard_cluster,
                    binary=args_.binary,
                    phi_model=args.phi_model
                )

        logs_path = os.path.join(logs_dir, "task_{}".format(task_id))
        os.makedirs(logs_path, exist_ok=True)
        logger = SummaryWriter(logs_path)

        if CLIENT_TYPE[args_.method] == "conceptEM_tune" and "train" in logs_dir:
           
            client = get_client(
                client_type=CLIENT_TYPE[args_.method],
                learners_ensemble=learners_ensemble,
                q=args_.q,
                train_iterator=train_iterator,
                val_iterator=val_iterator,
                test_iterator=test_iterator,
                logger=logger,
                local_steps=args_.local_steps,
                tune_locally=True,
                data_type = client_types[task_id],
                feature_type = feature_types[task_id],
                class_number = class_number
            )
        else:
  
            client = get_client_for_store_history_version_2(
                client_type=CLIENT_TYPE[args_.method],
                learners_ensemble=learners_ensemble,
                q=args_.q,
                train_iterator=train_iterator, 
                val_iterator=val_iterator,
                test_iterator=test_iterator,
                last_train_iterator=last_train_iterator,
                last_val_iterator=last_val_iterator,
                last_test_iterator=last_test_iterator,
                logger=logger,
                local_steps=args_.local_steps,
                tune_locally=args_.locally_tune_clients,
                data_type = client_types[task_id],
                feature_type = feature_types[task_id],
                last_data_type = last_client_types[task_id],
                last_feature_type= last_feature_types[task_id],
                class_number = class_number
            )

        clients_.append(client) 

    return clients_


def check_whether_clients_concept_shifts(clients):
    """

    Parameters
    ----------
    clients

    Returns
    -------
    shift_set,clean_set
    """
    LID_list = []
    for client in clients:
        output = client.get_output() 
 
        LID = client.get_LID(output, output)
        LID_list.append(LID)



    gmm_LID_accumulative = GaussianMixture(n_components=2, random_state=args.seed).fit(
        np.array(LID_list).reshape(-1,1))  
    labels_LID_accumulative = gmm_LID_accumulative.predict(np.array(LID_list).reshape(-1, 1))

    clean_label = np.argsort(gmm_LID_accumulative.means_[:, 0])[0]


    shift_set = np.where(labels_LID_accumulative != clean_label)[0]

    clean_set = np.where(labels_LID_accumulative == clean_label)[0]
    return shift_set,clean_set

def check_whether_clients_concept_shifts_accuracy_based(clients):
    """

    Parameters
    ----------
    clients

    Returns
    -------
    shift_set,clean_set
    """
    shift_set = []
    clean_set = []
    for i, client in enumerate(clients):
        last_data_losses, last_data_accuracies = client.get_last_data_all_models_loss_and_accuracy()
        current_data_losses, current_data_accuracies = client.get_current_data_all_models_loss_and_accuracy()

        last_data_minimal_loss_model_index = np.argmin(last_data_losses)
        current_data_minimal_loss_model_index = np.argmin(current_data_losses)
        
        last_data_maximal_accuracy_model_index = np.argmax(last_data_accuracies)
        current_data_maximal_accuracy_model_index = np.argmax(current_data_accuracies)

 

        if last_data_minimal_loss_model_index == current_data_minimal_loss_model_index \
                and last_data_maximal_accuracy_model_index == current_data_maximal_accuracy_model_index:
           
            clean_set.append(i)
        else:
            
            shift_set.append(i)
    return shift_set, clean_set

def check_whether_clients_concept_shifts_cluster_center_based(
        clients,
        cluster_centers,
        diagnosis_mode='binary',
        thresholds=None):
    """

    Parameters
    ----------
    clients

    Returns
    -------
    shift_set,clean_set,drift_types,drift_metrics
    """
    shift_set = []
    clean_set = []
    drift_types = {}
    drift_metrics = {}
    for i, client in enumerate(clients):
      
        last_prototype = client.get_last_val_iterator_output_prototype()
        current_prototype = client.get_current_val_iterator_output_prototype()

  
        last_prototype = np.array(last_prototype)
        current_prototype = np.array(current_prototype)


        if last_prototype.ndim >1:
          
            last_prototype = last_prototype.flatten()

        if current_prototype.ndim >1:
         
            current_prototype = current_prototype.flatten()
      
        last_predicted_cluster = predict_cluster(last_prototype, cluster_centers)
        current_predicted_cluster = predict_cluster(current_prototype, cluster_centers)

        print("client:",i)
        print("last_predicted_cluster",last_predicted_cluster)
        print("current_predicted_cluster",current_predicted_cluster)
        cluster_changed = last_predicted_cluster != current_predicted_cluster

        metrics = {
            "prototype_shift": prototype_shift(last_prototype, current_prototype),
            "cluster_changed": float(cluster_changed),
            "label_shift": 0.0,
            "feature_shift": 0.0,
            "accuracy_drop": 0.0,
            "last_acc_shared": 0.0,
            "current_acc_shared": 0.0,
        }

        if diagnosis_mode == 'multiclass':
            last_dist = client.get_last_label_distribution()
            current_dist = client.get_current_label_distribution()
            metrics["label_shift"] = label_distribution_shift(last_dist, current_dist, metric="l1")

            last_logits, last_labels = client.get_last_features_and_labels()
            current_logits, current_labels = client.get_current_features_and_labels()

            if last_logits.size and current_logits.size:
                metrics["feature_shift"] = float(
                    np.linalg.norm(np.mean(last_logits, axis=0) - np.mean(current_logits, axis=0))
                )

            accuracy_drop, last_acc, current_acc = shared_label_accuracy_drop(
                last_logits, last_labels, current_logits, current_labels
            )
            metrics["accuracy_drop"] = accuracy_drop
            metrics["last_acc_shared"] = last_acc
            metrics["current_acc_shared"] = current_acc

            drift_type = diagnose_drift_type(metrics, thresholds=thresholds, cluster_changed=cluster_changed)
        else:
            drift_type = 'real' if cluster_changed else 'none'

        if drift_type == 'none':
            clean_set.append(i)
        else:
            shift_set.append(i)

        drift_types[i] = drift_type
        drift_metrics[i] = metrics

    return shift_set, clean_set, drift_types, drift_metrics


def get_real_shift_clients_set(clients):
    shift_label_list = []
    for client in clients:
        shift_label = client.get_real_shift_label()
        shift_label_list.append(shift_label)


    shift_labels = np.array(shift_label_list)


    real_clean_set = np.where(shift_labels == 0)[0]
    real_shift_set = np.where(shift_labels == 1)[0]

    return  real_shift_set.tolist(),real_clean_set.tolist()

def get_shift_clients_prediction_accuracy(shift_set, clean_set, real_shift_set, real_clean_set):


    correct_predictions = (
        len(set(shift_set) & set(real_shift_set)) +  
        len(set(clean_set) & set(real_clean_set))    
    )


    total_clients = len(shift_set) + len(clean_set)


    prediction_accuracy = correct_predictions / total_clients if total_clients > 0 else 0.0

    return prediction_accuracy

def get_shift_clients_precision(shift_set, real_shift_set):


    true_positive = (
        len(set(shift_set) & set(real_shift_set))   
    )


    true_positive_plus_false_positive = len(real_shift_set)

    precision = true_positive / true_positive_plus_false_positive if true_positive_plus_false_positive > 0 else 0.0
    return precision

def get_shift_clients_recall(shift_set, real_shift_set):

    true_positive = (
        len(set(shift_set) & set(real_shift_set))   
    )


    true_positive_plus_false_nagative = len(shift_set)


    recall = true_positive / true_positive_plus_false_nagative if true_positive_plus_false_nagative > 0 else 0.0
    return recall

def set_clients_concept_shift_flag(clients,shift_set):
    for i,client in enumerate(clients):
        if i in shift_set:
            client.set_concept_shift_flag(1)
        else:
            client.set_concept_shift_flag(0)

def update_clients_train_iterator_and_other_attribute(clients):
    for client in clients:
        if client.concept_shift_flag==0:
          
            merged_train_iterator, merged_val_iterator, merged_test_iterator = \
                client.get_merge_last_and_current_train_iterators()
            client.update_train_iterator_and_other_attributes(merged_train_iterator, merged_val_iterator,
                                                              merged_test_iterator)
        else:
      
            client.update_train_iterator_and_other_attributes(client.current_train_iterator, client.current_val_iterator,
                                                              client.current_test_iterator)


def rotate_120degree_update_clients_train_iterator_and_other_attribute(clients,rotate_degrees):
    """

    Parameters
    ----------
    clients
    rotate_degrees

    Returns
    -------

    """
    for client in clients:
        if client.concept_shift_flag == 0:
           
            merged_train_iterator, merged_val_iterator, merged_test_iterator = \
                client.rotate_120_get_merge_last_and_current_train_iterators(rotate_degrees=rotate_degrees)
            client.update_train_iterator_and_other_attributes(merged_train_iterator, merged_val_iterator,
                                                              merged_test_iterator)
        else:   
    
            client.update_train_iterator_and_other_attributes(client.current_train_iterator, client.current_val_iterator,
                                                              client.current_test_iterator)

def get_clients_output_prototype_version1(clients, dp_manager=None, max_norm=10.0):

    clients_output_prototype = []
    for client in clients:
        output_prototype = client.get_current_val_iterator_output_prototype()
        
        # Apply differential privacy if enabled
        if dp_manager is not None:
            output_prototype = dp_manager.privatize_prototype(output_prototype, max_norm=max_norm)
        
        clients_output_prototype.append(output_prototype)

    return clients_output_prototype

def determine_cluster_number(clients_output_prototype):

    X = np.array(clients_output_prototype)

    if X.ndim == 3:
        X = X.reshape(X.shape[0], -1)  
    silhouette_scores = []
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(X)
        score = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(score)

    cluster_number = silhouette_scores.index(max(silhouette_scores))
    cluster_number += 2 
    return cluster_number,silhouette_scores


def determine_cluster_number_return_centers(clients_output_prototype):

    X = np.array(clients_output_prototype)


    if X.ndim == 3:
        X = X.reshape(X.shape[0], -1) 

    silhouette_scores = []
    best_kmeans = None  

    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(X)
        score = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(score)


        if best_kmeans is None or score > max(silhouette_scores[:-1]):
            best_kmeans = kmeans

    cluster_number = silhouette_scores.index(max(silhouette_scores)) + 2  
    cluster_centers = best_kmeans.cluster_centers_  

    return cluster_number, silhouette_scores, cluster_centers

def predict_cluster(client_prototype, cluster_centers):




    distances = np.linalg.norm(cluster_centers - client_prototype, axis=1)


    cluster_index = np.argmin(distances)

    return cluster_index

def run_experiment(args_, exp_logger=None):
    torch.manual_seed(args_.seed)

    run_paths = None
    if exp_logger is not None:
        run_paths = {
            "logs": str(exp_logger.seed_logs_dir(args_.seed)),
            "plots": exp_logger.seed_plots_dir(args_.seed),
            "metrics": exp_logger.seed_metrics_dir(args_.seed),
        }

    objective_assignment = getattr(args_, "objective_assignment", None)
    
    # Initialize Differential Privacy Manager (if enabled)
    dp_manager = None
    if getattr(args_, "use_dp", False):
        dp_manager = DifferentialPrivacyManager(
            epsilon=getattr(args_, "dp_epsilon", 1.0),
            delta=getattr(args_, "dp_delta", 1e-5),
            mechanism=getattr(args_, "dp_mechanism", "gaussian")
        )
        print(f"==> DP enabled: ε={dp_manager.epsilon}, δ={dp_manager.delta}, mechanism={dp_manager.mechanism}")

    data_dir = get_data_dir('cifar10-c-60_client-simple2-iid-4concept-change-name-version2')
    data_root_path = './data/cifar10-c-60_client-simple2-iid-4concept-change-name-version2'

    if run_paths is not None:
        logs_dir = run_paths["logs"]
    elif "logs_dir" in args_:
        logs_dir = args_.logs_dir
    else:
        logs_dir = os.path.join("logs", args_to_string(args_))

    setattr(args_, "logs_dir", logs_dir)

    print("==> Clients initialization..")

    train_num = 60  

  
    numbers = list(range(30)) 

    numbers = numbers * 2

    random.shuffle(numbers)

    initial_data_indexes = numbers  
    current_data_indexes = copy.deepcopy(initial_data_indexes)


    cluster_num = 2

    clients = rot_120deg_init_60_client_store_history_simple2_iid_auto_clst_num_balance_concept(args_,
                                                                                    last_data_indexes=initial_data_indexes,
                                                                                    current_data_indexes=initial_data_indexes,
                                                                                    rotate_degrees=0,
                                                                                    root_path=os.path.join(data_dir,
                                                                                                           "train"),
                                                                                    logs_dir=os.path.join(logs_dir,
                                                                                                          "train"),
                                                                                    data_root_path=data_root_path,
                                                                                    cluster_num=cluster_num,
                                                                                    test_num=4
                                                                                                )


    print("==> Test Clients initialization..")

    test_clients_0degree = rot_120deg_init_60_client_store_history_simple2_iid_auto_clst_num_balance_concept(args_,
                                            last_data_indexes=initial_data_indexes,
                                             current_data_indexes=initial_data_indexes,
                                             root_path=os.path.join(data_dir, "test"),
                                             logs_dir=os.path.join(logs_dir, "test"),
                                             rotate_degrees=0,
                                             data_root_path=data_root_path,
                                             cluster_num=cluster_num,
                                             test_num=4
                                             )


    test_clients_120degree = rot_120deg_init_60_client_store_history_simple2_iid_auto_clst_num_balance_concept(args_, last_data_indexes=initial_data_indexes,
                                             current_data_indexes=initial_data_indexes,
                                             root_path=os.path.join(data_dir, "test"),
                                             logs_dir=os.path.join(logs_dir, "test"),
                                             rotate_degrees=120,
                                             data_root_path=data_root_path,
                                             cluster_num=cluster_num,
                                             test_num=4
                                             )

    test_clients_240degree = rot_120deg_init_60_client_store_history_simple2_iid_auto_clst_num_balance_concept(args_, last_data_indexes=initial_data_indexes,
                                             current_data_indexes=initial_data_indexes,
                                             root_path=os.path.join(data_dir, "test"),
                                             logs_dir=os.path.join(logs_dir, "test"),
                                             rotate_degrees=240,
                                             data_root_path=data_root_path,
                                             cluster_num=cluster_num,
                                             test_num=4
                                             )

    test_clients = test_clients_0degree+test_clients_120degree+test_clients_240degree

    # Initialize per-client test history (slot concepts) if requested
    if getattr(args_, 'eval_all_past_concepts', False):
        for c in clients:
            try:
                c._test_history = [copy.deepcopy(c.test_iterator)]
            except Exception:
                c._test_history = []
    
    # Assign objectives to clients (optional - only if you want static client goals)
    if objective_assignment:
        if objective_assignment == "random_50_50":
            # Random 50-50 split with reproducibility
            rng = np.random.RandomState(args_.seed)
            n = len(clients)
            objectives = ['G'] * (n // 2) + ['P'] * (n - n // 2)
            rng.shuffle(objectives)
            for client, obj in zip(clients, objectives):
                client.objective = obj
            # Same for test clients
            n_test = len(test_clients)
            test_objectives = ['G'] * (n_test // 2) + ['P'] * (n_test - n_test // 2)
            rng.shuffle(test_objectives)
            for client, obj in zip(test_clients, test_objectives):
                client.objective = obj
            print(f"==> Random objective assignment (seed={args_.seed}): {Counter(objectives)}")
        elif not objective_assignment.startswith("drift_based:"):
            assign_objectives(clients, objective_assignment)
            assign_objectives(test_clients, objective_assignment)

    logs_path = os.path.join(logs_dir, "train", "global")
    os.makedirs(logs_path, exist_ok=True)
    global_train_logger = SummaryWriter(logs_path)

    logs_path = os.path.join(logs_dir, "test", "global")
    os.makedirs(logs_path, exist_ok=True)
    global_test_logger = SummaryWriter(logs_path)

    # Build a fixed global test iterator (optional)
    global_test_iterator = None
    if getattr(args_, "global_eval", False):
        try:
            # Use the initial 0-degree test clients to construct a fixed global test dataset
            # Copy datasets so later time-slot updates do not affect this baseline
            from torch.utils.data import ConcatDataset
            from torch.utils.data import DataLoader as TorchDataLoader

            global_test_dataset = ConcatDataset([
                copy.deepcopy(c.test_iterator.dataset) for c in test_clients_0degree
            ])
            global_test_iterator = TorchDataLoader(
                global_test_dataset,
                batch_size=args_.bz,
                shuffle=False,
                drop_last=False,
            )
            print("==> Global test iterator initialized (fixed baseline, 0° rotation)")
        except Exception as e:
            print(f"[WARN] Failed to build global test iterator: {e}")
            global_test_iterator = None


    if args_.split:
 
        global_learners_ensemble = \
        get_split_learners_ensemble(
            n_learners=args_.n_learners,
            client_type=CLIENT_TYPE[args_.method],
            name=args_.experiment,
            device=args_.device,
            optimizer_name=args_.optimizer,
            scheduler_name=args_.lr_scheduler,
            initial_lr=args_.lr,
            input_dim=args_.input_dimension,
            output_dim=args_.output_dimension,
            n_rounds=args_.n_rounds,
            seed=args_.seed,
            mu=args_.mu,
            embedding_dim=args_.embedding_dimension,
            n_gmm=args_.n_gmm,
            domain_disc=args_.domain_disc,
            hard_cluster=args_.hard_cluster,
            binary=args_.binary
        )
    else:
        global_learners_ensemble = \
            get_learners_ensemble(
                n_learners=args_.n_learners,
                client_type=CLIENT_TYPE[args_.method],
                name=args_.experiment,
                device=args_.device,
                optimizer_name=args_.optimizer,
                scheduler_name=args_.lr_scheduler,
                initial_lr=args_.lr,
                input_dim=args_.input_dimension,
                output_dim=args_.output_dimension,
                n_rounds=args_.n_rounds,
                seed=args_.seed,
                mu=args_.mu,
                embedding_dim=args_.embedding_dimension,
                n_gmm=args_.n_gmm,
                hard_cluster=args_.hard_cluster,
                binary=args_.binary,
                phi_model=args.phi_model
            )


    if args_.decentralized:
        aggregator_type = 'decentralized'
    else:
        aggregator_type = AGGREGATOR_TYPE[args_.method]


    T = args_.T
    prediction_accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    # Across-time aggregates (store per time-slot means)
    time_local_all_means = []
    time_global_all_means = []
    time_objective_aware_means = []
    time_objective_aware_old_means = []
    time_objective_aware_global_means = []
    event_records = []
    logs_base_path = Path(logs_dir)
    logs_base_path.mkdir(parents=True, exist_ok=True)
    for t in range(T):
        print("==> time slot {} starts..".format(t))
        current_events = {}
        
        if t != 0:
            # UPDATE EXISTING CLIENTS instead of recreating them
            print("==> Updating client data for new time slot..")
            
            if t % 3 == 0:
                rotate_degrees = 0
                cluster_num = 4
                
                torch.manual_seed(torch.seed())
                last_data_indexes = copy.deepcopy(current_data_indexes)  
                current_data_indexes = torch.randperm(train_num)  
                torch.manual_seed(args_.seed)

            elif t % 3 == 1:
                rotate_degrees = 120

                if t == 1:
                    cluster_num = 3
                    torch.manual_seed(torch.seed())
                    last_data_indexes = copy.deepcopy(current_data_indexes)  
                    range1 = list(range(0, 15)) 
                    range2 = list(range(15, 30))  
                    range3 = list(range(30, 45)) 
                    
                    numbers = []
                    numbers.extend(range1)  
                    numbers.extend(range2)  
                    numbers.extend(range3)  
                    
                    numbers.extend(random.choices(range1, k=20 - len(range1)))  
                    numbers.extend(random.choices(range2, k=20 - len(range2)))  
                    numbers.extend(random.choices(range3, k=20 - len(range3)))  
                    
                    random.shuffle(numbers)
                    current_data_indexes = numbers  
                    torch.manual_seed(args_.seed)

                else:
                    cluster_num = 4
                    torch.manual_seed(torch.seed())
                    last_data_indexes = copy.deepcopy(current_data_indexes)  
                    current_data_indexes = torch.randperm(train_num)  
                    torch.manual_seed(args_.seed)

            elif t % 3 == 2:
                rotate_degrees = 240
                cluster_num = 4
                torch.manual_seed(torch.seed())
                last_data_indexes = copy.deepcopy(current_data_indexes)  
                current_data_indexes = torch.randperm(train_num)  
                torch.manual_seed(args_.seed)

            # Update data loaders for existing clients (NO RECREATION!)
            update_client_data_loaders(
                clients=clients,
                last_data_indexes=last_data_indexes,
                current_data_indexes=current_data_indexes,
                rotate_degrees=rotate_degrees,
                data_root_path=data_root_path,
                args_=args_,
                is_test=False
            )
            
            # Update test clients
            update_client_data_loaders(
                clients=test_clients_0degree,
                last_data_indexes=initial_data_indexes,
                current_data_indexes=initial_data_indexes,
                rotate_degrees=0,
                data_root_path=data_root_path,
                args_=args_,
                is_test=True,
                test_num=4
            )
            
            update_client_data_loaders(
                clients=test_clients_120degree,
                last_data_indexes=initial_data_indexes,
                current_data_indexes=initial_data_indexes,
                rotate_degrees=120,
                data_root_path=data_root_path,
                args_=args_,
                is_test=True,
                test_num=4
            )
            
            update_client_data_loaders(
                clients=test_clients_240degree,
                last_data_indexes=initial_data_indexes,
                current_data_indexes=initial_data_indexes,
                rotate_degrees=240,
                data_root_path=data_root_path,
                args_=args_,
                is_test=True,
                test_num=4
            )

            # Copy global model to clients (same as before)
            for client in clients:
                for learner_id, learner in enumerate(global_learners_ensemble): 
                    copy_model(client.learners_ensemble[learner_id].model, learner.model)

            print(f"==> Clients updated for time slot {t} (rotation={rotate_degrees}°)")

        else:
            # t == 0: First time slot - still need to initialize test clients
            print("==> Test Clients initialization..")
            test_clients_0degree = rot_120deg_init_60_client_store_history_simple2_iid_auto_clst_num_balance_concept(args_,
                                                                                                         last_data_indexes=initial_data_indexes,
                                                                                                         current_data_indexes=initial_data_indexes,
                                                                                                         root_path=os.path.join(data_dir,"test"),
                                                                                                         logs_dir=os.path.join(logs_dir,"test"),
                                                                                                         rotate_degrees=0,
                                                                                                         data_root_path=data_root_path,
                                                                                                         cluster_num=cluster_num,
                                                                                                         test_num = 4)

            test_clients_120degree = rot_120deg_init_60_client_store_history_simple2_iid_auto_clst_num_balance_concept(args_,
                                                                                                           last_data_indexes=initial_data_indexes,
                                                                                                           current_data_indexes=initial_data_indexes,
                                                                                                           root_path=os.path.join(data_dir,"test"),
                                                                                                           logs_dir=os.path.join(logs_dir,"test"),
                                                                                                           rotate_degrees=120,
                                                                                                           data_root_path=data_root_path,
                                                                                                           cluster_num=cluster_num,
                                                                                                           test_num = 4)

            test_clients_240degree = rot_120deg_init_60_client_store_history_simple2_iid_auto_clst_num_balance_concept(args_,
                                                                                                          last_data_indexes=initial_data_indexes,
                                                                                                          current_data_indexes=initial_data_indexes,
                                                                                                          root_path=os.path.join(data_dir,"test"),
                                                                                                          logs_dir=os.path.join(logs_dir,"test"),
                                                                                                          rotate_degrees=240,
                                                                                                          data_root_path=data_root_path,
                                                                                                          cluster_num=cluster_num,
                                                                                                          test_num = 4)

            test_clients = test_clients_0degree + test_clients_120degree + test_clients_240degree
            
            # Assign objectives to test clients (same logic as training clients)
            if objective_assignment:
                if objective_assignment == "random_50_50":
                    # Random 50-50 split with reproducibility (same seed = same assignment)
                    rng = np.random.RandomState(args_.seed)
                    n = len(test_clients)
                    test_objectives = ['G'] * (n // 2) + ['P'] * (n - n // 2)
                    rng.shuffle(test_objectives)
                    for client, obj in zip(test_clients, test_objectives):
                        client.objective = obj
                elif not objective_assignment.startswith("drift_based:"):
                    assign_objectives(test_clients, objective_assignment)


            for client in test_clients:
                for learner_id, learner in enumerate(global_learners_ensemble):  
                    copy_model(client.learners_ensemble[learner_id].model, learner.model)

            
            if args_.split:

                new_global_learners_ensemble = \
                    get_split_learners_ensemble(
                        n_learners=cluster_num,
                        client_type=CLIENT_TYPE[args_.method],
                        name=args_.experiment,
                        device=args_.device,
                        optimizer_name=args_.optimizer,
                        scheduler_name=args_.lr_scheduler,
                        initial_lr=args_.lr,
                        input_dim=args_.input_dimension,
                        output_dim=args_.output_dimension,
                        n_rounds=args_.n_rounds,
                        seed=args_.seed,
                        mu=args_.mu,
                        embedding_dim=args_.embedding_dimension,
                        n_gmm=args_.n_gmm,
                        domain_disc=args_.domain_disc,
                        hard_cluster=args_.hard_cluster,
                        binary=args_.binary
                    )
            else:
                new_global_learners_ensemble = \
                    get_learners_ensemble(
                        n_learners=cluster_num,
                        client_type=CLIENT_TYPE[args_.method],
                        name=args_.experiment,
                        device=args_.device,
                        optimizer_name=args_.optimizer,
                        scheduler_name=args_.lr_scheduler,
                        initial_lr=args_.lr,
                        input_dim=args_.input_dimension,
                        output_dim=args_.output_dimension,
                        n_rounds=args_.n_rounds,
                        seed=args_.seed,
                        mu=args_.mu,
                        embedding_dim=args_.embedding_dimension,
                        n_gmm=args_.n_gmm,
                        hard_cluster=args_.hard_cluster,
                        binary=args_.binary,
                        phi_model=args.phi_model
                    )
            
            for learner_id, learner in enumerate(global_learners_ensemble):  
                copy_model(new_global_learners_ensemble[learner_id].model, learner.model)

            global_learners_ensemble = new_global_learners_ensemble

            
            clients_output_prototype = get_clients_output_prototype_version1(
                clients,
                dp_manager=dp_manager,
                max_norm=getattr(args_, "dp_max_norm", 10.0)
            )
            
            # Log privacy budget if DP is enabled
            if dp_manager is not None:
                privacy_spent = dp_manager.get_privacy_spent()
                print(f"==> Time slot {t}: Privacy spent = ε:{privacy_spent['total_epsilon']:.4f}, δ:{privacy_spent['delta']:.2e}, releases:{privacy_spent['num_releases']}")
            
            cluster_num, silhouette_scores,cluster_centers = \
                determine_cluster_number_return_centers(clients_output_prototype)

            # print("Cluster Number:", cluster_num)
            # print("silhouette_scores", silhouette_scores)
            
            determine_cluster_path = logs_base_path / f"determine_cluster_number-{args_.method}-{args_.gamma}-{args_.suffix}.txt"
            with open(determine_cluster_path, 'a+') as f:

                f.write('{}'.format(cluster_num))
                f.write('\n')
                f.write('{}'.format(silhouette_scores))
                f.write('\n')


            shift_set, clean_set, drift_types, drift_metrics = \
                check_whether_clients_concept_shifts_cluster_center_based(
                    clients,
                    cluster_centers,
                    diagnosis_mode=args_.diagnosis_mode,
                    thresholds=None if getattr(args_, "adaptive_thresholds", False) else DEFAULT_THRESHOLDS
                )

            binary_shift_set = [idx for idx, drift_type in drift_types.items() if drift_type == 'real']
            binary_clean_set = [idx for idx in range(len(clients)) if idx not in binary_shift_set]

            for idx, client in enumerate(clients):
                client.drift_type = drift_types.get(idx, 'none')

            if args_.diagnosis_mode == 'multiclass':
                type_counts = Counter(drift_types.values())
                drift_log_path = logs_base_path / f"drift-types-{args_.method}-{args_.gamma}-{args_.suffix}.txt"
                with open(drift_log_path, 'a+') as f:
                    f.write(f"time_slot={t}\n")
                    f.write(f"counts={dict(type_counts)}\n")
                    for client_id in sorted(drift_types):
                        metrics_payload = drift_metrics.get(client_id, {})
                        f.write(f"{client_id},{drift_types[client_id]},{json.dumps(metrics_payload)}\n")

            shift_eval = binary_shift_set if args_.diagnosis_mode == 'multiclass' else shift_set
            clean_eval = binary_clean_set if args_.diagnosis_mode == 'multiclass' else clean_set

            for client_idx, drift_type in drift_types.items():
                if drift_type == 'none':
                    continue
                client = clients[client_idx]
                baseline_old = evaluate_iterator_accuracy(
                    client.learners_ensemble,
                    getattr(client, "last_test_iterator", None)
                )
                client.record_event_baseline(baseline_old)
                current_events[client_idx] = {
                    "drift_type": drift_type,
                    "baseline_old": baseline_old,
                    "objective": client.objective,
                    "time_slot": t,
                }

            set_clients_concept_shift_flag(clients, binary_shift_set)

            if args_.objective_aware:
                merge_policy = getattr(args_, "merge_policy", None)
                diagnosis_mode = getattr(args_, "diagnosis_mode", "binary")
                apply_objective_policy(clients, merge_policy_str=merge_policy, diagnosis_mode=diagnosis_mode)
                objective_log_path = logs_base_path / f"objective-actions-{args_.method}-{args_.gamma}-{args_.suffix}.txt"
                with open(objective_log_path, 'a+') as f:
                    f.write(f"time_slot={t}\n")
                    for idx, client in enumerate(clients):
                        action = 'reset' if client.concept_shift_flag else 'merge'
                        objective = getattr(client, "objective", "N/A")
                        f.write(f"{idx},{objective},{client.drift_type},{action}\n")

            # Ablations: override for non-real drift only
            if getattr(args_, 'abl_all_reset', False) or getattr(args_, 'abl_all_merge', False):
                for idx, client in enumerate(clients):
                    if getattr(client, 'drift_type', 'none') != 'real':
                        if getattr(args_, 'abl_all_reset', False):
                            client.set_concept_shift_flag(1)
                        elif getattr(args_, 'abl_all_merge', False):
                            client.set_concept_shift_flag(0)

            # Apply merge/no-merge policy to training iterators based on flags
            try:
                update_clients_train_iterator_and_other_attribute(clients)
            except Exception:
                pass

            # Append current test iterator to history if requested
            if getattr(args_, 'eval_all_past_concepts', False):
                for c in clients:
                    try:
                        c._test_history.append(copy.deepcopy(c.test_iterator))
                    except Exception:
                        pass


            real_shift_set, real_clean_set = get_real_shift_clients_set(clients)
            # print("real_shift_set:",real_shift_set)
            # print("real_clean_set:",real_clean_set)

            # NOTE: Data loaders are now updated in the "if t != 0" block above using update_client_data_loaders()
            # No need to call rotate_120degree_update_clients_train_iterator_and_other_attribute() here anymore

            prediction_accuracy = get_shift_clients_prediction_accuracy(shift_set=shift_eval,clean_set=clean_eval,
                                                                        real_shift_set=real_shift_set,real_clean_set=real_clean_set)
            prediction_accuracy_list.append(prediction_accuracy)
            # print("shift client prediction accuracy:",prediction_accuracy)

            precision = get_shift_clients_precision(shift_set=shift_eval,real_shift_set=real_shift_set)
            precision_list.append(precision)
            # print("precision:",precision)

            recall = get_shift_clients_recall(shift_set=shift_eval,real_shift_set=real_shift_set)
            recall_list.append(recall)
            # F1 per time slot
            denom = (precision + recall + 1e-8)
            f1 = 0.0 if denom <= 1e-8 else (2.0 * precision * recall) / denom
            f1_list.append(f1)
            # print("recall:",recall)




        aggregator =\
            get_aggregator(
                aggregator_type=aggregator_type,
                clients=clients,
                global_learners_ensemble=global_learners_ensemble,
                lr_lambda=args_.lr_lambda,
                lr=args_.lr,
                q=args_.q,
                mu=args_.mu,
                communication_probability=args_.communication_probability,
                sampling_rate=args_.sampling_rate, 
                log_freq=args_.log_freq,
                global_train_logger=global_train_logger,
                global_test_logger=global_test_logger,
                test_clients=test_clients,
                verbose=args_.verbose,
                seed=args_.seed,
                experiment = args_.experiment,
                method = args_.method,
                suffix = args_.suffix,
                split = args_.split,
                domain_disc=args_.domain_disc,
                em_step=args_.em_step
            )

        print("Training..")
        pbar = tqdm(total=args_.n_rounds)
        current_round = 0
        pre_action = 0

        while current_round < args_.n_rounds:

      
            if pre_action == 0:
                aggregator.mix(diverse=False)
            else:
                aggregator.mix(diverse=False)

            C = CLASS_NUMBER[LOADER_TYPE[args_.experiment]]
            n_learner = aggregator.n_learners 
            cluster_label_weights = [[0] * C for _ in range(n_learner)]
            cluster_weights = [0 for _ in range(n_learner)]
            global_flags = [[] for _ in range(n_learner)]


            if 'shakespeare' not in args_.experiment:
                sample_weight_path = logs_base_path / f"sample-weight-{args_.method}-{args_.suffix}.txt"
                with open(sample_weight_path, 'w') as f:
                    for client_index, client in enumerate(clients):
           
                        for i in range(len(client.train_iterator.dataset.targets)):
                            if args_.method == 'FedSoft':
                                f.write('{},{},{}, {}\n'.format(client.data_type, client.train_iterator.dataset.targets[i], client.feature_types[i], aggregator.clusters_weights[client_index]))
                            else:

                                f.write('{},{},{}, {}\n'.format(client.data_type, client.train_iterator.dataset.targets[i], client.feature_types[i], client.samples_weights.T[i]))

                            for j in range(len(cluster_label_weights)):
                                cluster_weights[j] += client.samples_weights[j][i]
                        f.write('\n')
            else:
                for client_index, client in enumerate(clients):
                    for i in range(len(client.train_iterator.dataset.targets)): 
                        for j in range(len(cluster_label_weights)): 
                                cluster_weights[j] += client.samples_weights[j][i] 
 
            mean_i_path = logs_base_path / f"mean-I-{args_.method}-{args_.gamma}-{args_.suffix}.txt"
            with open(mean_i_path, 'a+') as f:
                mean_Is = torch.zeros((len(clients),))
                clusters = torch.zeros((len(clients),))
                client_types = torch.zeros((len(clients),))
                for i, client in enumerate(clients):
                    
                    mean_Is[i] = client.mean_I
                    client_types[i] = client.data_type
                    
                f.write('{}'.format(mean_Is))
                f.write('\n')
            cluster_weight_path = logs_base_path / f"cluster-weights-{args_.method}-{args_.gamma}-{args_.suffix}.txt"
            with open(cluster_weight_path, 'a+') as f:
                
                f.write('{}'.format(cluster_weights))
                f.write('\n')

            for client in clients:
                client_labels_learner_weights = client.labels_learner_weights
                for j in range(len(cluster_label_weights)):
                    for k in range(C):
                        
                        cluster_label_weights[j][k] += client_labels_learner_weights[j][k]
            for j in range(len(cluster_label_weights)):
                for i in range(len(cluster_label_weights[j])):
                    if cluster_label_weights[j][i] < 1e-8:
                        cluster_label_weights[j][i] = 1e-8
                cluster_label_weights[j] = [i / sum(cluster_label_weights[j]) for i in cluster_label_weights[j]]

          
            for client in clients:
                client.update_labels_weights(cluster_label_weights)
    
            for client in test_clients:
                print(client.mean_I, client.cluster, torch.nonzero(client.cluster==torch.max(client.cluster)).squeeze())

            if aggregator.c_round != current_round:
                pbar.update(1)
                current_round = aggregator.c_round

        
        # Per time-slot objective-aware/global/local means (across clients)
        try:
            # Local current test accuracy per client
            local_accs = []
            for c in clients:
                acc = evaluate_iterator_accuracy(c.learners_ensemble, c.test_iterator)
                if acc is not None:
                    local_accs.append(acc)
            if len(local_accs) > 0:
                time_local_all_means.append(float(np.mean(local_accs)))

            # Global test accuracy per client (if enabled)
            if 'global_test_iterator' in locals() and global_test_iterator is not None:
                global_accs = []
                for c in clients:
                    acc = evaluate_iterator_accuracy(c.learners_ensemble, global_test_iterator)
                    if acc is not None:
                        global_accs.append(acc)
                if len(global_accs) > 0:
                    time_global_all_means.append(float(np.mean(global_accs)))

            # Objective-aware (P: local current; G: global if available else old)
            oa_accs = []
            for c in clients:
                obj = getattr(c, 'objective', 'G')
                if obj == 'P':
                    acc = evaluate_iterator_accuracy(c.learners_ensemble, c.test_iterator)
                else:
                    if 'global_test_iterator' in locals() and global_test_iterator is not None:
                        acc = evaluate_iterator_accuracy(c.learners_ensemble, global_test_iterator)
                    else:
                        acc = evaluate_iterator_accuracy(c.learners_ensemble, getattr(c, 'last_test_iterator', None))
                if acc is not None:
                    oa_accs.append(acc)
            if len(oa_accs) > 0:
                time_objective_aware_means.append(float(np.mean(oa_accs)))

            # Objective-aware OLD (P: local; G: old)
            oa_old_accs = []
            for c in clients:
                if getattr(c, 'objective', 'G') == 'P':
                    acc = evaluate_iterator_accuracy(c.learners_ensemble, c.test_iterator)
                else:
                    acc = evaluate_iterator_accuracy(c.learners_ensemble, getattr(c, 'last_test_iterator', None))
                if acc is not None:
                    oa_old_accs.append(acc)
            if len(oa_old_accs) > 0:
                time_objective_aware_old_means.append(float(np.mean(oa_old_accs)))

            # Objective-aware GLOBAL (P: local; G: global)
            if 'global_test_iterator' in locals() and global_test_iterator is not None:
                oa_global_accs = []
                for c in clients:
                    if getattr(c, 'objective', 'G') == 'P':
                        acc = evaluate_iterator_accuracy(c.learners_ensemble, c.test_iterator)
                    else:
                        acc = evaluate_iterator_accuracy(c.learners_ensemble, global_test_iterator)
                    if acc is not None:
                        oa_global_accs.append(acc)
                if len(oa_global_accs) > 0:
                    time_objective_aware_global_means.append(float(np.mean(oa_global_accs)))
        except Exception:
            pass

        if "save_dir" in args_:
            save_dir = os.path.join(args_.save_dir)

            os.makedirs(save_dir, exist_ok=True)
            aggregator.save_state(save_dir)


        drift_prediction_path = logs_base_path / f"drift-prediction-result-{args_.method}-{args_.gamma}-{args_.suffix}.txt"
        with open(drift_prediction_path, 'a+') as f:
            f.write("prediction_accuracy_list:")
            f.write(f"{prediction_accuracy_list}\n")
            f.write("precision_list:")
            f.write(f"{precision_list}\n")
            f.write("recall_list:")
            f.write(f"{recall_list}\n")
            f.write("f1_list:")
            f.write(f"{f1_list}\n")

        aggregator = None
        gc.collect()

    global_train_logger.flush()
    global_train_logger.close()
    global_test_logger.flush()
    global_test_logger.close()

    detection_stats = {
        "accuracy": sample_stats(prediction_accuracy_list),
        "precision": sample_stats(precision_list),
        "recall": sample_stats(recall_list),
        "f1": sample_stats(f1_list),
    }

    p_scores = []
    g_scores = []
    forgetting_ratios = []
    retention_by_type = defaultdict(list)
    event_client_ids = set()

    for record in event_records:
        event_client_ids.add(record["client_id"])
        if record["post_current"] is not None:
            retention_by_type[record["drift_type"]].append(record["post_current"])
            if record["objective"] == 'P':
                p_scores.append(record["post_current"])
        if record["post_old"] is not None and record["objective"] == 'G':
            g_scores.append(record["post_old"])
        if record["baseline_old"] is not None and record["post_old"] is not None:
            forgetting = max(0.0, record["baseline_old"] - record["post_old"]) / (record["baseline_old"] + 1e-8)
            forgetting_ratios.append(forgetting)

    final_current_accs = []
    final_old_accs = []
    for idx, client in enumerate(clients):
        current_acc = evaluate_iterator_accuracy(client.learners_ensemble, client.test_iterator)
        old_acc = evaluate_iterator_accuracy(client.learners_ensemble, getattr(client, "last_test_iterator", None))
        if current_acc is not None:
            final_current_accs.append(current_acc)
        if old_acc is not None:
            final_old_accs.append(old_acc)
        if idx not in event_client_ids:
            if client.objective == 'P' and current_acc is not None:
                p_scores.append(current_acc)
            elif client.objective == 'G':
                g_scores.append(old_acc if old_acc is not None else current_acc)
                if old_acc is not None:
                    forgetting_ratios.append(0.0)

    retention_stats = {dtype: sample_stats(values) for dtype, values in retention_by_type.items()}
    p_stats = sample_stats(p_scores)
    g_stats = sample_stats([score for score in g_scores if score is not None])
    forgetting_stats = sample_stats(forgetting_ratios)
    overall_current_stats = sample_stats(final_current_accs)
    overall_old_stats = sample_stats(final_old_accs)

    # Objective-aware final scores (P: local current, G: global if enabled else old)
    objective_aware_scores = []
    for c in clients:
        if getattr(c, 'objective', 'G') == 'P':
            acc = evaluate_iterator_accuracy(c.learners_ensemble, c.test_iterator)
        else:
            if 'global_test_iterator' in locals() and global_test_iterator is not None:
                acc = evaluate_iterator_accuracy(c.learners_ensemble, global_test_iterator)
            else:
                acc = evaluate_iterator_accuracy(c.learners_ensemble, getattr(c, 'last_test_iterator', None))
        if acc is not None:
            objective_aware_scores.append(acc)
    objective_aware_stats = sample_stats(objective_aware_scores)

    # Objective-aware variants
    objective_aware_old_scores = []
    objective_aware_global_scores = []
    for c in clients:
        # P always local
        p_local = evaluate_iterator_accuracy(c.learners_ensemble, c.test_iterator) if getattr(c, 'objective', 'G') == 'P' else None
        if getattr(c, 'objective', 'G') == 'P':
            if p_local is not None:
                objective_aware_old_scores.append(p_local)
                objective_aware_global_scores.append(p_local)
        else:
            # G: old variant
            old_acc = evaluate_iterator_accuracy(c.learners_ensemble, getattr(c, 'last_test_iterator', None))
            if old_acc is not None:
                objective_aware_old_scores.append(old_acc)
            # G: global variant
            if 'global_test_iterator' in locals() and global_test_iterator is not None:
                g_acc = evaluate_iterator_accuracy(c.learners_ensemble, global_test_iterator)
                if g_acc is not None:
                    objective_aware_global_scores.append(g_acc)
    objective_aware_old_stats = sample_stats(objective_aware_old_scores)
    objective_aware_global_stats = sample_stats(objective_aware_global_scores)

    # Global (all clients) final stats
    global_all_stats = None
    if 'global_test_iterator' in locals() and global_test_iterator is not None:
        global_all_accs = [evaluate_iterator_accuracy(c.learners_ensemble, global_test_iterator) for c in clients]
        global_all_accs = [a for a in global_all_accs if a is not None]
        global_all_stats = sample_stats(global_all_accs)

    # Across-time aggregates
    across_time_stats = {
        "local_all": sample_stats(time_local_all_means),
        "global_all": sample_stats(time_global_all_means),
        "objective_aware": sample_stats(time_objective_aware_means),
        "objective_aware_old": sample_stats(time_objective_aware_old_means),
        "objective_aware_global": sample_stats(time_objective_aware_global_means),
    }

    # All past concepts @ final (per-client mean over test history)
    all_past_stats = {}
    if getattr(args_, 'eval_all_past_concepts', False):
        all_past_all = []
        all_past_P = []
        all_past_G = []
        for c in clients:
            history = getattr(c, '_test_history', None)
            if not history:
                continue
            accs = []
            for it in history:
                a = evaluate_iterator_accuracy(c.learners_ensemble, it)
                if a is not None:
                    accs.append(a)
            if not accs:
                continue
            m = float(np.mean(accs))
            all_past_all.append(m)
            if getattr(c, 'objective', 'G') == 'P':
                all_past_P.append(m)
            else:
                all_past_G.append(m)
        all_past_stats = {
            'all_clients': sample_stats(all_past_all),
            'P_clients': sample_stats(all_past_P),
            'G_clients': sample_stats(all_past_G),
        }

    # Optional: objective-grouped Global evaluation on a fixed baseline test set
    global_p_scores = []
    global_g_scores = []
    global_p_stats = None
    global_g_stats = None
    if global_test_iterator is not None:
        for client in clients:
            acc = evaluate_iterator_accuracy(client.learners_ensemble, global_test_iterator)
            if acc is None:
                continue
            if getattr(client, "objective", "G") == 'P':
                global_p_scores.append(acc)
            else:
                global_g_scores.append(acc)
        global_p_stats = sample_stats(global_p_scores)
        global_g_stats = sample_stats(global_g_scores)

    # Per-objective local/global breakdowns
    p_on_local = []
    g_on_local = []
    p_on_global = []
    g_on_global = []
    for client in clients:
        obj = getattr(client, 'objective', 'G')
        local_acc = evaluate_iterator_accuracy(client.learners_ensemble, client.test_iterator)
        global_acc = None
        if 'global_test_iterator' in locals() and global_test_iterator is not None:
            global_acc = evaluate_iterator_accuracy(client.learners_ensemble, global_test_iterator)
        if obj == 'P':
            if local_acc is not None:
                p_on_local.append(local_acc)
            if global_acc is not None:
                p_on_global.append(global_acc)
        else:
            if local_acc is not None:
                g_on_local.append(local_acc)
            if global_acc is not None:
                g_on_global.append(global_acc)

    p_on_local_stats = sample_stats(p_on_local)
    g_on_local_stats = sample_stats(g_on_local)
    p_on_global_stats = sample_stats(p_on_global) if (global_test_iterator is not None) else sample_stats([])
    g_on_global_stats = sample_stats(g_on_global) if (global_test_iterator is not None) else sample_stats([])

    metrics_output = {
        "seed": args_.seed,
        "event_records": event_records,
        "p_scores": p_scores,
        "g_scores": g_scores,
        "forgetting_ratios": forgetting_ratios,
        "retention": {dtype: values for dtype, values in retention_by_type.items()},
        "detection": detection_stats,
        "overall_current": final_current_accs,
        "overall_old": final_old_accs,
        "stats": {
            "p_scores": p_stats,
            "g_scores": g_stats,
            "objective_aware": objective_aware_stats,
            "forgetting": forgetting_stats,
            "overall_current": overall_current_stats,
            "overall_old": overall_old_stats,
            "overall_global": (global_all_stats or {}),
            "across_time": across_time_stats,
            "objective_aware_old": objective_aware_old_stats,
            "objective_aware_global": objective_aware_global_stats,
            "all_past_concepts": all_past_stats,
            "retention": {dtype: sample_stats(values) for dtype, values in retention_by_type.items()},
        },
    }

    if global_test_iterator is not None:
        metrics_output["global_scores"] = {"P": global_p_scores, "G": global_g_scores}
        metrics_output.setdefault("stats", {})["global_P"] = global_p_stats
        metrics_output.setdefault("stats", {})["global_G"] = global_g_stats
    metrics_output.setdefault("stats", {})["P_on_local"] = p_on_local_stats
    metrics_output.setdefault("stats", {})["G_on_local"] = g_on_local_stats
    metrics_output.setdefault("stats", {})["P_on_global"] = p_on_global_stats
    metrics_output.setdefault("stats", {})["G_on_global"] = g_on_global_stats

    if exp_logger is not None:
        exp_logger.write_metrics(args_.seed, metrics_output)
        fig = build_bar_plot(p_stats, g_stats)
        exp_logger.write_plot(args_.seed, "objective_scores.png", fig)
        plt.close(fig)
        
        # Log objective-grouped test accuracy summary
        objective_summary_path = logs_base_path / f"objective-grouped-accuracy-{args_.method}-{args_.suffix}.txt"
        with open(objective_summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            # Legend for clarity
            f.write("LEGEND\n")
            f.write("-" * 80 + "\n")
            f.write("Personalization (P) Clients: P-clients on current local test (final slot).\n")
            f.write("Generalization (G) Clients: G-clients on previous (old) local test (final slot; falls back to current if missing).\n")
            f.write("Overall (All Clients): All clients on current local test (final slot).\n")
            f.write("Objective-aware OLD: Mix by objective, P→local current; G→old (final slot).\n")
            f.write("Objective-aware GLOBAL: Mix by objective, P→local current; G→fixed global test set (final slot).\n")
            f.write("GLOBAL TEST (Global P/G): P/G-clients on the fixed global test set (baseline).\n")
            f.write("P_on_local / G_on_local: P/G-clients on current local test (note: P_on_local equals P Clients).\n")
            f.write("P_on_global / G_on_global: P/G-clients on the fixed global test set (duplicates Global P/G).\n")
            f.write("Overall Global (All Clients): All clients on the fixed global test set (baseline).\n")
            f.write("ACROSS-TIME: For each slot t, average over clients; then summarize those T values (mean/std/min/median/max).\n")
            f.write("TASKS@FINAL (Rotation): Mean accuracy at final slot on rotation-stratified test sets (0°, 120°, 240°).\n\n")
            f.write("=" * 80 + "\n")
            f.write("FINAL TEST ACCURACY GROUPED BY OBJECTIVE\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Personalization (P) Clients (n={len(p_scores)}):\n")
            f.write(f"  Mean: {p_stats['mean']:.4f}\n")
            f.write(f"  Std:  {p_stats['std']:.4f}\n")
            f.write(f"  Min:  {p_stats['min']:.4f}\n")
            f.write(f"  Median: {p_stats['median']:.4f}\n")
            f.write(f"  Max:  {p_stats['max']:.4f}\n\n")
            
            f.write(f"Generalization (G) Clients (n={len(g_scores)}):\n")
            f.write(f"  Mean: {g_stats['mean']:.4f}\n")
            f.write(f"  Std:  {g_stats['std']:.4f}\n")
            f.write(f"  Min:  {g_stats['min']:.4f}\n")
            f.write(f"  Median: {g_stats['median']:.4f}\n")
            f.write(f"  Max:  {g_stats['max']:.4f}\n\n")
            
            f.write(f"Overall (All Clients, n={len(final_current_accs)}):\n")
            f.write(f"  Mean: {overall_current_stats['mean']:.4f}\n")
            f.write(f"  Std:  {overall_current_stats['std']:.4f}\n")
            f.write(f"  Min:  {overall_current_stats['min']:.4f}\n")
            f.write(f"  Median: {overall_current_stats['median']:.4f}\n")
            f.write(f"  Max:  {overall_current_stats['max']:.4f}\n\n")

            # (Mixed) objective-aware omitted to avoid duplication with GLOBAL when --global_eval is on

            # Objective-aware (variants)
            # Report counts to reflect effective sample size (clients with valid iterators)
            nP = sum(1 for c in clients if getattr(c, 'objective', 'G') == 'P')
            nG = sum(1 for c in clients if getattr(c, 'objective', 'G') == 'G')
            f.write(f"Objective-aware OLD (P: local, G: old) (n={objective_aware_old_stats.get('count', 0)}; P={nP}, G={nG})\n")
            f.write(f"  Mean: {objective_aware_old_stats['mean']:.4f}\n")
            f.write(f"  Std:  {objective_aware_old_stats['std']:.4f}\n")
            f.write(f"  Min:  {objective_aware_old_stats['min']:.4f}\n")
            f.write(f"  Median: {objective_aware_old_stats['median']:.4f}\n")
            f.write(f"  Max:  {objective_aware_old_stats['max']:.4f}\n\n")
            if global_test_iterator is not None:
                f.write(f"Objective-aware GLOBAL (P: local, G: global) (n={objective_aware_global_stats.get('count', 0)}; P={nP}, G={nG})\n")
                f.write(f"  Mean: {objective_aware_global_stats['mean']:.4f}\n")
                f.write(f"  Std:  {objective_aware_global_stats['std']:.4f}\n")
                f.write(f"  Min:  {objective_aware_global_stats['min']:.4f}\n")
                f.write(f"  Median: {objective_aware_global_stats['median']:.4f}\n")
                f.write(f"  Max:  {objective_aware_global_stats['max']:.4f}\n\n")
            
            # Optional: Global evaluation summary
            if global_test_iterator is not None:
                f.write("GLOBAL TEST (Fixed baseline, objective-grouped)\n")
                f.write("-" * 80 + "\n")
                f.write(f"Global P (n={len(global_p_scores)}):\n")
                f.write(f"  Mean: {global_p_stats['mean']:.4f}\n")
                f.write(f"  Std:  {global_p_stats['std']:.4f}\n")
                f.write(f"  Min:  {global_p_stats['min']:.4f}\n")
                f.write(f"  Median: {global_p_stats['median']:.4f}\n")
                f.write(f"  Max:  {global_p_stats['max']:.4f}\n\n")
                f.write(f"Global G (n={len(global_g_scores)}):\n")
                f.write(f"  Mean: {global_g_stats['mean']:.4f}\n")
                f.write(f"  Std:  {global_g_stats['std']:.4f}\n")
                f.write(f"  Min:  {global_g_stats['min']:.4f}\n")
                f.write(f"  Median: {global_g_stats['median']:.4f}\n")
                f.write(f"  Max:  {global_g_stats['max']:.4f}\n\n")
            
            # Per-objective breakdowns
            f.write("PER-OBJECTIVE BREAKDOWNS\n")
            f.write("-" * 80 + "\n")
            pol = metrics_output["stats"]["P_on_local"]
            gol = metrics_output["stats"]["G_on_local"]
            pog = metrics_output["stats"]["P_on_global"]
            gog = metrics_output["stats"]["G_on_global"]
            f.write("P_on_local:\n")
            f.write(f"  Mean: {pol['mean']:.4f}\n")
            f.write(f"  Std:  {pol['std']:.4f}\n")
            f.write(f"  Min:  {pol['min']:.4f}\n")
            f.write(f"  Median: {pol['median']:.4f}\n")
            f.write(f"  Max:  {pol['max']:.4f}\n\n")
            f.write("G_on_local:\n")
            f.write(f"  Mean: {gol['mean']:.4f}\n")
            f.write(f"  Std:  {gol['std']:.4f}\n")
            f.write(f"  Min:  {gol['min']:.4f}\n")
            f.write(f"  Median: {gol['median']:.4f}\n")
            f.write(f"  Max:  {gol['max']:.4f}\n\n")
            f.write("P_on_global:\n")
            f.write(f"  Mean: {pog['mean']:.4f}\n")
            f.write(f"  Std:  {pog['std']:.4f}\n")
            f.write(f"  Min:  {pog['min']:.4f}\n")
            f.write(f"  Median: {pog['median']:.4f}\n")
            f.write(f"  Max:  {pog['max']:.4f}\n\n")
            f.write("G_on_global:\n")
            f.write(f"  Mean: {gog['mean']:.4f}\n")
            f.write(f"  Std:  {gog['std']:.4f}\n")
            f.write(f"  Min:  {gog['min']:.4f}\n")
            f.write(f"  Median: {gog['median']:.4f}\n")
            f.write(f"  Max:  {gog['max']:.4f}\n\n")

            # Overall Global (All Clients) if available
            if metrics_output["stats"].get("overall_global"):
                og = metrics_output["stats"]["overall_global"]
                f.write(f"Overall Global (All Clients):\n")
                f.write(f"  Mean: {og.get('mean', float('nan')):.4f}\n")
                f.write(f"  Std:  {og.get('std', float('nan')):.4f}\n")
                f.write(f"  Min:  {og.get('min', float('nan')):.4f}\n")
                f.write(f"  Median: {og.get('median', float('nan')):.4f}\n")
                f.write(f"  Max:  {og.get('max', float('nan')):.4f}\n\n")

            # Across-time (per time-slot means)
            f.write("ACROSS-TIME (per time-slot means)\n")
            f.write("-" * 80 + "\n")
            at_local = metrics_output["stats"]["across_time"].get("local_all", {})
            f.write("Local all clients:\n")
            f.write(f"  Mean: {at_local.get('mean', float('nan')):.4f}\n")
            f.write(f"  Std:  {at_local.get('std', float('nan')):.4f}\n")
            f.write(f"  Min:  {at_local.get('min', float('nan')):.4f}\n")
            f.write(f"  Median: {at_local.get('median', float('nan')):.4f}\n")
            f.write(f"  Max:  {at_local.get('max', float('nan')):.4f}\n\n")

            at_global = metrics_output["stats"]["across_time"].get("global_all", {})
            f.write("Global all clients:\n")
            f.write(f"  Mean: {at_global.get('mean', float('nan')):.4f}\n")
            f.write(f"  Std:  {at_global.get('std', float('nan')):.4f}\n")
            f.write(f"  Min:  {at_global.get('min', float('nan')):.4f}\n")
            f.write(f"  Median: {at_global.get('median', float('nan')):.4f}\n")
            f.write(f"  Max:  {at_global.get('max', float('nan')):.4f}\n\n")

            # Suppress mixed OA across-time to avoid confusion/duplication
            at_oa_old = metrics_output["stats"]["across_time"].get("objective_aware_old", {})
            f.write("Objective-aware OLD (P: local, G: old):\n")
            f.write(f"  Mean: {at_oa_old.get('mean', float('nan')):.4f}\n")
            f.write(f"  Std:  {at_oa_old.get('std', float('nan')):.4f}\n")
            f.write(f"  Min:  {at_oa_old.get('min', float('nan')):.4f}\n")
            f.write(f"  Median: {at_oa_old.get('median', float('nan')):.4f}\n")
            f.write(f"  Max:  {at_oa_old.get('max', float('nan')):.4f}\n\n")
            at_oa_global = metrics_output["stats"]["across_time"].get("objective_aware_global", {})
            f.write("Objective-aware GLOBAL (P: local, G: global):\n")
            f.write(f"  Mean: {at_oa_global.get('mean', float('nan')):.4f}\n")
            f.write(f"  Std:  {at_oa_global.get('std', float('nan')):.4f}\n")
            f.write(f"  Min:  {at_oa_global.get('min', float('nan')):.4f}\n")
            f.write(f"  Median: {at_oa_global.get('median', float('nan')):.4f}\n")
            f.write(f"  Max:  {at_oa_global.get('max', float('nan')):.4f}\n\n")

            # Tasks@final (Rotation)
            try:
                def _mean_acc_over(test_clients_list):
                    vals = []
                    for tc in test_clients_list:
                        a = evaluate_iterator_accuracy(global_learners_ensemble, tc.test_iterator)
                        if a is not None:
                            vals.append(a)
                    return float(np.mean(vals)) if vals else float('nan')
                rot0 = _mean_acc_over(test_clients_0degree)
                rot120 = _mean_acc_over(test_clients_120degree)
                rot240 = _mean_acc_over(test_clients_240degree)
                f.write("TASKS@FINAL (Rotation)\n")
                f.write("-" * 80 + "\n")
                f.write(f"(rotation=0°)   Mean: {rot0:.4f}\n")
                f.write(f"(rotation=120°) Mean: {rot120:.4f}\n")
                f.write(f"(rotation=240°) Mean: {rot240:.4f}\n\n")
            except Exception:
                pass
            
            # All Past Concepts @ Final
            if all_past_stats:
                f.write("ALL PAST CONCEPTS @ FINAL\n")
                f.write("-" * 80 + "\n")
                ap_all = all_past_stats.get('all_clients', {})
                ap_p = all_past_stats.get('P_clients', {})
                ap_g = all_past_stats.get('G_clients', {})
                f.write("All clients:\n")
                f.write(f"  Mean: {ap_all.get('mean', float('nan')):.4f}\n")
                f.write(f"  Std:  {ap_all.get('std', float('nan')):.4f}\n")
                f.write(f"  Min:  {ap_all.get('min', float('nan')):.4f}\n")
                f.write(f"  Median: {ap_all.get('median', float('nan')):.4f}\n")
                f.write(f"  Max:  {ap_all.get('max', float('nan')):.4f}\n\n")
                f.write("P clients:\n")
                f.write(f"  Mean: {ap_p.get('mean', float('nan')):.4f}\n")
                f.write(f"  Std:  {ap_p.get('std', float('nan')):.4f}\n")
                f.write(f"  Min:  {ap_p.get('min', float('nan')):.4f}\n")
                f.write(f"  Median: {ap_p.get('median', float('nan')):.4f}\n")
                f.write(f"  Max:  {ap_p.get('max', float('nan')):.4f}\n\n")
                f.write("G clients:\n")
                f.write(f"  Mean: {ap_g.get('mean', float('nan')):.4f}\n")
                f.write(f"  Std:  {ap_g.get('std', float('nan')):.4f}\n")
                f.write(f"  Min:  {ap_g.get('min', float('nan')):.4f}\n")
                f.write(f"  Median: {ap_g.get('median', float('nan')):.4f}\n")
                f.write(f"  Max:  {ap_g.get('max', float('nan')):.4f}\n\n")

            if dp_manager is not None:
                privacy_spent = dp_manager.get_privacy_spent()
                f.write(f"Privacy Budget Spent:\n")
                f.write(f"  Epsilon: {privacy_spent['epsilon']:.4f}\n")
                f.write(f"  Delta:   {privacy_spent['delta']:.2e}\n\n")
        
        print(f"\n==> Objective-grouped accuracy saved to: {objective_summary_path}")

    release_client_resources(test_clients)
    release_client_resources(clients)
    gc.collect()

    return metrics_output


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()
    seeds = parse_seed_argument(args.seeds, args.seed)
    logger = ExperimentLogger(args, seeds)
    seed_metrics = []
    for seed in seeds:
        seed_args = copy.deepcopy(args)
        seed_args.seed = seed
        metrics = run_experiment(seed_args, exp_logger=logger)
        seed_metrics.append(metrics)

    aggregate = aggregate_summary(seed_metrics)

    logger.save_summary({
        "per_seed": seed_metrics,
        "aggregate": aggregate,
    })

    print("\n=== FedORA Summary ===")
    print(f"Seeds: {seeds}")
    if 'p_scores' in aggregate:
        print(f"P-Score mean±std: {aggregate['p_scores']['mean']:.4f} ± {aggregate['p_scores']['std']:.4f}")
    if 'g_scores' in aggregate:
        print(f"G-Score mean±std: {aggregate['g_scores']['mean']:.4f} ± {aggregate['g_scores']['std']:.4f}")
    if 'forgetting' in aggregate:
        print(f"Forgetting ratio mean±std: {aggregate['forgetting']['mean']:.4f} ± {aggregate['forgetting']['std']:.4f}")

# -*- coding: utf-8 -*-
"""
Enhanced data generator for multiclass drift detection.

This version creates separate client pools for each drift type:
1. Real drift (concept): Label remapping cohorts
2. Feature drift only: Same labels, different rotations
3. Label drift only: Different class distributions (Dirichlet), same mapping
4. No drift: Fixed distribution, fixed mapping, fixed rotation

Usage:
    python make_cifar_c-MULTICLASS_DRIFT.py --drift_type <type> --alpha <value>
"""

import os
from PIL import Image
import os.path
import time
import torch
import torchvision.datasets as dset
import torchvision.transforms as trn
import torch.utils.data as data
import numpy as np
import torch.distributions.dirichlet as dirichlet
import random
from torchvision import transforms
from PIL import Image
import pickle
import argparse
from pathlib import Path

# Import all distortion helpers from original script
import skimage as sk
from skimage.filters import gaussian
from io import BytesIO
from wand.image import Image as WandImage
from wand.api import library as wandlibrary
import wand.color as WandColor
import ctypes
from PIL import Image as PILImage
import cv2
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.interpolation import map_coordinates
import warnings
warnings.simplefilter("ignore", UserWarning)

# ============================================================================
# CONFIGURATION
# ============================================================================

C = 10  # Number of classes (CIFAR-10)
CLIENT_NUMBER = 60  # 15 per drift type
CLIENTS_PER_TYPE = 15

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def save_data(l, path_):
    Path(path_).parent.mkdir(parents=True, exist_ok=True)
    with open(path_, "wb") as f:
        pickle.dump(l, f)

def convert_img(img):
    """Convert tensors or PIL images into numpy array."""
    if isinstance(img, torch.Tensor):
        arr = img.detach().cpu()
        if arr.ndim == 3:
            arr = arr.permute(1, 2, 0)
        return arr.numpy()
    return np.asarray(img)

def divide_by_label(dataset, class_num=10):
    """Group indices by label."""
    index_map = [[] for i in range(class_num)]
    len_map = [0 for _ in range(class_num)]
    for i in range(len(dataset)):
        index_map[dataset[i][1]].append(i)
        len_map[dataset[i][1]] += 1
    return index_map, len_map

def reweight(q, empty_class):
    """Reweight distribution after exhausting a class."""
    q[empty_class] = 0
    q = q / sum(q)
    return q

# ============================================================================
# IID PARTITIONING (for feature-only and no-drift)
# ============================================================================

def get_iid_class_and_labels(original_images, original_labels, N, class_num=10):
    """
    IID partition: uniform sampling across classes.
    """
    M = len(original_labels) // N

    clients_images = [[] for _ in range(N)]
    clients_labels = [[] for _ in range(N)]
    classes_by_index = [[] for _ in range(class_num)]
    classes_by_index_len = [0 for _ in range(class_num)]
    
    for i, label in enumerate(original_labels):
        classes_by_index[label].append(i)
        classes_by_index_len[label] += 1

    for i in range(N):
        p = torch.tensor(classes_by_index_len, dtype=torch.float32) / sum(classes_by_index_len)
        
        while len(clients_labels[i]) < M:
            sampled_class = torch.multinomial(p, 1).item()
            if classes_by_index_len[sampled_class] == 0:
                p = reweight(p, sampled_class)
            else:
                sampled_index = random.randint(0, classes_by_index_len[sampled_class] - 1)
                sampled_original_index = classes_by_index[sampled_class][sampled_index]
                clients_images[i].append(original_images[sampled_original_index])
                clients_labels[i].append(original_labels[sampled_original_index])
                classes_by_index[sampled_class].pop(sampled_index)
                classes_by_index_len[sampled_class] -= 1

    return clients_images, clients_labels

# ============================================================================
# DIRICHLET PARTITIONING (for label-drift)
# ============================================================================

def get_dirichlet_class_and_labels(original_images, original_labels, N, alpha, class_num=10):
    """
    Non-IID partition using Dirichlet distribution.
    Lower alpha → more skewed (heterogeneous) label distributions.
    
    This creates P(Y) differences while keeping P(Y|X) fixed (no remapping).
    """
    M = len(original_labels) // N

    clients_images = [[] for _ in range(N)]
    clients_labels = [[] for _ in range(N)]
    classes_by_index = [[] for _ in range(class_num)]
    classes_by_index_len = [0 for _ in range(class_num)]
    
    for i, label in enumerate(original_labels):
        classes_by_index[label].append(i)
        classes_by_index_len[label] += 1

    for i in range(N):
        # Sample from Dirichlet to get client-specific class proportions
        p = torch.tensor(classes_by_index_len, dtype=torch.float32) / sum(classes_by_index_len)
        p = p + 1e-6  # Avoid zeros
        q = dirichlet.Dirichlet(alpha * p).sample()
        
        while len(clients_labels[i]) < M:
            sampled_class = torch.multinomial(q, 1).item()
            if classes_by_index_len[sampled_class] == 0:
                q = reweight(q, sampled_class)
            else:
                sampled_index = random.randint(0, classes_by_index_len[sampled_class] - 1)
                sampled_original_index = classes_by_index[sampled_class][sampled_index]
                clients_images[i].append(original_images[sampled_original_index])
                clients_labels[i].append(original_labels[sampled_original_index])
                classes_by_index[sampled_class].pop(sampled_index)
                classes_by_index_len[sampled_class] -= 1

    return clients_images, clients_labels

# ============================================================================
# LABEL REMAPPING (for concept drift)
# ============================================================================

def apply_label_remapping(labels, mapping_type, class_num=10):
    """
    Apply different label remapping strategies.
    mapping_type:
        1 = identity (no change)
        2 = reversed (C-1-label)
        3 = shift +1
        4 = shift +2
    """
    remapped = []
    for label in labels:
        if mapping_type == 1:
            remapped.append(label)
        elif mapping_type == 2:
            remapped.append(class_num - 1 - label)
        elif mapping_type == 3:
            remapped.append((label + 1) % class_num)
        elif mapping_type == 4:
            remapped.append((label + 2) % class_num)
        else:
            raise ValueError(f"Unknown mapping_type: {mapping_type}")
    return remapped

# ============================================================================
# MAIN GENERATION
# ============================================================================

def generate_multiclass_drift_dataset(output_dir, dirichlet_alpha=0.5):
    """
    Generate 4 client pools for multiclass drift detection:
    
    Pool 1 (clients 0-14): REAL DRIFT (concept)
        - IID split, but with 4 different label mappings (~4 clients each)
        - Use for real/concept drift detection
    
    Pool 2 (clients 15-29): FEATURE DRIFT ONLY
        - IID split, identity mapping (type 1)
        - Feature drift introduced at runtime via rotation
        - Use for pure feature drift detection
    
    Pool 3 (clients 30-44): LABEL DRIFT ONLY
        - Dirichlet split (skewed P(Y)), identity mapping (type 1)
        - Different alpha values per client to vary P(Y) over time
        - Use for pure label-prior drift detection
    
    Pool 4 (clients 45-59): NO DRIFT (clean)
        - IID split, identity mapping (type 1)
        - Fixed indices, fixed rotation at runtime
        - Use for baseline (no drift)
    """
    
    print("="*80)
    print("MULTICLASS DRIFT DATASET GENERATOR")
    print("="*80)
    
    # Load CIFAR-10
    print("\nLoading CIFAR-10...")
    train_data = dset.CIFAR10('./data/cifar10-c/origin/', train=True, download=True)
    test_data = dset.CIFAR10('./data/cifar10-c/origin/', train=False, download=True)
    
    original_images_tr = [X for X, Y in train_data]
    original_labels_tr = [Y for X, Y in train_data]
    original_images_te = [X for X, Y in test_data]
    original_labels_te = [Y for X, Y in test_data]
    
    all_clients_images = []
    all_clients_labels = []
    all_clients_types = []
    
    # ========================================================================
    # POOL 1: REAL DRIFT (concept drift via label remapping)
    # ========================================================================
    print("\n[1/4] Generating REAL DRIFT pool (clients 0-14)...")
    print("      - 4 cohorts with different label mappings (~4 clients each)")
    
    # Split into 4 sub-cohorts: 4, 4, 4, 3 clients
    clients_per_mapping = [4, 4, 4, 3]
    for mapping_type in range(1, 5):  # 1, 2, 3, 4
        n_clients = clients_per_mapping[mapping_type - 1]
        start_id = sum(clients_per_mapping[:mapping_type-1])
        end_id = start_id + n_clients - 1
        print(f"      - Mapping type {mapping_type}: clients {start_id} to {end_id}")
        
        clients_imgs, clients_lbls = get_iid_class_and_labels(
            original_images_tr, 
            original_labels_tr, 
            N=n_clients,
            class_num=C
        )
        
        for i in range(n_clients):
            # Apply label remapping
            remapped_labels = apply_label_remapping(clients_lbls[i], mapping_type, C)
            converted_imgs = [np.uint8(convert_img(img)) for img in clients_imgs[i]]
            
            all_clients_images.append(converted_imgs)
            all_clients_labels.append(remapped_labels)
            all_clients_types.append(f"real_drift_map{mapping_type}")
    
    # ========================================================================
    # POOL 2: FEATURE DRIFT ONLY (rotation at runtime, no remapping)
    # ========================================================================
    print("\n[2/4] Generating FEATURE DRIFT ONLY pool (clients 15-29)...")
    print("      - IID split, identity mapping, rotation applied at runtime")
    
    clients_imgs, clients_lbls = get_iid_class_and_labels(
        original_images_tr, 
        original_labels_tr, 
        N=CLIENTS_PER_TYPE,
        class_num=C
    )
    
    for i in range(CLIENTS_PER_TYPE):
        converted_imgs = [np.uint8(convert_img(img)) for img in clients_imgs[i]]
        all_clients_images.append(converted_imgs)
        all_clients_labels.append(clients_lbls[i])  # Identity mapping
        all_clients_types.append("feature_drift_only")
    
    # ========================================================================
    # POOL 3: LABEL DRIFT ONLY (Dirichlet P(Y) shift, no remapping)
    # ========================================================================
    print("\n[3/4] Generating LABEL DRIFT ONLY pool (clients 30-44)...")
    print(f"      - Dirichlet split (alpha={dirichlet_alpha}), identity mapping")
    
    clients_imgs, clients_lbls = get_dirichlet_class_and_labels(
        original_images_tr, 
        original_labels_tr, 
        N=CLIENTS_PER_TYPE,
        alpha=dirichlet_alpha,
        class_num=C
    )
    
    for i in range(CLIENTS_PER_TYPE):
        converted_imgs = [np.uint8(convert_img(img)) for img in clients_imgs[i]]
        all_clients_images.append(converted_imgs)
        all_clients_labels.append(clients_lbls[i])  # Identity mapping
        all_clients_types.append("label_drift_only")
    
    # ========================================================================
    # POOL 4: NO DRIFT (clean baseline)
    # ========================================================================
    print("\n[4/4] Generating NO DRIFT pool (clients 45-59)...")
    print("      - IID split, identity mapping, fixed at runtime")
    
    clients_imgs, clients_lbls = get_iid_class_and_labels(
        original_images_tr, 
        original_labels_tr, 
        N=CLIENTS_PER_TYPE,
        class_num=C
    )
    
    for i in range(CLIENTS_PER_TYPE):
        converted_imgs = [np.uint8(convert_img(img)) for img in clients_imgs[i]]
        all_clients_images.append(converted_imgs)
        all_clients_labels.append(clients_lbls[i])  # Identity mapping
        all_clients_types.append("no_drift")
    
    # ========================================================================
    # SAVE CLIENT FILES
    # ========================================================================
    print("\n" + "="*80)
    print("SAVING CLIENT FILES")
    print("="*80)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for i in range(len(all_clients_images)):
        client_data = {
            'images': np.uint8(np.array(all_clients_images[i])),
            'labels': np.uint8(np.array(all_clients_labels[i])),
            'drift_type': all_clients_types[i]
        }
        
        save_path = output_path / f"{i}.pkl"
        save_data(client_data, str(save_path))
        
        if i % 20 == 0:
            print(f"  Saved client {i:3d} - {all_clients_types[i]}")
    
    print(f"\n✓ Saved {len(all_clients_images)} client files to {output_dir}")
    
    # ========================================================================
    # SAVE TEST FILES (one per mapping type)
    # ========================================================================
    print("\n" + "="*80)
    print("SAVING TEST FILES")
    print("="*80)
    
    test_images = [np.uint8(convert_img(img)) for img in original_images_te]
    
    for mapping_type in range(1, 5):
        test_labels = apply_label_remapping(original_labels_te, mapping_type, C)
        
        test_data_dict = {
            'images': np.uint8(np.array(test_images)),
            'labels': np.uint8(np.array(test_labels)),
            'drift_type': f'test_map{mapping_type}'
        }
        
        save_path = output_path / f"test-{mapping_type}.pkl"
        save_data(test_data_dict, str(save_path))
        print(f"  Saved test-{mapping_type}.pkl (mapping type {mapping_type})")
    
    print("\n" + "="*80)
    print("GENERATION COMPLETE")
    print("="*80)
    print("\nDataset structure:")
    print("  Clients  0-14: REAL DRIFT (4 label mappings)")
    print("  Clients 15-29: FEATURE DRIFT ONLY (identity mapping)")
    print("  Clients 30-44: LABEL DRIFT ONLY (Dirichlet, identity mapping)")
    print("  Clients 45-59: NO DRIFT (IID, identity mapping)")
    print(f"\nTotal: {len(all_clients_images)} clients + 4 test files")
    print("="*80)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate multiclass drift detection dataset")
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./data/cifar10-c-60_client-multiclass-drift",
        help="Output directory for generated client files"
    )
    parser.add_argument(
        "--dirichlet_alpha",
        type=float,
        default=0.5,
        help="Dirichlet alpha for label-drift clients (lower = more skewed)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    generate_multiclass_drift_dataset(args.output_dir, args.dirichlet_alpha)

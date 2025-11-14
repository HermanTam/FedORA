import os
import pickle
import string
import bisect

import torch
from torchvision.datasets import CIFAR10, CIFAR100, EMNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
from PIL import Image


class TabularDataset(Dataset):
    """
    Constructs a torch.utils.Dataset object from a pickle file;
    expects pickle file stores tuples of the form (x, y) where x is vector and y is a scalar

    Attributes
    ----------
    data: iterable of tuples (x, y)

    Methods
    -------
    __init__
    __len__
    __getitem__
    """

    def __init__(self, path):
        """
        :param path: path to .pkl file
        """
        with open(path, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.int64), idx


class SubImagenetC(Dataset):
    """
    Constructs a subset of CIFAR10-C dataset from a npy file;
    expects pickle file to store list of indices

    Attributes
    ----------
    indices: iterable of integers
    transform
    data
    targets

    Methods
    -------
    __init__
    __len__
    __getitem__
    """
    def __init__(self, indices, cifar10_data=None, cifar10_targets=None, transform=None):
        """
        :param path: path to .pkl file; expected to store list of indices
        :param cifar10_data: Cifar-10 dataset inputs stored as torch.tensor
        :param cifar10_targets: Cifar-10 dataset labels stored as torch.tensor
        :param transform:
        """

        self.indices = indices

        if transform is None:
            self.transform = \
                Compose([
                    ToTensor(),
                    Normalize(
                        (0.4802, 0.4481, 0.3975),
                        (0.2770, 0.2691, 0.2821)
                    )
                ])

        if cifar10_data is None or cifar10_targets is None:
            self.data, self.targets = get_cifar10()
        else:
            self.data, self.targets = cifar10_data, cifar10_targets

        self.data = self.data[self.indices]
        self.targets = self.targets[self.indices]
        # print(self.targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        target = target

        return img, target, index

class SubCIFAR10C(Dataset):
    """
    Constructs a subset of CIFAR10-C dataset from a npy file;
    expects pickle file to store list of indices

    Attributes
    ----------
    indices: iterable of integers
    transform
    data
    targets

    Methods
    -------
    __init__
    __len__
    __getitem__
    """
    def __init__(self, indices, cifar10_data=None, cifar10_targets=None, transform=None):
        """
        :param path: path to .pkl file; expected to store list of indices
        :param cifar10_data: Cifar-10 dataset inputs stored as torch.tensor
        :param cifar10_targets: Cifar-10 dataset labels stored as torch.tensor
        :param transform:
        """

        self.indices = indices

        if transform is None:
            self.transform = \
                Compose([
                    ToTensor(),
                    Normalize(
                        (0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)
                    )
                ])
        else:
            self.transform = transform

        if cifar10_data is None or cifar10_targets is None:
            self.data, self.targets = get_cifar10()
        else:
            self.data, self.targets = cifar10_data, cifar10_targets

        self.data = self.data[self.indices]
        self.targets = self.targets[self.indices]
        # print(self.targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        target = target

        return img, target, index

class SubPowerSupply(Dataset):
    """
    Constructs a subset of FEMNIST dataset corresponding to one client;
    Initialized with the path to a `.pt` file;
    `.pt` file is expected to hold a tuple of tensors (data, targets) storing the images and there corresponding labels.

    Attributes
    ----------
    transform
    data: iterable of integers
    targets

    Methods
    -------
    __init__
    __len__
    __getitem__
    """
    def __init__(self, indices, cifar10_data=None, cifar10_targets=None, transform=None):
        if transform is None:
            self.transform = Compose([
                ToTensor()
            ])

        self.indices = indices

        if cifar10_data is None or cifar10_targets is None:
            self.data, self.targets = get_cifar10()
        else:
            self.data, self.targets = cifar10_data, cifar10_targets
        
        self.data = self.data[self.indices]
        self.targets = self.targets[self.indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # print(img)

        # img = np.uint8(img * 255)
        # img = Image.fromarray(img, mode='L')
        # print(img)

        

        return torch.tensor(img), target, index

class SubFEMNISTC(Dataset):
    """
    Constructs a subset of FEMNIST dataset corresponding to one client;
    Initialized with the path to a `.pt` file;
    `.pt` file is expected to hold a tuple of tensors (data, targets) storing the images and there corresponding labels.

    Attributes
    ----------
    transform
    data: iterable of integers
    targets

    Methods
    -------
    __init__
    __len__
    __getitem__
    """
    def __init__(self, indices, cifar10_data=None, cifar10_targets=None, transform=None):
        if transform is None:
            self.transform = Compose([
                ToTensor(),
                Normalize((0.1307,), (0.3081,))
                # Normalize((0.5,), (0.5,))
            ])
        else:
            self.transform = transform #added

        self.indices = indices

        if cifar10_data is None or cifar10_targets is None:
            self.data, self.targets = get_cifar10()
        else:
            self.data, self.targets = cifar10_data, cifar10_targets
        
        self.data = self.data[self.indices]
        self.targets = self.targets[self.indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # print(img)

        # img = np.uint8(img * 255)
        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index

class SubFEMNIST(Dataset):
    """
    Constructs a subset of FEMNIST dataset corresponding to one client;
    Initialized with the path to a `.pt` file;
    `.pt` file is expected to hold a tuple of tensors (data, targets) storing the images and there corresponding labels.

    Attributes
    ----------
    transform
    data: iterable of integers
    targets

    Methods
    -------
    __init__
    __len__
    __getitem__
    """
    def __init__(self, path):
        self.transform = Compose([
            ToTensor(),
            Normalize((0.1307,), (0.3081,))
        ])

        self.data, self.targets = torch.load(path)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        img = np.uint8(img.numpy() * 255)
        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index


class SubEMNIST(Dataset):
    """
    Constructs a subset of EMNIST dataset from a pickle file;
    expects pickle file to store list of indices

    Attributes
    ----------
    indices: iterable of integers
    transform
    data
    targets

    Methods
    -------
    __init__
    __len__
    __getitem__
    """

    def __init__(self, path, emnist_data=None, emnist_targets=None, transform=None):
        """
        :param path: path to .pkl file; expected to store list of indices
        :param emnist_data: EMNIST dataset inputs
        :param emnist_targets: EMNIST dataset labels
        :param transform:
        """
        with open(path, "rb") as f:
            self.indices = pickle.load(f)

        if transform is None:
            self.transform =\
                Compose([
                    ToTensor(),
                    Normalize((0.1307,), (0.3081,))
                ])

        if emnist_data is None or emnist_targets is None:
            self.data, self.targets = get_emnist()
        else:
            self.data, self.targets = emnist_data, emnist_targets

        self.data = self.data[self.indices]
        self.targets = self.targets[self.indices]

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index


class SubCIFAR10(Dataset):
    """
    Constructs a subset of CIFAR10 dataset from a pickle file;
    expects pickle file to store list of indices

    Attributes
    ----------
    indices: iterable of integers
    transform
    data
    targets

    Methods
    -------
    __init__
    __len__
    __getitem__
    """
    def __init__(self, path, cifar10_data=None, cifar10_targets=None, transform=None):
        """
        :param path: path to .pkl file; expected to store list of indices
        :param cifar10_data: Cifar-10 dataset inputs stored as torch.tensor
        :param cifar10_targets: Cifar-10 dataset labels stored as torch.tensor
        :param transform:
        """
        with open(path, "rb") as f:
            self.indices = pickle.load(f)

        if transform is None:
            self.transform = \
                Compose([
                    ToTensor(),
                    Normalize(
                        (0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)
                    )
                ])

        if cifar10_data is None or cifar10_targets is None:
            self.data, self.targets = get_cifar10()
        else:
            self.data, self.targets = cifar10_data, cifar10_targets

        self.data = self.data[self.indices]
        self.targets = self.targets[self.indices]

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img.numpy())

        if self.transform is not None:
            img = self.transform(img)

        target = target

        return img, target, index


class SubCIFAR100(Dataset):
    """
    Constructs a subset of CIFAR100 dataset from a pickle file;
    expects pickle file to store list of indices

    Attributes
    ----------
    indices: iterable of integers
    transform
    data
    targets

    Methods
    -------
    __init__
    __len__
    __getitem__
    """
    def __init__(self, path, cifar100_data=None, cifar100_targets=None, transform=None):
        """
        :param path: path to .pkl file; expected to store list of indices:
        :param cifar100_data: CIFAR-100 dataset inputs
        :param cifar100_targets: CIFAR-100 dataset labels
        :param transform:
        """
        with open(path, "rb") as f:
            self.indices = pickle.load(f)

        if transform is None:
            self.transform = \
                Compose([
                    ToTensor(),
                    Normalize(
                        (0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)
                    )
                ])

        if cifar100_data is None or cifar100_targets is None:
            self.data, self.targets = get_cifar100()

        else:
            self.data, self.targets = cifar100_data, cifar100_targets

        self.data = self.data[self.indices]
        self.targets = self.targets[self.indices]

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img.numpy())

        if self.transform is not None:
            img = self.transform(img)

        target = target

        return img, target, index


class CharacterDataset(Dataset):
    def __init__(self, file_path, chunk_len):
        """
        Dataset for next character prediction, each sample represents an input sequence of characters
         and a target sequence of characters representing to next sequence of the input
        :param file_path: path to .txt file containing the training corpus
        :param chunk_len: (int) the length of the input and target sequences
        """
        self.all_characters = string.printable
        self.vocab_size = len(self.all_characters)
        self.n_characters = len(self.all_characters)
        self.chunk_len = chunk_len

        with open(file_path, 'r') as f:
            self.text = f.read()

        self.tokenized_text = torch.zeros(len(self.text), dtype=torch.long)

        self.inputs = torch.zeros(self.__len__(), self.chunk_len, dtype=torch.long)
        self.targets = torch.zeros(self.__len__(), self.chunk_len, dtype=torch.long)

        self.__build_mapping()
        self.__tokenize()
        self.__preprocess_data()

    def __tokenize(self):
        for ii, char in enumerate(self.text):
            self.tokenized_text[ii] = self.char2idx[char]

    def __build_mapping(self):
        self.char2idx = dict()
        for ii, char in enumerate(self.all_characters):
            self.char2idx[char] = ii

    def __preprocess_data(self):
        for idx in range(self.__len__()):
            self.inputs[idx] = self.tokenized_text[idx:idx+self.chunk_len]
            self.targets[idx] = self.tokenized_text[idx+1:idx+self.chunk_len+1]

    def __len__(self):
        return max(0, len(self.text) - self.chunk_len)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], idx

class MergedDataset(Dataset):
    def __init__(self, datasets,transform=None):
        self.datasets = datasets
        self.data = []
        self.targets = []

        for dataset in datasets:
            self.data.append(dataset.data)
            self.targets.append(dataset.targets)


        self.data = np.concatenate(self.data)  
        self.targets = np.concatenate(self.targets)  


        # self.data = torch.tensor(self.data)
        # self.targets = torch.tensor(self.targets)

        if transform is None:
            self.transform = \
                Compose([
                    ToTensor(),
                    Normalize(
                        (0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)
                    )
                ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        target = target

        return img, target, index

class RotateMergedDataset(Dataset):
    def __init__(self, datasets,rotate_degrees,transform=None):
        self.datasets = datasets
        self.data = []
        self.targets = []

        for i, dataset in enumerate(datasets):

            if i == 1:

                rotated_images = []
                for img_array in dataset.data:
                    img = Image.fromarray(img_array)  
                    rotated_img = img.rotate(rotate_degrees)
                    rotated_images.append(np.array(rotated_img))  


                self.data.append(rotated_images)
                self.targets.append(dataset.targets)
            else:

                rotated_images = []
                for img_array in dataset.data:
                    img = Image.fromarray(img_array)  
                    rotated_img = img.rotate(rotate_degrees-90)  

                    rotated_images.append(np.array(rotated_img))  

                self.data.append(rotated_images)
                self.targets.append(dataset.targets)


        self.data = np.concatenate(self.data)  
        self.targets = np.concatenate(self.targets)  


        # self.data = torch.tensor(self.data)
        # self.targets = torch.tensor(self.targets)

        if transform is None:
            self.transform = \
                Compose([
                    ToTensor(),
                    Normalize(
                        (0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)
                    )
                ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        target = target

        return img, target, index

class Rotate120MergedDataset(Dataset):
    """
    Merge multiple datasets and apply rotation to all samples.
    
    Handles both regular datasets (with .data attribute) and BufferedDataset (ER buffers).
    Regular datasets are pre-computed for efficiency, while BufferedDataset samples
    are fetched on-demand to save memory.
    
    Rotation logic:
    - Dataset at index 1: rotated by rotate_degrees
    - All other datasets: rotated by (rotate_degrees - 120)
    """
    def __init__(self, datasets, rotate_degrees, transform=None):
        """
        Parameters
        ----------
        datasets : list of Dataset
            Can be regular datasets (with .data attribute) or BufferedDataset (ER buffers)
        rotate_degrees : int
            Rotation angle (0, 120, 240) - applied to dataset at index 1
            Other datasets get rotated by (rotate_degrees - 120)
        transform : optional
            Transform to apply after rotation
        """
        self.datasets = datasets
        self.rotate_degrees = rotate_degrees
        self.transform = transform  # Store transform
        self.cumulative_sizes = self._cumsum([len(d) for d in datasets])
        
        # Precompute rotated data for regular datasets (optimization)
        self.rotated_data = []
        self.rotated_targets = []
        self.dataset_types = []  # Track which datasets are pre-computed vs on-demand
        
        for i, dataset in enumerate(datasets):
            # Determine rotation for this dataset index
            if i == 1:
                dataset_rotation = rotate_degrees
            else:
                dataset_rotation = rotate_degrees - 120
            
            # Check if dataset has .data attribute (regular CIFAR dataset)
            if hasattr(dataset, 'data') and hasattr(dataset, 'targets'):
                # Regular dataset - precompute rotations
                rotated_images = []
                for img_array in dataset.data:
                    img = Image.fromarray(img_array)
                    rotated_img = img.rotate(dataset_rotation, expand=False)
                    rotated_images.append(np.array(rotated_img))
                
                self.rotated_data.append(np.array(rotated_images))
                self.rotated_targets.append(np.array(dataset.targets))
                self.dataset_types.append('precomputed')
            else:
                # BufferedDataset (ER buffer) - mark as None for on-demand fetching
                self.rotated_data.append(None)
                self.rotated_targets.append(None)
                self.dataset_types.append(('ondemand', dataset_rotation))
        
        # Set default transform if none provided
        if self.transform is None:
            self.transform = Compose([
                ToTensor(),
                Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)
                )
            ])
    
    @staticmethod
    def _cumsum(sequence):
        """Calculate cumulative sum for efficient dataset indexing."""
        r, s = [], 0
        for e in sequence:
            r.append(e + s)
            s += e
        return r
    
    def __len__(self):
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        # Find which dataset this index belongs to
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        sample_idx = idx - (self.cumulative_sizes[dataset_idx - 1] if dataset_idx > 0 else 0)
        
        # Get sample from appropriate source
        if self.dataset_types[dataset_idx] == 'precomputed':
            # Precomputed rotation (regular dataset)
            img = self.rotated_data[dataset_idx][sample_idx]
            target = self.rotated_targets[dataset_idx][sample_idx]
            img = Image.fromarray(img)
        else:
            # On-demand fetch (BufferedDataset from ER)
            dataset_rotation = self.dataset_types[dataset_idx][1]
            sample = self.datasets[dataset_idx][sample_idx]
            
            # Handle tuple unpacking (BufferedDataset returns (data, label) or (data, label, metadata))
            if isinstance(sample, tuple):
                if len(sample) == 2:
                    img, target = sample
                elif len(sample) >= 3:
                    img, target, *_ = sample  # Discard metadata using extended unpacking
                else:
                    raise ValueError(f"Unexpected sample format: {len(sample)} values")
            else:
                raise ValueError("BufferedDataset must return tuple")
            
            # Apply rotation if needed
            if dataset_rotation != 0:
                # If img is tensor, convert to PIL
                if isinstance(img, torch.Tensor):
                    if img.dim() == 3 and img.size(0) in [1, 3]:
                        # CHW format - denormalize before converting to PIL
                        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
                        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
                        denormalized = img * std + mean
                        denormalized = torch.clamp(denormalized, 0, 1)
                        img = transforms.ToPILImage()(denormalized)
                    else:
                        raise ValueError(f"Unexpected tensor shape for rotation: {img.shape}")
                
                # Rotate PIL image
                if isinstance(img, Image.Image):
                    img = img.rotate(dataset_rotation, expand=False)
                elif isinstance(img, np.ndarray):
                    # If still numpy array, convert to PIL first
                    img = Image.fromarray(img)
                    img = img.rotate(dataset_rotation, expand=False)
        
        # Apply transforms
        if hasattr(self.datasets[dataset_idx], 'transform') and \
           self.datasets[dataset_idx].transform is not None:
            img = self.datasets[dataset_idx].transform(img)
        elif self.transform is not None:
            img = self.transform(img)
        
        # Handle target conversion
        if isinstance(target, torch.Tensor):
            target = target.item() if target.numel() == 1 else target
        
        # Return 3-tuple to match original behavior (img, target, index)
        return img, target, idx
class Rotate120MergedDatasetFmnist(Dataset):
    def __init__(self, datasets,rotate_degrees,transform=None):
        self.datasets = datasets
        self.data = []
        self.targets = []

        for i, dataset in enumerate(datasets):
            
            if i == 1:
                
                rotated_images = []
                for img_array in dataset.data:
                    img = Image.fromarray(img_array)  
                    rotated_img = img.rotate(rotate_degrees)
                    rotated_images.append(np.array(rotated_img))  

                
                self.data.append(rotated_images)
                self.targets.append(dataset.targets)
            else:
                
                rotated_images = []
                for img_array in dataset.data:
                    img = Image.fromarray(img_array)  
                    rotated_img = img.rotate(rotate_degrees-120)  
                    
                    rotated_images.append(np.array(rotated_img))  

                self.data.append(rotated_images)
                self.targets.append(dataset.targets)


        self.data = np.concatenate(self.data)  
        self.targets = np.concatenate(self.targets)  

        
        # self.data = torch.tensor(self.data)
        # self.targets = torch.tensor(self.targets)

        if transform is None:
            self.transform = \
                Compose([
                    ToTensor(),
                    Normalize((0.1307,), (0.3081,))
                    # Normalize((0.5,), (0.5,))
                ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        target = target

        return img, target, index

def get_emnist():
    """
    gets full (both train and test) EMNIST dataset inputs and labels;
    the dataset should be first downloaded (see data/emnist/README.md)
    :return:
        emnist_data, emnist_targets
    """
    emnist_path = os.path.join("data", "emnist", "raw_data")
    assert os.path.isdir(emnist_path), "Download EMNIST dataset!!"

    emnist_train =\
        EMNIST(
            root=emnist_path,
            split="byclass",
            download=True,
            train=True
        )

    emnist_test =\
        EMNIST(
            root=emnist_path,
            split="byclass",
            download=True,
            train=True
        )

    emnist_data =\
        torch.cat([
            emnist_train.data,
            emnist_test.data
        ])

    emnist_targets =\
        torch.cat([
            emnist_train.targets,
            emnist_test.targets
        ])

    return emnist_data, emnist_targets


def get_cifar10():
    """
    gets full (both train and test) CIFAR10 dataset inputs and labels;
    the dataset should be first downloaded (see data/emnist/README.md)
    :return:
        cifar10_data, cifar10_targets
    """
    cifar10_path = os.path.join("data", "cifar10", "raw_data")
    assert os.path.isdir(cifar10_path), "Download cifar10 dataset!!"

    cifar10_train =\
        CIFAR10(
            root=cifar10_path,
            train=True, download=False
        )

    cifar10_test =\
        CIFAR10(
            root=cifar10_path,
            train=False,
            download=False)

    cifar10_data = \
        torch.cat([
            torch.tensor(cifar10_train.data),
            torch.tensor(cifar10_test.data)
        ])

    cifar10_targets = \
        torch.cat([
            torch.tensor(cifar10_train.targets),
            torch.tensor(cifar10_test.targets)
        ])

    return cifar10_data, cifar10_targets


def get_cifar100():
    """
    gets full (both train and test) CIFAR100 dataset inputs and labels;
    the dataset should be first downloaded (see data/cifar100/README.md)
    :return:
        cifar100_data, cifar100_targets
    """
    cifar100_path = os.path.join("data", "cifar100", "raw_data")
    assert os.path.isdir(cifar100_path), "Download cifar10 dataset!!"

    cifar100_train =\
        CIFAR100(
            root=cifar100_path,
            train=True, download=False
        )

    cifar100_test =\
        CIFAR100(
            root=cifar100_path,
            train=False,
            download=False)

    cifar100_data = \
        torch.cat([
            torch.tensor(cifar100_train.data),
            torch.tensor(cifar100_test.data)
        ])

    cifar100_targets = \
        torch.cat([
            torch.tensor(cifar100_train.targets),
            torch.tensor(cifar100_test.targets)
        ])

    return cifar100_data, cifar100_targets

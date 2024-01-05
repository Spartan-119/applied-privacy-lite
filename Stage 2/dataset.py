import torch
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST

def get_mnist(data_path: str = "./data"):
    """
    Download MNIST dataset and apply minimal transformation.

    Args:
        data_path: Path to store the dataset.

    Returns:
        trainset: MNIST training dataset.
        testset: MNIST test dataset.
    """
    # Define data transformation
    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    # Download MNIST datasets
    trainset = MNIST(data_path, train=True, download=True, transform=transform)
    testset = MNIST(data_path, train=False, download=True, transform=transform)

    return trainset, testset

def prepare_dataset(num_partitions: int, batch_size: int, val_ratio: float = 0.1):
    """
    Download MNIST dataset and generate IID partitions for federated learning.

    Args:
        num_partitions: Number of partitions (clients) for federated learning.
        batch_size: Batch size for data loaders.
        val_ratio: Ratio of training examples to use for validation.

    Returns:
        trainloaders: List of training data loaders for each client.
        valloaders: List of validation data loaders for each client.
        testloader: DataLoader for the test dataset.
    """
    # Download MNIST dataset
    trainset, testset = get_mnist()

    # Determine number of training examples per partition
    num_images = len(trainset) // num_partitions

    # List of partition lengths (all partitions are of equal size)
    partition_len = [num_images] * num_partitions

    # Randomly split the trainset into partitions
    trainsets = random_split(
        trainset, partition_len, generator=torch.Generator().manual_seed(2023)
    )

    # Create dataloaders with train+val support
    trainloaders = []
    valloaders = []

    # For each train set, put aside some training examples for validation
    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(
            trainset_, [num_train, num_val], generator=torch.Generator().manual_seed(2023)
        )

        # Construct data loaders and append to their respective list.
        # In this way, each client will get the corresponding loader
        trainloaders.append(
            DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2)
        )
        valloaders.append(
            DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2)
        )

    # Leave the test set intact (not partitioned)
    testloader = DataLoader(testset, batch_size=128)

    return trainloaders, valloaders, testloader

import torch
import torchvision
import torchvision.transforms as transforms

def mnist_loader(batch_size, download_path="data"):
    """Returns a training dataset loader and a test dataset loader.

    Args:
        batch_size (int): The number of images per batch.
        download_path (string): The path of MNIST data

    Example:
        >>> test_loader, train_loader = mnist_loader(100, "./data")
    """
    train_dataset = torchvision.datasets.MNIST(download_path, train=True, download=True, transform=transforms.ToTensor())
    test_dataset = torchvision.datasets.MNIST(download_path, train=False, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size)
    return train_loader, test_loader

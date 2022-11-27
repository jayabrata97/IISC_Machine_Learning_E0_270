import torch
import torchvision
import torchvision.transforms as transforms

def cifar_loader(batch_size, download_path="data"):
    """
    Returns a training dataset loader and a test dataset loader

    batch_size: The number of images per batch.

    TODO: Write example usage
    """
    train_dataset = torchvision.datasets.CIFAR10(download_path, train=True, download=True, transform=transforms.ToTensor())
    test_dataset = torchvision.datasets.CIFAR10(download_path, train=False, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size)
    return train_loader, test_loader

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    train_loader, test_loader = cifar_loader(100)
    plt.imshow(train_loader.dataset[1][0].permute(1,2,0).detach().numpy())
    plt.show()

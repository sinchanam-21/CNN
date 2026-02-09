import torch
import torchvision
import torchvision.transforms as transforms
import ssl

# Disable SSL verification to avoid potential certificate errors during download
ssl._create_default_https_context = ssl._create_unverified_context


def get_dataloaders(batch_size=32, num_workers=0):
    """
    Creates and returns the data loaders for CIFAR-10 dataset.
    
    Args:
        batch_size (int): Batch size for training and testing.
        num_workers (int): Number of subprocesses to use for data loading.
        
    Returns:
        tuple: (trainloader, testloader, classes)
    """
    transform_train = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.RandomRotation(10),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)


    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
               
    return trainloader, testloader, classes

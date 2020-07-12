import torch
import torchvision
import matplotlib.pyplot as plt
import torch.utils.data.dataloader as dataloader
import numpy as np
from src import utils

""" N-way Omniglot Task Setup """

# 0.1 dataset-augmentation: random rotations of 90 degrees
data = torchvision.datasets.Omniglot(
    root="./data", download=True, transform=torchvision.transforms.ToTensor()
)
print(data._characters[:2])
degs = [90, 180, 270]
rotations = [torchvision.transforms.RandomAffine((deg, deg)) for deg in degs]
rand_rotations = torchvision.transforms.RandomChoice(rotations)
augmented = torchvision.datasets.Omniglot(
    root="./data", download=True,
    transform=torchvision.transforms.Compose([rand_rotations] + [torchvision.transforms.ToTensor()])
)
dataset = torch.utils.data.ConcatDataset([data, augmented])


def some_plots():
    image, label = data[0]
    plt.imshow(image[0], cmap='gray')
    plt.show()
    image, _ = augmented[0]
    plt.imshow(image[0], cmap='gray')
    plt.show()
    image, _ = dataset[0]
    plt.imshow(image[0], cmap='gray')
    plt.show()
    image, _ = dataset[1]
    plt.imshow(image[0], cmap='gray')
    plt.show()


# 0.2 1200 characters for training, rest for eval
# data_loader = dataloader.DataLoader(dataset)
batch_size = 2
test_split = .25
# shuffle_dataset = False  # False makes it easier to identify unused character classes
random_seed = 42

dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(test_split * dataset_size))
#
# if shuffle_dataset:
#     np.random.seed(random_seed)
#     np.random.shuffle(indices)


train_sampler = torch.utils.data.SubsetRandomSampler(indices[split:])
eval_sampler = torch.utils.data.SubsetRandomSampler(indices[:split])

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           sampler=train_sampler)
eval_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          sampler=eval_sampler)

images, targets = next(iter(train_loader))
images = images.squeeze()
plt.imshow(images[0], cmap='gray')
plt.show()
a = 1
# 1. pick N unseen character classes, independent of alphabet, as L
# 2. provide the model with one drawing of each of the N characters as samples S~L
# 3. and a batch B~L of samples

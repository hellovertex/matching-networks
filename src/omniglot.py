import torch
import torchvision
import matplotlib.pyplot as plt
import torch.utils.data.dataloader as dataloader
import numpy as np
from PIL import Image
from os.path import join
import os
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive, check_integrity, list_dir, list_files
from pytorch_metric_learning import samplers

""" N-way-k-shot Omniglot Data Setup """
# N = 5
# k = 1
degs = [90, 180, 270]  # rotations of class examples to augment data

optional_transforms = [torchvision.transforms.CenterCrop(size=28)]


class OmniglotDataSet(VisionDataset):
    """`Omniglot <https://github.com/brendenlake/omniglot>`_ Dataset.
    Custom implementation based on torchvision.datasets.OmniGlot
    Args:
        root (string): Root directory of dataset where directory
            ``omniglot-py`` exists.
        background (bool, optional): If True, creates dataset from the "background" set, otherwise
            creates from the "evaluation" set. This terminology is defined by the authors.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset zip files from the internet and
            puts it in root directory. If the zip files are already downloaded, they are not
            downloaded again.
    """
    folder = 'omniglot-py'
    download_url_prefix = 'https://github.com/brendenlake/omniglot/raw/master/python'
    zips_md5 = {
        'images_background': '68d2efa1b9178cc56df9314c21c6e718',
        'images_evaluation': '6b91aef0f799c5bb55b94e3f2daec811'
    }

    # NUM_TOTAL_EXAMPLES = 1623 * 20
    # NUM_TRAIN_EXAMPLES = 1200 * 20
    # NUM_EVAL_EXAMPLES = NUM_TOTAL_EXAMPLES - NUM_TRAIN_EXAMPLES

    def __init__(self, root, background=True, transform=None, target_transform=None,
                 download=False, training=True, train_set_classes=1200):
        super(OmniglotDataSet, self).__init__(join(root, self.folder), transform=transform,
                                              target_transform=target_transform)
        self.background = background

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.target_folder = join(self.root, self._get_target_folder())
        self._alphabets = list_dir(self.target_folder)
        self._characters = sum([[join(a, c) for c in list_dir(join(self.target_folder, a))]
                                for a in self._alphabets], [])
        if training:
            self._character_images = [[(image, idx) for image in
                                       list_files(join(self.target_folder, character), '.png')]
                                      for idx, character in enumerate(self._characters)][:train_set_classes]
        else:
            self._character_images = [[(image, idx) for image in
                                       list_files(join(self.target_folder, character), '.png')]
                                      for idx, character in enumerate(self._characters)][train_set_classes:]
        self._flat_character_images = sum(self._character_images, [])

    def __len__(self):
        return len(self._flat_character_images)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        image_name, character_class = self._flat_character_images[index]
        image_path = join(self.target_folder, self._characters[character_class], image_name)
        image = Image.open(image_path, mode='r').convert('L')

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            character_class = self.target_transform(character_class)

        return image, character_class

    def _check_integrity(self):
        zip_filename = self._get_target_folder()
        if not check_integrity(join(self.root, zip_filename + '.zip'), self.zips_md5[zip_filename]):
            return False
        return True

    def download(self, prints=False):
        if self._check_integrity():
            if prints:
                print('Files already downloaded and verified')
            return

        filename = self._get_target_folder()
        zip_filename = filename + '.zip'
        url = self.download_url_prefix + '/' + zip_filename
        download_and_extract_archive(url, self.root, filename=zip_filename, md5=self.zips_md5[filename])

    def _get_target_folder(self):
        return 'images_background' if self.background else 'images_evaluation'


# 0. Load data for first 1200 classes
training_data = OmniglotDataSet(
    root="./data", download=True, training=True, train_set_classes=1200,
    transform=torchvision.transforms.Compose(optional_transforms + [torchvision.transforms.ToTensor()])
)


def add_rotated_classes(data, rotations_at_random=False):
    # rotate each class by [90, 180, 270]
    rotations = [torchvision.transforms.RandomAffine((deg, deg)) for deg in degs]
    if rotations_at_random:  # only one out of [90, 180, 270] will be selected for each class
        rotations = torchvision.transforms.RandomChoice(rotations)
    augmented = data
    for rot in rotations:
        rotated_data = OmniglotDataSet(
            root="./data", download=True, training=True,
            transform=torchvision.transforms.Compose(optional_transforms + [rot] + [torchvision.transforms.ToTensor()])
        )
        # index % 1200 + (4800 / index) * 1200
        augmented = torch.utils.data.ConcatDataset([augmented, rotated_data])
    return augmented


# 1. rotate classes to augment dataset to size 4 * len(training_data)
augmented_dataset = add_rotated_classes(training_data)


# 2. provide the model with one drawing of each of the N characters as samples S
# 3. and a batch B of unlabelled examples
def get_data_loader(dataset, num_classes=1200, N=5, k=1):
    """ B is generated implicitly by calling this function with k = k + size of B """
    # labels for base dataset
    labels = np.array([np.repeat([i], 20) for i in range(num_classes)]).flatten()
    # labels for augmented dataset (including rotations)
    labels = np.tile(labels, len(degs) + 1)
    assert len(labels) == len(
        dataset), 'MPerClassSampler requires labels[x] should be the label of the xth element in your dataset'
    sampler = samplers.MPerClassSampler(labels, m=k)

    return torch.utils.data.DataLoader(augmented_dataset, batch_size=N * k,  # N*k required by MPerSampler
                                       sampler=sampler)


def some_plots():
    image, label = training_data[0]
    plt.imshow(image[0], cmap='gray')
    plt.show()
    image, _ = augmented_dataset[0]
    plt.imshow(image[0], cmap='gray')
    plt.show()
    image, _ = training_data[0]
    plt.imshow(image[0], cmap='gray')
    plt.show()
    image, _ = training_data[1]
    plt.imshow(image[0], cmap='gray')
    plt.show()

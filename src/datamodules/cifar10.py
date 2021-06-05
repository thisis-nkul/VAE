import pytorch_lightning as pl
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split

"""
Sample DataModule for CIFAR10 Dataset.
"""


class CIFAR10Data(pl.LightningDataModule):

    def __init__(self, data_dir='../../data', batch_size=128,
                 num_workers=2, shuffle_train=True):

        super(CIFAR10Data, self).__init__()

        self.batch_size = batch_size
        self.data_dir = data_dir
        self.nb_workers = num_workers
        self.shuffle = shuffle_train

        self.train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(45),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None):

        if stage in ('fit', None):
            CIFAR10_full = CIFAR10(self.data_dir,
                                   train=True,
                                   download=True,
                                   transform=self.train_transforms)

            self.train, self.val = random_split(CIFAR10_full,
                                                [56000, 4000])

        if stage in ('test', 'fit'):
            self.test = CIFAR10(self.data_dir, train=False,
                                download=True, transform=self.test_transforms)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size,
                          shuffle=self.shuffle, num_workers=self.nb_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size,
                          num_workers=self.nb_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size,
                          num_workers=self.nb_workers)

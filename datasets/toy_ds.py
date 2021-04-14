import os
import sys

import torch
import torch.nn
import numpy as np

from torch.utils.data import Dataset

class ToyDS(Dataset):
    def __init__(self, shape=[3,256,256], n_class=10, n_num=1000):
        self.shape = shape
        self.n_class = n_class
        self.n_num = n_num

    def __len__(self):
        return self.n_num

    def __getitem__(self, item):
        image_aug1 = torch.randn(self.shape)
        image_aug2 = torch.randn(self.shape)
        label = np.random.randint(0, self.n_class)

        return [image_aug1, image_aug2, label]


def test_ToyDS():
    from torch.utils.data import DataLoader
    ds = ToyDS()
    dataloader = DataLoader(ds, batch_size=2)

    for index, (images) in enumerate(dataloader):
        if index > 10:
            break
        print(images.shape,'\t', labels)

if __name__ == '__main__':
    test_ToyDS()

        
import torchvision
from PIL import Image
from torchvision import transforms
import torch
import numpy as np

class BinarizedMNIST(torchvision.datasets.MNIST):
    def __init__(self, train, root_path):
        super(BinarizedMNIST, self).__init__(
            train=train, root=root_path, download=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.train:
            img, target = self.data[idx], self.targets[idx]
        else:
            img, target = self.test_data[idx], self.test_labels[idx]

        img = self.data[idx]
        img = Image.fromarray(img.numpy(), mode='L')
        img = transforms.ToTensor()(img)
        # TODO: Try torch.bernoulli()?
        img = img > 0.5
        img = img.float()
        return img, target

    def get_train_bias(self):
       return -np.log(1./np.clip(self.get_train_mean(), 0.001, 0.999)-1.)

    def get_train_mean(self):
        imgs = self.data[:len(self)].type(torch.float)
        mean_img = imgs.mean(0).reshape(-1).numpy()
        return mean_img

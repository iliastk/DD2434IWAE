import torchvision
from PIL import Image
from torchvision import transforms
import torch


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
        # TODO: Try torch.bernoulli()
        img = torch.bernoulli(img)
        return img, target

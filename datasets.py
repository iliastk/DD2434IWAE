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
        img, target = self.data[idx], self.targets[idx]

        img = self.data[idx]
        img = Image.fromarray(img.numpy(), mode='L')
        img = transforms.ToTensor()(img)
        img = img > 0.5
        img = img.float()
        # img = torch.bernoulli(img)
        return img, target

    def get_bias(self):
       mean_img = self.get_mean()
       mean_img = np.clip(mean_img, 1e-8, 1.0 - 1e-7)
       mean_img_logit = -np.log(1./ (1.0 - mean_img))
       return mean_img_logit

    def get_mean(self):
        imgs = self.data[:len(self)].type(torch.float) / 255
        mean_img = imgs.mean(0).reshape(-1).numpy()
        return mean_img

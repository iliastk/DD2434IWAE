import torchvision
from PIL import Image
from torchvision import transforms
import torch
import numpy as np


class BinarizedMNIST(torchvision.datasets.MNIST):
    def __init__(self, train, root_path, cut_off_0_5 = False):
        super(BinarizedMNIST, self).__init__(
            train=train, root=root_path, download=True)
        self.cut_off_0_5 = cut_off_0_5

        imgs = [Image.fromarray(img.numpy(), mode='L')
                for img in self.data
                ]
        tensors = [transforms.ToTensor()(img) for img in imgs]
        tensor = torch.stack(tensors)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        tensor.to(self.device)
        if self.cut_off_0_5:
            tensor = tensor > 0.5
        sz, a, n, m = tensor.size()
        assert sz == len(self) and n == 28 and m == 28 and a == 1
        self.tensor = tensor.view(sz, 28*28)
                

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.tensor[idx]
        if self.cut_off_0_5: return img
        else:
            img = torch.bernoulli(img)
        return img, 0

    def get_bias(self):
        mean_img = self.get_mean()
        mean_img = np.clip(mean_img, 1e-8, 1.0 - 1e-7)
        mean_img_logit = -np.log(1. / (1.0 - mean_img))
        return mean_img_logit

    def get_mean(self):
        imgs = self.data[:len(self)].type(torch.float) / 255
        mean_img = imgs.mean(0).reshape(-1).numpy()
        return mean_img

    
def estimate_log_likelihood(model, root_path, num_samples=5000,
                                num_points=None, binarize='bernoulli'):
        """
        Allowed values for `binarize` are: 'bernoulli', 'cut-off-0.5', 'no'
        """
        import evaluation
        ds_test = None
        if binarize in ['bernoulli', 'cut-off-0.5']:
            ds_test = BinarizedMNIST(train=False, root_path=root_path,
                                     cut_off_0_5 = (binarize == 'cut-off-0.5'))
        elif binarize == 'no':
            ds_test = torchvision.datasets.MNIST(
            train=False, root=root_path, download=True)
        assert ds_test is not None
                   
        ds_test = BinarizedMNIST(train=False, root_path=root_path)
        
        data_loader = torch.utils.data.DataLoader(
                dataset=ds_test, batch_size=20,
                shuffle=False, num_workers=1)
        if num_points is None:
            num_points = len(ds_test)
        assert num_points <= len(ds_test)
        
        return evaluation.measure_estimated_log_likelihood(
            data_loader, model,
            num_samples, num_points)
        
            
# NOT the same as authors data, gotta find it
class BinarizedOMNIGLOT(torchvision.datasets.Omniglot):  # train: 24,345, test: 8,070
    def __init__(self, root_path):
        super(BinarizedOMNIGLOT, self).__init__(root=root_path, download=True)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]

        img = self.data[idx]
        img = Image.fromarray(img.numpy(), mode='L')
        img = transforms.ToTensor()(img)
        # img = img > 0.5
        # img = img.float()
        img = torch.bernoulli(img)
        return img, target

    def get_bias(self):
        mean_img = self.get_mean()
        mean_img = np.clip(mean_img, 1e-8, 1.0 - 1e-7)
        mean_img_logit = -np.log(1. / (1.0 - mean_img))
        return mean_img_logit

    def get_mean(self):
        imgs = self.data[:len(self)].type(torch.float) / 255
        mean_img = imgs.mean(0).reshape(-1).numpy()
        return mean_img

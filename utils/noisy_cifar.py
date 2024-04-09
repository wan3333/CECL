import numpy as np
import torch
from torchvision.datasets import CIFAR100
from PIL import Image

class ClenCIFAR100(CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False,
                 openset_ratio=0.2):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform,
                         download=download)

        self.nb_classes = 100
        if self.train:
            res = torch.load('./res_stage1/res_cifar100_symmetric_0.2.pkl')
            r1 = res['r1'].cpu().numpy()
            c1, c2 = res['c1'].cpu().numpy(), res['c2'].cpu().numpy()
            ch1, ch2 = res['ch1'].cpu().numpy(), res['ch2'].cpu().numpy()
            ch = ((ch1 == ch2) * (ch1 == True) * (ch2 == True))
            self.not_change = c1
            self.is_choosed = ch1
            self.is_agree = ch

            self.data = self.data[c1]
            self.noisy_labels = r1[c1]
            true_labels = np.array(self.targets)[c1]

            isood = true_labels >= int(self.nb_classes * (1 - openset_ratio))
            self.ground_truth_labels = np.array(self.targets)[c1]
            self.ground_truth_labels[isood] = -1


        else:
            self.targets = np.array(self.targets)
            id_index = self.targets < int(self.nb_classes * (1 - openset_ratio))
            self.data = self.data[id_index]
            self.targets = self.targets[id_index]

        self.num_samples = len(self.data)

    def __getitem__(self, index):
        if self.train:
            img, target, true_label = self.data[index], self.noisy_labels[index], self.ground_truth_labels[index]
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)
            return index, img, target, true_label

        else:
            img, target = self.data[index], self.targets[index]
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return index, img, target

    def __len__(self):
        return len(self.data)


class NoisyCIFAR100(CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False,
                 openset_ratio=0.2):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform,
                         download=download)

        self.nb_classes = 100
        if self.train:
            res = torch.load('./save/res_cifar100_symmetric_0.2.pkl')
            r1 = res['r1'].cpu().numpy()
            c1, c2 = res['c1'].cpu().numpy(), res['c2'].cpu().numpy()
            ch1, ch2 = res['ch1'].cpu().numpy(), res['ch2'].cpu().numpy()
            ch = ((ch1 == ch2) * (ch1 == True) * (ch2 == True))
            self.not_change = c1
            self.is_choosed = ch1
            self.is_agree = ch

            # self.data = self.data
            self.noisy_labels = r1
            true_labels = np.array(self.targets)

            isood = true_labels >= int(self.nb_classes * (1 - openset_ratio))
            self.ground_truth_labels = np.array(self.targets)
            self.ground_truth_labels[isood] = -1


        else:
            self.targets = np.array(self.targets)
            id_index = self.targets < int(self.nb_classes * (1 - openset_ratio))
            self.data = self.data[id_index]
            self.targets = self.targets[id_index]

        self.num_samples = len(self.data)

    def __getitem__(self, index):
        if self.train:
            img, target, true_label = self.data[index], self.noisy_labels[index], self.ground_truth_labels[index]
            not_change = self.not_change[index]
            choosed = self.is_choosed[index]
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)
            return index, img, target, true_label, not_change, choosed

        else:
            img, target = self.data[index], self.targets[index]
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return index, img, target

    def __len__(self):
        return len(self.data)



# This is a modified version of original (cifar dataloader):
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/cifar.py

# This file and the mvtec data directory must be in the same directory, such that:
# /.../this_directory/mvtec.py
# /.../this_directory/mvtec/bottle/...
# /.../this_directory/mvtec/cable/...
# and so on

import os
import numpy as np
from PIL import Image
import matplotlib.image as mpimg
from typing import Any, Callable, Optional, Tuple

from torchvision import transforms
from torchvision.datasets.vision import VisionDataset


class MVTEC(VisionDataset):
    """`MVTEC <https://www.mvtec.com/company/research/datasets/mvtec-ad/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directories
            ``bottle``, ``cable``, etc., exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        resize (int, optional): Desired output image size (H=W=resize).
        interpolation (int, optional): Interpolation method for downsizing image. If 'resize'
            is not None, a value for interpolation must be provided.
            See https://pytorch.org/vision/main/_modules/torchvision/transforms/functional.html
        category (string, optional): bottle, cable, capsule, etc.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        resize: Optional[int] = None,
        interpolation: int = 2,
        category: str = 'carpet',
    ) -> None:
            
        super().__init__(root,
                         transform=transform,
                         target_transform=target_transform)
        
        self.root = os.path.expanduser(root)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.resize = resize
        self.interpolation = interpolation
        self.category = category

        self.data = []
        self.targets = []
        
        if self.train:
            # load images for training
            cwd = os.getcwd()
            trainFolder = os.path.join(self.root, self.category, 'train/good/')
            os.chdir(trainFolder)

            for file in os.scandir():
                img = mpimg.imread(file.name)
                img = img*255
                img = img.astype(np.uint8)
                self.data.append(img)
                
                # label 1 = 'good' image
                self.targets.append(1)
                
            os.chdir(cwd)      
        else:
            # load images for testing
            cwd = os.getcwd()
            testFolder = os.path.join(self.root, self.category, 'test/')
            os.chdir(testFolder)
            subfolders = [subfolder.name for subfolder in os.scandir() if subfolder.is_dir()]
            cwsd = os.getcwd()
            
            # for every subfolder in test folder
            for subfolder in subfolders:
                # label 0 = 'defective' image
                label = 0
                if subfolder == 'good':
                    label = 1
                    
                os.chdir(subfolder)
                #filenames = [file.name for file in os.scandir()]
                #for file in filenames:
                for file in os.scandir():
                    img = mpimg.imread(file.name)
                    img = img*255
                    img = img.astype(np.uint8)
                    self.data.append(img)
                    self.targets.append(label)
                    
                os.chdir(cwsd)
                
            os.chdir(cwd)

        # data (images) is a numpy array,
        # targets (labels) is a list
        self.data = np.array(self.data)

        # print original data shape to screen
        print('original data shape: (N, H, W, C)', self.data.shape)
                
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is 0 for 'defective' images
                and 1 for 'good' images
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        
        # if resizing image
        # See: https://pytorch.org/vision/main/generated/torchvision.transforms.Resize.html
        if self.resize:
            resizeTransf = transforms.Resize(self.resize, self.interpolation)
            img = resizeTransf(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target

    def __len__(self) -> int:
        """
        Args:
            None
        Returns:
            int: length of data
        """
        return len(self.data)

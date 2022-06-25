# Copyright 2021 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import random
import glob
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Type, Union
import numpy as np

import torch
import torchvision
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.datasets import STL10, ImageFolder
import torchvision.transforms.functional as TF

from torchvision.utils import save_image
from torchvision.ops import masks_to_boxes
#
# import tensorflow as tf


class Mask_boxes_randomcrpoe(object):
    def __init__(self, size,
                 horizontal_flip_prob: float = 0.5,
                 min_scale: float = 0.08,
                 max_scale: float = 1.0,
                 min_iou: float = 0.2,
                 recrop_num: int = 5):

        self.size = size
        self.resize = transforms.Resize(size=(self.size, self.size),interpolation = transforms.InterpolationMode.BICUBIC)
        self.min_iou = min_iou
        self.recrop_num = recrop_num
        self.horizontal_flip_prob=horizontal_flip_prob
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, img, mask):

        # boxes = masks_to_boxes(transforms.ToTensor()(mask))
        # boxes = boxes.tolist()
        # if len(boxes) == 0:
        #     boxes.append([0,0,img.size[0],img.size[1]])
        # boxe = random.choice(boxes)
        # # boxe = boxe.tolist()
        # #
        # # for i in range(4):
        # #     boxe[i] = boxe[i] + random.randrange(-50, 51)
        # #     if
        #
        # img = TF.crop(img, boxe[0].item(), boxe[1].item(), boxe[2].item(), boxe[3].item())
        # mask = TF.crop(mask, boxe[0].item(), boxe[1].item(), boxe[2].item(), boxe[3].item())
        for _ in range(self.recrop_num):
            #get the Crop size
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                        img,
                        scale=(self.min_scale, self.max_scale),
                        ratio=(3.0/4, 4.0/3))
            mask_temp = TF.resized_crop(mask, i, j, h, w, size = self.size)
            mask_temp_tensor = (transforms.ToTensor()(mask_temp) > 0.0).type(torch.int16)
            # Determine whether to include foreground
            if mask_temp_tensor.view(1, -1).sum()/(mask_temp_tensor.shape[-1]*mask_temp_tensor.shape[-2]) > self.min_iou:
                break
        img = TF.resized_crop(img, i, j, h, w, size = self.size)
        mask = mask_temp

        img = self.resize(img)
        mask = self.resize(mask)

        # Random horizontal flipping
        if random.random() > self.horizontal_flip_prob:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > self.horizontal_flip_prob:
            img = TF.vflip(img)
            mask = TF.vflip(mask)

        return img, mask # Not sure how to make random_t in here

    def __repr__(self):
        return "Mask_boxes_randomcrpoe"

class Mask_center_randomcrpoe(object):
    def __init__(self, size,
                 horizontal_flip_prob: float = 0.5,
                 min_scale: float = 0.08,
                 max_scale: float = 1.0):
        self.size = size
        self.resize = transforms.Resize(size=(self.size, self.size),interpolation = transforms.InterpolationMode.BICUBIC)
        # self.scale = scale
        self.horizontal_flip_prob=horizontal_flip_prob
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, img, mask):

        tensor_mask = transforms.ToTensor()(mask)
        coordinate = torch.nonzero(tensor_mask).numpy()
        image_size = tensor_mask.shape
        if coordinate.shape[0] > 0:
            index = np.random.randint(0,coordinate.shape[0])
            coordinate = coordinate[index]

            # print(coordinate)

            # print(image_size)

            i = np.random.randint(0, coordinate[1]) if coordinate[1] != 0 else 0
            j = np.random.randint(0, coordinate[2]) if coordinate[2] != 0 else 0

            h = np.random.randint(0.08*(image_size[1]-i), image_size[1]-i)
            w = np.random.randint(0.08*(image_size[2]-j), image_size[2]-j)

            h = h if h + i < image_size[1] else image_size[1] - i
            w = w if w + j < image_size[2] else image_size[2] - j

        if coordinate.shape[0] == 0 or h == 0 or w == 0:
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                        img,
                        scale=(self.min_scale, self.max_scale),
                        ratio=(3.0/4, 4.0/3))

        #print(i, j, h, w)

        img = TF.resized_crop(img, i, j, h, w, size = self.size)
        mask = TF.resized_crop(mask, i, j, h, w, size = self.size)

        img = self.resize(img)
        mask = self.resize(mask)

        # Random horizontal flipping
        if random.random() > self.horizontal_flip_prob:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > self.horizontal_flip_prob:
            img = TF.vflip(img)
            mask = TF.vflip(mask)

        return img, mask # Not sure how to make random_t in here

    def __repr__(self):
        return "Mask_center_randomcrpoe"

class Mask_randomcrpoe(object):
    def __init__(self, size,
                 horizontal_flip_prob: float = 0.5,
                 min_scale: float = 0.08,
                 max_scale: float = 1.0):
        self.size = size
        self.resize = transforms.Resize(size=(self.size, self.size),interpolation = transforms.InterpolationMode.BICUBIC)
        self.horizontal_flip_prob = horizontal_flip_prob
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, img, mask):

        image_left = img  # load image with index from self.left_image_paths
        image_right = mask  # load image with index from self.right_image_paths
        # Resize

        # Random crop
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            image_left,
            scale=(self.min_scale, self.max_scale),
            ratio=(3.0 / 4, 4.0 / 3))
        image_left = TF.resized_crop(image_left, i, j, h, w, size=224,
                                     interpolation=transforms.InterpolationMode.BICUBIC)
        image_right = TF.resized_crop(image_right, i, j, h, w, size=224,
                                      interpolation=transforms.InterpolationMode.BICUBIC)

        image_left = self.resize(image_left)
        image_right = self.resize(image_right)

        # Random horizontal flipping
        if random.random() > self.horizontal_flip_prob:
            image_left = TF.hflip(image_left)
            image_right = TF.hflip(image_right)

        # Random vertical flipping
        if random.random() > self.horizontal_flip_prob:
            image_left = TF.vflip(image_left)
            image_right = TF.vflip(image_right)

        return image_left, image_right

    def __repr__(self):
        return "Mask_center_randomcrpoe"


def dataset_with_index(DatasetClass: Type[Dataset]) -> Type[Dataset]:
    """Factory for datasets that also returns the data index.

    Args:
        DatasetClass (Type[Dataset]): Dataset class to be wrapped.

    Returns:
        Type[Dataset]: dataset with index.
    """

    class DatasetWithIndex(DatasetClass):
        def __getitem__(self, index):
            data = super().__getitem__(index)
            return (index, *data)

    return DatasetWithIndex


class CustomDatasetWithoutLabels(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.images = os.listdir(root)

    def __getitem__(self, index):
        path = self.root / self.images[index]
        x = Image.open(path).convert("RGB")
        if self.transform is not None:
            x = self.transform(x)
        return x, -1

    def __len__(self):
        return len(self.images)

class ImageNet_With_Fowardground_and_Backgrond_Mask(ImageFolder):
    def __init__(self, img_root, msk_root, transform=None):
        super().__init__(img_root, transform=transform)
        self.root = Path(img_root)
        self.mask_dir = msk_root
        self.msk_path_list = []

        self.Get_Mask_Path_List()
        self.transform = transform

    def Get_Mask_Path_List(self):
        for img_path, _ in self.samples:
            mask_path = Path(str(img_path).replace("train", str(self.mask_dir)).replace("JPEG", "png"))
            # print(mask_path)
            assert(os.path.isfile(mask_path))
            self.msk_path_list.append(mask_path)

    def __getitem__(self, index):
        path, target = self.samples[index]
        mask_path = self.msk_path_list[index]
        img = Image.open(path).convert("RGB") # img = Image.open(img_path).convert("RGB")
        msk = Image.open(mask_path).convert("L") # msk = Image.open(msk_path).convert("L")

        if self.transform is not None:
            transform_result = self.transform([img, msk])
            #[[img,mask],[img,mask]]
            imgs = [temp[0] for temp in transform_result]
            msks = [temp[1] for temp in transform_result]

        fg_msks = [(msk > 0.0).type(imgs[0].type()) for msk in msks]
        bg_msks = [1 - fg_msk for fg_msk in fg_msks]


        return (imgs, target), fg_msks,  bg_msks


class GaussianBlur:
    def __init__(self, sigma: Sequence[float] = None):
        """Gaussian blur as a callable object.

        Args:
            sigma (Sequence[float]): range to sample the radius of the gaussian blur filter.
                Defaults to [0.1, 2.0].
        """

        if sigma is None:
            sigma = [0.1, 2.0]

        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Applies gaussian blur to an input image.

        Args:
            x (torch.Tensor): an image in the tensor format.

        Returns:
            torch.Tensor: returns a blurred image.
        """

        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class Solarization:
    """Solarization as a callable object."""

    def __call__(self, img: Image) -> Image:
        """Applies solarization to an input image.

        Args:
            img (Image): an image in the PIL.Image format.

        Returns:
            Image: a solarized image.
        """

        return ImageOps.solarize(img)


class NCropAugmentation:
    def __init__(self, transform: Callable, num_crops: int):
        """Creates a pipeline that apply a transformation pipeline multiple times.

        Args:
            transform (Callable): transformation pipeline.
            num_crops (int): number of crops to create from the transformation pipeline.
        """

        self.transform = transform
        self.num_crops = num_crops

    def __call__(self, x: Image) -> List[torch.Tensor]:
        """Applies transforms n times to generate n crops.

        Args:
            x (Image): an image in the PIL.Image format.

        Returns:
            List[torch.Tensor]: an image in the tensor format.
        """

        return [self.transform(x) for _ in range(self.num_crops)]

    def __repr__(self) -> str:
        return f"{self.num_crops} x [{self.transform}]"


class FullTransformPipeline:
    def __init__(self, transforms: Callable) -> None:
        self.transforms = transforms

    def __call__(self, x: Image) -> List[torch.Tensor]:
        """Applies transforms n times to generate n crops.

        Args:
            x (Image): an image in the PIL.Image format.

        Returns:
            List[torch.Tensor]: an image in the tensor format.
        """

        out = []
        for transform in self.transforms:
            out.extend(transform(x))
        return out

    def __repr__(self) -> str:
        return "\n".join([str(transform) for transform in self.transforms])


class BaseTransform:
    """Adds callable base class to implement different transformation pipelines."""

    def __call__(self, x: Image) -> torch.Tensor:
        return self.transform(x)
    
    def __repr__(self) -> str:
        return str(self.transform)


class ImageNet_With_Mask_Transfrom(BaseTransform):
    """MNCRL transform"""
    def __init__(
        self,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
        color_jitter_prob: float = 0.8,
        gray_scale_prob: float = 0.2,
        horizontal_flip_prob: float = 0.5,
        gaussian_prob: float = 0.5,
        solarization_prob: float = 0.0,
        min_scale: float = 0.08,
        max_scale: float = 1.0,
        crop_size: int = 224,
        crop_style: str = "Random_crop",
    ):

        assert crop_style in ["Random_crop","Mask_center","Mask_boxes"], "Crop style need including in Random_crop,Mask_center,Mask_boxes"
        if crop_style == "Random_crop":
            self.image_mask_transforms = Mask_randomcrpoe(crop_size,
                                                            horizontal_flip_prob,
                                                            min_scale,
                                                            max_scale
                                                        )
        elif crop_style == "Mask_center":
            self.image_mask_transforms = Mask_center_randomcrpoe(crop_size,
                                                            horizontal_flip_prob,
                                                            min_scale,
                                                            max_scale
                                                        )
        else:
            self.image_mask_transforms = Mask_boxes_randomcrpoe(crop_size,
                                                            horizontal_flip_prob,
                                                            min_scale,
                                                            max_scale
                                                        )

        self.mask_transforms=transforms.Compose([transforms.Resize([7, 7]),
                                                 #transforms.Grayscale(1),
                                                 transforms.ToTensor(),])
        self.image_transforms = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness, contrast, saturation, hue)],
                    p=color_jitter_prob,
                ),
                transforms.RandomGrayscale(p=gray_scale_prob),
                transforms.RandomApply([GaussianBlur()], p=gaussian_prob),
                transforms.RandomApply([Solarization()], p=solarization_prob),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225)),
            ]
        )

    def __call__(self, x: Image):
        mask = x[1]
        x = x[0]
        crop_img, crop_msk = self.image_mask_transforms(x, mask)
        transforms_img = self.image_transforms(crop_img)
        transforms_masl = self.mask_transforms(crop_msk)

        return [transforms_img, transforms_masl]

    def image_and_image_aug(self, img, mask):
        '''https://discuss.pytorch.org/t/how-to-apply-same-transform-on-a-pair-of-picture/14914?u=ssgosh'''
        image_left =  img # load image with index from self.left_image_paths
        image_right =  mask # load image with index from self.right_image_paths
        # Resize
        resize = transforms.Resize(size=(224, 224),interpolation = transforms.InterpolationMode.BICUBIC)

        #Random crop
        i, j, h, w = transforms.RandomResizedCrop.get_params(
                    image_left,
                    scale=(0.08, 1.0),
                    ratio=( 0.75, 1.3333333333333333))
        image_left = TF.resized_crop(image_left, i, j, h, w, size = 224, interpolation = transforms.InterpolationMode.BICUBIC)
        image_right = TF.resized_crop(image_right, i, j, h, w, size = 224, interpolation = transforms.InterpolationMode.BICUBIC)

        image_left = resize(image_left)
        image_right = resize(image_right)


        # Random horizontal flipping
        if random.random() > 0.5:
            image_left = TF.hflip(image_left)
            image_right = TF.hflip(image_right)

        # Random vertical flipping
        if random.random() > 0.5:
            image_left = TF.vflip(image_left)
            image_right = TF.vflip(image_right)

        return image_left, image_right

    def __repr__(self) -> str:
        return "ImageNet With Mask Transfrom"

class CifarTransform(BaseTransform):
    def __init__(
        self,
        cifar: str,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
        color_jitter_prob: float = 0.8,
        gray_scale_prob: float = 0.2,
        horizontal_flip_prob: float = 0.5,
        gaussian_prob: float = 0.5,
        solarization_prob: float = 0.0,
        min_scale: float = 0.08,
        max_scale: float = 1.0,
        crop_size: int = 32,
    ):
        """Class that applies Cifar10/Cifar100 transformations.

        Args:
            cifar (str): type of cifar, either cifar10 or cifar100.
            brightness (float): sampled uniformly in [max(0, 1 - brightness), 1 + brightness].
            contrast (float): sampled uniformly in [max(0, 1 - contrast), 1 + contrast].
            saturation (float): sampled uniformly in [max(0, 1 - saturation), 1 + saturation].
            hue (float): sampled uniformly in [-hue, hue].
            color_jitter_prob (float, optional): probability of applying color jitter.
                Defaults to 0.8.
            gray_scale_prob (float, optional): probability of converting to gray scale.
                Defaults to 0.2.
            horizontal_flip_prob (float, optional): probability of flipping horizontally.
                Defaults to 0.5.
            gaussian_prob (float, optional): probability of applying gaussian blur.
                Defaults to 0.0.
            solarization_prob (float, optional): probability of applying solarization.
                Defaults to 0.0.
            min_scale (float, optional): minimum scale of the crops. Defaults to 0.08.
            max_scale (float, optional): maximum scale of the crops. Defaults to 1.0.
            crop_size (int, optional): size of the crop. Defaults to 32.
        """

        super().__init__()

        if cifar == "cifar10":
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2470, 0.2435, 0.2616)
        else:
            mean = (0.5071, 0.4865, 0.4409)
            std = (0.2673, 0.2564, 0.2762)

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (crop_size, crop_size),
                    scale=(min_scale, max_scale),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomApply(
                    [transforms.ColorJitter(
                        brightness, contrast, saturation, hue)],
                    p=color_jitter_prob,
                ),
                transforms.RandomGrayscale(p=gray_scale_prob),
                transforms.RandomApply([GaussianBlur()], p=gaussian_prob),
                transforms.RandomApply([Solarization()], p=solarization_prob),
                transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )


class STLTransform(BaseTransform):
    def __init__(
        self,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
        color_jitter_prob: float = 0.8,
        gray_scale_prob: float = 0.2,
        horizontal_flip_prob: float = 0.5,
        gaussian_prob: float = 0.5,
        solarization_prob: float = 0.0,
        min_scale: float = 0.08,
        max_scale: float = 1.0,
        crop_size: int = 96,
    ):
        """Class that applies STL10 transformations.

        Args:
            brightness (float): sampled uniformly in [max(0, 1 - brightness), 1 + brightness].
            contrast (float): sampled uniformly in [max(0, 1 - contrast), 1 + contrast].
            saturation (float): sampled uniformly in [max(0, 1 - saturation), 1 + saturation].
            hue (float): sampled uniformly in [-hue, hue].
            color_jitter_prob (float, optional): probability of applying color jitter.
                Defaults to 0.8.
            gray_scale_prob (float, optional): probability of converting to gray scale.
                Defaults to 0.2.
            horizontal_flip_prob (float, optional): probability of flipping horizontally.
                Defaults to 0.5.
            gaussian_prob (float, optional): probability of applying gaussian blur.
                Defaults to 0.0.
            solarization_prob (float, optional): probability of applying solarization.
                Defaults to 0.0.
            min_scale (float, optional): minimum scale of the crops. Defaults to 0.08.
            max_scale (float, optional): maximum scale of the crops. Defaults to 1.0.
            crop_size (int, optional): size of the crop. Defaults to 96.
        """

        super().__init__()
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (crop_size, crop_size),
                    scale=(min_scale, max_scale),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness, contrast, saturation, hue)],
                    p=color_jitter_prob,
                ),
                transforms.RandomGrayscale(p=gray_scale_prob),
                transforms.RandomApply([GaussianBlur()], p=gaussian_prob),
                transforms.RandomApply([Solarization()], p=solarization_prob),
                transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
            ]
        )

class ImagenetTransform(BaseTransform):
    def __init__(
        self,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
        color_jitter_prob: float = 0.8,
        gray_scale_prob: float = 0.2,
        horizontal_flip_prob: float = 0.5,
        gaussian_prob: float = 0.5,
        solarization_prob: float = 0.0,
        min_scale: float = 0.08,
        max_scale: float = 1.0,
        crop_size: int = 224,
    ):
        """Class that applies Imagenet transformations.

        Args:
            brightness (float): sampled uniformly in [max(0, 1 - brightness), 1 + brightness].
            contrast (float): sampled uniformly in [max(0, 1 - contrast), 1 + contrast].
            saturation (float): sampled uniformly in [max(0, 1 - saturation), 1 + saturation].
            hue (float): sampled uniformly in [-hue, hue].
            color_jitter_prob (float, optional): probability of applying color jitter.
                Defaults to 0.8.
            gray_scale_prob (float, optional): probability of converting to gray scale.
                Defaults to 0.2.
            horizontal_flip_prob (float, optional): probability of flipping horizontally.
                Defaults to 0.5.
            gaussian_prob (float, optional): probability of applying gaussian blur.
                Defaults to 0.0.
            solarization_prob (float, optional): probability of applying solarization.
                Defaults to 0.0.
            min_scale (float, optional): minimum scale of the crops. Defaults to 0.08.
            max_scale (float, optional): maximum scale of the crops. Defaults to 1.0.
            crop_size (int, optional): size of the crop. Defaults to 224.
        """

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    crop_size,
                    scale=(min_scale, max_scale),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness, contrast, saturation, hue)],
                    p=color_jitter_prob,
                ),
                transforms.RandomGrayscale(p=gray_scale_prob),
                transforms.RandomApply([GaussianBlur()], p=gaussian_prob),
                transforms.RandomApply([Solarization()], p=solarization_prob),
                transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225)),
            ]
        )

class CustomTransform(BaseTransform):
    def __init__(
        self,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
        color_jitter_prob: float = 0.8,
        gray_scale_prob: float = 0.2,
        horizontal_flip_prob: float = 0.5,
        gaussian_prob: float = 0.5,
        solarization_prob: float = 0.0,
        min_scale: float = 0.08,
        max_scale: float = 1.0,
        crop_size: int = 224,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.228, 0.224, 0.225),
    ):
        """Class that applies Custom transformations.
        If you want to do exoteric augmentations, you can just re-write this class.

        Args:
            brightness (float): sampled uniformly in [max(0, 1 - brightness), 1 + brightness].
            contrast (float): sampled uniformly in [max(0, 1 - contrast), 1 + contrast].
            saturation (float): sampled uniformly in [max(0, 1 - saturation), 1 + saturation].
            hue (float): sampled uniformly in [-hue, hue].
            color_jitter_prob (float, optional): probability of applying color jitter.
                Defaults to 0.8.
            gray_scale_prob (float, optional): probability of converting to gray scale.
                Defaults to 0.2.
            horizontal_flip_prob (float, optional): probability of flipping horizontally.
                Defaults to 0.5.
            gaussian_prob (float, optional): probability of applying gaussian blur.
                Defaults to 0.0.
            solarization_prob (float, optional): probability of applying solarization.
                Defaults to 0.0.
            min_scale (float, optional): minimum scale of the crops. Defaults to 0.08.
            max_scale (float, optional): maximum scale of the crops. Defaults to 1.0.
            crop_size (int, optional): size of the crop. Defaults to 224.
            mean (Sequence[float], optional): mean values for normalization.
                Defaults to (0.485, 0.456, 0.406).
            std (Sequence[float], optional): std values for normalization.
                Defaults to (0.228, 0.224, 0.225).
        """

        super().__init__()
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    crop_size,
                    scale=(min_scale, max_scale),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness, contrast, saturation, hue)],
                    p=color_jitter_prob,
                ),
                transforms.RandomGrayscale(p=gray_scale_prob),
                transforms.RandomApply([GaussianBlur()], p=gaussian_prob),
                transforms.RandomApply([Solarization()], p=solarization_prob),
                transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )


def prepare_transform(dataset: str, **kwargs) -> Any:
    """Prepares transforms for a specific dataset. Optionally uses multi crop.

    Args:
        dataset (str): name of the dataset.

    Returns:
        Any: a transformation for a specific dataset.
    """

    if dataset in ["cifar10", "cifar100"]:
        return CifarTransform(cifar=dataset, **kwargs)
    elif dataset == "stl10":
        return STLTransform(**kwargs)
    elif dataset in ["imagenet", "imagenet100"]:
        return ImagenetTransform(**kwargs)
    elif dataset == "imagenet_with_mask" :
        #print('dataset == "imagenet_with_mask"')
        #return Imagenet_image_mask_Transform_MNCRL(**kwargs)
        return ImageNet_With_Mask_Transfrom(**kwargs)
    elif dataset == "custom":
        return CustomTransform(**kwargs)
    else:
        raise ValueError(f"{dataset} is not currently supported.")


def prepare_n_crop_transform(
    transforms: List[Callable], num_crops_per_aug: List[int]
) -> NCropAugmentation:
    """Turns a single crop transformation to an N crops transformation.

    Args:
        transforms (List[Callable]): list of transformations.
        num_crops_per_aug (List[int]): number of crops per pipeline.

    Returns:
        NCropAugmentation: an N crop transformation.
    """

    assert len(transforms) == len(num_crops_per_aug)

    T = []
    for transform, num_crops in zip(transforms, num_crops_per_aug):
        T.append(NCropAugmentation(transform, num_crops))
    return FullTransformPipeline(T)


def prepare_datasets(
    dataset: str,
    transform: Callable,
    data_dir: Optional[Union[str, Path]] = None,
    train_dir: Optional[Union[str, Path]] = None,
    mask_dir: Optional[Union[str, str]] = None,
    no_labels: Optional[Union[str, Path]] = False,
    download: bool = True,
) -> Dataset:
    """Prepares the desired dataset.

    Args:
        dataset (str): the name of the dataset.
        transform (Callable): a transformation.
        data_dir (Optional[Union[str, Path]], optional): the directory to load data from.
            Defaults to None.
        train_dir (Optional[Union[str, Path]], optional): training data directory
            to be appended to data_dir. Defaults to None.
        no_labels (Optional[bool], optional): if the custom dataset has no labels.

    Returns:
        Dataset: the desired dataset with transformations.
    """

    if data_dir is None:
        sandbox_folder = Path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        data_dir = sandbox_folder / "datasets"

    if train_dir is None:
        train_dir = Path(f"{dataset}/train")
    else:
        train_dir = Path(train_dir)

    if dataset in ["cifar10", "cifar100"]:
        DatasetClass = vars(torchvision.datasets)[dataset.upper()]
        train_dataset = dataset_with_index(DatasetClass)(
            data_dir / train_dir,
            train=True,
            download=download,
            transform=transform,
        )

    elif dataset == "stl10":
        train_dataset = dataset_with_index(STL10)(
            data_dir / train_dir,
            split="train+unlabeled",
            download=download,
            transform=transform,
        )

    elif dataset in ["imagenet", "imagenet100"]:
        train_dir = data_dir / train_dir

        #train_dataset = dataset_with_index(ImageFolder)(train_dir, transform)
        train_dataset = dataset_with_index(MNCRLDataset)(train_dir, "train_binary_mask_by_USS", transform)

    elif dataset == "imagenet_with_mask" :
        train_path = os.path.join(data_dir, train_dir)
        # mask_path = os.path.join(data_dir, mask_dir)
        train_dataset = dataset_with_index(ImageNet_With_Fowardground_and_Backgrond_Mask)(train_path, mask_dir, transform)

    elif dataset == "custom":
        train_dir = data_dir / train_dir

        if no_labels:
            dataset_class = CustomDatasetWithoutLabels
        else:
            dataset_class = ImageFolder

        train_dataset = dataset_with_index(dataset_class)(train_dir, transform)
    elif dataset == "MNCRL_imagenet":
        train_dir = data_dir / train_dir
        train_dataset = dataset_with_index(MNCRLDataset)(train_dir, "train_binary_mask_by_USS", transform)

    return train_dataset


def prepare_dataloader(
    train_dataset: Dataset, batch_size: int = 64, num_workers: int = 4
) -> DataLoader:
    """Prepares the training dataloader for pretraining.
    Args:
        train_dataset (Dataset): the name of the dataset.
        batch_size (int, optional): batch size. Defaults to 64.
        num_workers (int, optional): number of workers. Defaults to 4.
    Returns:
        DataLoader: the training dataloader with the desired dataset.
    """

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader

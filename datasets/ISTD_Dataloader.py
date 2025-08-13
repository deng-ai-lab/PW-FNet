import os.path
import time
import cv2
import torch.utils.data as data
from PIL import Image
from random import randrange
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
import torch
import random
import numpy as np
import torchvision.transforms.functional as TF
from math import ceil
import math

def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.
    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.
    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)

def imfrombytes(content, flag='color', float32=False):
    """Read an image from bytes.
    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.
        float32 (bool): Whether to change to float32., If True, will also norm
            to [0, 1]. Default: False.
    Returns:
        ndarray: Loaded image array.
    """
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {
        'color': cv2.IMREAD_COLOR,
        'grayscale': cv2.IMREAD_GRAYSCALE,
        'unchanged': cv2.IMREAD_UNCHANGED
    }
    if img_np is None:
        raise Exception('None .. !!!')
    img = cv2.imdecode(img_np, imread_flags[flag])
    if float32:
        img = img.astype(np.float32) / 255.
    return img


def read_img255(filename):
    # img0 = cv2.imread(filename)
    # img1 = img0[:, :, ::-1].astype('float32') / 1.0
    # return img1
    with open(filename, 'rb') as f:
        value_buf = f.read()
    return value_buf

def read_mask(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32)
    img = img/255.
    img = img[:, :, np.newaxis]
    return img

def augment(imgs=[], size=256, edge_decay=0., only_h_flip=False):
    H, W, _ = imgs[0].shape
    # print("imgs[0].shape: ", imgs[0].shape)
    # print("imgs[1].shape: ", imgs[1].shape)
    Hc, Wc = [size, size]

    H, W, _ = imgs[0].shape
    # simple re-weight for the edge
    if random.random() < Hc / H * edge_decay:
        Hs = 0 if random.randint(0, 1) == 0 else H - Hc
    else:
        Hs = random.randint(0, H - Hc)

    if random.random() < Wc / W * edge_decay:
        Ws = 0 if random.randint(0, 1) == 0 else W - Wc
    else:
        Ws = random.randint(0, W - Wc)

    for i in range(len(imgs)):
        imgs[i] = imgs[i][Hs:(Hs + Hc), Ws:(Ws + Wc), :]

    # horizontal flip
    if random.randint(0, 1) == 1:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1)

    if not only_h_flip:
        # bad data augmentations for outdoor
        rot_deg = random.randint(0, 3)
        for i in range(len(imgs)):
            imgs[i] = np.rot90(imgs[i], rot_deg, (0, 1))

    return imgs


def align_for_test(imgs=[], local_size=32):
    H, W, _ = imgs[0].shape
    Hc = local_size * (H // local_size)
    Wc = local_size * (W // local_size)
    Hs = (H - Hc) // 2
    Ws = (W - Wc) // 2
    for i in range(len(imgs)):
        imgs[i] = imgs[i][Hs:(Hs + Hc), Ws:(Ws + Wc), :]
    return imgs

def align(imgs=[], size_H=448, size_W=608):
    H, W, _ = imgs[0].shape
    Hc = size_H
    Wc = size_W
    Hs = (H - Hc) // 2
    Ws = (W - Wc) // 2
    for i in range(len(imgs)):
        imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :]
    return imgs


# --- Training dataset --- #
# train_data_dir: /home/jxy/projects_dir/datasets/Rain/train
class TrainData(data.Dataset):
    def __init__(self, crop_size, train_data_dir, only_h_flip=False):
        super().__init__()
        train_list_rain = '/home/jxy/projects_dir/datasets/ISTD_Dataset/train.txt'

        with open(train_list_rain) as f:
            contents = f.readlines()
            gt_names = [i.strip() for i in contents]
            rain_names = gt_names
            mask_names = gt_names
        self.rain_names = rain_names
        self.gt_names = gt_names
        self.mask_names = mask_names
        self.train_data_dir = train_data_dir
        self.crop_size = crop_size
        self.only_h_flip = only_h_flip

    def get_images(self, index):
        rain_name = self.rain_names[index]
        gt_name = self.gt_names[index]
        mask_name = self.mask_names[index]
        rain_path = os.path.join(self.train_data_dir, 'train_A', rain_name)
        gt_path = os.path.join(self.train_data_dir, 'train_C', gt_name)
        mask_path = os.path.join(self.train_data_dir, 'train_B', mask_name)
        rain_img = imfrombytes(read_img255(rain_path), float32=True)
        gt_img = imfrombytes(read_img255(gt_path), float32=True)
        mask_img = read_mask(mask_path)
        [rain_img, gt_img, mask_img] = augment([rain_img, gt_img, mask_img], size=self.crop_size, edge_decay=0.,
                                     only_h_flip=self.only_h_flip)
        rain = img2tensor(rain_img, bgr2rgb=True, float32=True)
        gt = img2tensor(gt_img, bgr2rgb=True, float32=True)
        mask = img2tensor(mask_img, bgr2rgb=True, float32=True)

        return {'source': rain, 'target': gt, 'mask': mask, 'filename': rain_name}

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.rain_names)


class TrainData_fixed(data.Dataset):
    def __init__(self, crop_size, train_data_dir, only_h_flip=False):
        super().__init__()
        train_list_rain = '/home/jxy/projects_dir/datasets/ISTD_Dataset/train.txt'

        with open(train_list_rain) as f:
            contents = f.readlines()
            gt_names = [i.strip() for i in contents]
            rain_names = gt_names
            mask_names = gt_names
        self.rain_names = rain_names
        self.gt_names = gt_names
        self.mask_names = mask_names
        self.train_data_dir = train_data_dir
        self.crop_size = crop_size
        self.only_h_flip = only_h_flip

    def get_images(self, index):
        rain_name = self.rain_names[index]
        gt_name = self.gt_names[index]
        mask_name = self.mask_names[index]
        rain_path = os.path.join(self.train_data_dir, 'train_A', rain_name)
        gt_path = os.path.join(self.train_data_dir, 'train_C_fixed_ours', gt_name)
        mask_path = os.path.join(self.train_data_dir, 'train_B', mask_name)
        rain_img = imfrombytes(read_img255(rain_path), float32=True)
        gt_img = imfrombytes(read_img255(gt_path), float32=True)
        mask_img = read_mask(mask_path)
        [rain_img, gt_img, mask_img] = augment([rain_img, gt_img, mask_img], size=self.crop_size, edge_decay=0.,
                                               only_h_flip=self.only_h_flip)
        rain = img2tensor(rain_img, bgr2rgb=True, float32=True)
        gt = img2tensor(gt_img, bgr2rgb=True, float32=True)
        mask = img2tensor(mask_img, bgr2rgb=True, float32=True)
        # ############## data aug ############
        p = 0.3
        hue_interval = [-p, p]
        saturation_interval = [1-p, 1+p]
        fn_ids = torch.randperm(2)
        h_value = torch.empty(1).uniform_(hue_interval[0], hue_interval[1])
        s_value = torch.empty(1).uniform_(saturation_interval[0], saturation_interval[1])
        for fn_id in fn_ids:
            if fn_id == 0:
                gt = TF.adjust_saturation(gt, s_value)
                rain = TF.adjust_saturation(rain, s_value)
            if fn_id == 1:
                gt = TF.adjust_hue(gt, h_value)
                rain = TF.adjust_hue(rain, h_value)
        return {'source': rain, 'target': gt, 'mask': mask, 'filename': rain_name}

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.rain_names)


class TestData(data.Dataset):
    def __init__(self, local_size, val_data_dir, flag=True):
        super().__init__()

        val_list_rain = "/home/jxy/projects_dir/datasets/ISTD_Dataset/test.txt"

        with open(val_list_rain) as f:
            contents = f.readlines()
            gt_names = [i.strip() for i in contents]
            rain_names = gt_names
            mask_names = rain_names

        self.rain_names = rain_names
        self.gt_names = gt_names
        self.mask_names = mask_names
        self.val_data_dir = val_data_dir
        self.flag = flag
        self.local_size = local_size

    def get_images(self, index):
        rain_name = self.rain_names[index]
        gt_name = self.gt_names[index]
        mask_name = self.mask_names[index]

        rain_path = os.path.join(self.val_data_dir, 'test_A', rain_name)
        gt_path = os.path.join(self.val_data_dir, 'test_C', gt_name)
        mask_path = os.path.join(self.val_data_dir, 'test_B', mask_name)
        rain_img = imfrombytes(read_img255(rain_path), float32=True)
        gt_img = imfrombytes(read_img255(gt_path), float32=True)
        mask_img =read_mask(mask_path)
        h, w, c = rain_img.shape
        h = h - h % self.local_size
        w = w - w % self.local_size
        # [rain_img, gt_img] = align_for_test([rain_img, gt_img], local_size=self.local_size)
        [rain_img, gt_img, mask_img] = align([rain_img, gt_img, mask_img], size_H=h, size_W=w)
        rain = img2tensor(rain_img, bgr2rgb=True, float32=True)
        gt = img2tensor(gt_img, bgr2rgb=True, float32=True)
        mask = img2tensor(mask_img, bgr2rgb=True, float32=True)

        return {'source': rain, 'target': gt, 'mask': mask, 'filename': rain_name}

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.rain_names)


class TestData_fixed(data.Dataset):
    def __init__(self, local_size, val_data_dir, flag=True):
        super().__init__()

        val_list_rain = "/home/jxy/projects_dir/datasets/ISTD_Dataset/test.txt"

        with open(val_list_rain) as f:
            contents = f.readlines()
            gt_names = [i.strip() for i in contents]
            rain_names = gt_names
            mask_names = rain_names

        self.rain_names = rain_names
        self.gt_names = gt_names
        self.mask_names = mask_names
        self.val_data_dir = val_data_dir
        self.flag = flag
        self.local_size = local_size

    def get_images(self, index):
        rain_name = self.rain_names[index]
        gt_name = self.gt_names[index]
        mask_name = self.mask_names[index]

        rain_path = os.path.join(self.val_data_dir, 'test_A', rain_name)
        gt_path = os.path.join(self.val_data_dir, 'test_C_fixed_official', gt_name)
        mask_path = os.path.join(self.val_data_dir, 'test_B', mask_name)
        rain_img = imfrombytes(read_img255(rain_path), float32=True)
        gt_img = imfrombytes(read_img255(gt_path), float32=True)
        mask_img = read_mask(mask_path)
        h, w, c = rain_img.shape
        h = h - h % self.local_size
        w = w - w % self.local_size
        # [rain_img, gt_img] = align_for_test([rain_img, gt_img], local_size=self.local_size)
        [rain_img, gt_img, mask_img] = align([rain_img, gt_img, mask_img], size_H=h, size_W=w)
        rain = img2tensor(rain_img, bgr2rgb=True, float32=True)
        gt = img2tensor(gt_img, bgr2rgb=True, float32=True)
        mask = img2tensor(mask_img, bgr2rgb=True, float32=True)

        return {'source': rain, 'target': gt, 'mask': mask, 'filename': rain_name}

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.rain_names)
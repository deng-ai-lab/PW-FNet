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
from math import ceil

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
        img = torch.from_numpy(img.transpose(2, 0, 1))
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

def augment(imgs=[], size=256, edge_decay=0., only_h_flip=False, scale='x2'):
    H, W, _ = imgs[0].shape
    # print("imgs[0].shape: ", imgs[0].shape)
    # print("imgs[1].shape: ", imgs[1].shape)
    Hc, Wc = [size, size]

    # simple re-weight for the edge
    if random.random() < Hc / H * edge_decay:
        Hs = 0 if random.randint(0, 1) == 0 else H - Hc
    else:
        Hs = random.randint(0, H - Hc)

    if random.random() < Wc / W * edge_decay:
        Ws = 0 if random.randint(0, 1) == 0 else W - Wc
    else:
        Ws = random.randint(0, W - Wc)

    imgs[0] = imgs[0][Hs:(Hs + Hc), Ws:(Ws + Wc), :]
    if scale == 'x2':
        imgs[1] = imgs[1][Hs * 2:(Hs + Hc) * 2, Ws * 2:(Ws + Wc) * 2, :]
    elif scale == 'x3':
        imgs[1] = imgs[1][Hs * 3:(Hs + Hc) * 3, Ws * 3:(Ws + Wc) * 3, :]
    elif scale == 'x4':
        imgs[1] = imgs[1][Hs * 4:(Hs + Hc) * 4, Ws * 4:(Ws + Wc) * 4, :]

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


# --- Training dataset --- #
# train_data_dir: /home/jxy/projects_dir/datasets/Rain/train
class TrainData(data.Dataset):
    def __init__(self, crop_size, train_data_dir, scale, only_h_flip=False):
        super().__init__()

        if scale == 'x2':
            train_list_rain = '/home/jxy/projects_dir/datasets/DIV2K/trainx2.txt'
        elif scale == 'x3':
            train_list_rain = '/home/jxy/projects_dir/datasets/DIV2K/trainx3.txt'
        elif scale == 'x4':
            train_list_rain = '/home/jxy/projects_dir/datasets/DIV2K/trainx4.txt'

        with open(train_list_rain) as f:
            contents = f.readlines()
            rain_names = [i.strip() for i in contents]
            gt_names = [i.replace(scale, '') for i in rain_names]
        self.rain_names = rain_names
        self.gt_names = gt_names
        self.train_data_dir = train_data_dir
        self.crop_size = crop_size
        self.only_h_flip = only_h_flip
        self.dataset = []
        self.scale = scale
        self.__preprocess__()

    def __preprocess__(self):
        """Preprocess the Artworks dataset."""
        for l in range(len(self.rain_names)):
            rain_name = self.rain_names[l]
            gt_name = self.gt_names[l]
            if self.scale == 'x2':
                rain_path = os.path.join(self.train_data_dir, 'DIV2K_train_LR_bicubic_X2', rain_name)
            elif self.scale == 'x3':
                rain_path = os.path.join(self.train_data_dir, 'DIV2K_train_LR_bicubic_X3', rain_name)
            elif self.scale == 'x4':
                rain_path = os.path.join(self.train_data_dir, 'DIV2K_train_LR_bicubic_X4', rain_name)
            gt_path = os.path.join(self.train_data_dir, 'DIV2K_train_HR', gt_name)
            rain_img = imfrombytes(read_img255(rain_path), float32=True)
            gt_img = imfrombytes(read_img255(gt_path), float32=True)

            self.dataset.append((rain_img, gt_img))

        print("Finish to read the dataset!")


    def get_images(self, index):
        rain_name = self.rain_names[index]
        # gt_name = self.gt_names[index]
        # if self.scale == 'x2':
        #     rain_path = os.path.join(self.train_data_dir, 'DIV2K_train_LR_bicubic_X2', rain_name)
        # elif self.scale == 'x3':
        #     rain_path = os.path.join(self.train_data_dir, 'DIV2K_train_LR_bicubic_X3', rain_name)
        # elif self.scale == 'x4':
        #     rain_path = os.path.join(self.train_data_dir, 'DIV2K_train_LR_bicubic_X4', rain_name)
        # gt_path = os.path.join(self.train_data_dir, 'DIV2K_train_HR', gt_name)
        # rain_img = imfrombytes(read_img255(rain_path), float32=True)
        # gt_img = imfrombytes(read_img255(gt_path), float32=True)
        rain_img = self.dataset[index][0]
        gt_img = self.dataset[index][1]
        [rain_img, gt_img] = augment([rain_img, gt_img], size=self.crop_size, edge_decay=0.,
                                     only_h_flip=self.only_h_flip, scale=self.scale)

        rain = img2tensor(rain_img, bgr2rgb=True, float32=True)
        gt = img2tensor(gt_img, bgr2rgb=True, float32=True)

        return {'source': rain, 'target': gt, 'filename': rain_name}

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.rain_names)


# val_data_dir: /home/jxy/projects_dir/datasets/Rain/test
class TestData_B100(data.Dataset):
    def __init__(self, val_data_dir, scale):
        super().__init__()
        val_list_rain = "/home/jxy/projects_dir/datasets/DIV2K/testB100.txt"

        with open(val_list_rain) as f:
            contents = f.readlines()
            rain_names = [i.strip().replace('x2', scale) for i in contents]
            gt_names = [i.replace('LRBI', 'HR').replace('x2', scale) for i in rain_names]

        self.rain_names = rain_names
        self.gt_names = gt_names
        self.val_data_dir = val_data_dir
        self.scale = scale

    def get_images(self, index):
        rain_name = self.rain_names[index]
        gt_name = self.gt_names[index]
        rain_path = os.path.join(self.val_data_dir, "LR", "LRBI", 'B100', self.scale, rain_name)
        gt_path = os.path.join(self.val_data_dir, "HR", "B100", self.scale, gt_name)
        rain_img = imfrombytes(read_img255(rain_path), float32=True)
        gt_img = imfrombytes(read_img255(gt_path), float32=True)
        rain = img2tensor(rain_img, bgr2rgb=True, float32=True)
        gt = img2tensor(gt_img, bgr2rgb=True, float32=True)

        return {'source': rain, 'target': gt, 'filename': rain_name}

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.rain_names)

class TestData_Manga109(data.Dataset):
    def __init__(self, val_data_dir, scale):
        super().__init__()
        val_list_rain = "/home/jxy/projects_dir/datasets/DIV2K/testManga109.txt"

        with open(val_list_rain) as f:
            contents = f.readlines()
            rain_names = [i.strip().replace('x2', scale) for i in contents]
            gt_names = [i.replace('LRBI', 'HR').replace('x2', scale) for i in rain_names]

        self.rain_names = rain_names
        self.gt_names = gt_names
        self.val_data_dir = val_data_dir
        self.scale = scale

    def get_images(self, index):
        rain_name = self.rain_names[index]
        gt_name = self.gt_names[index]
        rain_path = os.path.join(self.val_data_dir, "LR", "LRBI", 'Manga109', self.scale, rain_name)
        gt_path = os.path.join(self.val_data_dir, "HR", "Manga109", self.scale, gt_name)
        rain_img = imfrombytes(read_img255(rain_path), float32=True)
        gt_img = imfrombytes(read_img255(gt_path), float32=True)
        rain = img2tensor(rain_img, bgr2rgb=True, float32=True)
        gt = img2tensor(gt_img, bgr2rgb=True, float32=True)

        return {'source': rain, 'target': gt, 'filename': rain_name}

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.rain_names)

class TestData_Manga109_full(data.Dataset):
    def __init__(self, val_data_dir, scale):
        super().__init__()
        val_list_rain = "/home/jxy/projects_dir/datasets/DIV2K/testManga109full.txt"

        with open(val_list_rain) as f:
            contents = f.readlines()
            rain_names = [i.strip().replace('x2', scale) for i in contents]
            gt_names = [i.replace('LRBI', 'HR').replace('x2', scale) for i in rain_names]

        self.rain_names = rain_names
        self.gt_names = gt_names
        self.val_data_dir = val_data_dir
        self.scale = scale

    def get_images(self, index):
        rain_name = self.rain_names[index]
        gt_name = self.gt_names[index]
        rain_path = os.path.join(self.val_data_dir, "LR", "LRBI", 'Manga109', self.scale, rain_name)
        gt_path = os.path.join(self.val_data_dir, "HR", "Manga109", self.scale, gt_name)
        rain_img = imfrombytes(read_img255(rain_path), float32=True)
        gt_img = imfrombytes(read_img255(gt_path), float32=True)
        rain = img2tensor(rain_img, bgr2rgb=True, float32=True)
        gt = img2tensor(gt_img, bgr2rgb=True, float32=True)

        return {'source': rain, 'target': gt, 'filename': rain_name}

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.rain_names)

class TestData_Set14(data.Dataset):
    def __init__(self, val_data_dir, scale):
        super().__init__()
        val_list_rain = "/home/jxy/projects_dir/datasets/DIV2K/testSet14.txt"

        with open(val_list_rain) as f:
            contents = f.readlines()
            rain_names = [i.strip().replace('x2', scale) for i in contents]
            gt_names = [i.replace('LRBI', 'HR').replace('x2', scale) for i in rain_names]

        self.rain_names = rain_names
        self.gt_names = gt_names
        self.val_data_dir = val_data_dir
        self.scale = scale

    def get_images(self, index):
        rain_name = self.rain_names[index]
        gt_name = self.gt_names[index]
        rain_path = os.path.join(self.val_data_dir, "LR", "LRBI", 'Set14', self.scale, rain_name)
        gt_path = os.path.join(self.val_data_dir, "HR", "Set14", self.scale, gt_name)
        rain_img = imfrombytes(read_img255(rain_path), float32=True)
        gt_img = imfrombytes(read_img255(gt_path), float32=True)
        rain = img2tensor(rain_img, bgr2rgb=True, float32=True)
        gt = img2tensor(gt_img, bgr2rgb=True, float32=True)

        return {'source': rain, 'target': gt, 'filename': rain_name}

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.rain_names)

class TestData_Set5(data.Dataset):
    def __init__(self, val_data_dir, scale):
        super().__init__()
        val_list_rain = "/home/jxy/projects_dir/datasets/DIV2K/testSet5.txt"

        with open(val_list_rain) as f:
            contents = f.readlines()
            rain_names = [i.strip().replace('x2', scale) for i in contents]
            gt_names = [i.replace('LRBI', 'HR').replace('x2', scale) for i in rain_names]

        self.rain_names = rain_names
        self.gt_names = gt_names
        self.val_data_dir = val_data_dir
        self.scale = scale

    def get_images(self, index):
        rain_name = self.rain_names[index]
        gt_name = self.gt_names[index]
        rain_path = os.path.join(self.val_data_dir, "LR", "LRBI", 'Set5', self.scale, rain_name)
        gt_path = os.path.join(self.val_data_dir, "HR", "Set5", self.scale, gt_name)
        rain_img = imfrombytes(read_img255(rain_path), float32=True)
        gt_img = imfrombytes(read_img255(gt_path), float32=True)
        rain = img2tensor(rain_img, bgr2rgb=True, float32=True)
        gt = img2tensor(gt_img, bgr2rgb=True, float32=True)

        return {'source': rain, 'target': gt, 'filename': rain_name}

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.rain_names)

class TestData_Urban100(data.Dataset):
    def __init__(self, val_data_dir, scale):
        super().__init__()
        val_list_rain = "/home/jxy/projects_dir/datasets/DIV2K/testUrban100.txt"

        with open(val_list_rain) as f:
            contents = f.readlines()
            rain_names = [i.strip().replace('x2', scale) for i in contents]
            gt_names = [i.replace('LRBI', 'HR').replace('x2', scale) for i in rain_names]

        self.rain_names = rain_names
        self.gt_names = gt_names
        self.val_data_dir = val_data_dir
        self.scale = scale

    def get_images(self, index):
        rain_name = self.rain_names[index]
        gt_name = self.gt_names[index]
        rain_path = os.path.join(self.val_data_dir, "LR", "LRBI", 'Urban100', self.scale, rain_name)
        gt_path = os.path.join(self.val_data_dir, "HR", "Urban100", self.scale, gt_name)
        rain_img = imfrombytes(read_img255(rain_path), float32=True)
        gt_img = imfrombytes(read_img255(gt_path), float32=True)
        rain = img2tensor(rain_img, bgr2rgb=True, float32=True)
        gt = img2tensor(gt_img, bgr2rgb=True, float32=True)

        return {'source': rain, 'target': gt, 'filename': rain_name}

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.rain_names)
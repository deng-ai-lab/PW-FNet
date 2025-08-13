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

def cubic(x):
    """Implementation of `cubic` function in Matlab under Python language.

    Args:
        x: Element vector.

    Returns:
        Bicubic interpolation.
    """

    absx = torch.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return (1.5 * absx3 - 2.5 * absx2 + 1) * ((absx <= 1).type_as(absx)) + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * (
        ((absx > 1) * (absx <= 2)).type_as(absx))


def calculate_weights_indices(in_length: int, out_length: int, scale: float, kernel_width: int, antialiasing: bool):
    """Implementation of `calculate_weights_indices` function in Matlab under Python language.

    Args:
        in_length (int): Input length.
        out_length (int): Output length.
        scale (float): Scale factor.
        kernel_width (int): Kernel width.
        antialiasing (bool): Whether to apply antialiasing when down-sampling operations.
            Caution: Bicubic down-sampling in PIL uses antialiasing by default.

    """

    if (scale < 1) and antialiasing:
        # Use a modified kernel (larger kernel width) to simultaneously
        # interpolate and antialiasing
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = torch.linspace(1, out_length, out_length)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5 + scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = torch.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    p = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = left.view(out_length, 1).expand(out_length, p) + torch.linspace(0, p - 1, p).view(1, p).expand(
        out_length, p)

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = u.view(out_length, 1).expand(out_length, p) - indices

    # apply cubic kernel
    if (scale < 1) and antialiasing:
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)

    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, p)

    # If a column in weights is all zero, get rid of it. only consider the
    # first and last column.
    weights_zero_tmp = torch.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, p - 2)
        weights = weights.narrow(1, 1, p - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, p - 2)
        weights = weights.narrow(1, 0, p - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)


def imresize(image, scale_factor: float, antialiasing: bool = True):
    """Implementation of `imresize` function in Matlab under Python language.

    Args:
        image: The input image.
        scale_factor (float): Scale factor. The same scale applies for both height and width.
        antialiasing (bool): Whether to apply antialiasing when down-sampling operations.
            Caution: Bicubic down-sampling in `PIL` uses antialiasing by default. Default: ``True``.

    Returns:
        np.ndarray: Output image with shape (c, h, w), [0, 1] range, w/o round.
    """
    squeeze_flag = False
    if type(image).__module__ == np.__name__ or 1:  # numpy type
        numpy_type = True
        if image.ndim == 2:
            image = image[:, :, None]
            squeeze_flag = True
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
    else:
        numpy_type = False
        if image.ndim == 2:
            image = image.unsqueeze(0)
            squeeze_flag = True

    in_c, in_h, in_w = image.size()
    out_h, out_w = math.ceil(in_h * scale_factor), math.ceil(in_w * scale_factor)
    kernel_width = 4

    # get weights and indices
    weights_h, indices_h, sym_len_hs, sym_len_he = calculate_weights_indices(in_h, out_h, scale_factor, kernel_width, antialiasing)
    weights_w, indices_w, sym_len_ws, sym_len_we = calculate_weights_indices(in_w, out_w, scale_factor, kernel_width, antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_c, in_h + sym_len_hs + sym_len_he, in_w)
    img_aug.narrow(1, sym_len_hs, in_h).copy_(image)

    sym_patch = image[:, :sym_len_hs, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, 0, sym_len_hs).copy_(sym_patch_inv)

    sym_patch = image[:, -sym_len_he:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, sym_len_hs + in_h, sym_len_he).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(in_c, out_h, in_w)
    kernel_width = weights_h.size(1)
    for i in range(out_h):
        idx = int(indices_h[i][0])
        for j in range(in_c):
            out_1[j, i, :] = img_aug[j, idx:idx + kernel_width, :].transpose(0, 1).mv(weights_h[i])

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(in_c, out_h, in_w + sym_len_ws + sym_len_we)
    out_1_aug.narrow(2, sym_len_ws, in_w).copy_(out_1)

    sym_patch = out_1[:, :, :sym_len_ws]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, 0, sym_len_ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, :, -sym_len_we:]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, sym_len_ws + in_w, sym_len_we).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(in_c, out_h, out_w)
    kernel_width = weights_w.size(1)
    for i in range(out_w):
        idx = int(indices_w[i][0])
        for j in range(in_c):
            out_2[j, :, i] = out_1_aug[j, :, idx:idx + kernel_width].mv(weights_w[i])

    if squeeze_flag:
        out_2 = out_2.squeeze(0)
    if numpy_type:
        out_2 = out_2.numpy()
        if not squeeze_flag:
            out_2 = out_2.transpose(1, 2, 0)

    return out_2

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
class TrainData1(data.Dataset):
    def __init__(self, crop_size, train_data_dir, only_h_flip=False):
        super().__init__()
        train_list_rain = '/home/jxy/projects_dir/datasets/C-CUHK/CUHK_CR1_train.txt'

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
        rain_path = os.path.join(self.train_data_dir, 'CUHK-CR1', 'train', 'cloud', rain_name)
        gt_path = os.path.join(self.train_data_dir, 'CUHK-CR1', 'train', 'label', gt_name)
        mask_path = os.path.join(self.train_data_dir, 'nir', 'CUHK-CR1', 'train', 'cloud', mask_name)
        mask_gt_path = os.path.join(self.train_data_dir, 'nir', 'CUHK-CR1', 'train', 'label', mask_name)
        rain_img = imfrombytes(read_img255(rain_path), float32=True)
        gt_img = imfrombytes(read_img255(gt_path), float32=True)
        mask_img = read_mask(mask_path)
        mask_gt_img = read_mask(mask_gt_path)
        rain_img = imresize(rain_img, 1/2)
        gt_img = imresize(gt_img, 1/2)
        mask_img = imresize(mask_img, 1/2)
        mask_gt_img = imresize(mask_gt_img, 1/2)
        [rain_img, gt_img, mask_img, mask_gt_img] = augment([rain_img, gt_img, mask_img, mask_gt_img], size=self.crop_size, edge_decay=0.,
                                     only_h_flip=self.only_h_flip)
        rain = img2tensor(rain_img, bgr2rgb=True, float32=True)
        gt = img2tensor(gt_img, bgr2rgb=True, float32=True)
        mask = img2tensor(mask_img, bgr2rgb=True, float32=True)
        mask_gt = img2tensor(mask_gt_img, bgr2rgb=True, float32=True)
        rain = torch.cat([rain, mask], dim=0)
        gt = torch.cat([gt, mask_gt], dim=0)

        return {'source': rain, 'target': gt, 'filename': rain_name}

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.rain_names)


class TrainData2(data.Dataset):
    def __init__(self, crop_size, train_data_dir, only_h_flip=False):
        super().__init__()
        train_list_rain = '/home/jxy/projects_dir/datasets/C-CUHK/CUHK_CR2_train.txt'

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
        rain_path = os.path.join(self.train_data_dir, 'CUHK-CR2', 'train', 'cloud', rain_name)
        gt_path = os.path.join(self.train_data_dir, 'CUHK-CR2', 'train', 'label', gt_name)
        mask_path = os.path.join(self.train_data_dir, 'nir', 'CUHK-CR2', 'train', 'cloud', mask_name)
        mask_gt_path = os.path.join(self.train_data_dir, 'nir', 'CUHK-CR2', 'train', 'label', mask_name)
        rain_img = imfrombytes(read_img255(rain_path), float32=True)
        gt_img = imfrombytes(read_img255(gt_path), float32=True)
        mask_img = read_mask(mask_path)
        mask_gt_img = read_mask(mask_gt_path)
        rain_img = imresize(rain_img, 1/2)
        gt_img = imresize(gt_img, 1/2)
        mask_img = imresize(mask_img, 1/2)
        mask_gt_img = imresize(mask_gt_img, 1/2)
        [rain_img, gt_img, mask_img, mask_gt_img] = augment([rain_img, gt_img, mask_img, mask_gt_img],
                                                            size=self.crop_size, edge_decay=0.,
                                                            only_h_flip=self.only_h_flip)
        rain = img2tensor(rain_img, bgr2rgb=True, float32=True)
        gt = img2tensor(gt_img, bgr2rgb=True, float32=True)
        mask = img2tensor(mask_img, bgr2rgb=True, float32=True)
        mask_gt = img2tensor(mask_gt_img, bgr2rgb=True, float32=True)

        rain = torch.cat([rain, mask], dim=0)
        gt = torch.cat([gt, mask_gt], dim=0)

        return {'source': rain, 'target': gt, 'filename': rain_name}

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.rain_names)


class TestData1(data.Dataset):
    def __init__(self, local_size, val_data_dir, flag=True):
        super().__init__()

        val_list_rain = "/home/jxy/projects_dir/datasets/C-CUHK/CUHK_CR1_test.txt"

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
        rain_path = os.path.join(self.val_data_dir, 'CUHK-CR1', 'test', 'cloud', rain_name)
        gt_path = os.path.join(self.val_data_dir, 'CUHK-CR1', 'test', 'label', gt_name)
        mask_path = os.path.join(self.val_data_dir, 'nir', 'CUHK-CR1', 'test', 'cloud', mask_name)
        mask_gt_path = os.path.join(self.val_data_dir, 'nir', 'CUHK-CR1', 'test', 'label', mask_name)
        rain_img = imfrombytes(read_img255(rain_path), float32=True)
        gt_img = imfrombytes(read_img255(gt_path), float32=True)
        mask_img = read_mask(mask_path)
        mask_gt_img = read_mask(mask_gt_path)
        rain_img = imresize(rain_img, 1/2)
        gt_img = imresize(gt_img, 1/2)
        mask_img = imresize(mask_img, 1/2)
        mask_gt_img = imresize(mask_gt_img, 1/2)

        h, w, c = rain_img.shape
        h = h - h % self.local_size
        w = w - w % self.local_size
        # [rain_img, gt_img] = align_for_test([rain_img, gt_img], local_size=self.local_size)
        [rain_img, gt_img, mask_img, mask_gt_img] = align([rain_img, gt_img, mask_img, mask_gt_img], size_H=h, size_W=w)
        rain = img2tensor(rain_img, bgr2rgb=True, float32=True)
        gt = img2tensor(gt_img, bgr2rgb=True, float32=True)
        mask = img2tensor(mask_img, bgr2rgb=True, float32=True)
        mask_gt = img2tensor(mask_gt_img, bgr2rgb=True, float32=True)

        rain = torch.cat([rain, mask], dim=0)
        gt = torch.cat([gt, mask_gt], dim=0)

        return {'source': rain, 'target': gt, 'mask': mask, 'filename': rain_name}

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.rain_names)


class TestData2(data.Dataset):
    def __init__(self, local_size, val_data_dir, flag=True):
        super().__init__()

        val_list_rain = "/home/jxy/projects_dir/datasets/C-CUHK/CUHK_CR2_test.txt"

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
        rain_path = os.path.join(self.val_data_dir, 'CUHK-CR2', 'test', 'cloud', rain_name)
        gt_path = os.path.join(self.val_data_dir, 'CUHK-CR2', 'test', 'label', gt_name)
        mask_path = os.path.join(self.val_data_dir, 'nir', 'CUHK-CR2', 'test', 'cloud', mask_name)
        mask_gt_path = os.path.join(self.val_data_dir, 'nir', 'CUHK-CR2', 'test', 'label', mask_name)
        rain_img = imfrombytes(read_img255(rain_path), float32=True)
        gt_img = imfrombytes(read_img255(gt_path), float32=True)
        mask_img = read_mask(mask_path)
        mask_gt_img = read_mask(mask_gt_path)
        rain_img = imresize(rain_img, 1/2)
        gt_img = imresize(gt_img, 1/2)
        mask_img = imresize(mask_img, 1/2)
        mask_gt_img = imresize(mask_gt_img, 1/2)

        h, w, c = rain_img.shape
        h = h - h % self.local_size
        w = w - w % self.local_size
        # [rain_img, gt_img] = align_for_test([rain_img, gt_img], local_size=self.local_size)
        [rain_img, gt_img, mask_img, mask_gt_img] = align([rain_img, gt_img, mask_img, mask_gt_img], size_H=h, size_W=w)
        rain = img2tensor(rain_img, bgr2rgb=True, float32=True)
        gt = img2tensor(gt_img, bgr2rgb=True, float32=True)
        mask = img2tensor(mask_img, bgr2rgb=True, float32=True)
        mask_gt = img2tensor(mask_gt_img, bgr2rgb=True, float32=True)

        rain = torch.cat([rain, mask], dim=0)
        gt = torch.cat([gt, mask_gt], dim=0)

        return {'source': rain, 'target': gt, 'mask': mask, 'filename': rain_name}

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.rain_names)

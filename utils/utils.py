import numpy as np
from pytorch_msssim import ssim
from skimage.metrics import structural_similarity as compare_ssim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from einops import rearrange

class MixUp_AUG:
    def __init__(self):
        self.dist = torch.distributions.beta.Beta(torch.tensor([1.2]), torch.tensor([1.2]))

    def aug(self, rgb_gt, rgb_noisy, gray_mask):
        bs = rgb_gt.size(0)
        indices = torch.randperm(bs)
        rgb_gt2 = rgb_gt[indices]
        rgb_noisy2 = rgb_noisy[indices]
        gray_mask2 = gray_mask[indices]
        # gray_contour2 = gray_mask[indices]
        lam = self.dist.rsample((bs,1)).view(-1,1,1,1).cuda()

        rgb_gt    = lam * rgb_gt + (1-lam) * rgb_gt2
        rgb_noisy = lam * rgb_noisy + (1-lam) * rgb_noisy2
        gray_mask = lam * gray_mask + (1-lam) * gray_mask2
        # gray_mask = torch.where(gray_mask>0.01, torch.ones_like(gray_mask), torch.zeros_like(gray_mask))
        # gray_contour = lam * gray_contour + (1-lam) * gray_contour2
        return rgb_gt, rgb_noisy, gray_mask

def to_fft(x_sp):
    ffted = torch.fft.rfft2(x_sp, norm='ortho')
    x_fft_real = torch.unsqueeze(torch.real(ffted), dim=-1)
    x_fft_imag = torch.unsqueeze(torch.imag(ffted), dim=-1)
    ffted = torch.cat((x_fft_real, x_fft_imag), dim=-1)
    x_fr = rearrange(ffted, 'b c h w d -> b (c d) h w').contiguous()

    return x_fr

def to_sp(x_sp, x_fr):
    _, _, h, w = x_sp.size()
    ffted = rearrange(x_fr, 'b (c d) h w -> b c h w d', d=2).contiguous()
    ffted = torch.view_as_complex(ffted)
    ifft_fr = torch.fft.irfft2(ffted, s=(h, w), norm='ortho')

    return ifft_fr

def to_ssim_skimage(dehaze, gt):
  dehaze_list = torch.split(dehaze, 1, dim=0)
  gt_list = torch.split(gt, 1, dim=0)

  dehaze_list_np = [dehaze_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in
                    range(len(dehaze_list))]
  gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
  ssim_list = [compare_ssim(dehaze_list_np[ind], gt_list_np[ind], data_range=1, multichannel=True) for ind in
               range(len(dehaze_list))]

  return ssim_list

def _convert_input_type_range(img):
  """Convert the type and range of the input image.
  It converts the input image to np.float32 type and range of [0, 1].
  It is mainly used for pre-processing the input image in colorspace
  convertion functions such as rgb2ycbcr and ycbcr2rgb.
  Args:
    img (ndarray): The input image. It accepts:
        1. np.uint8 type with range [0, 255];
        2. np.float32 type with range [0, 1].
  Returns:
      (ndarray): The converted image with type of np.float32 and range of
          [0, 1].
  """
  img_type = img.dtype
  img = img.astype(np.float32)
  if img_type == np.float32:
    pass
  elif img_type == np.uint8:
    img /= 255.
  else:
    raise TypeError('The img type should be np.float32 or np.uint8, '
                    f'but got {img_type}')
  return img


def _convert_output_type_range(img, dst_type):
  """Convert the type and range of the image according to dst_type.
  It converts the image to desired type and range. If `dst_type` is np.uint8,
  images will be converted to np.uint8 type with range [0, 255]. If
  `dst_type` is np.float32, it converts the image to np.float32 type with
  range [0, 1].
  It is mainly used for post-processing images in colorspace convertion
  functions such as rgb2ycbcr and ycbcr2rgb.
  Args:
    img (ndarray): The image to be converted with np.float32 type and
        range [0, 255].
    dst_type (np.uint8 | np.float32): If dst_type is np.uint8, it
        converts the image to np.uint8 type with range [0, 255]. If
        dst_type is np.float32, it converts the image to np.float32 type
        with range [0, 1].
  Returns:
    (ndarray): The converted image with desired type and range.
  """
  if dst_type not in (np.uint8, np.float32):
    raise TypeError('The dst_type should be np.float32 or np.uint8, '
                    f'but got {dst_type}')
  if dst_type == np.uint8:
    img = img.round()
  else:
    img /= 255.

  return img.astype(dst_type)

def rgb2ycbcr(img, y_only=False):
  """Convert a RGB image to YCbCr image.
  This function produces the same results as Matlab's `rgb2ycbcr` function.
  It implements the ITU-R BT.601 conversion for standard-definition
  television. See more details in
  https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.
  It differs from a similar function in cv2.cvtColor: `RGB <-> YCrCb`.
  In OpenCV, it implements a JPEG conversion. See more details in
  https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.
  Args:
    img (ndarray): The input image. It accepts:
        1. np.uint8 type with range [0, 255];
        2. np.float32 type with range [0, 1].
    y_only (bool): Whether to only return Y channel. Default: False.
  Returns:
    ndarray: The converted YCbCr image. The output image has the same type
        and range as input image.
  """
  img_type = img.dtype
  img = _convert_input_type_range(img)
  if y_only:
    out_img = np.dot(img, [65.481, 128.553, 24.966]) + 16.0
  else:
    out_img = np.matmul(img,
                        [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                         [24.966, 112.0, -18.214]]) + [16, 128, 128]
  out_img = _convert_output_type_range(out_img, img_type)
  return out_img


def to_y_channel(img):
  """Change to Y channel of YCbCr.
  Args:
    img (ndarray): Images with range [0, 255].
  Returns:
    (ndarray): Images with range [0, 255] (float type) without round.
  """
  img = img.astype(np.float32) / 255.
  img = rgb2ycbcr(img, y_only=True)
  img = img[..., None]
  return img * 255.


def calculate_psnr_torch(img1, img2):
  b, c, h, w = img1.shape
  v = torch.tensor([[65.481/255], [128.553/255], [24.966/255]]).cuda()
  img1 = torch.mm(img1.permute(0, 2, 3, 1).reshape(-1, c), v) + 16./255
  img2 = torch.mm(img2.permute(0, 2, 3, 1).reshape(-1, c), v) + 16./255
  img1 = img1.reshape(b, h, w, -1)
  img2 = img2.reshape(b, h, w, -1)
  mse_loss = F.mse_loss(img1, img2, reduction='none').mean((1, 2, 3))
  psnr_full = 10 * torch.log10(1 / mse_loss).mean()
  sim = ssim(img1.permute(0, 3, 1, 2), img2.permute(0, 3, 1, 2), data_range=1, size_average=False).mean()
  # sim_ = 0
  # for i in range(b):
  #   sim_ += _ssim_cly(img1[i, :, :, 0].cpu().numpy()*255, img2[i, :, :, 0].cpu().numpy()*255) / b
  # print(mse)
  return psnr_full, sim

import math

def pixel_weight(height, width, sample, beta):
    h = height
    w = width
    k = sample
    alpha = (1 - beta / k) / (beta - 1)

    w_weight = np.empty([h, w], dtype=float)
    h_weight = np.empty([h, w], dtype=float)
    # 宽度方向
    if w > 2 * k:
        for x in range(w):
            if x < k:
                weight = ((1 / (x+1)) + alpha) / (2 * (math.log(k)+0.5772) + (w - 2 * k)/k + w * alpha)
                for i in range(h): w_weight[i][x] = weight
            elif x < (w-k):
                weight = (1 / k + alpha) / (2 * (math.log(k)+0.5772) + (w - 2 * k)/k + w * alpha)
                for i in range(h): w_weight[i][x] = weight
            else:
                weight = (1 / (w-x) + alpha) / (2 * (math.log(k)+0.5772) + (w - 2 * k)/k + w * alpha)
                for i in range(h): w_weight[i][x] = weight
    else:
        for x in range(w):
            if x < k:
                weight = ((1 / (x + 1)) + alpha) / (2 * (math.log(k)+0.5772) + (w - 2 * k) / k + w * alpha)
                for i in range(h): w_weight[i][x] = weight
            else:
                weight = (1 / (w - x) + alpha) / (2 * (math.log(k)+0.5772) + (w - 2 * k) / k + w * alpha)
                for i in range(h): w_weight[i][x] = weight

    # 高度方向
    if h > 2 * k:
        for x in range(h):
            if x < k:
                weight = ((1 / (x + 1)) + alpha) / (2 * (math.log(k)+0.5772) + (h - 2 * k) / k + h * alpha)
                h_weight[x][:] = weight
            elif x < (h - k):
                weight = (1 / k + alpha) / (2 * (math.log(k)+0.5772) + (h - 2 * k) / k + h * alpha)
                h_weight[x][:] = weight
            else:
                weight = (1 / (h - x) + alpha) / (2 * (math.log(k)+0.5772) + (h - 2 * k) / k + h * alpha)
                h_weight[x][:] = weight
    else:
        for x in range(w):
            if x < k:
                weight = ((1 / (x + 1)) + alpha) / (2 * (math.log(k)+0.5772) + (h - 2 * k) / k + h * alpha)
                h_weight[x][:] = weight
            else:
                weight = (1 / (h - x) + alpha) / (2 * (math.log(k)+0.5772) + (h - 2 * k) / k + h * alpha)
                h_weight[x][:] = weight

    # 总权重
    total_weight = (w_weight + h_weight) / (h+w)

    print(total_weight)

    return total_weight


import cv2

def calculate_psnr(img1,
                   img2,
                   test_y_channel=False):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img1 (ndarray/tensor): Images with range [0, 255]/[0, 1].
        img2 (ndarray/tensor): Images with range [0, 255]/[0, 1].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: psnr result.
    """

    if type(img1) == torch.Tensor:
        if len(img1.shape) == 4:
            img1 = img1.squeeze(0)
        img1 = img1.detach().cpu().numpy().transpose(1, 2, 0)
    if type(img2) == torch.Tensor:
        if len(img2.shape) == 4:
            img2 = img2.squeeze(0)
        img2 = img2.detach().cpu().numpy().transpose(1, 2, 0)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    def _psnr(img1, img2):
        if test_y_channel:
            img1 = to_y_channel(img1)
            img2 = to_y_channel(img2)

        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        max_value = 1. if img1.max() <= 1 else 255.
        return 20. * np.log10(max_value / np.sqrt(mse))

    return _psnr(img1, img2)


def _ssim(img1, img2, max_value):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    C1 = (0.01 * max_value) ** 2
    C2 = (0.03 * max_value) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def _3d_gaussian_calculator(img, conv3d):
    out = conv3d(img.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    return out


def _generate_3d_gaussian_kernel():
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    kernel_3 = cv2.getGaussianKernel(11, 1.5)
    kernel = torch.tensor(np.stack([window * k for k in kernel_3], axis=0))
    conv3d = torch.nn.Conv3d(1, 1, (11, 11, 11), stride=1, padding=(5, 5, 5), bias=False, padding_mode='replicate')
    conv3d.weight.requires_grad = False
    conv3d.weight[0, 0, :, :, :] = kernel
    return conv3d


def _ssim_3d(img1, img2, max_value):
    assert len(img1.shape) == 3 and len(img2.shape) == 3
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255]/[0, 1] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255]/[0, 1] with order 'HWC'.

    Returns:
        float: ssim result.
    """
    C1 = (0.01 * max_value) ** 2
    C2 = (0.03 * max_value) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    kernel = _generate_3d_gaussian_kernel().cuda()

    img1 = torch.tensor(img1).float().cuda()
    img2 = torch.tensor(img2).float().cuda()

    mu1 = _3d_gaussian_calculator(img1, kernel)
    mu2 = _3d_gaussian_calculator(img2, kernel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = _3d_gaussian_calculator(img1 ** 2, kernel) - mu1_sq
    sigma2_sq = _3d_gaussian_calculator(img2 ** 2, kernel) - mu2_sq
    sigma12 = _3d_gaussian_calculator(img1 * img2, kernel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())


def _ssim_cly(img1, img2):
    assert len(img1.shape) == 2 and len(img2.shape) == 2
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    kernel = cv2.getGaussianKernel(11, 1.5)
    # print(kernel)
    window = np.outer(kernel, kernel.transpose())

    bt = cv2.BORDER_REPLICATE

    mu1 = cv2.filter2D(img1, -1, window, borderType=bt)
    mu2 = cv2.filter2D(img2, -1, window, borderType=bt)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window, borderType=bt) - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window, borderType=bt) - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window, borderType=bt) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1,
                   img2,
                   test_y_channel=False,
                   ssim3d=True):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: ssim result.
    """


    if type(img1) == torch.Tensor:
        if len(img1.shape) == 4:
            img1 = img1.squeeze(0)
        img1 = img1.detach().cpu().numpy().transpose(1, 2, 0)
    if type(img2) == torch.Tensor:
        if len(img2.shape) == 4:
            img2 = img2.squeeze(0)
        img2 = img2.detach().cpu().numpy().transpose(1, 2, 0)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)


    def _cal_ssim(img1, img2):
        if test_y_channel:
            img1 = to_y_channel(img1)
            img2 = to_y_channel(img2)
            return _ssim_cly(img1[..., 0], img2[..., 0])

        ssims = []
        # ssims_before = []

        # skimage_before = skimage.metrics.structural_similarity(img1, img2, data_range=255., multichannel=True)
        # print('.._skimage',
        #       skimage.metrics.structural_similarity(img1, img2, data_range=255., multichannel=True))
        max_value = 1 if img1.max() <= 1 else 255
        with torch.no_grad():
            final_ssim = _ssim_3d(img1, img2, max_value) if ssim3d else _ssim(img1, img2, max_value)
            ssims.append(final_ssim)

        # for i in range(img1.shape[2]):
        #     ssims_before.append(_ssim(img1, img2))

        # print('..ssim mean , new {:.4f}  and before {:.4f} .... skimage before {:.4f}'.format(np.array(ssims).mean(), np.array(ssims_before).mean(), skimage_before))
        # ssims.append(skimage.metrics.structural_similarity(img1[..., i], img2[..., i], multichannel=False))

        return np.array(ssims).mean()

    return _cal_ssim(img1, img2)


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]


class C2R(nn.Module):
    def __init__(self, ablation=False):

        super(C2R, self).__init__()
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.ab = ablation
        print('*******************use normal 6 neg clcr loss****************')

    def forward(self, pred, source, target):
        pred1 = pred[2]
        pred2 = nn.PixelShuffle(2)(pred[1])
        pred4 = nn.PixelShuffle(4)(pred[0])
        pred1_vgg, pred2_vgg, pred4_vgg, source_vgg, target_vgg = self.vgg(pred1), self.vgg(pred2), self.vgg(pred4), self.vgg(source), self.vgg(target)
        loss = 0
        for i in range(len(pred1_vgg)):
            d_ap1 = self.l1(pred1_vgg[i], target_vgg[i].detach())
            d_ap2 = self.l1(pred2_vgg[i], target_vgg[i].detach())
            d_ap4 = self.l1(pred4_vgg[i], target_vgg[i].detach())
            d_ap21 = self.l1(pred2_vgg[i], pred1_vgg[i].detach())
            d_ap41 = self.l1(pred4_vgg[i], pred1_vgg[i].detach())
            d_ap42 = self.l1(pred4_vgg[i], pred2_vgg[i].detach())
            if not self.ab:
                d_an1 = self.l1(pred1_vgg[i], source_vgg[i].detach())
                d_an2 = self.l1(pred2_vgg[i], source_vgg[i].detach())
                d_an4 = self.l1(pred4_vgg[i], source_vgg[i].detach())
                d_an12 = self.l1(pred1_vgg[i], pred2_vgg[i].detach())
                d_an14 = self.l1(pred1_vgg[i], pred4_vgg[i].detach())
                d_an24 = self.l1(pred2_vgg[i], pred4_vgg[i].detach())
                contrastive1 = d_ap1 / (d_an1 / 3 + d_an12 / 3 + d_an14 / 3)
                contrastive2 = (d_ap2 / 2 + d_ap21 / 2) / (d_an2 / 2 + d_an24 / 2)
                contrastive4 = (d_ap4 / 3 + d_ap42 / 3 + d_ap41 / 3) / d_an4


            loss += self.weights[i] * (contrastive1 + contrastive2 + contrastive4)
        return loss

def caluate_contra(pred, source, target, criterion):
    pred_4_1 = nn.PixelShuffle(4)(pred[0])
    pred_4_2 = nn.PixelShuffle(2)(pred[0])
    pred_4_4 = pred[0]
    pred_2_1 = nn.PixelShuffle(2)(pred[1])
    pred_2_2 = pred[1]
    pred_2_4 = nn.PixelUnshuffle(2)(pred[1])
    pred_1_1 = pred[2]
    pred_1_2 = nn.PixelUnshuffle(2)(pred[2])
    pred_1_4 = nn.PixelUnshuffle(4)(pred[2])

    target_1 = target
    target_2 = nn.PixelUnshuffle(2)(target)
    target_4 = nn.PixelUnshuffle(4)(target)

    source_1 = source
    source_2 = nn.PixelUnshuffle(2)(source)
    source_4 = nn.PixelUnshuffle(4)(source)

    target1_fft = torch.fft.fft2(target_1, dim=(-2, -1))
    target1_fft = torch.stack((target1_fft.real, target1_fft.imag), -1)

    target2_fft = torch.fft.fft2(target_2, dim=(-2, -1))
    target2_fft = torch.stack((target2_fft.real, target2_fft.imag), -1)

    target4_fft = torch.fft.fft2(target_4, dim=(-2, -1))
    target4_fft = torch.stack((target4_fft.real, target4_fft.imag), -1)

    source1_fft = torch.fft.fft2(source_1, dim=(-2, -1))
    source1_fft = torch.stack((source1_fft.real, source1_fft.imag), -1)

    source2_fft = torch.fft.fft2(source_2, dim=(-2, -1))
    source2_fft = torch.stack((source2_fft.real, source2_fft.imag), -1)

    source4_fft = torch.fft.fft2(source_4, dim=(-2, -1))
    source4_fft = torch.stack((source4_fft.real, source4_fft.imag), -1)

    pred11_fft = torch.fft.fft2(pred_1_1, dim=(-2, -1))
    pred11_fft = torch.stack((pred11_fft.real, pred11_fft.imag), -1)

    pred12_fft = torch.fft.fft2(pred_1_2, dim=(-2, -1))
    pred12_fft = torch.stack((pred12_fft.real, pred12_fft.imag), -1)

    pred14_fft = torch.fft.fft2(pred_1_4, dim=(-2, -1))
    pred14_fft = torch.stack((pred14_fft.real, pred14_fft.imag), -1)

    pred21_fft = torch.fft.fft2(pred_2_1, dim=(-2, -1))
    pred21_fft = torch.stack((pred21_fft.real, pred21_fft.imag), -1)

    pred22_fft = torch.fft.fft2(pred_2_2, dim=(-2, -1))
    pred22_fft = torch.stack((pred22_fft.real, pred22_fft.imag), -1)

    pred24_fft = torch.fft.fft2(pred_2_4, dim=(-2, -1))
    pred24_fft = torch.stack((pred24_fft.real, pred24_fft.imag), -1)

    pred41_fft = torch.fft.fft2(pred_4_1, dim=(-2, -1))
    pred41_fft = torch.stack((pred41_fft.real, pred41_fft.imag), -1)

    pred42_fft = torch.fft.fft2(pred_4_2, dim=(-2, -1))
    pred42_fft = torch.stack((pred42_fft.real, pred42_fft.imag), -1)

    pred44_fft = torch.fft.fft2(pred_4_4, dim=(-2, -1))
    pred44_fft = torch.stack((pred44_fft.real, pred44_fft.imag), -1)

    d_ap1_1 = criterion(pred11_fft, target1_fft) * 0.1 + criterion(pred_1_1, target_1)
    d_an1_1 = criterion(pred11_fft, source1_fft) * 0.1 + criterion(pred_1_1, source_1)
    d_an2_1 = criterion(pred11_fft, pred21_fft.detach()) * 0.1 + criterion(pred_1_1, pred_2_1.detach())
    d_an3_1 = criterion(pred11_fft, pred41_fft.detach()) * 0.1 + criterion(pred_1_1, pred_4_1.detach())
    contrastive_1 = d_ap1_1 / (1/3 * d_an1_1 + 1/3 * d_an2_1 + 1/3 * d_an3_1) * 0.1 + d_ap1_1

    d_ap1_2 = criterion(pred22_fft, target2_fft) * 0.1 + criterion(pred_2_2, target_2)
    d_ap2_2 = criterion(pred22_fft, pred12_fft.detach()) * 0.1 + criterion(pred_2_2, pred_1_2.detach())
    d_an1_2 = criterion(pred22_fft, source2_fft) * 0.1 + criterion(pred_2_2, source_2)
    d_an2_2 = criterion(pred22_fft, pred42_fft.detach()) * 0.1 + criterion(pred_2_2, pred_4_2.detach())
    contrastive_2 = (1/2 * d_ap1_2 + 1/2 * d_ap2_2) / (1/2 * d_an1_2 + 1/2 * d_an2_2) * 0.1 + d_ap1_2

    d_ap1_4 = criterion(pred44_fft, target4_fft) * 0.1 + criterion(pred_4_4, target_4)
    d_ap2_4 = criterion(pred44_fft, pred14_fft.detach()) * 0.1 + criterion(pred_4_4, pred_1_4.detach())
    d_ap3_4 = criterion(pred44_fft, pred24_fft.detach()) * 0.1 + criterion(pred_4_4, pred_2_4.detach())
    d_an1_4 = criterion(pred44_fft, source4_fft) * 0.1 + criterion(pred_4_4, source_4)
    contrastive_3 = (1/3 * d_ap1_4 + 1/3 * d_ap2_4 + 1/3 * d_ap3_4) / d_an1_4 * 0.1 + d_ap1_4

    return contrastive_1 + contrastive_2 + contrastive_3


def caluate_contra_v3(pred, source, target, criterion):
    # pred_4_1 = nn.PixelShuffle(4)(pred[0])
    # pred_4_2 = nn.PixelShuffle(2)(pred[0])
    pred_4_4 = pred[0]
    # pred_2_1 = nn.PixelShuffle(2)(pred[1])
    pred_2_2 = pred[1]
    # pred_2_4 = nn.PixelUnshuffle(2)(pred[1])
    pred_1_1 = pred[2]
    # pred_1_2 = nn.PixelUnshuffle(2)(pred[2])
    # pred_1_4 = nn.PixelUnshuffle(4)(pred[2])

    target_1 = target
    target_2 = nn.PixelUnshuffle(2)(target)
    target_4 = nn.PixelUnshuffle(4)(target)

    source_1 = source
    source_2 = nn.PixelUnshuffle(2)(source)
    source_4 = nn.PixelUnshuffle(4)(source)

    target1_fft = torch.fft.fft2(target_1, dim=(-2, -1))
    target1_fft = torch.stack((target1_fft.real, target1_fft.imag), -1)

    target2_fft = torch.fft.fft2(target_2, dim=(-2, -1))
    target2_fft = torch.stack((target2_fft.real, target2_fft.imag), -1)

    target4_fft = torch.fft.fft2(target_4, dim=(-2, -1))
    target4_fft = torch.stack((target4_fft.real, target4_fft.imag), -1)

    source1_fft = torch.fft.fft2(source_1, dim=(-2, -1))
    source1_fft = torch.stack((source1_fft.real, source1_fft.imag), -1)

    source2_fft = torch.fft.fft2(source_2, dim=(-2, -1))
    source2_fft = torch.stack((source2_fft.real, source2_fft.imag), -1)

    source4_fft = torch.fft.fft2(source_4, dim=(-2, -1))
    source4_fft = torch.stack((source4_fft.real, source4_fft.imag), -1)

    pred11_fft = torch.fft.fft2(pred_1_1, dim=(-2, -1))
    pred11_fft = torch.stack((pred11_fft.real, pred11_fft.imag), -1)

    # pred12_fft = torch.fft.fft2(pred_1_2, dim=(-2, -1))
    # pred12_fft = torch.stack((pred12_fft.real, pred12_fft.imag), -1)
    #
    # pred14_fft = torch.fft.fft2(pred_1_4, dim=(-2, -1))
    # pred14_fft = torch.stack((pred14_fft.real, pred14_fft.imag), -1)
    #
    # pred21_fft = torch.fft.fft2(pred_2_1, dim=(-2, -1))
    # pred21_fft = torch.stack((pred21_fft.real, pred21_fft.imag), -1)
    #
    pred22_fft = torch.fft.fft2(pred_2_2, dim=(-2, -1))
    pred22_fft = torch.stack((pred22_fft.real, pred22_fft.imag), -1)

    # pred24_fft = torch.fft.fft2(pred_2_4, dim=(-2, -1))
    # pred24_fft = torch.stack((pred24_fft.real, pred24_fft.imag), -1)
    #
    # pred41_fft = torch.fft.fft2(pred_4_1, dim=(-2, -1))
    # pred41_fft = torch.stack((pred41_fft.real, pred41_fft.imag), -1)
    #
    # pred42_fft = torch.fft.fft2(pred_4_2, dim=(-2, -1))
    # pred42_fft = torch.stack((pred42_fft.real, pred42_fft.imag), -1)

    pred44_fft = torch.fft.fft2(pred_4_4, dim=(-2, -1))
    pred44_fft = torch.stack((pred44_fft.real, pred44_fft.imag), -1)

    d_ap1_1 = criterion(pred11_fft, target1_fft) * 0.1 + criterion(pred_1_1, target_1)
    d_an1_1 = criterion(pred11_fft, source1_fft) * 0.1 + criterion(pred_1_1, source_1)
    # d_an2_1 = criterion(pred11_fft, pred21_fft.detach())
    # d_an3_1 = criterion(pred11_fft, pred41_fft.detach())
    contrastive_1 = 0.1 * d_ap1_1 / d_an1_1 + d_ap1_1

    d_ap1_2 = criterion(pred22_fft, target2_fft) * 0.1 + criterion(pred_2_2, target_2)
    # d_ap2_2 = criterion(pred22_fft, pred12_fft.detach())
    d_an1_2 = criterion(pred22_fft, source2_fft) * 0.1 + criterion(pred_2_2, target_2)
    # d_an2_2 = criterion(pred22_fft, pred42_fft.detach())
    contrastive_2 = 0.1 * d_ap1_2 / d_an1_2 + d_ap1_2

    d_ap1_4 = criterion(pred44_fft, target4_fft) * 0.1 + criterion(pred_4_4, target_4)
    # d_ap2_4 = criterion(pred44_fft, pred14_fft.detach())
    # d_ap3_4 = criterion(pred44_fft, pred24_fft.detach())
    d_an1_4 = criterion(pred44_fft, source4_fft) * 0.1 + criterion(pred_4_4, target_4)
    contrastive_3 = 0.1 * d_ap1_4 / d_an1_4 + d_ap1_4

    return contrastive_1 + contrastive_2 + contrastive_3


def caluate_contra_v2(pred, source, target, criterion):
    pred_4_1 = nn.PixelShuffle(4)(pred[0])
    pred_4_2 = nn.PixelShuffle(2)(pred[0])
    pred_4_4 = pred[0]
    pred_2_1 = nn.PixelShuffle(2)(pred[1])
    pred_2_2 = pred[1]
    pred_2_4 = nn.PixelUnshuffle(2)(pred[1])
    pred_1_1 = pred[2]
    pred_1_2 = nn.PixelUnshuffle(2)(pred[2])
    pred_1_4 = nn.PixelUnshuffle(4)(pred[2])

    target_1 = target
    target_2 = nn.PixelUnshuffle(2)(target)
    target_4 = nn.PixelUnshuffle(4)(target)

    source_1 = source
    source_2 = nn.PixelUnshuffle(2)(source)
    source_4 = nn.PixelUnshuffle(4)(source)

    target1_fft = torch.fft.fft2(target_1, dim=(-2, -1))
    target1_fft = torch.stack((target1_fft.real, target1_fft.imag), -1)

    target2_fft = torch.fft.fft2(target_2, dim=(-2, -1))
    target2_fft = torch.stack((target2_fft.real, target2_fft.imag), -1)

    target4_fft = torch.fft.fft2(target_4, dim=(-2, -1))
    target4_fft = torch.stack((target4_fft.real, target4_fft.imag), -1)

    source1_fft = torch.fft.fft2(source_1, dim=(-2, -1))
    source1_fft = torch.stack((source1_fft.real, source1_fft.imag), -1)

    source2_fft = torch.fft.fft2(source_2, dim=(-2, -1))
    source2_fft = torch.stack((source2_fft.real, source2_fft.imag), -1)

    source4_fft = torch.fft.fft2(source_4, dim=(-2, -1))
    source4_fft = torch.stack((source4_fft.real, source4_fft.imag), -1)

    pred11_fft = torch.fft.fft2(pred_1_1, dim=(-2, -1))
    pred11_fft = torch.stack((pred11_fft.real, pred11_fft.imag), -1)

    pred12_fft = torch.fft.fft2(pred_1_2, dim=(-2, -1))
    pred12_fft = torch.stack((pred12_fft.real, pred12_fft.imag), -1)

    pred14_fft = torch.fft.fft2(pred_1_4, dim=(-2, -1))
    pred14_fft = torch.stack((pred14_fft.real, pred14_fft.imag), -1)

    # pred21_fft = torch.fft.fft2(pred_2_1, dim=(-2, -1))
    # pred21_fft = torch.stack((pred21_fft.real, pred21_fft.imag), -1)

    pred22_fft = torch.fft.fft2(pred_2_2, dim=(-2, -1))
    pred22_fft = torch.stack((pred22_fft.real, pred22_fft.imag), -1)

    pred24_fft = torch.fft.fft2(pred_2_4, dim=(-2, -1))
    pred24_fft = torch.stack((pred24_fft.real, pred24_fft.imag), -1)

    # pred41_fft = torch.fft.fft2(pred_4_1, dim=(-2, -1))
    # pred41_fft = torch.stack((pred41_fft.real, pred41_fft.imag), -1)
    #
    # pred42_fft = torch.fft.fft2(pred_4_2, dim=(-2, -1))
    # pred42_fft = torch.stack((pred42_fft.real, pred42_fft.imag), -1)

    pred44_fft = torch.fft.fft2(pred_4_4, dim=(-2, -1))
    pred44_fft = torch.stack((pred44_fft.real, pred44_fft.imag), -1)

    d_ap1_1 = criterion(pred11_fft, target1_fft)
    d_an1_1 = criterion(pred11_fft, source1_fft)
    # d_an2_1 = criterion(pred11_fft, pred21_fft.detach())
    # d_an3_1 = criterion(pred11_fft, pred41_fft.detach())
    contrastive_1 = d_ap1_1 / d_an1_1

    d_ap1_2 = criterion(pred22_fft, target2_fft)
    d_ap2_2 = criterion(pred22_fft, pred12_fft.detach())
    d_an1_2 = criterion(pred22_fft, source2_fft)
    # d_an2_2 = criterion(pred22_fft, pred42_fft.detach())
    contrastive_2 = (0.625 * d_ap1_2 + 0.375 * d_ap2_2) / d_an1_2

    d_ap1_4 = criterion(pred44_fft, target4_fft)
    d_ap2_4 = criterion(pred44_fft, pred14_fft.detach())
    d_ap3_4 = criterion(pred44_fft, pred24_fft.detach())
    d_an1_4 = criterion(pred44_fft, source4_fft)
    contrastive_3 = (0.5 * d_ap1_4 + 0.375 * d_ap2_4 + 0.125 * d_ap3_4) / d_an1_4

    return contrastive_1 + contrastive_2 + contrastive_3


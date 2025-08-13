
# -*- coding: utf-8 -*-
import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = "2, 4, 5, 6"
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils import AverageMeter
from datasets.Rain_Dataloader import TrainData_for_DID, TestData_for_DID
from numpy import *

from models import *
from utils.utils import *
from pytorch_ssim import SSIM

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='PW_FNet_8448', type=str, help='model name')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
parser.add_argument('--exp', default='rain', type=str, help='experiment setting')
args = parser.parse_args()

torch.manual_seed(8001)
import pywt
import pywt.data
import torch.nn.functional as F

def train(train_loader, network, criterion, optimizer):
    losses = AverageMeter()

    # torch.cuda.empty_cache()

    network.train()
    for batch in train_loader:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        pred_img = network(source_img)
        label_img = target_img

        ya, (yh, yv, yd) = network.module.wavedec(label_img)
        label_img2 = torch.cat([ya, yh, yv, yd], dim=1)

        label_fft2 = torch.fft.fft2(label_img2, dim=(-2, -1))
        label_fft2 = torch.stack((label_fft2.real, label_fft2.imag), -1)

        label_fft3 = torch.fft.fft2(label_img, dim=(-2, -1))
        label_fft3 = torch.stack((label_fft3.real, label_fft3.imag), -1)

        pred_fft1 = torch.fft.fft2(pred_img[0], dim=(-2, -1))
        pred_fft1 = torch.stack((pred_fft1.real, pred_fft1.imag), -1)

        pred_fft2 = torch.fft.fft2(pred_img[1], dim=(-2, -1))
        pred_fft2 = torch.stack((pred_fft2.real, pred_fft2.imag), -1)

        pred_fft4 = torch.fft.fft2(pred_img[2], dim=(-2, -1))
        pred_fft4 = torch.stack((pred_fft4.real, pred_fft4.imag), -1)

        pred_fft5 = torch.fft.fft2(pred_img[3], dim=(-2, -1))
        pred_fft5 = torch.stack((pred_fft5.real, pred_fft5.imag), -1)

        f1 = criterion(pred_fft1, label_fft3)
        f2 = criterion(pred_fft2, label_fft2)
        f4 = criterion(pred_fft4, label_fft2)
        f5 = criterion(pred_fft5, label_fft3)
        loss_fft = f1 + f2 + f4 + f5

        loss = 0.1 * loss_fft
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg


def valid(val_loader_full, network):
    PSNR_full = AverageMeter()
    SSIM_full = AverageMeter()

    PSNR_full_1 = AverageMeter()
    SSIM_full_1 = AverageMeter()

    PSNR_full_2 = AverageMeter()
    SSIM_full_2 = AverageMeter()

    PSNR_full_3 = AverageMeter()
    SSIM_full_3 = AverageMeter()

    PSNR_full_4 = AverageMeter()
    SSIM_full_4 = AverageMeter()

    # torch.cuda.empty_cache()

    network.eval()

    for batch in val_loader_full:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        with torch.no_grad():  # torch.no_grad() may cause warning
            output_list = network(source_img)  # we change this to [0,1]?
            output = output_list[3].clamp_(0, 1)
            output_1 = output_list[0].clamp_(0, 1)
            output_2 = output_list[1]
            output_4 = output_list[2]

            ya, yh, yv, yd = torch.chunk(output_2, 4, dim=1)
            output_2 = network.module.waverec([ya, (yh, yv, yd)], None).clamp_(0, 1)
            ya, yh, yv, yd = torch.chunk(output_4, 4, dim=1)
            output_4 = network.module.waverec([ya, (yh, yv, yd)], None).clamp_(0, 1)


        psnr_full, sim = calculate_psnr_torch(target_img, output)
        PSNR_full.update(psnr_full.item(), source_img.size(0))

        ssim_full = sim
        SSIM_full.update(ssim_full.item(), source_img.size(0))

        psnr_full_1, sim_1 = calculate_psnr_torch(target_img, output_1)
        PSNR_full_1.update(psnr_full_1.item(), source_img.size(0))

        ssim_full_1 = sim_1
        SSIM_full_1.update(ssim_full_1.item(), source_img.size(0))

        psnr_full_2, sim_2 = calculate_psnr_torch(target_img, output_2)
        PSNR_full_2.update(psnr_full_2.item(), source_img.size(0))

        ssim_full_2 = sim_2
        SSIM_full_2.update(ssim_full_2.item(), source_img.size(0))

        psnr_full_4, sim_4 = calculate_psnr_torch(target_img, output_4)
        PSNR_full_4.update(psnr_full_4.item(), source_img.size(0))

        ssim_full_4 = sim_4
        SSIM_full_4.update(ssim_full_4.item(), source_img.size(0))

    return PSNR_full.avg, SSIM_full.avg, PSNR_full_1.avg, SSIM_full_1.avg, PSNR_full_2.avg, SSIM_full_2.avg, PSNR_full_3.avg, SSIM_full_3.avg, PSNR_full_4.avg, SSIM_full_4.avg


if __name__ == '__main__':
    setting_filename = os.path.join('configs', args.exp, args.model + '.json')
    print(setting_filename)
    if not os.path.exists(setting_filename):
        setting_filename = os.path.join('configs', args.exp, 'default.json')
    with open(setting_filename, 'r') as f:
        setting = json.load(f)

    device_index = [0, 1, 2, 3]
    network = eval(args.model.replace('-', '_'))()
    network = nn.DataParallel(network, device_ids=device_index).cuda()
    # network.load_state_dict(torch.load('/home/jxy/projects_dir/PW-FNet/saved_models/rain/Backbone_mimo_iccv_did_8448_best.pth')['state_dict'])

    criterion = nn.L1Loss()

    if setting['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=setting['lr'], eps=1e-8)
    elif setting['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(network.parameters(), lr=setting['lr'])
    else:
        raise Exception("ERROR: wrunsupported optimizer")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=setting['epochs'],
                                                           eta_min=1e-6)

    train_dir = '/home/jxy/projects_dir/datasets/DIDTrain'
    test_dir = '/home/jxy/projects_dir/datasets/DIDTest'
    train_dataset = TrainData_for_DID(256, train_dir)
    train_loader = DataLoader(train_dataset,
                              batch_size=setting['batch_size'],
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)

    test_dataset = TestData_for_DID(8, test_dir)
    test_loader = DataLoader(test_dataset,
                             batch_size=8,
                             shuffle=True,
                             num_workers=args.num_workers,
                             pin_memory=True)

    save_dir = os.path.join(args.save_dir, args.exp)
    os.makedirs(save_dir, exist_ok=True)

    # change test_str when you development new exp
    test_str = 'iccv_did'

    if not os.path.exists(os.path.join(save_dir, args.model + test_str + '.pth')):
        print('==> Start training, current model name: ' + args.model)

        writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp, args.model, test_str))

        best_psnr = 0
        best_ssim = 0

        for epoch in tqdm(range(setting['epochs'] + 1)):

            train_loss = train(train_loader, network, criterion, optimizer)

            writer.add_scalar('train_loss', train_loss, epoch)

            scheduler.step()  # TODO

            if epoch % setting['eval_freq'] == 0:

                avg_psnr, avg_ssim, avg_psnr_1, avg_ssim_1, avg_psnr_2, avg_ssim_2, avg_psnr_3, avg_ssim_3, avg_psnr_4, avg_ssim_4 = valid(test_loader, network)
                print(avg_psnr, avg_ssim, avg_psnr_1, avg_ssim_1, avg_psnr_2, avg_ssim_2, avg_psnr_3, avg_ssim_3, avg_psnr_4, avg_ssim_4)

                writer.add_scalar('valid_psnr', avg_psnr, epoch)
                writer.add_scalar('valid_ssim', avg_ssim, epoch)

                torch.save({'state_dict': network.state_dict()},
                           os.path.join(save_dir, args.model + test_str + '_newest' + '.pth'))

                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    torch.save({'state_dict': network.state_dict()},
                               os.path.join(save_dir, args.model + test_str + '_best' + '.pth'))
                writer.add_scalar('best_psnr', best_psnr, epoch)

                if avg_ssim > best_ssim:
                    best_ssim = avg_ssim
                writer.add_scalar('best_ssim', best_ssim, epoch)

    else:
        print('==> Existing trained model')
        exit(1)

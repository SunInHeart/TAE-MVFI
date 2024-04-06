import os
import sys
import time
import copy
import shutil
import random
import pdb

import torch
import numpy as np
from tqdm import tqdm

import config
import myutils
import torchvision.utils as utils
import math
import torch.nn.functional as F

from models import TAE_MVFI

torch.set_grad_enabled(False)  # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True  # make sure to use cudnn for computational performance

# arguments_intDevice = 0

##### Parse CmdLine Arguments #####
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
args, unparsed = config.get_args()
cwd = os.getcwd()

# torch.cuda.set_device(arguments_intDevice)

device = torch.device('cuda' if args.cuda else 'cpu')

torch.manual_seed(args.random_seed)
if args.cuda:
    torch.cuda.manual_seed(args.random_seed)


if args.dataset == "vimeo90K_triplet":
    from datasets.vimeo90k_triplet import get_loader
    test_loader = get_loader('test', args.data_root, args.test_batch_size, shuffle=False, num_workers=args.num_workers)
elif args.dataset == "vimeo90K_septuplet":
    from datasets.vimeo90k_septuplet import get_loader
    test_loader = get_loader('test', args.data_root, args.test_batch_size, shuffle=False, num_workers=args.num_workers)
elif args.dataset == "ucf101":
    from datasets.ucf101_test import get_loader
    test_loader = get_loader(args.model, args.data_root, args.test_batch_size, shuffle=False, num_workers=args.num_workers)
elif args.dataset == "Davis":
    from datasets.Davis_test import get_loader
    test_loader = get_loader(args.model, args.data_root, args.test_batch_size, shuffle=False, num_workers=args.num_workers, test_mode=True)

if args.model == 'TAE_MVFI_s':
    model = TAE_MVFI.Network(isMultiple=False).to(device)
elif args.model == 'TAE_MVFI_m':
    model = TAE_MVFI.Network(isMultiple=True).to(device)

print("Building model: %s" % args.model)
model = torch.nn.DataParallel(model).to(device)
print("#params", sum([p.numel() for p in model.parameters()]))

# TAE_MVFI_s
def test_TAE_MVFI_s(args):
    time_taken = []
    losses, psnrs, ssims = myutils.init_meters(args.loss)
    model.eval()

    with torch.no_grad():
        for i, (images, gt_image) in enumerate(tqdm(test_loader)):

            images = [img_.to(device) for img_ in images]
            gt = gt_image.to(device)

            torch.cuda.synchronize()
            start_time = time.time()
            out = model(images)

            torch.cuda.synchronize()
            time_taken.append(time.time() - start_time)

            myutils.eval_metrics(out, gt, psnrs, ssims)

    print("PSNR: %f, SSIM: %fn" %
          (psnrs.avg, ssims.avg))
    print("Time , ", sum(time_taken)/len(time_taken))

    return psnrs.avg

# TAE_MVFI_m
def test_TAE_MVFI_m(args):
    time_taken = []
    losses, psnrs, ssims = myutils.init_meters(args.loss)
    model.eval()

    with torch.no_grad():
        # for i, (images, gt_image, _) in enumerate(tqdm(test_loader)):
        for i, (images, gt_images) in enumerate(tqdm(test_loader)):

            images = [img_.to(device) for img_ in images]
            num = len(gt_images) + 1
            floatTimes = [num_ * (1 / num) for num_ in range(1, num)]

            for j, gt_image in enumerate(gt_images):
                time_torch = torch.ones((images[0].shape[0], 1, int(images[0].shape[2] / 2), int(images[0].shape[3] / 2))) * floatTimes[j]
                gt = gt_image.to(device)

                torch.cuda.synchronize()
                start_time = time.time()
                out = model([images[0], images[1], time_torch.to(device)])  # images is a list of neighboring frames

                torch.cuda.synchronize()
                time_taken.append(time.time() - start_time)

                myutils.eval_metrics(out, gt, psnrs, ssims)

    print("PSNR: %f, SSIM: %fn" %
          (psnrs.avg, ssims.avg))
    print("Time , ", sum(time_taken)/len(time_taken))

    return psnrs.avg


# select which mdoel to test
if args.model == 'TAE_MVFI_s':
    test = test_TAE_MVFI_s
elif args.model == 'TAE_MVFI_m':
    test = test_TAE_MVFI_m

""" Entry Point """
def main(args):
    
    assert args.load_from is not None

    # model_dict = model.state_dict()
    # original model test
    # model.load_state_dict(torch.load(args.load_from, map_location=lambda storage, loc: storage)['model_state'])
    # self-trained model test
    model.load_state_dict(torch.load(args.load_from)["state_dict"], strict=True)
    test(args)


if __name__ == "__main__":
    main(args)

import random
import time

import torch
from tqdm import tqdm

import config
import myutils
from loss import Loss
import shutil
import os

from models import TAE_MVFI

def load_checkpoint(args, model, optimizer, path):
    print("loading checkpoint %s" % path)
    checkpoint = torch.load(path)
    args.start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr = checkpoint.get("lr", args.lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


##### Parse CmdLine Arguments #####
args, unparsed = config.get_args()
cwd = os.getcwd()
print(args)

save_loc = os.path.join(args.checkpoint_dir, "checkpoints")

device = torch.device('cuda' if args.cuda else 'cpu')
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

torch.manual_seed(args.random_seed)
if args.cuda:
    torch.cuda.manual_seed(args.random_seed)


if args.model == 'TAE_MVFI_s':
    model = TAE_MVFI.Network(isMultiple=False).to(device)
    from datasets.vimeo90k_triplet import get_loader
elif args.model == 'TAE_MVFI_m':
    model = TAE_MVFI.Network(isMultiple=True).to(device)
    from datasets.vimeo90k_septuplet import get_loader
cropped_train_loader = get_loader('train', args.data_root, args.batch_size, shuffle=True, num_workers=args.num_workers)
finetune_train_loader = get_loader('train', args.data_root, args.batch_size, shuffle=True, num_workers=args.num_workers, finetune=True)
test_loader = get_loader('test', args.data_root, args.test_batch_size, shuffle=False, num_workers=args.num_workers)

save_loc = os.path.join(save_loc, args.model)
if not os.path.exists(save_loc):
    os.makedirs(save_loc)

print("Building model: %s" % args.model)
# model = UNet_3D_3D( n_inputs=args.nbr_frame, joinType=args.joinType)
model = torch.nn.DataParallel(model).to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('the number of network parameters: {}'.format(total_params))

##### Define Loss & Optimizer #####
criterion = Loss(args)

## ToDo: Different learning rate schemes for different parameters
# from torch.optim import Adamax
# optimizer = Adamax(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
from torch.optim import Adam
optimizer = Adam(params=model.parameters(), lr=args.lr)

# TAE_MVFI_s
def train_TAE_MVFI_s(args, epoch):
    torch.cuda.empty_cache()
    losses, psnrs, ssims = myutils.init_meters(args.loss)
    model.train()
    criterion.train()

    if epoch < args.max_epoch:
        train_loader = cropped_train_loader
    else:
        train_loader = finetune_train_loader

    for i, (images, gt_image) in enumerate(train_loader):

        # Build input batch
        images = [img_.to(device) for img_ in images]

        # Forward
        optimizer.zero_grad()
        # out_ll, out_l, out = model(images)
        out = model(images)

        gt = gt_image.to(device)

        loss, _ = criterion(out, gt)
        overall_loss = loss

        losses['total'].update(loss.item())

        overall_loss.backward()
        optimizer.step()

        # Calc metrics & print logs
        if i % args.log_iter == 0:
            myutils.eval_metrics(out, gt, psnrs, ssims)

            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}\tPSNR: {:.4f}  Lr:{:.6f}'.format(
                epoch, i, len(train_loader), losses['total'].avg, psnrs.avg,
                optimizer.param_groups[0]['lr'], flush=True))

            # Reset metrics
            losses, psnrs, ssims = myutils.init_meters(args.loss)


def test_TAE_MVFI_s(args, epoch):
    print('Evaluating for epoch = %d' % epoch)
    losses, psnrs, ssims = myutils.init_meters(args.loss)
    model.eval()
    criterion.eval()
    torch.cuda.empty_cache()

    with torch.no_grad():
        # for i, (images, gt_image, _) in enumerate(tqdm(test_loader)):
        for i, (images, gt_image) in enumerate(tqdm(test_loader)):

            images = [img_.to(device) for img_ in images]
            gt = gt_image.to(device)

            out = model(images)  # images is a list of neighboring frames

            # Save loss values
            loss, loss_specific = criterion(out, gt)
            for k, v in losses.items():
                if k != 'total':
                    v.update(loss_specific[k].item())
            losses['total'].update(loss.item())

            # Evaluate metrics
            myutils.eval_metrics(out, gt, psnrs, ssims)

    return losses['total'].avg, psnrs.avg, ssims.avg

# TAE_MVFI_m
def train_TAE_MVFI_m(args, epoch):
    torch.cuda.empty_cache()
    losses, psnrs, ssims = myutils.init_meters(args.loss)
    model.train()
    criterion.train()

    if epoch < args.max_epoch:
        train_loader = cropped_train_loader
    else:
        train_loader = finetune_train_loader

    for i, (images, gt_images) in enumerate(train_loader):
        # print('images, gt_images: ', len(images), len(gt_images))

        # Build input batch
        images = [img_.to(device) for img_ in images]
        num = len(gt_images) + 1  # 6
        floatTimes = [num_ * (1 / num) for num_ in range(1, num)]  # [0.16666666666666666, 0.3333333333333333, 0.5, 0.6666666666666666, 0.8333333333333333]

        randnum = random.sample(range(0, 5), 2)
        for j in randnum:
            # Forward
            optimizer.zero_grad()
            # print(images[0].shape)
            time_torch = torch.ones((images[0].shape[0], 1, int(images[0].shape[2] / 2), int(images[0].shape[3] / 2))) * \
                         floatTimes[j]
            # out_ll, out_l, out = model(images)
            out = model([images[0], images[1], time_torch.to(device)])

            gt = gt_images[j].to(device)
            # print(out.shape, gt.shape)
            loss, _ = criterion(out, gt)
            overall_loss = loss

            losses['total'].update(loss.item())

            overall_loss.backward()
            optimizer.step()

        # Calc metrics & print logs
        if i % args.log_iter == 0:
            myutils.eval_metrics(out, gt, psnrs, ssims)

            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}\tPSNR: {:.4f}  Lr:{:.6f}'.format(
                epoch, i, len(train_loader), losses['total'].avg, psnrs.avg,
                optimizer.param_groups[0]['lr'], flush=True))

            # Reset metrics
            losses, psnrs, ssims = myutils.init_meters(args.loss)


def test_TAE_MVFI_m(args, epoch):
    print('Evaluating for epoch = %d' % epoch)
    losses, psnrs, ssims = myutils.init_meters(args.loss)
    model.eval()
    criterion.eval()
    torch.cuda.empty_cache()

    with torch.no_grad():
        # for i, (images, gt_image, _) in enumerate(tqdm(test_loader)):
        for i, (images, gt_images) in enumerate(tqdm(test_loader)):

            images = [img_.to(device) for img_ in images]
            num = len(gt_images) + 1
            floatTimes = [num_ * (1 / num) for num_ in range(1, num)]

            for j, gt_image in enumerate(gt_images):
                time_torch = torch.ones((images[0].shape[0], 1, int(images[0].shape[2] / 2), int(images[0].shape[3] / 2))) * floatTimes[j]

                out = model([images[0], images[1], time_torch.to(device)])  # images is a list of neighboring frames

                gt = gt_image.to(device)
                # Save loss values
                loss, loss_specific = criterion(out, gt)
                for k, v in losses.items():
                    if k != 'total':
                        v.update(loss_specific[k].item())
                losses['total'].update(loss.item())

                # Evaluate metrics
                myutils.eval_metrics(out, gt, psnrs, ssims)

    return losses['total'].avg, psnrs.avg, ssims.avg


def print_log(epoch, model_name, num_epochs, one_epoch_time, oup_pnsr, oup_ssim, Lr):
    print('({0:.0f}s) Epoch [{1}/{2}], Val_PSNR:{3:.2f}, Val_SSIM:{4:.4f}'
          .format(one_epoch_time, epoch, num_epochs, oup_pnsr, oup_ssim))
    # write training log
    with open('./training_log/' + model_name + '_train_log.txt', 'a') as f:
        print(
            'Date: {0}s, Time_Cost: {1:.0f}s, Epoch: [{2}/{3}], Val_PSNR:{4:.2f}, Val_SSIM:{5:.4f}, Lr:{6}'
            .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    one_epoch_time, epoch, num_epochs, oup_pnsr, oup_ssim, Lr), file=f)


if args.batch_size == 2:
    lr_schular = [5e-5, 2.5e-5, 1.25e-5, 5e-6, 2.5e-6]
if args.batch_size == 4:
    lr_schular = [1e-4, 5e-5, 2.5e-5, 1.25e-5, 5e-6]
if args.batch_size == 8:
    lr_schular = [2e-4, 1e-4, 5e-5, 2.5e-5, 1.25e-5]
if args.batch_size == 16:
    lr_schular = [4e-4, 2e-4, 1e-4, 5e-5, 2.5e-5]
training_schedule = [40, 70, 90, 100, 110]


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for i in range(len(training_schedule)):
        if epoch < training_schedule[i]:
            current_learning_rate = lr_schular[i]
            break

    for param_group in optimizer.param_groups:
        param_group['lr'] = current_learning_rate
        print('Learning rate sets to {}.'.format(param_group['lr']))

# select which model to train
if args.model == 'TAE_MVFI_s':
    train = train_TAE_MVFI_s
    test = test_TAE_MVFI_s
elif args.model == 'TAE_MVFI_m':
    train = train_TAE_MVFI_m
    test = test_TAE_MVFI_m

""" Entry Point """
def main(args):
    # load_checkpoint(args, model, optimizer, save_loc+'/epoch20/model_best.pth')
    # test_loss, psnr, ssim = test(args, args.start_epoch)
    # print(psnr)
    if args.resume:
        load_checkpoint(args, model, optimizer, save_loc+'/checkpoint.pth')  # load checkbreakpoint for resume training

    if args.start_epoch == 0:
        best_psnr = 0
    else:
        _, best_psnr, _ = test(args, args.start_epoch)

    for epoch in range(args.start_epoch, args.max_epoch+args.finetune_epochs):
        adjust_learning_rate(optimizer, epoch)
        start_time = time.time()
        train(args, epoch)

        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr': optimizer.param_groups[-1]['lr']
        }, os.path.join(save_loc, 'checkpoint.pth'))

        test_loss, psnr, ssim = test(args, epoch)

        # save checkpoint
        is_best = psnr > best_psnr
        best_psnr = max(psnr, best_psnr)
        if is_best:
            shutil.copyfile(os.path.join(save_loc, 'checkpoint.pth'), os.path.join(save_loc, 'model_best.pth'))

        # if epoch+1 == args.max_epoch:
        #     shutil.copyfile(os.path.join(save_loc, 'checkpoint.pth'), os.path.join(save_loc, 'checkpoint_max_epoch.pth'))

        one_epoch_time = time.time() - start_time
        print_log(epoch, args.model, args.max_epoch+args.finetune_epochs, one_epoch_time, psnr, ssim, optimizer.param_groups[-1]['lr'])

if __name__ == "__main__":
    main(args)

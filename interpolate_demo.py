import os
import torch

import config
from PIL import Image
from torchvision import transforms
import torchvision.utils as utils

import math
from models import TAE_MVFI

torch.set_grad_enabled(False)  # make sure to not compute gradients for computational performance
torch.backends.cudnn.enabled = True  # make sure to use cudnn for computational performance

# Parse CmdLine Arguments
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# interpolate
interp_arg = config.add_argument_group("Interp")
interp_arg.add_argument('--input_path', type=str, default='./inter_data/')
interp_arg.add_argument('--first', type=str, default='im1.png')
interp_arg.add_argument('--second', type=str, default='im2.png')
interp_arg.add_argument('--out_path', type=str, default='./out_data/')
interp_arg.add_argument('--time', type=float, default=0.5)

args, unparsed = config.get_args()
cwd = os.getcwd()

device = torch.device('cuda' if args.cuda else 'cpu')

torch.manual_seed(args.random_seed)
if args.cuda:
    torch.cuda.manual_seed(args.random_seed)

print("Building model: %s" % args.model)
if args.model == 'TAE_MVFI_s':
    model = TAE_MVFI.Network(isMultiple=False).to(device)
elif args.model == 'TAE_MVFI_m':
    model = TAE_MVFI.Network(isMultiple=True).to(device)
# use self-trained model open it
model = torch.nn.DataParallel(model).to(device)
print("#params", sum([p.numel() for p in model.parameters()]))


def generate(args):
    model.eval()

    # transform
    trans = transforms.Compose([
        transforms.ToTensor()
    ])

    # get images
    img1 = trans(Image.open(args.input_path + args.first)).to(device).unsqueeze(0)
    img2 = trans(Image.open(args.input_path + args.second)).to(device).unsqueeze(0)

    assert img1.shape == img2.shape

    # addPadding = torch.nn.ReplicationPad2d(
    #     [0, int((math.ceil(img1.size(3) / 32.0) * 32 - img1.size(3))), 0,
    #      int((math.ceil(img1.size(2) / 32.0) * 32 - img1.size(2)))])
    # removePadding = torch.nn.ReplicationPad2d(
    #     [0, 0 - int((math.ceil(img1.size(3) / 32.0) * 32 - img1.size(3))), 0,
    #      0 - int((math.ceil(img1.size(2) / 32.0) * 32 - img1.size(2)))])

    # img1_padded = addPadding(img1)
    # img2_padded = addPadding(img2)

    with torch.no_grad():

        # img1_padded = img1_padded.unsqueeze(0)
        # img2_padded = img2_padded.unsqueeze(0)
        if args.model == 'TAE_MVFI_s':
            # out = model([img1_padded, img2_padded])
            out = model([img1, img2])
        elif args.model == 'TAE_MVFI_m':
            # time_torch = torch.ones((1, 1, int(img1_padded.shape[2] / 2), int(img1_padded.shape[3] / 2))) * args.time
            time_torch = torch.ones((1, 1, int(img1.shape[2] / 2), int(img1.shape[3] / 2))) * args.time
            # out = model([img1_padded, img2_padded, time_torch.to(device)])
            out = model([img1, img2, time_torch.to(device)])
        # out = removePadding(out)
        # print(out)
        utils.save_image(out, args.out_path + 'out.'+args.first.split('.')[-1])
        print("result saved!")


""" Entry Point """
def main(args):

    assert args.load_from is not None
    # use original model
    # model.load_state_dict(torch.load(args.load_from, map_location=lambda storage, loc: storage)['model_state'])
    # use self-trained model
    model.load_state_dict(torch.load(args.load_from)["state_dict"], strict=True)
    generate(args)


if __name__ == "__main__":
    main(args)

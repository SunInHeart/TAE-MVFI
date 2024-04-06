import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random

class VimeoTriplet(Dataset):
    def __init__(self, data_root, is_training, input_frames="13", mode='full', finetune=False):  # original mode='mini'
        """
        Creates a Vimeo Septuplet object.
        Inputs.
            data_root: Root path for the Vimeo dataset containing the sep tuples.
            is_training: Train/Test.
            input_frames: Which frames to input for frame interpolation network.
        """
        self.data_root = data_root
        self.image_root = os.path.join(self.data_root, 'sequences')
        self.training = is_training
        self.inputs = input_frames

        train_fn = os.path.join(self.data_root, 'tri_trainlist.txt')
        test_fn = os.path.join(self.data_root, 'tri_testlist.txt')
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()

        if mode != 'full':
            tmp = []
            for i, value in enumerate(self.testlist):
                if i % 38 == 0:
                    tmp.append(value)
            self.testlist = tmp

        if self.training:
            if finetune:
                self.transforms = transforms.Compose([
                    transforms.ToTensor()
                ])
            else:
                self.transforms = transforms.Compose([
                    transforms.RandomCrop(256),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    # transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
                    transforms.ToTensor()
                ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor()
            ])

    def __getitem__(self, index):
        if self.training:
            imgpath = os.path.join(self.image_root, self.trainlist[index])
        else:
            imgpath = os.path.join(self.image_root, self.testlist[index])

        imgpaths = [imgpath + f'/im{i}.png' for i in range(1, 4)]

        pth_ = imgpaths

        # Load images
        images = [Image.open(pth) for pth in imgpaths]  # [im1, im2, im3]

        ## Select only relevant inputs
        # inputs = [int(e)-1 for e in list(self.inputs)]  # [0, 2]
        # inputs = inputs[:len(inputs)//2] + [1] + inputs[len(inputs)//2:]  # [0, 1, 2]
        # images = [images[i] for i in inputs]  # [im1, im2, im3]
        # imgpaths = [imgpaths[i] for i in inputs]

        # Data augmentation
        if self.training:
            seed = random.randint(0, 2**32)
            images_ = []
            for img_ in images:
                random.seed(seed)
                images_.append(self.transforms(img_))
            images = images_
            # Random Temporal Flip
            if random.random() >= 0.5:
                images = images[::-1]
                imgpaths = imgpaths[::-1]

        else:
            T = self.transforms
            images = [T(img_) for img_ in images]

            # imgpath = '_'.join(imgpath.split('/')[-2:])

        gt = images[1]  # im2
        images = images[:1] + images[-1:]  # [im1, im3]
        # return images, gt, imgpath
        return images, gt

    def __len__(self):
        if self.training:
            return len(self.trainlist)
        else:
            return len(self.testlist)
            # return 1

def get_loader(mode, data_root, batch_size, shuffle, num_workers, test_mode=None, finetune=False):
    if mode == 'train':
        is_training = True
    else:
        is_training = False
    dataset = VimeoTriplet(data_root, is_training=is_training, finetune=finetune)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


if __name__ == "__main__":

    dataset = VimeoTriplet("/home/zhihao/DATA-M2/video_interpolation//vimeo_septuplet/", is_training=True)
    print(dataset[0])
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=32, pin_memory=True)

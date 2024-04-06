import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# TAE_MVFI_s
class TAE_MVFI_s_Davis(Dataset):
    def __init__(self, data_root, ext="png"):

        super().__init__()

        self.data_root = data_root
        self.images_sets = []

        for label_id in os.listdir(self.data_root):

            ctg_imgs_ = sorted(os.listdir(os.path.join(self.data_root, label_id)))
            ctg_imgs_ = [os.path.join(self.data_root, label_id, img_id) for img_id in ctg_imgs_]
            for start_idx in range(0, len(ctg_imgs_)-2, 2):
                add_files = ctg_imgs_[start_idx: start_idx+3]
                self.images_sets.append(add_files)

        self.transforms = transforms.Compose([
                transforms.CenterCrop((480, 848)),
                transforms.ToTensor()
            ])

        print(len(self.images_sets))

    def __getitem__(self, idx):

        imgpaths = self.images_sets[idx]
        images = [Image.open(img) for img in imgpaths]
        images = [self.transforms(img) for img in images]

        return images[:1] + images[-1:], images[1]

    def __len__(self):

        return len(self.images_sets)

# TAE_MVFI_m
class TAE_MVFI_m_Davis(Dataset):
    def __init__(self, data_root, ext="png"):

        super().__init__()

        self.data_root = data_root
        self.images_sets = []

        for label_id in os.listdir(self.data_root):

            ctg_imgs_ = sorted(os.listdir(os.path.join(self.data_root, label_id)))
            ctg_imgs_ = [os.path.join(self.data_root, label_id, img_id) for img_id in ctg_imgs_]
            for start_idx in range(0, len(ctg_imgs_)-6, 2):
                add_files = ctg_imgs_[start_idx: start_idx+7]
                self.images_sets.append(add_files)

        self.transforms = transforms.Compose([
                transforms.CenterCrop((480, 848)),
                transforms.ToTensor()
            ])

        print(len(self.images_sets))

    def __getitem__(self, idx):

        imgpaths = self.images_sets[idx]
        images = [Image.open(img) for img in imgpaths]
        images = [self.transforms(img) for img in images]

        return images[:1] + images[-1:], images[1:-1]

    def __len__(self):

        return len(self.images_sets)

def get_loader(model, data_root, batch_size, shuffle, num_workers, test_mode=True):
    if model == 'TAE_MVFI_s':
        Davis = TAE_MVFI_s_Davis
    elif model == 'TAE_MVFI_m':
        Davis = TAE_MVFI_m_Davis

    dataset = Davis(data_root)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

if __name__ == "__main__":

    dataset = Davis("./Davis_test/")

    print(len(dataset))

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
import os
import cv2
# from torch.utils.tensorboard import SummaryWriter
import torch.nn
from torch.utils.data import Dataset
from torchvision import transforms
import natsort
import random
from PIL import Image
import numpy as np

from networks import *

def get_random_crop(image, crop_height, crop_width):

    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    crop = image[y: y + crop_height, x: x + crop_width]

    return crop

class ImageLoader:
    def __init__(self, path_to_folder: str):
        self.input_folder_path: str = path_to_folder

    def __iter__(self):
        for file in sorted(os.listdir(self.input_folder_path), key=lambda x: int(x[:-4])):
            img = cv2.imread(os.path.join(self.input_folder_path, file), cv2.IMREAD_COLOR)
            yield img

class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform = None):
        self.main_dir = main_dir
        # self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    # def __getitem__(self, idx):
    #     img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
    #     image = Image.open(img_loc).convert("RGB")
    #     # tensor_image = self.transform(image)
    #     return image

    def get_batch(self, batch_size=10, crop_size=32):
        batch = []
        for i in range(batch_size):
            random_ind = random.randrange(len(self))
            im = cv2.imread(f"{self.main_dir}//{str(random_ind).zfill(4)}.png")
            random_crop = get_random_crop(im, 32, 32)
            batch.append(random_crop)
            # print(f"{os.getcwd()}//{str(random_ind).zfill(4)}.png")
            # print(im.shape)
        return batch

if __name__ == "__main__":
    dataset = CustomDataSet(os.getcwd() + "//DIV2K_train_HR")
    b = dataset.get_batch()
    print([im.shape for im in b])
    # it = enumerate(dataset)
    # print(type(dataset))
    # print(type(it))
    # im = dataset[5]
    # print(im.size)
    # transforms_ = torch.nn.Sequential(
    #     transforms.ToTensor(),
    #     transforms.RandomCrop(size=(32, 32))
    #
    # )
    # scripted_transforms = torch.jit.script(transforms_)
    # image_loader = enumerate(ImageLoader(os.getcwd() + "//DIV2K_train_HR"))
    # net = SuperResolutionNetwork(G0=4, G=4, d=2)
    # # print(net)
    # img = cv2.imread(os.getcwd() + "//DIV2K_train_HR//b.png")
    # tensor_ = torch.Tensor(img)
    # tensor_ = torch.permute(tensor_, (2, 1, 0))
    # tensor_ = torch.unsqueeze(tensor_, 0)
    # # print(tensor_.shape)
    # b = net(tensor_)
    # print(b.shape)
    # #index, image = next(image_loader)
    # # image = torch.Tensor(image)
    # # writer.add_graph(net, image)
    # # writer.close()
    #
    # # for idx, image in enumerate(image_loader):
    # #     print(idx)
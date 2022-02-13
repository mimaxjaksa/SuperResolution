import os
import cv2
import numpy as np
import torch
# import torch.nn as nn
# import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
import natsort
from networks import *

class ImageLoader:
    def __init__(self, path_to_folder: str):
        self.input_folder_path: str = path_to_folder

    def __iter__(self):
        for file in sorted(os.listdir(self.input_folder_path), key=lambda x: int(x[:-4])):
            img = cv2.imread(os.path.join(self.input_folder_path, file), cv2.IMREAD_COLOR)
            yield img

class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image

if __name__ == "__main__":
    writer = SummaryWriter('example')
    image_loader = enumerate(ImageLoader(os.getcwd() + "//DIV2K_train_HR"))
    net = SuperResolutionNetwork(G0=4, G=4, d=2)
    # print(net)
    img = cv2.imread(os.getcwd() + "//DIV2K_train_HR//b.png")
    tensor_ = torch.Tensor(img)
    tensor_ = torch.permute(tensor_, (2, 1, 0))
    tensor_ = torch.unsqueeze(tensor_, 0)
    # print(tensor_.shape)
    b = net(tensor_)
    print(b.shape)
    #index, image = next(image_loader)
    # image = torch.Tensor(image)
    # writer.add_graph(net, image)
    # writer.close()

    # for idx, image in enumerate(image_loader):
    #     print(idx)
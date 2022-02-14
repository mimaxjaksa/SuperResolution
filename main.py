import os
import cv2
# from torch.utils.tensorboard import SummaryWriter
import torch.nn
from torch.utils.data import Dataset
import torch.optim as optim
from torchvision import transforms
import natsort
import random
from PIL import Image
import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
import time
import argparse

from networks import *

def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('--mode', dest = 'mode', type = str, default = 'train')
	parser.add_argument('--model', dest='model_name', type=str, default=f"{os.getcwd()}//models//1.pt")

	args = parser.parse_args()

	return args

time_format = '%D_%M_%S'

def downscale_image(input_image = None, ratio = 2, method = 'simple'):
    if input_image is not None:
        AssertionError("No input image")
    if method != 'simple':
        return
    w, h, c = input_image.shape
    if w%ratio != 0 or h%ratio != 0:
        AssertionError("Image not resizeable")
    new_im = np.zeros([int(w/ratio), int(h/ratio), 3])
    im = input_image.astype(int)
    for i in range(new_im.shape[0]):
        for j in range(new_im.shape[1]):
            new_im[i, j, :] = (im[2*i, 2*j, :] + im[2*i + 1, 2*j, :] + im[2*i, 2*j + 1, :] + im[2*i + 1, 2*j + 1, :])/4
    return new_im.astype(np.uint8)

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
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)
        self.train_len = int(0.8*len(self))
        self.test_len = int(0.1*len(self))
        self.val_len = len(self) - self.train_len - self.test_len

    def __len__(self):
        return len(self.total_imgs)

    def get_train_batch(self, batch_size=10, crop_size=32):
        batch = []
        for i in range(batch_size):
            random_ind = random.randrange(1, self.train_len)
            im = cv2.imread(f"{self.main_dir}//{str(random_ind).zfill(4)}.png")
            random_crop = get_random_crop(im, crop_size, crop_size)
            batch.append(random_crop)
        return batch

    def get_random_test_image(self):
        random_ind = random.randrange(len(self) - self.test_len, len(self))
        return cv2.imread(f"{self.main_dir}//{str(random_ind).zfill(4)}.png")

def write_loss_to_file(loss):
    with open("loss.txt", "a") as f:
        f.write(str(loss) + '\n')

if __name__ == "__main__":
    args = parse_args()
    # Hyperparameters
    crop_size = 32
    batch_size = 16
    epoch_count = 200
    backpropagations_per_epoch = 512 # -> 512/16 = 32 batches per epoch
    G0 = 64
    G = 64
    d = 15
    c = 6
    ratio = 2
    lr = 1e-4

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = CustomDataSet(os.getcwd() + "//DIV2K_train_HR")
    loss_fn = torch.nn.MSELoss()
    net = SuperResolutionNetwork(G0=G0, G=G, c=c, d=d, ratio=ratio).to(device)
    if args.mode == 'train':
        print(f"Starting training on {device} at {time.strftime(time_format)}")
        optimizer = optim.Adam(net.parameters(), lr=lr)

        losses = []
        for epoch_i in range(epoch_count):
            running_loss = 0.0
            for batch_i in range(int(backpropagations_per_epoch/batch_size)):
                batch = dataset.get_train_batch(batch_size=batch_size, crop_size=crop_size)
                for im in batch:
                    optimizer.zero_grad()
                    input = torch.Tensor(downscale_image(im))
                    input = torch.unsqueeze(torch.permute(input, (2, 1, 0)), 0).to(device)
                    output = net(input)
                    target = torch.unsqueeze(torch.permute(torch.Tensor(im), (2, 1, 0)), 0).to(device)
                    loss = loss_fn(output, target)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
            write_loss_to_file(running_loss/backpropagations_per_epoch)
            if epoch_i % 10 == 0:
                print(f"Finished {epoch_i}th epoch, current loss: {running_loss} at {time.strftime(time_format)}")
                torch.save(net, f"{os.getcwd()}//models//{epoch_i}.pt")
        print(losses)
    else:
        # print(args.model_name)
        # checkpoint = torch.jit.load(args.model_name)
        # net.load_state_dict(checkpoint['model_state_dict'])
        net = torch.load(args.model_name).to('cpu')
        net.eval()
        while True:
            test_im = dataset.get_random_test_image()
            test_im = get_random_crop(test_im, crop_height=500, crop_width=500)
            input_image = downscale_image(test_im, ratio=2)
            input_image = torch.unsqueeze(torch.permute(torch.Tensor(input_image), (2, 1, 0)), 0)
            output = net(input_image)
            loss = loss_fn(output, torch.unsqueeze(torch.permute(torch.Tensor(test_im), (2, 1, 0)), 0))
            print(loss.item())
            a = torch.permute(torch.squeeze(output), (2, 1, 0)).detach().numpy().astype(np.uint8)
            cv2.imshow("OUTPUT", a)
            cv2.imshow("TARGET", test_im)
            cv2.waitKey(0)
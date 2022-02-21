import os
import time

import cv2
import torch.optim as optim
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

from networks import SuperResolutionNetwork
from utils import *


import torch


if __name__ == "__main__":
    args = parse_args()
    hyperparameters = {
        'crop_size': 32,
        'batch_size': 16,
        'epoch_count': 201,
        'backpropagations_per_epoch': 1024,
        'G0': 64,
        'G': 64,
        'd': 20,
        'c': 6,
        'ratio': 2,
        'lr': 1e-4
    }

    dataset = CustomDataSet(os.getcwd() + "//DIV2K_train_HR")
    device = 'cpu' if args.mode == 'test' or not torch.cuda.is_available() else 'cuda'
    net = SuperResolutionNetwork(**hyperparameters).to(device)
    optimizer = optim.Adam(net.parameters(), lr=hyperparameters["lr"])
    if args.mode == 'train':
        loss_fn = SSIM() # Loss to minimize
        hyperparameters['loss'] = loss_fn.name
        start_time = time.strftime(time_format)
        print(f"Starting training on {device} at {start_time}")
        models_folder_name = f"models_{start_time}"
        os.mkdir(models_folder_name)
        save_hyperparameters(models_folder_name, hyperparameters)

        losses = []
        for epoch_i in range(hyperparameters["epoch_count"]):
            if epoch_i % 40:
                hyperparameters["lr"] = hyperparameters["lr"] / 5
                optimizer = optim.Adam(net.parameters(), lr=hyperparameters["lr"])
            running_loss = 0.0
            for batch_i in range(int(hyperparameters["backpropagations_per_epoch"]/hyperparameters["batch_size"])):
                batch = dataset.get_train_batch(batch_size=hyperparameters["batch_size"], crop_size=hyperparameters["crop_size"])
                for im in batch:
                    optimizer.zero_grad()
                    input = np_to_tensor(downscale_image(im), device)
                    target = np_to_tensor(im, device)
                    output = net(input)
                    loss = loss_fn(output, target)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()/hyperparameters["backpropagations_per_epoch"]
            if epoch_i % 10 == 0:
                print(f"Finished {epoch_i}th epoch, current loss: {running_loss} at {time.strftime(time_format)}")
                save_all(net, optimizer, running_loss, os.path.join(os.getcwd(), models_folder_name, f"{epoch_i}.pt"))
    elif args.mode == "test":
        mse_loss = torch.nn.MSELoss()
        psnr_loss = PSNR()
        combined_loss = Combined_loss()
        ssim_loss = pytorch_ssim.SSIM(window_size=11)
        checkpoint = torch.load(args.model_name)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        train_loss = checkpoint['loss']
        net.eval()
        test_im = dataset.get_random_test_image(crop_len=500)
        input_image = downscale_image(test_im, ratio=2)
        upscaled_image = upscale_image(input_image, ratio = 2)
        input_image = np_to_tensor(input_image, device)
        output = net(input_image)
        output = tensor_to_np(output)
        cv2.imwrite("1output.png", output)
        cv2.imwrite("2target.png", test_im)
        cv2.imwrite("0upscaled.png", upscaled_image)
    else: #Plotting loss values for models in a given folder
        folder_name = args.folder_name
        losses = []
        for name in os.listdir(os.path.join(os.getcwd(), folder_name)):
            checkpoint = torch.load(os.path.join(os.getcwd(), folder_name, name))
            losses.append(checkpoint['loss'])
        plt.plot(losses)
        plt.grid()
        plt.ylabel("MSE+PSNR")
        plt.xlabel("epochs")
        plt.title("Loss function")
        plt.waitforbuttonpress()

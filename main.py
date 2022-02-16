import os
import time

import cv2
import numpy as np
import torch.optim as optim

from networks import SuperResolutionNetwork
from utils import *

if __name__ == "__main__":
    args = parse_args()
    hyperparameters = {
        'crop_size': 32,
        'batch_size': 16,
        'epoch_count': 200,
        'backpropagations_per_epoch': 1024,
        'G0': 64,
        'G': 64,
        'd': 20,
        'c': 6,
        'ratio': 2,
        'lr': 1e-4
    }

    dataset = CustomDataSet(os.getcwd() + "//DIV2K_train_HR")
    loss_fn = torch.nn.MSELoss()
    device = 'cpu' if args.mode == 'test' or not torch.cuda.is_available() else 'cuda'
    net = SuperResolutionNetwork(**hyperparameters).to(device)
    optimizer = optim.Adam(net.parameters(), lr=hyperparameters["lr"])
    if args.mode == 'train':
        start_time = time.strftime(time_format)
        print(f"Starting training on {device} at {start_time}")
        models_folder_name = f"models_{start_time}"
        os.mkdir(models_folder_name)

        losses = []
        for epoch_i in range(hyperparameters["epoch_count"]):
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
            write_loss_to_file(running_loss)
            if epoch_i % 10 == 0:
                print(f"Finished {epoch_i}th epoch, current loss: {running_loss} at {time.strftime(time_format)}")
                save_all(net, optimizer, running_loss, os.path.join(os.getcwd(), models_folder_name, f"{epoch_i}.pt"))
                # torch.save(net, f"{os.getcwd()}//{models_folder_name}//{epoch_i}.pt")
    else:
        # net = torch.load(args.model_name).to(device)
        # net.eval()
        checkpoint = torch.load(args.model_name)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        train_loss = checkpoint['loss']
        net.eval()
        test_im = dataset.get_random_test_image(crop_len=500)
        input_image = downscale_image(test_im, ratio=2)
        cv2.imwrite("input_image.png", input_image)
        input_image = np_to_tensor(input_image, device)
        output = net(input_image)
        loss = loss_fn(output, np_to_tensor(test_im, device))
        output = tensor_to_np(output)
        diff = (output.astype(int) - test_im.astype(int)).astype(np.uint8)
        print(np.max(np.max(diff)))
        cv2.imwrite("0output.png", output)
        cv2.imwrite("1target.png", test_im)
        cv2.imwrite("2diff.png", diff)
        pass

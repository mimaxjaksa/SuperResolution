import time

import cv2
import torch.optim as optim

from networks import SuperResolutionNetwork
from utils import *

if __name__ == "__main__":
    args = parse_args()
    hyperparameters = {
        'crop_size': 32,
        'batch_size': 16,
        'epoch_count': 300,
        'backpropagations_per_epoch': 128,
        'G0': 64,
        'G': 64,
        'd': 20,
        'c': 6,
        'ratio': 2,
        'lr': 1e-4
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = CustomDataSet(os.getcwd() + "//DIV2K_train_HR")
    loss_fn = torch.nn.MSELoss()
    net = SuperResolutionNetwork(**hyperparameters).to(device)
    if args.mode == 'train':
        start_time = time.strftime(time_format)
        print(f"Starting training on {device} at {start_time}")
        models_folder_name = f"models_{start_time}"
        os.mkdir(models_folder_name)
        optimizer = optim.Adam(net.parameters(), lr=hyperparameters["lr"])

        losses = []
        for epoch_i in range(hyperparameters["epoch_count"]):
            running_loss = 0.0
            for batch_i in range(int(hyperparameters["backpropagations_per_epoch"]/hyperparameters["batch_size"])):
                batch = dataset.get_train_batch(batch_size=hyperparameters["batch_size"], crop_size=hyperparameters["crop_size"])
                for im in batch:
                    optimizer.zero_grad()
                    input = np_to_tensor(downscale_image(im))
                    target = np_to_tensor(im)
                    output = net(input)
                    loss = loss_fn(output, target)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()/hyperparameters["backpropagations_per_epoch"]
                    pass
            write_loss_to_file(running_loss)
            if epoch_i % 2 == 0:
                print(f"Finished {epoch_i}th epoch, current loss: {running_loss} at {time.strftime(time_format)}")
                torch.save(net, f"{os.getcwd()}//{models_folder_name}//{epoch_i}.pt")
    else:
        net = torch.load(args.model_name).to('cpu')
        net.eval()
        while True:
            test_im = dataset.get_random_test_image(crop_len=500)
            input_image = downscale_image(test_im, ratio=2)
            input_image = np_to_tensor(input_image)
            output = net(input_image)
            loss = loss_fn(output, np_to_tensor(test_im))
            cv2.imshow("OUTPUT", tensor_to_np(output))
            cv2.imshow("TARGET", test_im)
            cv2.waitKey(0)
import torch
import torch.nn as nn
import torch.nn.functional as F

class ShallowFeatureExtraction(nn.Module):
    def __init__(self, in_, out_):
        super(ShallowFeatureExtraction, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_, out_channels=out_, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        return self.conv(x)

class ResidualDenseBlock(nn.Module):
    def __init__(self, G0 = 64, G = 64, c = 6):
        super(ResidualDenseBlock, self).__init__()
        layer_list = []
        for i in range(c):
            in_dim = G0 + i*G
            layer_list.append(nn.Conv2d(in_channels=in_dim, out_channels=G, kernel_size=(3, 3), padding=1))
        self.layers = nn.ModuleList(layer_list)
        self.final_conv = nn.Conv2d(in_channels=(G0 + c*G), out_channels=G0, kernel_size=(1, 1))

    def forward(self, x_):
        ins = [x_]
        for layer in self.layers:
            x = F.relu(layer(torch.cat(ins, dim=1)))
            ins.append(x)
        x = torch.cat(ins, dim=1)
        x = self.final_conv(x)
        return (x + x_)

class SuperResolutionNetwork(nn.Module):
    def __init__(self, G0 = 64, G = 64, c = 6, d = 20, ratio = 2):
        super(SuperResolutionNetwork, self).__init__()
        if G0 % ratio**2:
            AssertionError(f"Feature map count (G0) has to be a divisible by upscale ratio")
        self.pixel_shuffle = nn.PixelShuffle(ratio)
        self.conv1 = ShallowFeatureExtraction(in_=3,  out_=G0)
        self.conv2 = ShallowFeatureExtraction(in_=G0, out_=G0)
        block_list = d*[ResidualDenseBlock(G0=G0, G=G, c=c)]
        self.blocks = nn.ModuleList(block_list)
        self.conv3 = nn.Conv2d(in_channels = d*G0, out_channels=G0, kernel_size=(1, 1))
        self.conv4 = nn.Conv2d(in_channels=G0, out_channels=G0, kernel_size=(3, 3), padding=1)
        self.upscale_conv = nn.Conv2d(in_channels=G0, out_channels=G0,kernel_size=(3, 3), padding=1)
        self.conv5 = nn.Conv2d(in_channels=int(G0/(ratio**2)), out_channels=3, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        F_1 = self.conv1(x)
        x = self.conv2(F_1)
        block_outputs = []
        for block in self.blocks:
            x = block(x)
            block_outputs.append(x)
        x = self.conv3(torch.cat(block_outputs, dim=1))
        x = self.conv4(x)
        x = x + F_1
        x = self.upscale_conv(x)
        x = self.pixel_shuffle(x)
        x = self.conv5(x)
        return x

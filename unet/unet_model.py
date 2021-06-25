""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
from torchvision import transforms
import time
from PIL import Image
import numpy as np

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == '__main__':
    net = UNet(n_channels=3, n_classes=1)
    net.eval()

    device = 'cpu'
    net.to(device=device)
    norm_img = transforms.Compose([
        transforms.Resize(256),  # DEBUG: predict 256 images
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    img = np.random.rand(512, 512, 3).astype('uint8')
    img = Image.fromarray(img)
    img = norm_img(img)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        net.load_state_dict(torch.load('../checkpoints3/CP_epoch20.pth', map_location=device))
        # for p in net.parameters():
        #     # p = np.random.rand(1).astype(np.float64) * np.random.rand(1).astype(np.float64)
        #     print(p)
        start1 = time.time()
        output = net(img)
        start2 = time.time()
        print(start2 - start1)
        # print(torch.max(output), torch.min(output))
        probs = torch.sigmoid(output)
        probs = probs.squeeze(0)
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(512),
                transforms.ToTensor()
            ]
        )
        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()
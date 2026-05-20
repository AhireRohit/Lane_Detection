import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SCNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base=32):
        super().__init__()

        self.enc1 = ConvBlock(in_channels, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)

        self.pool = nn.MaxPool2d(2)

        self.context = nn.Sequential(
            ConvBlock(base * 4, base * 4),
            ConvBlock(base * 4, base * 4),
        )

        self.dec2 = ConvBlock(base * 4 + base * 2, base * 2)
        self.dec1 = ConvBlock(base * 2 + base, base)

        self.head = nn.Conv2d(base, out_channels, 1)

    def spatial_message_passing(self, x):
        # simplified SCNN-style directional context
        out = x.clone()

        for i in range(1, out.shape[2]):
            out[:, :, i, :] = out[:, :, i, :] + out[:, :, i - 1, :]

        for i in range(out.shape[2] - 2, -1, -1):
            out[:, :, i, :] = out[:, :, i, :] + out[:, :, i + 1, :]

        for j in range(1, out.shape[3]):
            out[:, :, :, j] = out[:, :, :, j] + out[:, :, :, j - 1]

        for j in range(out.shape[3] - 2, -1, -1):
            out[:, :, :, j] = out[:, :, :, j] + out[:, :, :, j + 1]

        return out / 4.0

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.context(e3)
        b = self.spatial_message_passing(b)

        d2 = F.interpolate(b, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = F.interpolate(d2, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.head(d1)
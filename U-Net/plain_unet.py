import torch.nn as nn
import torch

def conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),  # 3은 kernel size
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class plain_unet(nn.Module):
    def __init__(self, n_classes=50):
        super(plain_unet, self).__init__()
        self.n_classes = n_classes
        self.convDown1 = conv(9, 64)
        self.convDown2 = conv(64, 128)
        self.convDown3 = conv(128, 256)
        self.convDown4 = conv(256, 512)
        self.convDown5 = conv(512, 1024)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.convUp4 = conv(1024+512, 512)
        self.convUp3 = conv(512+256, 256)
        self.convUp2 = conv(256+128, 128)
        self.convUp1 = conv(128+64, 64)
        self.convUp_fin = nn.Conv2d(64, self.n_classes, 1)



    def forward(self, x):
        conv1 = self.convDown1(x)
        x = self.maxpool(conv1)
        conv2 = self.convDown2(x)
        x = self.maxpool(conv2)
        conv3 = self.convDown3(x)
        x = self.maxpool(conv3)
        conv4 = self.convDown4(x)
        x = self.maxpool(conv4)

        conv5 = self.convDown5(x)
        print("conv5: ",conv5.shape)
        x = self.upsample(conv5)
        print("after upsample: ", x.shape)
        x = torch.cat((x,conv4), dim=1)

        x = self.convUp4(x)
        print("conv5: ",conv5.shape)
        x = self.upsample(x) #1, 512, 64, 64
        x = torch.cat((x,conv3), dim=1)

        x = self.convUp3(x)
        x = self.upsample(x) #1, 256, 128, 128
        x = torch.cat((x,conv2), dim=1)

        x = self.convUp2(x)
        x = self.upsample(x) #1, 128, 256, 256
        x = torch.cat((x,conv1), dim=1)

        x = self.convUp1(x) #1, 64, 256, 256

        out = self.convUp_fin(x) #1, 50, 256, 256
        return out

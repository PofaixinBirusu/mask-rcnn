import torch
from torch import nn


def conv_relu(in_channel, out_channel, kernel, stride=1, padding=0):
    conv = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel, stride, padding),
        nn.BatchNorm2d(out_channel, eps=1e-3),
        nn.ReLU(True),
    )
    return conv


# 定义incepion结构，见inception图
class InceptionV2(nn.Module):
    def __init__(self, in_channel, out1_1, out2_1, out2_3, out3_1, out3_5, out4_1):
        super(InceptionV2, self).__init__()
        self.branch1 = conv_relu(in_channel, out1_1, 1)
        self.branch2 = nn.Sequential(
            conv_relu(in_channel, out2_1, 1),
            conv_relu(out2_1, out2_3, 3, padding=1))
        self.branch3 = nn.Sequential(
            conv_relu(in_channel, out3_1, 1),
            conv_relu(out3_1, out3_5, 5, padding=2))
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            conv_relu(in_channel, out4_1, 1),
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        output = torch.cat([b1, b2, b3, b4], dim=1)
        return output


# GoogLeNet，见上表所示结构
class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.features = nn.Sequential(
            conv_relu(3, 64, 7, 2, 3), nn.MaxPool2d(2, stride=2),
            conv_relu(64, 64, 1), conv_relu(64, 192, 3, padding=1), nn.MaxPool2d(2, 2),
            InceptionV2(192, 64, 96, 128, 16, 32, 32),
            InceptionV2(256, 128, 128, 192, 32, 96, 64), nn.MaxPool2d(2, stride=2),
            InceptionV2(480, 192, 96, 208, 16, 48, 64),
            InceptionV2(512, 160, 112, 224, 24, 64, 64),
            InceptionV2(512, 128, 128, 256, 24, 64, 64),
            InceptionV2(512, 112, 144, 288, 32, 64, 64),
            InceptionV2(528, 256, 160, 320, 32, 128, 128),
            conv_relu(832, 512, 1)
        )

    def forward(self, x):
        out = self.features(x)
        # x = x.view(x.size(0), -1)
        # out = self.classifier(x)
        return out


if __name__ == '__main__':
    x = torch.rand(2, 3, 640, 720).cuda()
    net = GoogLeNet().cuda()
    out = net(x)
    print(out.shape)

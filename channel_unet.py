import torch.nn as nn
import torch
import torch.nn.functional as F


class GAU(nn.Module):
    def __init__(self, channels_high, channels_low, upsample=True):
        super(GAU, self).__init__()
        self.upsample = upsample
        self.conv3x3 = nn.Conv2d(channels_low, channels_low, kernel_size=3, padding=1, bias=False)
        self.bn_low = nn.BatchNorm2d(channels_low)

        self.conv1x1 = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
        self.bn_high = nn.BatchNorm2d(channels_low)

        if upsample:
            self.conv_upsample = nn.ConvTranspose2d(channels_high, channels_low, kernel_size=4, stride=2, padding=1,
                                                    bias=False)
            self.bn_upsample = nn.BatchNorm2d(channels_low)
        else:
            self.conv_reduction = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
            self.bn_reduction = nn.BatchNorm2d(channels_low)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, fms_high, fms_low, fm_mask=None):
        b, c, h, w = fms_high.shape

        fms_high_gp = nn.AvgPool2d(fms_high.shape[2:])(fms_high).view(len(fms_high), c, 1, 1)
        fms_high_gp = self.conv1x1(fms_high_gp)
        fms_high_gp = self.relu(fms_high_gp)

        fms_low_mask = self.conv3x3(fms_low)
        fms_low_mask = self.bn_low(fms_low_mask)

        fms_att = fms_low_mask * fms_high_gp
        if self.upsample:
            out = self.relu(
                self.bn_upsample(self.conv_upsample(fms_high)) + fms_att)
        else:
            out = self.relu(
                self.bn_reduction(self.conv_reduction(fms_high)) + fms_att)
        return out


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p, mid_channels=None):
        super(DoubleConv, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout_p)  # 添加 Dropout 层

        # 残差连接的卷积层
        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.res_conv(x)  # 残差连接

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)  # 在第一层后添加 Dropout

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)  # 在第二层后添加 Dropout

        x = self.conv3(x)
        x = self.bn3(x)
        # 残差连接
        x += residual
        x = self.relu(x)  # 最终激活
        x = self.dropout(x)  # 在第三层后添加 Dropout
        return x


class self_net(nn.Module):
    def __init__(self, in_ch=3, out_ch=4, dropout_p=0.1):
        super(self_net, self).__init__()
        filter = [16, 32, 64, 128]
        self.conv1 = DoubleConv(in_ch, filter[0], dropout_p=dropout_p)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = DoubleConv(filter[0], filter[1], dropout_p=dropout_p)
        self.pool2 = nn.AvgPool2d(2)
        self.conv3 = DoubleConv(filter[1], filter[2], dropout_p=dropout_p)
        self.pool3 = nn.AvgPool2d(2)
        self.conv4 = DoubleConv(filter[2], filter[3], dropout_p=dropout_p)
        self.pool4 = nn.AvgPool2d(2)
        self.up6 = nn.ConvTranspose2d(filter[3], filter[2], 2, stride=2)
        self.conv6 = DoubleConv(filter[2] * 2, filter[2], dropout_p=dropout_p)
        self.up7 = nn.ConvTranspose2d(filter[2], filter[1], 2, stride=2)
        self.conv7 = DoubleConv(filter[1] * 2, filter[1], dropout_p=dropout_p)
        self.up8 = nn.ConvTranspose2d(filter[1], filter[0], 2, stride=2)
        self.conv8 = DoubleConv(filter[0] * 2, filter[0], dropout_p=dropout_p)
        self.conv10 = nn.Conv2d(filter[0], out_ch, 1)

        self.gau_1 = GAU(filter[3], filter[2])
        self.gau_2 = GAU(filter[2], filter[1])
        self.gau_3 = GAU(filter[1], filter[0])

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)

        up_6 = self.up6(c4)
        gau1 = self.gau_1(c4, up_6)
        merge6 = torch.cat([up_6, gau1], dim=1)

        c6 = self.conv6(merge6)

        up_7 = self.up7(c6)
        gau2 = self.gau_2(c6, c2)
        merge7 = torch.cat([up_7, gau2], dim=1)

        c7 = self.conv7(merge7)

        up_8 = self.up8(c7)
        gau3 = self.gau_3(c7, c1)
        merge8 = torch.cat([up_8, gau3], dim=1)
        c8 = self.conv8(merge8)

        c10 = self.conv10(c8)

        return c10



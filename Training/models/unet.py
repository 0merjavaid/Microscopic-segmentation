
import torch.nn as nn
import torch.nn.functional as F
import torch


class SaveFeatures():
    features = None

    def __init__(self, m):
        """
        :param m:
        """
        self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        """
        :param module:
        :param input:
        :param output:
        :return:
        """
        self.features = output

    def remove(self):
        """
        :return:
        """
        self.hook.remove()


class UnetBlock(nn.Module):

    def __init__(self, up_in, x_in, n_out):
        """
        :param up_in:
        :param x_in:
        :param n_out:
        """
        super().__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, 1)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
        self.bn = nn.BatchNorm2d(n_out)

    def forward(self, up_p, x_p):
        """
        :param up_p:
        :param x_p:
        :return:
        """
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        return self.bn(F.relu(cat_p))


class Unet34(nn.Module):

    def __init__(self, rn):
        """
        :param rn:
        """
        super().__init__()
        self.rn = rn
        self.sfs = [SaveFeatures(rn[i]) for i in [2, 4, 5, 6]]
        self.up1 = UnetBlock(512, 256, 256)
        self.up2 = UnetBlock(256, 128, 256)
        self.up3 = UnetBlock(256, 64, 256)
        self.up4 = UnetBlock(256, 64, 256)
        self.up5 = nn.ConvTranspose2d(256, 1, 2, stride=2)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        x = F.relu(self.rn(x))
        x = self.up1(x, self.sfs[3].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        x = self.up5(x)
        return F.sigmoid(x[:, 0])

    def close(self):
        """
        :return:
        """
        for sf in self.sfs:
            sf.remove()

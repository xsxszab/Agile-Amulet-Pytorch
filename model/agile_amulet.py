
import numpy as np
import PIL
import torch
import torch.nn as nn
import torchvision


class VGG16(nn.Module):
    """VGG16 net with BN layer and pretrained params,
    this structure only includes five conv blocks.
    """

    def __init__(self):
        super().__init__()
        self.vgg_pre = []
        self.conv1 = nn.Sequential(  # VGG16 net conv block1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(  # VGG16 net conv block2
            nn.MaxPool2d(2, stride=2),  # 1/2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(  # VGG16 net conv block3
            nn.MaxPool2d(2, stride=2),  # 1/4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(  # VGG16 net conv block4
            nn.MaxPool2d(2, stride=2),  # 1/8
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )

        self.conv5 = nn.Sequential(  # VGG16 net conv block5
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/16
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )

        self.__copy_param()  # copy parameters from pretrained VGG16 net

    def forward(self, x):
        """Forward and generates five predicted saliency map.
         of different spatial resolution
         :param x: input tensor, should has shape [batch_size, 3, 256, 256].
         """
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        x = self.conv5(c4)
        return x

    def __copy_param(self):
        """Load parameters from pretrained VGG16 net."""
        vgg16 = torchvision.models.vgg16_bn(pretrained=True)
        DGG_features = list(self.conv1.children())
        DGG_features.extend(list(self.conv2.children()))
        DGG_features.extend(list(self.conv3.children()))
        DGG_features.extend(list(self.conv4.children()))
        DGG_features.extend(list(self.conv5.children()))
        DGG_features = nn.Sequential(*DGG_features)

        for layer_1, layer_2 in zip(vgg16.features, DGG_features):
            if (isinstance(layer_1, nn.Conv2d) and
                    isinstance(layer_2, nn.Conv2d)):
                assert layer_1.weight.size() == layer_2.weight.size()
                assert layer_1.bias.size() == layer_2.bias.size()
                layer_2.weight.data = layer_1.weight.data
                layer_2.bias.data = layer_1.bias.data


class SFA_Block(nn.Module):
    """Basic Side-output Feature Aggregation block."""
    channels_dict = {1: 64, 2: 128, 3: 256, 4: 512, 5: 512}  # level->channels num mapping

    def __init__(self, level=None):
        """Init module.
        :param level: int, value range from 1 to 5, indicating level of this module(module with highest spatial
         resolution has greatest level).
        """
        super().__init__()
        self.level = level
        if level == 5:
            self.feature_conv = nn.Conv2d(self.channels_dict[level], 32, 3, padding=1)
            self.feature_deconv = nn.ConvTranspose2d(32, 32,)  # TODO: complete deconvolution part.
            self.output_conv = nn.Conv2d(32, 64, 3, padding=1)
        else:
            self.feature_conv = nn.Conv2d(self.channels_dict[level], 32, 3, padding=1)
            self.feature_deconv = nn.ConvTranspose2d(32, 32, )  # TODO: complete deconvolution part.
            self.output_conv = nn.Conv2d(98, 64, 3, padding=1)  # 64+2+32 input channels

    def forward(self, sfa_in=None, sca_in=None, feature_in=None):
        """Forward calculation.
        :param sfa_in: sfa feature from lower level, set to None when level is 1.
        :param sca_in: sca feature from same level, set to None when level is 1.
        :param feature_in: feature map from same level feature-extraction network.
        """
        if self.level == 5:
            feature = self.feature_conv(feature_in)
            feature = self.feature_deconv(feature)
            output = self.output_conv(feature)
            return output

        else:
            feature = self.feature_conv(feature_in)
            feature = self.feature_deconv(feature)
            cat_feature = torch.cat([sfa_in, sca_in, feature], 1)  # concatenate three input feature maps
            output = self.output_conv(cat_feature)
            return output


class SCA_Block(nn.Module):
    """Basic Spatially Contextual Attention block."""
    channels_dict = {1: 74, 2: 72, 3: 70, 4: 68, 5: 64}

    def __init__(self, level=None):
        """Init module.
        :param level: int, value range from 1 to 5, indicating level of this module(module with highest spatial
        resolution has greatest level).
        """
        super().__init__()
        self.level = level
        self.ouptut_conv = nn.Conv2d(self.channels_dict[level], 64, 3, padding=1)

    def forward(self, sfa=None, sca_2=None, sca_3=None, sca_4=None, sca_5=None):
        if self.level == 5:
            return self.ouptut_conv(sfa)
        elif self.level == 4:
            cat_feature = torch.cat([sfa, sca_5], 1)
            return self.ouptut_conv(cat_feature)
        elif self.level == 3:
            cat_feature = torch.cat([sfa, sca_5, sca_4], 1)
            return self.ouptut_conv(cat_feature)
        elif self.level == 2:
            cat_feature = torch.cat([sfa, sca_5, sca_4, sca_3], 1)
            return self.ouptut_conv(cat_feature)
        elif self.level == 1:
            cat_feature = torch.cat([sfa, sca_5, sca_4, sca_3, sca_2], 1)
            return self.ouptut_conv(cat_feature)


class RSP_Block(nn.Module):
    """Basic Recursive Saliency Prediction block."""
    def __init__(self, level=None):
        """Init module.
        :param level: int, value range from 1 to 5, indicating level of this module(module with highest spatial
        resolution has greatest level).
        """
        super().__init__()
        self.level = level
        if level == 5:
            self.output_conv = nn.Conv2d(2, 2, 3, padding=1)
        else:
            self.output_conv = nn.Conv2d(6, 2, 3, padding=1)

    def forward(self, rsp=None, sca_1=None, sca_2=None):
        if self.level == 5:
            return self.output_conv(sca_2)
        else:
            feature = torch.add(rsp ,sca_1)
            feature = torch.add(feature, sca_2)
            feature = self.output_conv(feature)
            return feature


class Agile_Amulet(nn.Module):
    """Structure of Agile Amulet network."""

    def __init__(self):
        super().__init__()
        self.vgg = VGG16()

        self.sfa1 = SFA_Block(level=1)
        self.sfa2 = SFA_Block(level=2)
        self.sfa3 = SFA_Block(level=3)
        self.sfa4 = SFA_Block(level=4)
        self.sfa5 = SFA_Block(level=5)

        self.sca1 = SCA_Block(level=1)
        self.sca2 = SCA_Block(level=2)
        self.sca3 = SCA_Block(level=3)
        self.sca4 = SCA_Block(level=4)
        self.sca5 = SCA_Block(level=5)

        self.rsp1 = RSP_Block(level=1)
        self.rsp2 = RSP_Block(level=2)
        self.rsp3 = RSP_Block(level=3)
        self.rsp4 = RSP_Block(level=4)
        self.rsp5 = RSP_Block(level=5)


    def forward(self, x):
        x1 = self.vgg.conv1.forward(x)
        x2 = self.vgg.conv2.forward(x1)
        x3 = self.vgg.conv3.forward(x2)
        x4 = self.vgg.conv4.forward(x3)
        x5 = self.vgg.conv5.forward(x4)

        sfa5_out = self.sfa5.forward(feature_in=x5)
        sca5_out = self.sca5.forward(sfa=sfa5_out)
        rsp5_out = self.rsp5.forward(sca_2=sca5_out)

        sfa4_out = self.sfa4.forward(sfa_in=sfa5_out, sca_in=sca5_out, feature_in=x4)
        sca4_out = self.sca4.forward(sfa=sfa4_out, sca_5=sca5_out)
        rsp4_out = self.rsp4.forward(rsp=rsp5_out, sca_1=sca4_out, sca_2=sca5_out)

        sfa3_out = self.sfa3.forward(sfa_in=sfa4_out, sca_in=sca4_out, feature_in=x3)
        sca3_out = self.sca3.forward(sfa=sfa3_out, sca_5=sca5_out, sca_4=sca4_out)
        rsp3_out = self.rsp3.forward(rsp=rsp4_out, sca_1=sca3_out, sca_2=sca4_out)

        sfa2_out = self.sfa2.forward(sfa_in=sfa3_out, sca_in=sca3_out, feature_in=x2)
        sca2_out = self.sca2.forward(sfa=sfa2_out, sca_5=sca5_out, sca_4=sca4_out, sca_3=sca3_out)
        rsp2_out = self.rsp2.forward(rsp=rsp3_out, sca_1=sca2_out, sca_2=sca3_out)

        sfa1_out = self.sfa1.forward(sfa_in=sfa2_out, sca_in=sca2_out, feature_in=x1)
        sca1_out = self.sca1.forward(sfa=sfa1_out, sca_5=sca5_out, sca_4=sca4_out, sca_3=sca3_out, sca_2=sca2_out)
        rsp1_out = self.rsp1.forward(rsp=rsp2_out, sca_1=sca2_out, sca_2=sca3_out)

        return rsp1_out


if __name__ == '__main__':
    test = Agile_Amulet()
    img = PIL.Image.open('test.jpg', 'r')
    img = img.resize((256, 256))
    img = np.array(img)
    img = np.expand_dims(img, 0)
    img = np.transpose(img, (0, 3, 1, 2))
    img = torch.from_numpy(img).float()
    outputs = test.forward(img)

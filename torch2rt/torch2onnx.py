"""
# Code Adapted from:
# https://github.com/sthalles/deeplab_v3
#
# MIT License
#
# Copyright (c) 2018 Thalles Santos Silva
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
"""
import logging
import torch
from torch import nn
import sys
sys.path.insert(1, '../network')
from network import SEresnext
from network import Resnet
from network.wider_resnet import wider_resnet38_a2
from network.mynn import initialize_weights, Norm2d, Upsample
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms



class _AtrousSpatialPyramidPoolingModule(nn.Module):
    """
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    """

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=(6, 12, 18)):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          Norm2d(reduction_dim), nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                Norm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            Norm2d(reduction_dim), nn.ReLU(inplace=True))

    def forward(self, x):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = Upsample(img_features, x_size[2:])
        out = img_features

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out

class DeepWV3Plus_semantic(nn.Module):
    """
    WideResNet38 version of DeepLabV3
    mod1
    pool2
    mod2 bot_fine
    pool3
    mod3-7
    bot_aspp

    structure: [3, 3, 6, 3, 1, 1]
    channels = [(128, 128), (256, 256), (512, 512), (512, 1024), (512, 1024, 2048),
              (1024, 2048, 4096)]
    """

    def __init__(self, trunk='WideResnet38', criterion=None):

        super(DeepWV3Plus_semantic, self).__init__()
        self.criterion = criterion
        logging.info("Trunk: %s", trunk)

        tasks = ['semantic']
        wide_resnet = wider_resnet38_a2(classes=1000, dilation=True, tasks=tasks)
        wide_resnet = torch.nn.DataParallel(wide_resnet)
        if criterion is not None:
            try:
                checkpoint = torch.load('./pretrained_models/wider_resnet38.pth.tar', map_location='cpu')
#                wide_resnet.load_state_dict(checkpoint['state_dict'])

                net_state_dict = wide_resnet.state_dict()
                loaded_dict = checkpoint['state_dict']
                new_loaded_dict = {}
                for k in net_state_dict:
                    if k in loaded_dict and net_state_dict[k].size() == loaded_dict[k].size():
                        new_loaded_dict[k] = loaded_dict[k]
                    else:
                        logging.info("Skipped loading parameter %s", k)
                net_state_dict.update(new_loaded_dict)
                wide_resnet.load_state_dict(net_state_dict)

                del checkpoint
            except:
                print("Please download the ImageNet weights of WideResNet38 in our repo to ./pretrained_models/wider_resnet38.pth.tar.")
                raise RuntimeError("=====================Could not load ImageNet weights of WideResNet38 network.=======================")
        wide_resnet = wide_resnet.module

        self.mod1 = wide_resnet.mod1
        self.mod2 = wide_resnet.mod2
        self.mod3 = wide_resnet.mod3
        self.mod4 = wide_resnet.mod4
        self.mod5 = wide_resnet.mod5
        self.mod6 = wide_resnet.mod6
        self.mod7 = wide_resnet.mod7
        self.pool2 = wide_resnet.pool2
        self.pool3 = wide_resnet.pool3
        del wide_resnet

        self.aspp = _AtrousSpatialPyramidPoolingModule(4096, 256,
                                                       output_stride=8)

        self.bot_fine = nn.Conv2d(128, 48, kernel_size=1, bias=False)
        self.bot_aspp = nn.Conv2d(1280, 256, kernel_size=1, bias=False)

        self.final = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 21, kernel_size=1, bias=False))

        # for param in self.mod1.parameters():
        #     param.requires_grad = False
        # for param in self.mod2.parameters():
        #     param.requires_grad = False
        # for param in self.mod3.parameters():
        #     param.requires_grad = False
        # for param in self.mod4.parameters():
        #     param.requires_grad = False
        # for param in self.mod5.parameters():
        #     param.requires_grad = False

        # for param in self.pool2.parameters():
        #     param.requires_grad = False
        # for param in self.pool3.parameters():
        #     param.requires_grad = False

        # for param in self.mod6.block1.bn1.parameters():
        #     param.requires_grad = False
        # for param in self.mod6.block1.convs.parameters():
        #     param.requires_grad = False
        # for param in self.mod7.block1.bn1.parameters():
        #     param.requires_grad = False
        # for param in self.mod7.block1.convs.parameters():
        #     param.requires_grad = False

    def forward(self, inp, gts=None, task=None):

        x_size = inp.size()
        x = self.mod1(inp)
        m2 = self.mod2(self.pool2(x))
        x = self.mod3(self.pool3(m2))
        x = self.mod4(x)
        x = self.mod5(x)
        x = self.mod6(x, task=task)
        x = self.mod7(x, task=task)
        x = self.aspp(x)

        dec0_up = self.bot_aspp(x)
        dec0_fine = self.bot_fine(m2)
        dec0_up = Upsample(dec0_up, m2.size()[2:])
        dec0 = [dec0_fine, dec0_up]
        dec0 = torch.cat(dec0, 1)

        dec1 = self.final(dec0)
        out = Upsample(dec1, x_size[2:])

        if self.training:
            return self.criterion(out, gts)

        return out#[:,:num_class,:,:]

def read_image(img_path):
    input_img = Image.open(img_path)
    resize = transforms.Resize([640,480])
    input_img = resize(input_img)
    to_tensor = transforms.ToTensor()
    input_img = to_tensor(input_img)
    input_img = torch.unsqueeze(input_img, 0)
    return input_img

trained_model = DeepWV3Plus_semantic()
trained_model.load_state_dict(torch.load('pretrained_models/best_epoch_77_mean-iu_0.72510.pth'))
img_path = '/mnt/jetson_sdcard/dataset_forest_tiny/testing/images/image5001.png'
input_data = read_image(img_path).cuda()
torch.onnx.export(trained_model, input_data, "output/semantic_model.onnx")
# img_root = '/mnt/jetson_sdcard/dataset_forest/testing/images/'
# filename_list = os.listdir(img_root)
# filename_list.sort()
# for i in filename_list:
#     input_data = read_image(img_root+i).cuda()
#     torch.onnx.export(trained_model, input_data, "output/semantic_model.onnx")
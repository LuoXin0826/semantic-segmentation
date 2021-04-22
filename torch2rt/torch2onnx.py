#
# converts a saved PyTorch model to ONNX format
#
import os
import logging
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import sys
sys.path.extend('../')
from network import SEresnext
from network import Resnet
from network.deepv3 import DeepWV3Plus_semantic
from network.wider_resnet import wider_resnet38_a2
from network.mynn import initialize_weights, Norm2d, Upsample
import argparse


# parse command line
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='./pretrained_models/best_epoch_77_mean-iu_0.72510.pth',
                    help="path to input PyTorch model")
parser.add_argument('--output', type=str, default='',
                    help="desired path of converted ONNX model")
parser.add_argument('--model-dir', type=str, default='',
                    help="directory to look for the input PyTorch model in, and export the converted ONNX model to (if --output doesn't specify a directory)")
parser.add_argument('--num_classes', type=int, default=21,
                    help="Number of classes (default: 21)")
parser.add_argument('--width', type=int, default=640,
                    help="Width (default: 640)")
parser.add_argument('--height', type=int, default=480,
                    help="Height (default: 480)")
parser.add_argument('--arch', type=str, default='network.deepv3.DeepWV3Plus_semantic',
                    help='Network architecture. We have DeepSRNX50V3PlusD (backbone: ResNeXt50) \
                    and deepWV3Plus (backbone: WideResNet38) default:DeepWV3Plus_semantic.')

opt = parser.parse_args()
print(opt)

# format input model path
if opt.model_dir:
    opt.model_dir = os.path.expanduser(opt.model_dir)
    opt.input = os.path.join(opt.model_dir, opt.input)

# set the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('running on device ' + str(device))

# load the model checkpoint
print('loading checkpoint:  ' + opt.input)
checkpoint = torch.load(opt.input)

num_classes = opt.num_classes

# create the model architecture
print('num classes:  ' + str(num_classes))

model = network.get_net_ori(opt, criterion)

# load the model weights
model.load_state_dict(checkpoint['model'])

model.to(device)
model.eval()

print(model)
print('')

# create example image data
resolution = [opt.width, opt.height]
input = torch.ones((1, 3, resolution[0], resolution[1])).cuda()
print('input size:  {:d}x{:d}'.format(resolution[1], resolution[0]))

# format output model path
if not opt.output:
    opt.output = wise_resnet + '.onnx'

if opt.model_dir and opt.output.find('/') == -1 and opt.output.find('\\') == -1:
    opt.output = os.path.join(opt.model_dir, opt.output)

# export the model
input_names = ["input_0"]
output_names = ["output_0"]

print('exporting model to ONNX...')
torch.onnx.export(model, input, opt.output, verbose=True,
                  input_names=input_names, output_names=output_names)
print('model exported to:  {:s}'.format(opt.output))
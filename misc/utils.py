from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import collections
import torch
import torch.nn as nn
from torch.autograd import Variable
import misc.resnet as resnet
import os
import random

##################################################################################
# Convolutional Blocks
##################################################################################
def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        # print m.__class__.__name__
        m.weight.data.normal_(0.0, 0.02)

class LeakyReLUConv1d(nn.Module):
  def __init__(self, n_in, n_out, kernel_size, stride, padding=0):
    super(LeakyReLUConv1d, self).__init__()
    model = []
    model += [nn.Conv1d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)]
    model += [nn.LeakyReLU(inplace=True)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)

  def forward(self, x):
    return self.model(x)

class LeakyReLUBNConv1d(nn.Module):
  def __init__(self, n_in, n_out, kernel_size, stride, padding=0):
    super(LeakyReLUBNConv1d, self).__init__()
    model = []
    model += [nn.Conv1d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)]
    model += [nn.BatchNorm1d(n_out)]
    model += [nn.LeakyReLU(inplace=True)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)

  def forward(self, x):
    return self.model(x)

class LeakyReLUBNConv2d(nn.Module):
  def __init__(self, n_in, n_out, kernel_size, stride, padding=0):
    super(LeakyReLUBNConv2d, self).__init__()
    model = []
    model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)]
    model += [nn.BatchNorm2d(n_out)]
    model += [nn.LeakyReLU(inplace=True)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)

  def forward(self, x):
    return self.model(x)

class LeakyReLUConv2d(nn.Module):
  def __init__(self, n_in, n_out, kernel_size, stride, padding=0):
    super(LeakyReLUConv2d, self).__init__()
    model = []
    model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)]
    model += [nn.LeakyReLU(inplace=True)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)

  def forward(self, x):
    return self.model(x)
##################################################################################
# Residual Blocks
##################################################################################
class INSResBlock(nn.Module):
  def conv1x1(self, inplanes, out_planes, kernel=1, stride=1, padding=1):
    return nn.Conv2d(inplanes, out_planes, kernel_size=kernel, stride=stride, padding=padding)

  def __init__(self, inplanes, planes, kernel=1, stride=1, padding=1, dropout=0.0):
    super(INSResBlock, self).__init__()
    model = []
    model += [self.conv1x1(inplanes, planes, kernel, stride, padding)]
    model += [nn.InstanceNorm2d(planes)]
    model += [nn.ReLU(inplace=True)]
    model += [self.conv1x1(planes, planes, kernel, stride, padding)]
    model += [nn.InstanceNorm2d(planes)]
    if dropout > 0:
      model += [nn.Dropout(p=dropout)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)

  def forward(self, x):
    residual = x
    out = self.model(x)
    out += residual
    return out

def build_cnn(opt):
    net = getattr(resnet, opt.cnn_model)()
    if vars(opt).get('start_from', None) is None and vars(opt).get('cnn_weight', '') != '':
        net.load_state_dict(torch.load(opt.cnn_weight))
    net = nn.Sequential(\
        net.conv1,
        net.bn1,
        net.relu,
        net.maxpool,
        net.layer1,
        net.layer2,
        net.layer3,
        net.layer4)
    if vars(opt).get('start_from', None) is not None:
        net.load_state_dict(torch.load(os.path.join(opt.start_from, 'model-cnn.pth')))
    return net

def prepro_images(imgs, data_augment=False):
    # crop the image
    h,w = imgs.shape[2], imgs.shape[3]
    cnn_input_size = 224

    # cropping data augmentation, if needed
    if h > cnn_input_size or w > cnn_input_size:
        if data_augment:
          xoff, yoff = random.randint(0, w-cnn_input_size), random.randint(0, h-cnn_input_size)
        else:
          # sample the center
          xoff, yoff = (w-cnn_input_size)//2, (h-cnn_input_size)//2
    # crop.
    imgs = imgs[:,:, yoff:yoff+cnn_input_size, xoff:xoff+cnn_input_size]

    return imgs

def if_use_att(caption_model):
    # Decide if load attention feature according to caption model
    if caption_model in ['show_tell', 'all_img', 'fc']:
        return False
    return True

# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i,j].cpu().numpy()
            if ix > 0 :
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix)]
            else:
                break
        out.append(txt)
    return out

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

class LanguageModelCriterion(nn.Module):
    def __init__(opt):
        super(LanguageModelCriterion, opt).__init__()

    def forward(opt, input, target, mask, coverage=None):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        input = to_contiguous(input).view(-1, input.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        output = - input.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if hasattr(param, 'grad'):
                param.grad.data.clamp_(-grad_clip, grad_clip)

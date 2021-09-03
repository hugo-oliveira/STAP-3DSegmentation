import os
import yaml
import imageio
import nibabel as nib
from math import ceil

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


# Auxiliary functions.
def norm(arr):
    
    return (arr.astype(np.float32) - arr.min()) / float(arr.max() - arr.min())

def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv3d) or isinstance(module, nn.Linear) or isinstance(module, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm3d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

def get_config(config):
    
    with open(config, 'r') as stream:
        return yaml.load(stream)

def save_nifti(vol, path):
    
    vol_nifti = nib.nifti1.Nifti1Image(vol, None)
    
    nib.nifti1.save(vol_nifti, path)

def save_gif(vol_img, vol_msk, gif_path):
    
    vol_img = (norm(vol_img) * 255).astype(np.uint8)
    vol_msk = (norm(vol_msk) * 255).astype(np.uint8)
    
    gif_list = []
    for z in range(vol_img.shape[2]):
        
        v_img = vol_img[:,:,z]
        v_msk = vol_msk[:,:,z]
        
        v_rgb = np.zeros((v_img.shape[0], v_img.shape[1], 3), dtype=v_img.dtype)
        v_rgb[:,:,0] = v_msk
        v_rgb[:,:,1] = v_img
        v_rgb[:,:,2] = v_img
        
        gif_list.append(v_rgb)
        
    imageio.mimsave(gif_path, gif_list, duration=(4.0 / float(len(gif_list))))

###########################################################################
# Implementation from: https://github.com/hubutui/DiceLoss-PyTorch ########
###########################################################################
class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

###########################################################################
# Implementation from: https://github.com/hubutui/DiceLoss-PyTorch ########
###########################################################################
class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weight[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]

def dice_loss(prd, lab, num_classes=4, weight=[1.0, 1.0, 1.0, 1.0]):
    
    criterion = DiceLoss(weight=weight)
    return criterion(prd, F.one_hot(lab, num_classes=num_classes))

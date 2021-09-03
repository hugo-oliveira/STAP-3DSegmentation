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
#             elif isinstance(module, nn.BatchNorm3d) or isinstance(module, nn.GroupNorm):
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

# Implementation from: https://github.com/hubutui/DiceLoss-PyTorch
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

###########################################################################
# Implementation from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch ######
###########################################################################
class FocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class, alpha=[0.25,0.75], gamma=2, balance_index=-1, size_average=True):
        
        super(FocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
        self.eps = 1e-6

        if isinstance(self.alpha, (list, tuple)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.Tensor(list(self.alpha))
        elif isinstance(self.alpha, (float,int)):
            assert 0 < self.alpha < 1.0, 'alpha should be in `(0,1)`)'
            assert balance_index > -1
            alpha = torch.ones((self.num_class))
            alpha *= 1-self.alpha
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        elif isinstance(self.alpha, torch.Tensor):
            self.alpha = self.alpha
        else:
            raise TypeError('Not support alpha type, expect `int|float|list|tuple|torch.Tensor`')

    def forward(self, logit, target):

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.transpose(1, 2).contiguous() # [N,C,d1*d2..] -> [N,d1*d2..,C]
            logit = logit.view(-1, logit.size(-1)) # [N,d1*d2..,C]-> [N*d1*d2..,C]
        target = target.view(-1, 1) # [N,d1,d2,...]->[N*d1*d2*...,1]
        
        
        # ----------memory saving way--------
        pt = logit.gather(1, target).view(-1) + self.eps # avoid apply
        logpt = pt.log()

        if self.alpha.device != logpt.device:
            alpha = self.alpha.to(logpt.device)
            alpha_class = alpha.gather(0,target.view(-1))
            logpt = alpha_class*logpt
        loss = -1 * torch.pow(torch.sub(1.0, pt), self.gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

def focal_loss(lab, prd, num_classes=4):
    
    criterion = FocalLoss(num_class=num_classes, alpha=0.25, gamma=2.0, balance_index=2)
    return criterion(F.softmax(prd, dim=1), lab)

###########################################################################
# Implementation from: https://github.com/black0017/MedicalZooPytorch #####
###########################################################################
def loss_vae(recon_x, x, mu, logvar, type="BCE", h1=0.1, h2=0.1):
    """
    see Appendix B from VAE paper:
    Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param recon_x:
    :param x:
    :param mu,logvar: VAE parameters
    :param type: choices BCE,L1,L2
    :param h1: reconsrtruction hyperparam
    :param h2: KL div hyperparam
    :return: total loss of VAE
    """
    batch = recon_x.shape[0]
    assert recon_x.size() == x.size()
    assert recon_x.shape[0] == x.shape[0]
    rec_flat = recon_x.view(batch, -1)
    x_flat = x.view(batch, -1)
    if type=="BCE":
        loss_rec = F.binary_cross_entropy(rec_flat, x_flat, reduction='sum')
    elif type=="L1":
        loss_rec = torch.sum(torch.abs(rec_flat - x_flat))
    elif type =="L2":
        loss_rec = torch.sum(torch.sqrt(rec_flat * rec_flat - x_flat * x_flat))
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return loss_rec * h1, KLD * h2
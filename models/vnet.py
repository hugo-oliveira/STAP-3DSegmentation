import torch
import torch.nn.functional as F
from torch import nn

from utils import initialize_weights

###########################################################################
# Code adapted from: https://github.com/zyody/vnet.pytorch ################
###########################################################################

def passthrough(x, **kwargs):
    
    return x

def ELUCons(elu, nchan):
    
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)


# Normalization between sub-volumes is necessary for good performance.
class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    
    def _check_input_dim(self, input):
        
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
            
        #super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        
        self._check_input_dim(input)
        
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)


class LUConv(nn.Module):
    
    def __init__(self, nchan, elu):
        
        super(LUConv, self).__init__()
        
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(nchan)
        
        initialize_weights(self)

    def forward(self, x):
        
        out = self.relu1(self.bn1(self.conv1(x)))
        
        return out


def _make_nConv(nchan, depth, elu):
    
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
        
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    
    def __init__(self, inChans, outChans, elu):
        
        super(InputTransition, self).__init__()
        
        self.inChans = inChans
        self.outChans = outChans
        
        self.conv1 = nn.Conv3d(inChans, outChans, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(outChans)
        self.relu1 = ELUCons(elu, outChans)
        
        initialize_weights(self)

    def forward(self, x):
        
        # Conv3D and Batch Norm.
        out = self.bn1(self.conv1(x))
        
        # Repeat input outChans // inChans times.
        residual = x.repeat(1, self.outChans // self.inChans, 1, 1, 1)
        
        out = self.relu1(out + residual)
        
        return out


class DownTransition(nn.Module):
    
    def __init__(self, inChans, nConvs, elu, dropout=False):
        
        super(DownTransition, self).__init__()
        
        outChans = 2 * inChans
        
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)
        
        initialize_weights(self)

    def forward(self, x):
        
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(out + down)
        
        return out


class UpTransition(nn.Module):
    
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        
        super(UpTransition, self).__init__()
        
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)
        
        initialize_weights(self)

    def forward(self, x, skipx):
        
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(out + xcat)
        
        return out


class OutputTransition(nn.Module):
    
    def __init__(self, inChans, outChans, elu):
        
        super(OutputTransition, self).__init__()
        
        self.conv1 = nn.Conv3d(inChans, outChans, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(outChans)
        self.conv2 = nn.Conv3d(outChans, outChans, kernel_size=1)
        self.relu1 = ELUCons(elu, outChans)
        
        initialize_weights(self)

    def forward(self, x):
        
        # Convolve 32 down to 2 channels
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        
        return out


class VNet(nn.Module):
    
    # The number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent.
    def __init__(self, in_channels, num_classes, elu=True):
        
        super(VNet, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.elu = elu
        
        self.in_tr = InputTransition(self.in_channels, 16, elu)
        
        self.down_tr32 = DownTransition(16, 1, self.elu)
        self.down_tr64 = DownTransition(32, 2, self.elu)
        self.down_tr128 = DownTransition(64, 3, self.elu, dropout=True)
        self.down_tr256 = DownTransition(128, 2, self.elu, dropout=True)
        
        self.up_tr256 = UpTransition(256, 256, 2, self.elu, dropout=True)
        self.up_tr128 = UpTransition(256, 128, 2, self.elu, dropout=True)
        self.up_tr64 = UpTransition(128, 64, 1, self.elu)
        self.up_tr32 = UpTransition(64, 32, 1, self.elu)
        
        self.out_tr = OutputTransition(32, self.num_classes, self.elu)
        
        initialize_weights(self)
    
    def replace_classifier(self, num_classes):
        
        self.num_classes = num_classes
        
        self.out_tr = OutputTransition(32, self.num_classes, self.elu).cuda() #.to(self.in_tr.device)
        
        initialize_weights(self.out_tr)
    
    def set_trainable(self, trainable):
        
        # Setting model as trainable or not trainable.
        for param in self.parameters():
            param.requires_grad = trainable
    
    def partial_freeze(self):
        
        # Freezing all layers but classifier.
        for param in self.in_tr.parameters():
            param.requires_grad = False
            
        for param in self.down_tr32.parameters():
            param.requires_grad = False
        for param in self.down_tr64.parameters():
            param.requires_grad = False
        for param in self.down_tr128.parameters():
            param.requires_grad = False
        for param in self.down_tr256.parameters():
            param.requires_grad = False
            
        for param in self.up_tr256.parameters():
            param.requires_grad = False
        for param in self.up_tr128.parameters():
            param.requires_grad = False
        for param in self.up_tr64.parameters():
            param.requires_grad = False
        for param in self.up_tr32.parameters():
            param.requires_grad = False
            
        for param in self.out_tr.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        
        out16 = self.in_tr(x)
        
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        
        out = self.out_tr(out)
        
        return out

import torch
import torch.nn.functional as F
from torch import nn

from utils import initialize_weights

class _EncoderBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, dropout=False):
        
        super(_EncoderBlock, self).__init__()
        
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        ]
        
        if dropout:
            
            layers.append(nn.Dropout())
            
        layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
        
        self.encode = nn.Sequential(*layers)
        
        initialize_weights(self)
        
    def forward(self, x):
        
        return self.encode(x)


class _DecoderBlock(nn.Module):
    
    def __init__(self, in_channels, middle_channels, out_channels, dropout=False):
        
        super(_DecoderBlock, self).__init__()
        
        layers = [
            nn.Dropout3d(),
            nn.Conv3d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(middle_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(middle_channels, out_channels, kernel_size=2, stride=2, padding=0, output_padding=0)
        ]
        
        if dropout:
            
            layers.append(nn.Dropout())
        
        self.decode = nn.Sequential(*layers)
        
        initialize_weights(self)
        
    def forward(self, x):
        
        return self.decode(x)


class UNet(nn.Module):
    
    def __init__(self, in_channels, num_classes):
        
        super(UNet, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        self.enc1 = _EncoderBlock(self.in_channels, 32)
        self.enc2 = _EncoderBlock(32, 64)
        self.enc3 = _EncoderBlock(64, 128)
        self.enc4 = _EncoderBlock(128, 256, dropout=True)
        
        self.center = _DecoderBlock(256, 512, 256)
        
        self.dec4 = _DecoderBlock(512, 256, 128)
        self.dec3 = _DecoderBlock(256, 128, 64)
        self.dec2 = _DecoderBlock(128, 64, 32)
        
        self.dec1 = nn.Sequential(
            nn.Dropout3d(),
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        
        self.final = nn.Conv3d(32, self.num_classes, kernel_size=1)
        
        self.mode_interpolation='trilinear'
        
        initialize_weights(self)
    
    def replace_classifier(self, num_classes):
        
        self.num_classes = num_classes
        
        self.final = nn.Conv3d(32, self.num_classes, kernel_size=1).cuda() #to(self.enc1.device)
        
        initialize_weights(self.final)
    
    def set_trainable(self, trainable):
        
        # Setting model as trainable or not trainable.
        for param in self.parameters():
            param.requires_grad = trainable
    
    def partial_freeze(self):
        
        # Freezing all layers but classifier.
        for param in self.enc1.parameters():
            param.requires_grad = False
        for param in self.enc2.parameters():
            param.requires_grad = False
        for param in self.enc3.parameters():
            param.requires_grad = False
        for param in self.enc4.parameters():
            param.requires_grad = False
            
        for param in self.center.parameters():
            param.requires_grad = False
            
        for param in self.dec4.parameters():
            param.requires_grad = False
        for param in self.dec3.parameters():
            param.requires_grad = False
        for param in self.dec2.parameters():
            param.requires_grad = False
        for param in self.dec1.parameters():
            param.requires_grad = False
            
        for param in self.final.parameters():
            param.requires_grad = True
        
    def forward(self, x):
        
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        
        center = self.center(enc4)
        
        dec4 = self.dec4(torch.cat([center, F.interpolate(enc4, center.size()[2:], mode=self.mode_interpolation)], 1))
        dec3 = self.dec3(torch.cat([dec4, F.interpolate(enc3, dec4.size()[2:], mode=self.mode_interpolation)], 1))
        dec2 = self.dec2(torch.cat([dec3, F.interpolate(enc2, dec3.size()[2:], mode=self.mode_interpolation)], 1))
        dec1 = self.dec1(torch.cat([dec2, F.interpolate(enc1, dec2.size()[2:], mode=self.mode_interpolation)], 1))
        
        final = self.final(dec1)
        
        return F.interpolate(final, x.size()[2:], mode=self.mode_interpolation)

import datetime
import os
import time
import gc
import sys
import numpy as np
import skimage

from skimage import io
from skimage import util
from skimage import transform

from sklearn import metrics

from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch.nn.functional as F

from models import *
from utils import *
import list_dataset

cudnn.benchmark = True

# Predefining directories.
ckpt_path = './ckpt'
gif_path = './gifs'
out_path = './outputs'

# Reading config file path.
assert len(sys.argv) >= 2
config_path = sys.argv[1]

# Main function.
def main(config_path):
    
    # Loading config from yaml file.
    args = get_config(config_path)
    
    # Setting experiment name.
    exp_name = args['exp_name']
    
    # Setting datasets.
    test_set = list_dataset.ListDataset(args['root'], 'test', args['data_name'], args['task_name'], args['fold_name'], (args['h_size'], args['w_size'], args['z_size']))
    test_loader = DataLoader(test_set, batch_size=1, num_workers=0, shuffle=False)
    
    # Setting network architecture.
    num_classes = test_set.num_classes
    
    # Instanciating architecture.
    if (args['conv_name'] == 'highresnet'):
        
        net = HighRes3DNet(1, out_channels=num_classes)
        
    elif (args['conv_name'] == 'skipdensenet'):
        
        net = SkipDenseNet3D(1, num_classes=num_classes)
        
    elif (args['conv_name'] == 'mednet'):
        
        net = ResNetMed3D(1, num_classes=num_classes)
        
    elif (args['conv_name'] == 'unet'):
        
        net = UNet(1, num_classes=num_classes)
        
    elif (args['conv_name'] == 'vnet'):
        
        net = VNet(1, num_classes=num_classes)
        
    
    net = net.cuda()
    net = nn.DataParallel(net)
    print(net)
    print('%d trainable parameters...' % (sum(p.numel() for p in net.parameters() if p.requires_grad)))
    sys.stdout.flush()
    
    # Loading model.
    net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, 'model.pth')))
    
    # Making sure checkpoint and output directories are created.
    check_mkdir(out_path)
    check_mkdir(os.path.join(out_path, exp_name))
    check_mkdir(gif_path)
    check_mkdir(os.path.join(gif_path, exp_name))
    
    # Forwarding test dataset.
    test(exp_name, test_loader, net, args)

def test(exp_name, test_loader, net, args):
    
    # Setting network for evaluation mode.
    net.eval()
    
    # Lists.
    labs_all = []
    prds_all = []
    
    with torch.no_grad():
        
        # Starting time.
        tic = time.time()
        
        # Iterating over batches.
        for i, data in enumerate(test_loader):
            
            # Obtaining images, labels and paths for batch.
            inps_list, labs_list, off_list, size_list, strides, orig_shape, img_name = data
            
            print('    Test Sample %d/%d: "%s"' % (i + 1, len(test_loader), img_name[0]))
            sys.stdout.flush()
            
            # Resizing algorithm.
            inps_full, labs_full, prds_full = list_dataset.resize_reconstruction(net, inps_list, labs_list, off_list, size_list, strides, orig_shape)
            
            # Appending label and prediction lists for computing metrics.
            labs_all.extend(labs_full.ravel().tolist())
            prds_all.extend(prds_full.ravel().tolist())
            
            # Formatting images, labels and predictions for output.
            out_img = (norm(inps_full) * 255).astype(np.uint8)
            out_lab = labs_full.astype(np.uint8)
            out_prd = prds_full.astype(np.uint8)
            
            # Saving gifs.
            lab_path = os.path.join(gif_path, exp_name, img_name[0].replace('.nii.gz', '_lab.gif'))
            prd_path = os.path.join(gif_path, exp_name, img_name[0].replace('.nii.gz', '_prd.gif'))
            
            save_gif(out_img, out_lab, lab_path)
            save_gif(out_img, out_prd, prd_path)
            
            # Saving niftis.
            img_path = os.path.join(out_path, exp_name, img_name[0].replace('.nii.gz', '_img.nii.gz'))
            lab_path = os.path.join(out_path, exp_name, img_name[0].replace('.nii.gz', '_lab.nii.gz'))
            prd_path = os.path.join(out_path, exp_name, img_name[0].replace('.nii.gz', '_prd.nii.gz'))
            
            save_nifti(out_img, img_path)
            save_nifti(out_lab, lab_path)
            save_nifti(out_prd, prd_path)
            
        # Ending time.
        toc = time.time()
        
        # Printing test time.
        print('Test time %.2f' % (toc - tic))
        sys.stdout.flush()
        
        prds_np = np.asarray(prds_all).ravel()
        labs_np = np.asarray(labs_all).ravel()
        
        # Computing error metrics.
        if test_loader.dataset.num_classes == 2:
            iou = metrics.jaccard_score(labs_np, prds_np, average='binary')
            dice = metrics.f1_score(labs_np, prds_np, average='binary')
        else:
            iou = metrics.jaccard_score(labs_np, prds_np, average='macro')
            dice = metrics.f1_score(labs_np, prds_np, average='macro')
        
        cm = metrics.confusion_matrix(labs_np, prds_np)
        
        # Printing epoch metrics.
        print('--------------------------------------------------------------------')
        print('Test iou: %.4f, dice: %.4f' % (iou, dice))
        print(cm)
        print('--------------------------------------------------------------------')
        sys.stdout.flush()


if __name__ == '__main__':
    main(config_path)

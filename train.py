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
    train_set = list_dataset.ListDataset(args['root'], 'train', args['data_name'], args['task_name'], args['fold_name'], (args['h_size'], args['w_size'], args['z_size']), args['patching'])
    train_loader = DataLoader(train_set, batch_size=args['batch_size'], num_workers=args['num_workers'], shuffle=True, drop_last=True)
    
    test_set = list_dataset.ListDataset(args['root'], 'test', args['data_name'], args['task_name'], args['fold_name'], (args['h_size'], args['w_size'], args['z_size']), args['patching'])
    test_loader = DataLoader(test_set, batch_size=1, num_workers=0, shuffle=False)
    
    # Setting network architecture.
    if 'snapshot' in args and args['snapshot'] is not None and args['snapshot_n_classes'] is not None:
        num_classes = args['snapshot_n_classes']
    else:
        num_classes = train_set.num_classes
        
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
#     net = nn.DataParallel(net) # Uncomment for multi-GPU training.
    print(net)
    print('%d trainable parameters...' % (sum(p.numel() for p in net.parameters() if p.requires_grad)))
    sys.stdout.flush()
    
    try:
        if 'snapshot' in args and args['snapshot'] is not None and os.path.isfile(args['snapshot']):
            net.load_state_dict(torch.load(os.path.join(ckpt_path, args['snapshot'])))
            net.module.replace_classifier(train_set.num_classes)
    except:
        print('Unnable to load pretrained model "%s". Finishing...' % (args['snapshot']))
        exit(0)
    
    # Weights for each class.
    weights = train_set.weights
    weights = torch.FloatTensor(weights).cuda()
    
    # Setting optimizer.
    opt = optim.Adam([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'], 'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'], 'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ], betas=(args['momentum'], 0.99))
    
    # Setting scheduler.
    scheduler = optim.lr_scheduler.StepLR(opt, args['opt_step'] or args['epoch_num'] // 10, gamma=args['opt_gamma'] or 0.5)
    
    # Making sure checkpoint and output directories are created.
    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    check_mkdir(out_path)
    check_mkdir(os.path.join(out_path, exp_name))
    check_mkdir(gif_path)
    check_mkdir(os.path.join(gif_path, exp_name))
    
    # Training only classifier.
    if 'classifier_tuning_epochs' in args and args['classifier_tuning_epochs'] is not None and args['classifier_tuning_epochs'] > 0:

        # Freezing all but last layer for tuning.
        net.module.partial_freeze()

        # Iterating over classifier tuning epochs.
        for epoch in range(1, args['classifier_tuning_epochs'] + 1):

            # Reloading config from yaml file.
            args = get_config(config_path)

            # Training function.
            train(exp_name, train_loader, net, weights, opt, epoch, args, epoch % args['test_freq'] == 0)

            if epoch % args['test_freq'] == 0:

                # Computing validation loss and test loss.
                test(exp_name, test_loader, net, weights, opt, epoch, args, True)

        # Unfreezing network.
        net.module.set_trainable(True)
    
    # Iterating over full training epochs.
    for epoch in range(1, args['epoch_num'] + 1):
        
        # Reloading config from yaml file.
        args = get_config(config_path)
        
        # Training function.
        train(exp_name, train_loader, net, weights, opt, epoch, args, epoch % args['test_freq'] == 0)
        
        if epoch % args['test_freq'] == 0 or epoch == args['epoch_num']:
            
            # Computing validation loss and test loss.
            test(exp_name, test_loader, net, weights, opt, epoch, args, True)
            
        scheduler.step()

# Training function.
def train(exp_name, train_loader, net, weights, optimizer, epoch, args, generate_outputs):
    
    # Setting network for training mode.
    net.train()
    
    # Lists for whole epoch.
    train_loss = []
        
    # Starting time.
    tic = time.time()
    
    # Iterating over batches.
    for i, data in enumerate(train_loader):
        
        # Obtaining images, labels and paths for batch.
        inps, labs, img_name = data
        
        # Casting tensors to cuda.
        inps = inps.cuda()
        labs = labs.cuda()
        
        # Clears the gradients of optimizer.
        optimizer.zero_grad()
        
        # Forwarding.
        outs = net(inps)
        
        # Computing prediction.
        prds = outs.data.max(1)[1].squeeze().cpu().numpy()
        
        # Supervised loss component.
        outs_linear = outs.permute(0, 2, 3, 4, 1).contiguous().view(-1, outs.size(1))
        labs_linear = labs.view(-1)
        
        loss = F.cross_entropy(outs_linear, labs_linear, weight=weights) + \
               dice_loss(outs_linear, labs_linear, num_classes=train_loader.dataset.num_classes, weight=weights)
        
        # Computing backpropagation and updating model.
        loss.backward()
        optimizer.step()
        
        # Updating loss list.
        train_loss.append(loss.data.item())
        
        # Outputting images and predictions.
        if generate_outputs:
            
            # Saving gifs and niftis.
            for j in range(inps.size(0)):
                
                out_img = (norm(inps[j].detach().squeeze().cpu().numpy()) * 255).astype(np.uint8)
                out_lab = labs[j].detach().squeeze().cpu().numpy().astype(np.uint8)
                out_prd = prds[j].astype(np.uint8)
                
                # Saving gifs.
                lab_path = os.path.join(gif_path, exp_name, img_name[j] + '_lab.gif')
                prd_path = os.path.join(gif_path, exp_name, img_name[j] + '_prd.gif')
                
                save_gif(out_img, out_lab, lab_path)
                save_gif(out_img, out_prd, prd_path)
                
                # Saving niftis.
                img_path = os.path.join(out_path, exp_name, img_name[j].replace('.nii.gz', '_img.nii.gz'))
                lab_path = os.path.join(out_path, exp_name, img_name[j].replace('.nii.gz', '_lab.nii.gz'))
                prd_path = os.path.join(out_path, exp_name, img_name[j].replace('.nii.gz', '_prd.nii.gz'))
                
                save_nifti(out_img, img_path)
                save_nifti(out_lab, lab_path)
                save_nifti(out_prd, prd_path)
                
        # Printing.
        if (i + 1) % args['print_freq'] == 0:
            print('[epoch %d], [iter %d / %d], [train loss %.4f]' % (epoch, i + 1, len(train_loader), np.asarray(train_loss).mean()))
            sys.stdout.flush()
    
    # Ending time.
    toc = time.time()
    
    # Printing training time.
    print('Training time %.2f' % (toc - tic))
    sys.stdout.flush()

def test(exp_name, test_loader, net, weights, optimizer, epoch, args, save_model):
    
    # Setting network for evaluation mode.
    net.eval()
    
    if (save_model):
        
        torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, 'model.pth'))
        torch.save(optimizer.state_dict(), os.path.join(ckpt_path, exp_name, 'opt.pth'))
        
    # Lists for whole epoch loss.
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
            
            # Volume reconstruction.
            inps_full = None
            labs_full = None
            prds_full = None
            
            inps_full, labs_full, prds_full = list_dataset.resize_reconstruction(net, inps_list, labs_list, off_list, size_list, strides, orig_shape)
            
            # Appending label and prediction lists for computing metrics.
            labs_all.extend(labs_full.ravel().tolist())
            prds_all.extend(prds_full.ravel().tolist())
            
            # Saving gifs.
            out_img = (norm(inps_full) * 255).astype(np.uint8)
            out_lab = labs_full.astype(np.uint8)
            out_prd = prds_full.astype(np.uint8)
            
            # Saving gifs.
            lab_path = os.path.join(gif_path, exp_name, img_name[0] + '_lab.gif')
            prd_path = os.path.join(gif_path, exp_name, img_name[0] + '_prd.gif')
            
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
        
        # Computing error metrics for whole epoch.
        if test_loader.dataset.num_classes == 2:
            iou = metrics.jaccard_score(labs_np, prds_np, average='binary')
            dice = metrics.f1_score(labs_np, prds_np, average='binary')
        else:
            iou = metrics.jaccard_score(labs_np, prds_np, average='macro')
            dice = metrics.f1_score(labs_np, prds_np, average='macro')

        cm = metrics.confusion_matrix(labs_np, prds_np)
        
        # Printing epoch loss.
        print('--------------------------------------------------------------------')
        print('Test epoch %d, iou: %.4f, dice: %.4f' % (epoch, iou, dice))
        print(cm)
        print('--------------------------------------------------------------------')
        sys.stdout.flush()


if __name__ == '__main__':
    main(config_path)

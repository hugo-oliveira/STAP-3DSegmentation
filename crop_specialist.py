import os
import numpy as np
import nibabel as nib
from skimage import measure
from matplotlib import pyplot as plt
import sys
from utils import *
import glob
import shutil
from copy import copy

def save_nifti(vol, path):
    
    vol_nifti = nib.nifti1.Nifti1Image(vol, None)
    
    nib.nifti1.save(vol_nifti, path)

def crop_to_vol(img, lab, prd, valid_labels):
    
    bboxes = []
    
    for v in valid_labels:
        
        regions = measure.regionprops(measure.label(prd == v))
        
        index_max_vol = -1
        max_vol = 0
        
        for i, r in enumerate(regions):
            
            if r.area > max_vol:
                
                max_vol = r.area
                index_max_vol = i
                
        if max_vol == 0:
            bboxes.append((0, 0, 0, img.shape[0], img.shape[1], img.shape[2]))
        else:
            bboxes.append(regions[index_max_vol].bbox)
    
    bboxes = np.asarray(bboxes)
    
    full_bbox = (bboxes.min(axis=0)[:3] - 5).tolist() + (bboxes.max(axis=0)[3:] + 5).tolist()
    
    full_bbox = [
        max(0, full_bbox[0]),
        max(0, full_bbox[1]),
        max(0, full_bbox[2]),
        min(img.shape[0], full_bbox[3]),
        min(img.shape[1], full_bbox[4]),
        min(img.shape[2], full_bbox[5])
    ]
    
    img_crop = img[full_bbox[0]:full_bbox[3],
                   full_bbox[1]:full_bbox[4],
                   full_bbox[2]:full_bbox[5]]
    lab_crop = lab[full_bbox[0]:full_bbox[3],
                   full_bbox[1]:full_bbox[4],
                   full_bbox[2]:full_bbox[5]]
    
    lab_crop[np.in1d(lab_crop, valid_labels).reshape(lab_crop.shape) == False] = 0
    counts = [np.count_nonzero(lab_crop == c) for c in range(4)]
    #print(float(counts[0] + counts[1] + counts[2]) / counts[3])
    return img_crop, lab_crop, full_bbox



config_path = sys.argv[1]

# taking the arguments 
args = get_config(config_path)
arch = args['conv_name']
data_name = args['root'] + args['data_name']
exp = args['exp_name']
fold_name = args['fold_name']

# setting the paths
tst_config = f'{data_name}/all_tst_f{fold_name}.txt'
img_dir = f'{data_name}/images/'
lab_dir = f'{data_name}/ground_truths/all/'
prd_dir = f'./outputs/{exp}/'


# setting the testing 
f = lambda x: x.replace(".nii.gz\n", "").replace(".nii.gz","")
lines = open(tst_config, 'r').readlines()
test_imgs = list(map(f, lines))

for specialist_task, valid_labels in [('cerebellum', [3]), ('stem_ventricle', [1,2])]:
    
    path = '%s_%s_%s_f%d' % (args['data_name'], specialist_task, arch, fold_name)
    out_img_dir = f'./bbox/{path}/images/' 
    out_lab_dir = f'./bbox/{path}/ground_truths/all/' 
    coord_file = f'./bbox/bboxes_{arch}_{specialist_task}_{fold_name}.csv' 
    config_file_path = config_path[:-5] +  specialist_task + '.yaml'
    
    if not os.path.isdir(f'./bbox/{path}/'):
        os.mkdir(f'./bbox/{path}/')
    if not os.path.isdir(f'./bbox/{path}/images/'):
        os.mkdir(f'./bbox/{path}/images/')
    if not os.path.isdir(f'./bbox/{path}/ground_truths/'):
        os.mkdir(f'./bbox/{path}/ground_truths/')
    if not os.path.isdir(f'./bbox/{path}/ground_truths/all/'):
        os.mkdir(f'./bbox/{path}/ground_truths/all/')

    for file_txt in glob.glob(f'bbox/{specialist_task}/*.txt'):
        shutil.copyfile(file_txt, file_txt.replace(f'bbox/{specialist_task}/', f'./bbox/{path}/'))
    
    args_new = copy(args)
    args_new['data_name'] = path
    args_new['root'] = './bbox/'
    args_new['exp_name'] = path
    save_new_config(args_new, config_file_path)

    

    files = sorted([f for f in os.listdir(lab_dir) if os.path.isfile(os.path.join(lab_dir, f))])


    with open(coord_file, 'w') as csv_file:
        
        # Iterating over files.
        for i, f in enumerate(files):
            
            print('%d/%d: "%s"' % (i + 1, len(files), f))

            # Setting paths.
            img_path = os.path.join(img_dir, f)
            lab_path = os.path.join(lab_dir, f)
            out_img_path = os.path.join(out_img_dir, f)
            out_lab_path = os.path.join(out_lab_dir, f)
            prd_path = os.path.join(prd_dir, f.replace('.nii.gz', '_prd.nii.gz'))

            img_name = f[:-7]
            # Loading nifti files.
            img = nib.load(img_path)
            lab = nib.load(lab_path)
            prd = nib.load(prd_path)
            # Getting ndarray.
            img_np = img.get_fdata()
            lab_np = lab.get_fdata()
            prd_np = prd.get_fdata()
            
            if img_name in test_imgs:
                # Cropping images and labels according to predictions.
                img_crop, lab_crop, coords = crop_to_vol(img_np, lab_np, prd_np, valid_labels)
            else:
                img_crop, lab_crop, coords = crop_to_vol(img_np, lab_np, lab_np, valid_labels)

            print(img_crop.shape)
            
            csv_file.write('%s %d %d %d %d %d %d\n' % (f, coords[0], coords[1], coords[2], coords[3], coords[4], coords[5]))

            # Saving cropped volumes.
            save_nifti(img_crop, out_img_path)
            save_nifti(lab_crop, out_lab_path)

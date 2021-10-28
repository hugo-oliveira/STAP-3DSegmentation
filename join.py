import nibabel as nib
import numpy as np
import sys
from utils import *
import os
import re
from sklearn import metrics


def join_seg(prd_full, prd_cere, prd_vent, bbox_cere, bbox_vent):
    prd_full[:] = 0
    prd_full[bbox_cere[0]:bbox_cere[3],
             bbox_cere[1]:bbox_cere[4],
             bbox_cere[2]:bbox_cere[5]] = prd_cere

    index = np.where(
            prd_full[bbox_vent[0]:bbox_vent[3],
                     bbox_vent[1]:bbox_vent[4],
                     bbox_vent[2]:bbox_vent[5]] == 0
        )
    prd_full[index] = prd_vent[index]

    return prd_full




config_path = sys.argv[1]
config_path_cere = sys.argv[2]
config_path_vent = sys.argv[3]

# taking the arguments 
args = get_config(config_path)
args_cere = get_config(config_path_cere)
args_vent = get_config(config_path_vent)
arch = args['conv_name']
data_name = args['root'] + args['data_name']
exp = args['exp_name']
fold_name = args['fold_name']


tst_config = f'{data_name}/all_tst_f{fold_name}.txt'
img_dir = f'{data_name}/images/'
lab_dir = f'{data_name}/ground_truths/all/'
prd_dir = f'./outputs/{exp}'

prd_cere = f'./outputs/{args_cere["exp_name"]}/'
prd_vent = f'./outputs/{args_vent["exp_name"]}/'
coord_file_cere = f'./bbox/bboxes_{arch}_cerebellum.csv' 
coord_file_vent = f'./bbox/bboxes_{arch}_stem_ventricle.csv' 

f = lambda x: x.replace(".nii.gz\n", "").replace(".nii.gz","")
lines = open(tst_config, 'r').readlines()
test_imgs = list(map(f, lines))

tst_config = f'{data_name}/all_tst_f{fold_name}.txt'

path = '%s_specialists_%s_f%d' % (args['data_name'], arch, fold_name)

if not os.path.isdir(f'./prd_final/{path}/'):
    os.mkdir(f'./prd_final/{path}/')

files = sorted([f for f in os.listdir('./outputs/hc_pediatric_cerebellum_highresnet_f0/') if os.path.isfile(os.path.join(prd_cere, f)) and re.match(r'.*_prd\.nii\.gz', f)])

csv_file_cere = open(coord_file_cere, 'r')
csv_file_vent = open(coord_file_vent, 'r')

lines_cere = csv_file_cere.readlines()
lista_cere = [i.split() for i in lines_cere]
dir_cere = {i[0]: np.array(i[1:], dtype='int') for i in lista_cere}

lines_vent = csv_file_vent.readlines()
lista_vent = [i.split() for i in lines_vent]
dir_vent = {i[0]: np.array(i[1:], dtype='int') for i in lista_vent}

prds_np = []
labs_np = []
for i, f in enumerate(files):

    lab_path = os.path.join(lab_dir, f.replace('_prd', ''))
    prd_full_path = os.path.join(prd_dir, f)
    prd_cere_path = os.path.join(prd_cere, f)
    prd_vent_path = os.path.join(prd_vent, f)

    out_lab_path = os.path.join(f'./prd_final/{path}/', f)
    full = nib.load(prd_full_path)
    cere = nib.load(prd_cere_path)
    vent = nib.load(prd_vent_path)
    lab = nib.load(lab_path)

    lab_np = lab.get_fdata()
    full_np = full.get_fdata()
    cere_np = cere.get_fdata()
    vent_np = vent.get_fdata()


    prds_all.extend(full.ravel().tolist())
    labs_np.extend(lab_np.ravel().tolist())

    img_join = join_seg(full_np, cere_np, vent_np, dir_cere[f], dir_vent[f])

    save_nifti(img_join, out_lab_path)


prds_np = np.asarray(prds_all).ravel()
labs_np = np.asarray(lab_np).ravel()
iou = metrics.jaccard_score(labs_np, prds_np, average='macro')
dice = metrics.f1_score(labs_np, prds_np, average='macro')

cm = metrics.confusion_matrix(labs_np, prds_np)
print('--------------------------------------------------------------------')
print('iou: %.4f, dice: %.4f' % ( iou, dice))
print(cm)
print('--------------------------------------------------------------------')
sys.stdout.flush()
import os
import numpy as np
import argparse 

parser = argparse.ArgumentParser(description='Process the split of the dataset in training folders.')
parser.add_argument('dir', help="Root of the directory of your dataset.")
parser.add_argument('--random_seed', '-r', help="Random seed to perform split.", type=int, default=12345)

args = parser.parse_args()
seed = args.random_seed
root = args.dir

if root[-1] = '/'
    root = root[:-1]

# Root dirs.
img_dir = f'{root}/images/'
lab_dir = f'{root}/ground_truths/all/'

# Listing files.
files = sorted([f for f in os.listdir(lab_dir) if os.path.isfile(os.path.join(lab_dir, f))])

# Generating random permutation.
np.random.seed(seed)
perm = np.random.permutation(len(files))

# Reordering file list.
files = [files[perm[i]] for i in range(len(files))]

# Generating folds from permutation.
n_folds = 5

# Iterating over folds.
for fold in range(n_folds):
    
    fold_tst = sorted(files[fold::n_folds]) # Selecting test samples.
    fold_trn = sorted([f for f in files if f not in fold_tst]) # Splitting remaining training samples.
    
    with open(f'{root}/all_tst_f{fold}.txt', 'w') as tst_file:
        
        for i, f in enumerate(fold_tst):
            tst_file.write('%s' % (f))
            if i < len(fold_tst) - 1:
                tst_file.write('\n')
    
    with open(f'{root}/all_trn_f{fold}.txt', 'w') as trn_file:
        
        for i, f in enumerate(fold_trn):
            trn_file.write('%s' % (f))
            if i < len(fold_trn) - 1:
                trn_file.write('\n')
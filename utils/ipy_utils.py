import os, shutil
from IPython.display import FileLink
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from tqdm import tqdm

PREFIX = ''.join(['../']*(len(os.getcwd().split('Max')[-1].split('/'))-1))

def download_files(files, postfix=None):
    tmp_dir = PREFIX+'tmp_midi/download_batch'
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir)
    n_digits = int(np.ceil(np.log10(len(files)+1)))
    if postfix is None:
        postfix = ['']*len(files)
    for i,f in enumerate(files):
        fname = f.split('/')[-1]
        shutil.copy(f'{f}', f'{tmp_dir}/{i:>0{n_digits}} {postfix[i]} {fname}')
    shutil.make_archive(tmp_dir,'zip', tmp_dir)
    return FileLink(tmp_dir+'.zip')

def download_midis(midis):
    tmp_dir = PREFIX+'tmp_midi/download_midis'
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir)
    n_digits = int(np.ceil(np.log10(len(midis)+1)))
    for i,midi in enumerate(midis):
        midi.write(f'{tmp_dir}/midi_{i:>0{n_digits}}.mid')
    shutil.make_archive(tmp_dir,'zip', tmp_dir)
    return FileLink(tmp_dir+'.zip')

def show_tensor(tensor, transpose=None, normalize=None, figsize=(10,10), nrow=None, padding=2, verbose=True, **kwargs):
    '''Flexible tool for visulizing tensors of any shape. Support batch_size >= 1.'''
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(np.array(tensor))
    tensor = tensor.detach().cpu().float()
    
    if tensor.ndim == 4 and tensor.shape[1] == 1:
        if verbose: print('processing as black&white')
        tensor = tensor.repeat(1,3,1,1)
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 2:
        if verbose: print('processing as black&white')
        tensor = tensor.unsqueeze(0).repeat(3,1,1).unsqueeze(0)
        
    if normalize is None:
        if tensor.max() <= 1.0 and tensor.min() >= 0.0:
            normalize = False
        else:
            if verbose: print('tensor has been normalized to [0., 1.]')
            normalize = True
            
    if transpose is None:
        transpose = True if tensor.shape[1] != 3 else False
    if transpose:
        tensor = tensor.permute(0,3,1,2)
    
    if nrow is None:
        nrow = int(np.ceil(np.sqrt(tensor.shape[0])))
        
    grid = torchvision.utils.make_grid(tensor, normalize=normalize, nrow=nrow, padding=padding, **kwargs)
    plt.figure(figsize=figsize)
    return plt.imshow(grid.permute(1,2,0))


def multi_process(fn, x, start=0, n_proc=None):
    from concurrent.futures import ProcessPoolExecutor
    from joblib import cpu_count
    if not n_proc:
        n_proc = cpu_count()
    x = x[start:]
    with ProcessPoolExecutor(n_proc) as pool:
        return list(tqdm(pool.map(fn, x), position=0, total=len(x)))
    
def ewm(x, alpha=0.01):
    import pandas as pd
    return pd.Series(x).ewm(alpha=alpha).mean()
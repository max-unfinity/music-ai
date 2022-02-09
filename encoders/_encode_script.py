import os, sys, shutil
import json
import re
import itertools
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import time
import pretty_midi

sys.path.append('../')

from utils import ipy_utils
from encoders.BaseEncoder import BaseEncoder, DataInfo
from encoders.Midi2NumpyRelEncoder import Midi2NumpyRelEncoder


import hashlib

class HashedSaver:
    def __init__(self, out_dir, n0=2):
        self.out_dir = out_dir
        self.n0 = n0
    
    def save(self, file, splits):
        out_dir = self.out_dir
        n0 = self.n0
        saves = []
        
        path, fname = os.path.split(file)
        fname, ext = os.path.splitext(fname)
        h = hashlib.md5(file.encode()).hexdigest()
        for i,split in enumerate(splits):
            save_name = f'{fname}_{h}_{i:0>{n0}}.pt'
            try:
                torch.save(split, f'{out_dir}/{save_name}')
            except OSError:  # if file_name is too long
                save_name = f'{h}_{i:0>{n0}}.pt'
                torch.save(split, f'{out_dir}/{save_name}')
            saves += [save_name]
        return saves


main_dir = '../'
dataset_name = 'lakh'
data_list = 'data/_data_lists/lakh.pt'

encoder = Midi2NumpyRelEncoder()

info = DataInfo()
info.dataset_name = dataset_name
info.src_data_list = os.path.abspath(main_dir + data_list)
info.src_dir, info.src_files = torch.load(info.src_data_list)
info.out_dir = os.path.abspath(f'{main_dir}/data/{dataset_name}_{encoder.__class__.__name__}/data')
info.out_files = []
info.out2src = []

encoder.data_info = info
saver = HashedSaver(info.out_dir)

assert not os.path.exists(info.out_dir)
os.makedirs(info.out_dir)


def fn(file):
    info = encoder.data_info
    encoded = encoder.encode(info.src_dir+file)
    if not encoded: saves = []
#     splits = encoder.collate(encoded) if encoder.auto_collate else [encoded]
#     if not splits: saves = []
    else:
        saves = saver.save(file, [encoded])
    info.src2out.append(saves)
    return saves

ipy_utils.multi_process(fn, info.src_files)

print('len(info.src2out)', len(info.src2out))
info.out_files = [y for x in encoder.data_info.src2out for y in x if x]
info.out2src = [[i for y in x] for i,x in enumerate(info.src2out) if x]
print('len(info.out2src)', len(info.out2src))

ds_files = list(map(str, Path(info.out_dir).rglob('*.*')))
print('len(ds_files)', len(ds_files))

torch.save(info.out_files, f'{info.out_dir}/../data_list.pt')
encoder.save()

print('SAVED!')

os.system(f'ls {info.out_dir}/../ -lh')
os.system(f'du -sh {info.out_dir}')

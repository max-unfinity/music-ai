import os
import numpy as np
import torch

from utils import ipy_utils
from .BaseEncoder import BaseEncoder


class ImgEncoderMulti(BaseEncoder):
    def __init__(self, step=0.05, n_inst=4, max_len=256, dtype=np.float32):
        self.step = step
#         self.cache_len = cache_len
#         self.cache = np.exp(k*-x)
        self.n_inst
        self.max_len = max_len
        self.dtype = dtype

        self.max_sec = max_len*step

    def get_sepv(self, notes):
        starts,ends,pitches,vels = [x.T[0] for x in np.split(notes, 4, 1)]
        pitches = pitches.astype(int)
        vels = vels/128.
        starts = np.clip(np.round(starts/self.step).astype(int), 0, self.max_len-1)
        ends = np.clip(np.round(ends/self.step).astype(int), 0, self.max_len-1)
        return starts, ends, pitches, vels
        
    def encode(self, notes_multi, t0):
        t_max = max([x[-1,1] for x in notes_multi])
        t0 = np.maximum(t_max-self.max_sec/2, 0)*t0
        t1 = t0+self.max_sec

        img = np.zeros((self.n_inst,128,self.max_len), dtype=np.float32)

        for i in range(self.n_inst):
            if i >= len(notes_multi):
                continue    
            notes = notes_multi[i]
            a = notes[:,0]
            y = notes[:,1]
            notes = notes[(a>=t0) & (y<t1)]
            if notes.size == 0:
                continue    

            notes[:,:2] = notes[:,:2] - notes[0,:2].min()

            starts, ends, pitches, vels = self.get_sepv(notes)
            img[i, pitches, ends] = -1.
            img[i, pitches, starts] = vels + 1.0

        return img


class ImgEncoderSingle(BaseEncoder):
    def __init__(self, step=0.05, max_len=256, dtype=np.float32):
        self.step = step
#         self.cache_len = cache_len
#         self.cache = np.exp(k*-x)
        self.max_len = max_len
        self.dtype = dtype

        self.max_sec = max_len*step

    def get_sepv(self, notes):
        starts,ends,pitches,vels = [x.T[0] for x in np.split(notes, 4, 1)]
        pitches = pitches.astype(int)
        vels = vels/128.
        starts = np.clip(np.round(starts/self.step).astype(int), 0, self.max_len-1)
        ends = np.clip(np.round(ends/self.step).astype(int), 0, self.max_len-1)
        return starts, ends, pitches, vels
        
    def encode(self, notes, t0):
        t_max = notes[-50:,1].max()
        t0 = np.maximum(t_max-self.max_sec/2, 0)*t0
        t1 = t0+self.max_sec

        img = np.zeros((128,self.max_len), dtype=np.float32)

        a = notes[:,0]
        y = notes[:,1]
        notes = notes[(a>=t0) & (y<t1)]
        if len(notes) == 0:
#             print('empty notes:', t0,t1,t_max)
            return img

        notes[:,:2] = notes[:,:2] - notes[0,:2].min()

        starts, ends, pitches, vels = self.get_sepv(notes)
        img[pitches, ends] = -1.
        img[pitches, starts] = vels + 1.0

        return img

def tri2one(img, axis=0):
    return np.take(img, 0, axis)+np.take(img, 2, axis)-np.take(img, 1, axis)

def one2tri(img):
    res = np.zeros((3,)+img.shape, dtype=img.dtype)
    m0 = img>=1
    m1 = img<0
    res[0,m0] = 1
    res[1,m1] = 1
    res[2,m0] = img[m0]-1
    return res


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encoder):
        memory, memory_idx = torch.load(ipy_utils.PREFIX+'V2/data/lakh_Midi2NumpyEncoder/memory_4tracks_19k_single.pt')
        self.memory = memory
        self.encoder = encoder
        
    def __len__(self):
        return len(self.memory)
    
    def __getitem__(self, i):
        x = self.memory[i]
        t0 = np.random.rand()
        return self.encoder.encode(x, t0=t0)
    
def process_batch(batch, device):
    x = batch
    x = x[:,None]
    mask = (x>=1) | (x<0)
    return x.to(device), mask.to(device)


from sklearn.metrics import precision_score, recall_score, f1_score, mean_absolute_error

def metrics(x, rec, thr_on=0.5, thr_off=-0.1):
    x_m_on = (x>=thr_on).flatten()
    x_m_off = (x<thr_off).flatten()
    rec_m_on = (rec>=thr_on).flatten()
    rec_m_off = (rec<thr_off).flatten()

    pr_on = precision_score(x_m_on,rec_m_on), recall_score(x_m_on,rec_m_on)
    pr_off = precision_score(x_m_off,rec_m_off), recall_score(x_m_off,rec_m_off)

    f1_on, f1_off = f1_score(x_m_on,rec_m_on), f1_score(x_m_off,rec_m_off)

    mi_on = x_m_on&rec_m_on
    mu = x_m_on|rec_m_on
    iou_on = mi_on.sum()/mu.sum()

    mi = x_m_off&rec_m_off
    mu = x_m_off|rec_m_off
    iou_off = mi.sum()/mu.sum()

    vel_score = 1-mean_absolute_error(x.flatten()[mi_on]-1, rec.flatten()[mi_on]-1) if mi_on.sum()>0 else 0.0

    return (pr_on,pr_off), (f1_on,f1_off), (iou_on,iou_off), vel_score
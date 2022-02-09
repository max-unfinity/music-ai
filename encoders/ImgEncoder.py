import os
import numpy as np
from itertools import zip_longest
import torch

from utils import ipy_utils
from .BaseEncoder import BaseEncoder


class ImgEncoderMulti(BaseEncoder):
    def __init__(self, step=0.03125, n_inst=9, max_len=256, dtype=np.float32):
        self.step = step
        self.n_inst = n_inst
        self.max_len = max_len
        self.dtype = dtype

        self.max_sec = max_len*step
        self.n_pithes = 128

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

        img = np.zeros((self.n_inst,self.n_pithes,self.max_len), dtype=self.dtype)

        for i in range(self.n_inst):
            if i >= len(notes_multi):
                continue    
            notes = notes_multi[i]
            a = notes[:,0]
            y = notes[:,1]
            notes = notes[(a>=t0) & (a<t1)]
            if notes.size == 0:
                continue    

            notes[:,:2] = notes[:,:2] - t0

            starts, ends, pitches, vels = self.get_sepv(notes)
            img[i, pitches, ends] = -1.
            img[i, pitches, starts] = vels + 1.0

        return img
    
    def decode(self, img_multi, thr_on=0.5, thr_off=-0.5, strict_mode=False):
        assert img_multi.ndim == 3
        assert img_multi.shape[1] == self.n_pithes
        max_len = self.max_len
        step = self.step

        notes_multi = []
        for img in img_multi:
            ons = img >= thr_on
            offs = img < thr_off
            alls = ons | offs

            notes = []
            for pitch in range(self.n_pithes):
                alls_idxs = alls[pitch].nonzero()
                if not len(alls_idxs):
                    continue
                else:
                    alls_idxs = alls_idxs[0]
                for start_idx in ons[pitch].nonzero()[0]:
                    start = start_idx*step
                    vel = int((img[pitch,start_idx]-thr_on)/(2-thr_on)*127)
                    end_idx = alls_idxs[alls_idxs > start_idx]
                    if not len(end_idx):
                        if strict_mode:
                            continue
                        else:
                            end_idx = [max_len]    
                    end = end_idx[0]*step
                    notes.append([start, end, pitch, vel])
            notes = np.array(notes, dtype=self.dtype)
            if len(notes):
                notes = notes[np.argsort(notes[:,0])]
            notes_multi.append(notes)
        return notes_multi
    
    def notes2midi(self, notes_multi, programs=0, tempo=120.):
        if not hasattr(programs, '__iter__'): programs = [programs]*len(notes_multi)
        programs = programs[:len(notes_multi)]
        notes = [[pretty_midi.Note(int(x[3]),int(x[2]),x[0],x[1]) for x in notes_np] for notes_np in notes_multi]
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        for notes_i,program_i in zip_longest(notes, programs, fillvalue=0):
            if not len(notes_i): continue
            inst = pretty_midi.Instrument(0 if program_i==-1 else program_i, is_drum=program_i==-1, name=f'program_{program_i}')
            inst.notes = notes_i
            midi.instruments.append(inst)
        return midi

    
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
        memory = torch.load(ipy_utils.PREFIX+'V2/data/lakh_Midi2NumpyEncoder/memory_4tracks_19k_multi.pt')
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
#     x = x[:,None]
    mask = (x>=1) | (x<0)
    return x.to(device), mask.to(device)


from sklearn.metrics import precision_score, recall_score, f1_score, mean_absolute_error

def calc_metrics(x, rec, thr_on=0.5, thr_off=-0.5):
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
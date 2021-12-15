import os, sys, shutil
import json
import re
import itertools
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
import time
from IPython.display import FileLink
import pretty_midi

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .BaseEncoder import BaseEncoder


class CNNEncoder(BaseEncoder):
    def __init__(self, step=0.05, decay_shift=1, vel_min=0.0, cache_len=128, max_len=256, min_len=32, **kw):
        self.step = step
        self.decay_shift = decay_shift
        self.cache_len = cache_len
        self.cache = (np.arange(self.cache_len, dtype=np.float32)+self.decay_shift)**-0.5/self.decay_shift**-0.5
        self.vel_min = vel_min
#         self.b = 128*self.vel_min/(1-self.vel_min)
        self.max_len = max_len
        self.min_len = min_len

    def encode(self, notes):
        img_size = int(np.ceil(notes[:,1].max()/self.step))

        i_sort = np.argsort(notes[:,0])
        notes[i_sort]

        starts,ends,pitches,vels = [x.T[0] for x in np.split(notes, 4, 1)]
        pitches = pitches.astype(int)
        vels = vels/128
        starts = np.round(starts/self.step).astype(int)
        ends = np.clip(np.round(ends/self.step).astype(int), a_min=0, a_max=img_size-1)

        img = np.zeros((128,img_size), dtype=np.float32)
        for a,b,p,v in zip(starts,ends,pitches,vels):
            if b<a:
                continue
            l = b-a
            img[p,a:min(b,a+self.cache_len)] = self.cache[:min(l, self.cache_len)]
            img[p,a] = v+1
            img[p,b] = -1

        return img

    def decode(self, img):
        # 1. find NOTE_ONs
        # 2. extract VEL
        # 3. compare 512 right to NOTE_ON (while img == cache). Нота прирывается там, где она != cache
        return

    def to_midi(self, events):
        return
        pitch, length, vel, time_shifts = events
        
        notes = []
        overlap_tracking = {}  # {pitch: Note}
        t = 0
        for note_pitch, note_len, note_vel, ts in zip(pitch, length, vel, time_shifts):
            t += ts
            note_start = t
            note_end = t+note_len
            if overlap_tracking.get(note_pitch) is not None:
                note_prev = overlap_tracking[note_pitch]
                if note_start < note_prev.end:
                    # do NOTE_OFF earler
                    note_prev.end = note_start
            p_note = pretty_midi.Note(note_vel, note_pitch, note_start, note_end)
            notes.append(p_note)
            overlap_tracking[note_pitch] = p_note

        midi_out = pretty_midi.PrettyMIDI()
        midi_out.instruments.append(pretty_midi.Instrument(0, name='piano'))
        midi_out.instruments[0].notes = notes
        return midi_out
    
    def collate(self, img):
        l = img.shape[1]
        split_idxs = np.cumsum([self.max_len]*(l//self.max_len))
        splits = np.split(img, split_idxs, -1)
        last_len = splits[-1].shape[1]
        if last_len < self.min_len:
            splits.pop(-1)
        else:
            splits[-1] = np.pad(splits[-1], [[0,0],[0,self.max_len-last_len]])
        return splits
    
    
class CNNDataset(Dataset):
    def __init__(self, ds_files, prefix_path='', transform=None):
        self.transform = transform
        self.files = torch.load(ds_files)
        self.prefix_path = prefix_path
        self.genre2id = {'classic':0, 'jazz':1, 'calm':2, 'pop':3}
        
        genres = [f.split('/')[3] for f in self.files]
        assert len([0 for g in set(genres) if g not in self.genre2id]) == 0

        self.genres = [self.genre2id[g] for g in genres]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = torch.load(self.prefix_path + self.files[idx])
        if self.transform:
            x = self.transform(x)
        x = torch.from_numpy(x)
        genre = self.genres[idx]
        return x, genre, idx
    
def process_batch(batch, device):
    x, genre, idx = batch
    x = x[:,None]
    mask = (x>=1) | (x<0)
    return x.to(device), mask.to(device)
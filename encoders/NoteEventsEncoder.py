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


DEFAULT_SPECIAL_TOKENS = {'PAD':0, 'BOS':1, 'EOS':2, 'MASK':3}


def quantize_log(x, step, k):
    return np.round(np.log(x*k+1)/step)*step


def unquantize_log(x, step, k):
    return (np.exp(x)-1)/k


class NoteEventsEncoder(BaseEncoder):
    def __init__(self, n_len=35, n_vel=32, n_ts=65, len_quantize_step=0.15, len_quantize_k=20, ts_quantize_step=0.08, ts_quantize_k=16, max_seq_len=512, min_seq_len=32, special_tokens=None):
        self.n_len = n_len
        self.n_vel = n_vel
        self.n_ts = n_ts
        self.len_quantize_step = len_quantize_step
        self.len_quantize_k = len_quantize_k
        self.ts_quantize_step = ts_quantize_step
        self.ts_quantize_k = ts_quantize_k
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.special_tokens = special_tokens or DEFAULT_SPECIAL_TOKENS

    def encode(self, notes, tonal_shift):
        """Return: pitch, octave, length, vel, time_shifts"""
        i_sort = np.argsort(notes[:,0])
        notes = notes[i_sort]
        starts = notes[:,0]
        
        time_shifts = np.diff(starts, prepend=starts[0])  # add TS before every note-event
        time_shifts = np.clip(np.round(quantize_log(time_shifts, step=self.ts_quantize_step, k=self.ts_quantize_k)/self.ts_quantize_step).astype(int), a_min=0, a_max=self.n_ts-1)
        length = (notes[:,1] - notes[:,0])
        length = np.clip(np.round(quantize_log(length, step=self.len_quantize_step, k=self.len_quantize_k)/self.len_quantize_step).astype(int), a_min=0, a_max=self.n_len-1)
        vel = (notes[:,3]*self.n_vel/128).astype(int)
        pitch = notes[:,2].astype(int)
        pitch, octave = (pitch-tonal_shift)%12, (pitch-tonal_shift)//12
        return pitch, octave, length, vel, time_shifts

    def decode(self, events, tonal_shift):
        pitch, octave, length, vel, time_shifts = events

        pitch = octave*12+pitch+tonal_shift
        vel = np.clip(vel*128//self.n_vel, a_min=1, a_max=127)
        length = unquantize_log(length*self.len_quantize_step, step=self.len_quantize_step, k=self.len_quantize_k)
        time_shifts = unquantize_log(time_shifts*self.ts_quantize_step, step=self.ts_quantize_step, k=self.ts_quantize_k)
        
        return pitch, length, vel, time_shifts

    def to_midi(self, events):
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
    
    def collate(self, events, genre):
        pad, bos, eos = [self.special_tokens[x] for x in ['PAD', genre, 'EOS']]
        
        events = np.stack(events) + len(self.special_tokens)
        effective_len = self.max_seq_len - 1  # 1 for BOS
        E, L = events.shape        
        n_splits = L//effective_len
        if L%effective_len >= self.min_seq_len:
            n_splits += 1
        if n_splits <= 0:
            return []
        splits = [np.zeros((E,self.max_seq_len), dtype=events.dtype)+pad for i in range(n_splits)]
        
        i = 0
        for s in splits:
            s[:,0] = [bos]*E
            s[:,1:min(self.max_seq_len, L-i+1)] = events[:,i:i+effective_len]
            i += effective_len
        
        s = splits[-1]
        if s[0,-1] == pad:
            eos_idx = (s[0] == pad).nonzero()[0][0]
            s[:,eos_idx] = eos
        return splits
    
    def get_embed_dims(self):
        n_embeds = [12, 11, self.n_len, self.n_vel, self.n_ts]
        return [x+len(self.special_tokens) for x in n_embeds]


class NoteEventsDataset(Dataset):
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
    x0,genre,idx = batch
    x0 = x0.permute(1,2,0)
    x = x0[:,:-1].to(device)
    tgt = x0[:,1:].to(device)
    return x, tgt, genre, idx


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('weight', pe)

    def forward(self, x):
        x = x + self.weight[:x.size(0),None]
        return x

class NoteEventsEmbedding(nn.Module):
    def __init__(self, embed_dims, d_model=768, d_embed=320, max_len=512, ACT=nn.ReLU, dropout=0.1, use_positional_ecnoding=True, **kw):
        super().__init__()
        self.use_positional_ecnoding = use_positional_ecnoding
        self.n_embed = len(embed_dims)
        
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.embeddings = nn.ModuleList([nn.Embedding(n, d_embed, padding_idx=0) for n in embed_dims])
        self.embed_linear = nn.Sequential(nn.Linear(d_embed*self.n_embed, d_model), ACT(), nn.Dropout(dropout))
        
    def forward(self, x):
        # x: E,T,B
        assert x.shape[0] == len(self.embeddings)
        T = x.shape[1]
        x = torch.cat([emb(t) for emb,t in zip(self.embeddings, x)], -1)
        x = self.embed_linear(x)
        if self.use_positional_ecnoding:
            x = self.positional_encoding(x)
        return x
    
class NoteEventsHead(nn.Module):
    def __init__(self, embed_dims, variant=0, d_model=768, d_embed=320, ACT=nn.ReLU, **kw):
        super().__init__()
        self.n_embed = len(embed_dims)
        if variant == 0:
            self.out_proj1 = nn.Sequential(nn.Linear(d_model, d_embed*self.n_embed), ACT(), nn.LayerNorm(d_embed*self.n_embed))
            self.out_proj2 = nn.ModuleList([nn.Sequential(nn.Linear(d_embed, d_embed*2), ACT(), nn.LayerNorm(d_embed*2)) for n in embed_dims])
            self.out_proj3 = nn.ModuleList([nn.Linear(d_embed*2, n) for n in embed_dims])
        elif variant == 1:
            self.out_proj1 = nn.Sequential(nn.Linear(d_model, d_embed*self.n_embed), ACT())
            self.out_proj2 = nn.ModuleList([nn.Identity() for n in embed_dims])
            self.out_proj3 = nn.ModuleList([nn.Linear(d_embed, n) for n in embed_dims])
        
    def forward(self, x):
        # x: E,T,B
        x = self.out_proj1(x)
        x = torch.chunk(x, self.n_embed, -1)
        x = [m(t) for m,t in zip(self.out_proj2, x)]
        x = [m(t) for m,t in zip(self.out_proj3, x)]
        return x

class NoteEventsModel(nn.Module):
    def __init__(self, embedding, structure, head, pad_token):
        super().__init__()
        self.embedding, self.structure, self.head = embedding, structure, head
        self.pad_token = pad_token
    
    def forward(self, x):
        # x: E,T,B
        x0 = x
        x = self.embedding(x)
        src_key_padding_mask = x0[0].eq(self.pad_token).T
        x = self.structure(x, padding_mask=src_key_padding_mask)
        x = self.head(x)
        return x

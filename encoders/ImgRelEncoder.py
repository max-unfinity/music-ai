import os
import numpy as np
from itertools import zip_longest
import pretty_midi
import torch

from utils import ipy_utils
from .BaseEncoder import BaseEncoder


class ImgRelEncoder(BaseEncoder):
    """quantize=64, n_inst=12, n_bars=4, max_len=256, dtype=np.float32"""
    
    def __init__(self, quantize=64, n_inst=15, n_bars=4, max_len=256, dtype=np.float32):
        self.quantize = quantize
        self.n_inst = n_inst
        self.n_bars = n_bars
        self.max_len = max_len
        self.dtype = dtype
        self.n_pithes = 128

    def get_sepv(self, notes):
        starts,ends,pitches,vels = [x.T[0] for x in np.split(notes, 4, 1)]
        pitches = pitches.astype(int)
        vels = vels/128.
        starts = np.clip(starts, 0, self.max_len-1).astype(int)
        ends = np.clip(ends, 0, self.max_len-1).astype(int)
        return starts, ends, pitches, vels
        
    def encode(self, encoded, t0=None):
        notes_multi, programs, bar_info = encoded['notes'], encoded['programs'], encoded['bar_info']

        if t0 is None:
            t0 = np.random.randint(0, max(1,len(notes_multi)-(self.n_bars-1)))

        notes = notes_multi[t0:t0+self.n_bars]
        programs = np.array(programs)//8+1
        bar_info_part = bar_info[t0:t0+self.n_bars]

        img = np.zeros((self.n_inst,self.n_pithes,self.max_len), dtype=self.dtype)
        bar_cum = 0
        for i_bar, (bar,bar_info_i) in enumerate(zip(notes,bar_info_part)):
            for notes_i, p_i in zip(bar, programs):
                if p_i >= self.n_inst:
                    continue
                notes_i = np.array(notes_i, dtype=self.dtype)
                if not len(notes_i):
                    continue
                notes_i[:,:2] += bar_cum
                starts, ends, pitches, vels = self.get_sepv(notes_i)
                img[p_i, pitches, ends] = -1.
                img[p_i, pitches, starts] = vels + 1.0
            ts = bar_info_i[1]
            bar_cum += self.quantize*ts[0]/ts[1]

        return img
    
    def decode(self, img_multi, tempo=120., thr_on=0.5, thr_off=-0.5, strict_mode=False):
        assert img_multi.ndim == 3
        assert img_multi.shape[1] == self.n_pithes
        max_len = self.max_len
        step = 1/tempo*60*4/self.quantize

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
    
    def notes2midi(self, notes_multi, tempo=120., programs=None):
        if programs is None:
            programs = (np.arange(self.n_inst)-1)*8
            programs[0] = -1
        else:
            assert len(programs) == self.n_inst
        notes = [[pretty_midi.Note(int(x[3]),int(x[2]),x[0],x[1]) for x in notes_np] for notes_np in notes_multi]
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        for notes_i,program_i in zip_longest(notes, programs, fillvalue=0):
            if not len(notes_i): continue
            inst = pretty_midi.Instrument(0 if program_i==-1 else program_i, is_drum=program_i==-1, name=f'program_{program_i}')
            inst.notes = notes_i
            midi.instruments.append(inst)
        return midi
import numpy as np
import pretty_midi
import collections
import warnings
from itertools import zip_longest

from .BaseEncoder import BaseEncoder


def fix_overlaps(notes_sorted):
    overlap_tracking = {}  # pitch: i
    for i, note in enumerate(notes_sorted):
        start, end, pitch, vel = note.start, note.end, note.pitch, note.velocity
        if overlap_tracking.get(pitch) is not None:
            i_prev = overlap_tracking[pitch]
            if start < notes_sorted[i_prev].end:
                # do NOTE_OFF earler
                notes_sorted[i_prev].end = start
        overlap_tracking[pitch] = i
    
    # del redundant notes
    for i in range(len(notes_sorted)-1, -1, -1):
        note = notes_sorted[i]
        if note.end <= note.start:
            del notes_sorted[i]

class Midi2NumpyRelEncoder(BaseEncoder):
    def __init__(self, quantize=64, min_notes_for_inst=10, min_bars=4):
        """quantize=64, min_notes_for_inst=10, min_bars=4"""
        # not implemented:
        # control_changes (including sustain pedal)
        # pitch_bends
        self.min_notes_for_inst = min_notes_for_inst
        self.min_bars = min_bars
        self.quantize = quantize
        
    def _load_midi(self, file):
        with warnings.catch_warnings(record=True) as w:
            midi = pretty_midi.PrettyMIDI(file)
            w = list(filter(lambda x: 'Tempo, Key or Time' in x.message.args[0], w))
            is_corrupted = bool(w)
            return midi, is_corrupted
        
    def _get_notes(self, midi):
        mapp = collections.defaultdict(list)
        for inst in midi.instruments:
            ch = -1 if inst.is_drum else inst.program
            mapp[ch].append(inst)

        for k in mapp:
            inst = mapp[k][0]
            for inst2 in mapp[k][1:]:
                inst.notes += inst2.notes
            inst.notes = sorted(inst.notes, key=lambda x: x.start)
            fix_overlaps(inst.notes)
            mapp[k] = inst

        db = midi.get_downbeats()
        tc = midi.get_tempo_changes()
        ts = np.array([x.time for x in midi.time_signature_changes]), midi.time_signature_changes
        if not len(ts[1]):
            ts = np.array([0.]), [pretty_midi.TimeSignature(4,4,0.)]
        
        bar_info = []
        cur_tc_i = 0
        cur_ts_i = 0
        for db_ in db:
            while cur_tc_i+1 < len(tc[0]) and db_+1e-4 >= tc[0][cur_tc_i+1]:
                cur_tc_i += 1
            while cur_ts_i+1 < len(ts[0]) and db_+1e-4 >= ts[0][cur_ts_i+1]:
                cur_ts_i += 1
            x = ts[1][cur_ts_i]
            bar_info.append([tc[1][cur_tc_i], (x.numerator, x.denominator)])
                                               
        notes_multi = collections.defaultdict(list)
        div = midi.resolution*4/self.quantize
        for channel,inst in mapp.items():
            cur_db_i = 0
            notes_multi[channel].append([])
            for note in inst.notes:
                s = note.start
                while cur_db_i+1 < len(db) and (s >= db[cur_db_i+1] or np.isclose(s, db[cur_db_i+1])):
                    cur_db_i += 1
                    notes_multi[channel].append([])
                cur_db = midi.time_to_tick(db[cur_db_i])
                
                start_tick = midi.time_to_tick(note.start)-cur_db
                end_tick = midi.time_to_tick(note.end)-cur_db
                notes_multi[channel][cur_db_i].append([int(start_tick/div), int(end_tick/div), note.pitch, note.velocity])
#             notes_multi[channel] = np.array(notes_multi[channel], dtype=self.dtype)
        
        notes = list(zip_longest(*notes_multi.values(), fillvalue=[]))
        return notes, list(notes_multi.keys()), bar_info
        
    def encode(self, file):
        try:
            midi, is_corrupted = self._load_midi(file)
        except Exception as e:
            return None
        midi.instruments = [inst for inst in midi.instruments if len(inst.notes) >= self.min_notes_for_inst]
#         tempo_changes = midi.get_tempo_changes()
#         time_signatures = [[x.numerator, x.denominator, x.time] for x in midi.time_signature_changes]
        resolution = midi.resolution
        notes, programs, bar_info = self._get_notes(midi)
        if len(notes) < self.min_bars:
            return None
        out = dict(zip('notes,programs,bar_info,resolution'.split(','),
                       [notes,programs,bar_info,resolution]))
        return out
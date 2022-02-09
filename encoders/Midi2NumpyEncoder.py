import numpy as np
import warnings
import pretty_midi

from .BaseEncoder import BaseEncoder

class Midi2NumpyEncoder(BaseEncoder):
    def __init__(self, dtype=np.float32):
        # not implemented:
        # control_changes (including sustain pedal)
        # pitch_bends
        
        self.dtype = dtype
        self.auto_collate = False
        
    def _load_midi(self, file):
        with warnings.catch_warnings(record=True) as w:
            midi = pretty_midi.PrettyMIDI(file)
            w = list(filter(lambda x: 'Tempo, Key or Time' in x.message.args[0], w))
            is_corrupted = bool(w)
            return midi, is_corrupted
        
    def encode(self, file):
        try:
            midi, is_corrupted = self._load_midi(file)
        except Exception as e:
            return None
        midi.instruments = [inst for inst in midi.instruments if len(inst.notes) >= 5]
        notes = [np.array([[x.start,x.end,x.pitch,x.velocity] for x in inst.notes], dtype=self.dtype) for inst in midi.instruments]
        if not notes:
            return None
        _info = [[inst.program if not inst.is_drum else -1, inst.name] for inst in midi.instruments]
        track_programs, track_names = list(map(list, zip(*_info)))
        tempo_changes = midi.get_tempo_changes()
        time_signatures = [[x.numerator, x.denominator, x.time] for x in midi.time_signature_changes]
        key_signatures = [[x.key_number, x.time] for x in midi.key_signature_changes]
        beats = midi.get_beats()
        downbeats = midi.get_downbeats()
        resolution = midi.resolution
        
        out = dict(zip('notes,track_programs,track_names,tempo_changes,time_signatures,key_signatures,beats,downbeats,resolution,is_corrupted'.split(','),
                       [notes,track_programs,track_names,tempo_changes,time_signatures,key_signatures,beats,downbeats,resolution,is_corrupted]))
        return out
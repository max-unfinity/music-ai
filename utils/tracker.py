import re
import torch
import numpy as np
import warnings

class Tracker:
    def __init__(self, local_vars):
        self.local_vars = re.findall('[a-zA-Z0-9_]+', local_vars)
        [self.__dict__.update({x:[]}) for x in self.local_vars]
    
    def collect(self, context_vars):
        for k in self.local_vars:
            if k in context_vars:
                self.__dict__[k] += [self._format_value(context_vars[k])]
            else:
                warnings.warn(f'[Tracker] can\'t get "{k}", but it is on tracking')

    def __iter__(self):
        for k in self.local_vars:
            yield k, self.__dict__[k]
            
    def get_dict(self):
        return {k: self.__dict__[k] for k in self.local_vars}
    
    def get_last(self):
        return {k: self.__dict__[k][-1] if self.__dict__[k] else None for k in self.local_vars}
        
    def _format_value(self, x):
        if isinstance(x, torch.Tensor):
            if x.numel() == 1:
                return x.item()
            else:
                return x.tolist()
        elif isinstance(x, np.ndarray):
            if x.size == 1:
                return x.item(0)
            else:
                return x.tolist()
        return x
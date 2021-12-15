import torch
import inspect


class DataInfo:
    def __init__(self, **kw):
        for k,v in kw.items():
            self.__dict__[k] = v


def assign_args(l):
    it = iter(l.items())
    self = next(it)[1]
    self.__dict__.update(it)
    self._init_params = self.__dict__.copy()
    

def assigner(f):
    def wrapper(*args, **kwargs):
        sign = inspect.getfullargspec(f)
        assert not sign.varargs
        names = sign.args
        defaults = sign.defaults

        d = dict(zip(names, args))
        names = names[len(args):]
        for k,v in kwargs.items():
            d[k] = v
            if k in names: names.remove(k)
        if defaults:
            for k,v in zip(names, defaults):
                d[k] = v
        assign_args(d)
        f(*args, **kwargs)
    return wrapper


class BaseEncoder:
    def __init_subclass__(self):
        self.__init__ = assigner(self.__init__)
    
    def save(self, f=None):
        d = {'init_params': self._init_params}
        if hasattr(self, 'data_info') and self.data_info:
            d['data_info'] = self.data_info.__dict__
            if not f:
                f = self.data_info.out_dir+'/../encoder.pt'
        torch.save(d, f)
        
    @classmethod
    def load(cls, dataset_name, name='', main_dir='.', f=None):
        if not f:
            f = f"{main_dir}/data/{dataset_name}_{cls.__name__}{'_'+name if name else ''}/encoder.pt"
        d = torch.load(f)
        e = cls(**d['init_params'])
        if d.get('data_info'):
            e.data_info = DataInfo(**d['data_info'])
        return e
    
    def collate(self, x):
        return x
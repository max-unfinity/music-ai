import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(inp, out), nn.ReLU(), nn.BatchNorm1d(out))
        
    def forward(self, x):
        return self.seq(x)
    
class ResBlock(nn.Module):
    def __init__(self, inp, hid):
        super().__init__()
        self.seq1 = nn.Sequential(nn.Linear(inp, hid), nn.ReLU(), nn.BatchNorm1d(hid))
        self.seq2 = nn.Sequential(nn.Linear(hid, inp), nn.ReLU(), nn.BatchNorm1d(inp))
        
    def forward(self, x):
        x1 = self.seq1(x)
        x2 = self.seq2(x1)
        return x+x2

class SoftQuantize(nn.Module):
    def __init__(self, n_embed, embedding_dim):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_embed = n_embed
        self.temperature = 1.0
        self.embed = nn.Embedding(n_embed, embedding_dim)

    def forward(self, logits):
        # B, N_EMBED
        temp = self.temperature

        p = torch.softmax(logits, 1)
        entropy = (p * torch.log(p*p.shape[1] + 1e-10)).sum(1).mean()
        
        p = torch.softmax(logits/temp, 1)
        z = torch.einsum('bn,nd -> bd', p, self.embed.weight)

        return z, entropy

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(Block(128,256), ResBlock(256,256*2), Block(256,512), ResBlock(512,512*2), ResBlock(512,512*2), nn.Linear(512,512))
        self.quantize = SoftQuantize(512, 512)
        self.decoder = nn.Sequential(nn.Linear(512, 128))
        self.out_act = nn.Identity()

    def forward(self, x):
        # B,1,128,256
        B,_,X,Y = x.shape
        x = x.squeeze(1).permute(0,2,1).reshape(B*Y,128)
        x = self.encoder(x)
        x, vq_loss = self.quantize(x)
        x = self.decoder(x)
        x = self.out_act(x)
        x = x.view(B,Y,128).permute(0,2,1).unsqueeze(1).contiguous()
        return x, vq_loss
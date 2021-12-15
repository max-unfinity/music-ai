import os, sys, shutil
import json
import itertools
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel

from encoders.NoteEventsEncoder import *
from utils.inverse_power_with_warmup import InversePowerWithWarmupLRScheduler


params = dict(
    SEED = 0,
    NAME = 'youtube_23k_NoteEventsEncoder_v2',
    DS_DIR = 'data/youtube_23k_final_NoteEventsEncoder/',
    num_epochs = 300,
    batch_size = 48,
    num_workers = 3,
    val_every = 1000,
    save_every = 1000,
    lr = 1e-4,
    use_scheduler = True,
    peak_lr = 2e-4,
    warmup_steps = 3000,
    power = 1,
    shift = 53000,
    LOG_TOTAL_NORM = True,
    LOG_ALL_GRADS = False,
    CLIPPING = False,
    gpus = [0],
    DDP = False,
    save_optimizer = False
)

globals().update(params)


class DummyWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.module = model
        
    def forward(self, *a, **kw):
        return self.module(*a, **kw)


def create_dataloaders(batch_size, num_workers=0):
    print('loading data...')

    train_dataset = NoteEventsDataset(DS_DIR+'ds_files.pt', prefix_path='', transform=None)
    val_dataset = NoteEventsDataset(DS_DIR+'ds_files.pt', prefix_path='', transform=None)
    
    np.random.seed(0)
    idxs = np.random.permutation(len(train_dataset))
    vl, tr = np.split(idxs, [4000])
    
    train_dataset = Subset(train_dataset, tr)
    val_dataset = Subset(val_dataset, vl)
    
    sampler = DistributedSampler(train_dataset, world_size, rank, True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, pin_memory=False, num_workers=num_workers)
    sampler = DistributedSampler(val_dataset, world_size, rank, False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size*5, sampler=sampler, pin_memory=False, num_workers=num_workers)
    
    return train_loader, val_loader

def init_model(lr, seed=0):
    print('loading model...')
    from structures.music_transformer.music_transformer import MusicTransformer
    torch.manual_seed(seed)    
    model_params = dict(
        d_model = 768,
        d_embed = 320,
        dim_feedforward = 2048,
        n_layers = 6,
        num_heads = 8,
        ACT = nn.ReLU,
        structure_act = 'gelu',
        dropout = 0.1,
        max_len = 1024,
        variant = 1,
    )
    locals().update(model_params)
    params['config'] = model_params
    embed_dims = encoder.get_embed_dims()
    
    embedding = eval('NoteEventsEmbedding(embed_dims, d_model=d_model, d_embed=d_embed, ACT=ACT, dropout=dropout, max_len=max_len)')
    structure = eval('MusicTransformer(device, d_model=d_model, dim_feedforward=dim_feedforward, n_layers=n_layers, num_heads=num_heads, dropout=dropout, max_sequence=max_len, activation=structure_act)')
    head = eval('NoteEventsHead(embed_dims, variant=variant, d_model=d_model, d_embed=d_embed, ACT=ACT)')
    model = NoteEventsModel(embedding, structure, head, PAD_TOKEN).to(device)
    
    if ddp:
        model = DistributedDataParallel(model, device_ids=[rank])
    else:
        model = DummyWrapper(model)
    print(sum((torch.numel(x) for x in model.parameters()))/1e6, 'M parameters')
    optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-5)
    return model, optimizer

def validate(model, val_loader):
    n, CE, ACC = 0, 0, 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_loader):
            x, tgt, genre, idx = process_batch(batch, device)

            logits = model(x)

            losses = []
            preds = []
            for l,t in zip(logits, tgt):
                losses.append(F.cross_entropy(l.view(-1, l.shape[-1]), t.flatten(), ignore_index=PAD_TOKEN, reduction='sum'))
                preds += [l.argmax(-1)]
            loss = torch.stack(losses).sum()
            pred = torch.stack(preds)
            
            mask = tgt != PAD_TOKEN
            n += mask.sum().item()
            CE += loss.item()
            ACC += (pred[mask] == tgt[mask]).sum().item()
            
    model.train()
    return CE/n, ACC/n


def train_distributed(rank_, world_size_, ddp_=True):
    global device, NAME, SEED, rank, world_size, ddp, encoder, PAD_TOKEN
    rank, world_size, ddp = rank_, world_size_, ddp_
    
    if ddp:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        torch.distributed.init_process_group('nccl', rank=rank, world_size=world_size)
    
    device = torch.device(f'cuda:{rank}')
    print(device, rank)
    
    with open(DS_DIR+'encoder.json', 'r') as f:
        d = json.load(f)
        d.pop('meta')
    encoder = NoteEventsEncoder(**d)
    PAD_TOKEN = encoder.special_tokens['PAD']
    
    model, optimizer = init_model(lr, SEED)
    
    if use_scheduler:
        scheduler = InversePowerWithWarmupLRScheduler(optimizer, peak_lr=peak_lr, warmup_steps=warmup_steps, power=power, shift=shift)
    
    if rank == 0:
        save_dir = f'output/{NAME}'
        save_name = f'{NAME}'
        if os.path.exists(save_dir):
            print(f'WARNING: {save_dir} exists! It may rewrite useful files.\nClean this dir? [1/0]')
            if int(input()):
                shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        writer = SummaryWriter(f'output/{NAME}/tensorboard')


    # TRAIN
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    LS = {'loss':[], 'lr':[], 'val_ce':[], 'val_acc':[], 'grads':defaultdict(lambda:[]), 'total_grad':[]}

    i_step = -1
    best_ce = float('inf')
    patience = 0
    ep = -1
    loss_w = torch.tensor([1.0]*5).to(device)
    

    cmd = ''
    while cmd != '/stop':
        try:
            train_loader, val_loader = create_dataloaders(batch_size, num_workers)
            for _ in range(num_epochs):
                ep += 1
                model.train()
                train_loader.sampler.set_epoch(ep)
                if rank == 0:
                    bar = tqdm(train_loader, position=rank)
                else:
                    bar = train_loader
                for batch in bar:
                    i_step += 1
                    x, tgt, genre, idx = process_batch(batch, device)

                    logits = model(x)

                    losses = []
                    ls_names = ['pitch','octave','length','velocity','time_shift']
                    for l,t in zip(logits, tgt):
                        losses.append(F.cross_entropy(l.view(-1, l.shape[-1]), t.flatten(), ignore_index=PAD_TOKEN))
                    loss = (torch.stack(losses)*loss_w).mean()

                    optimizer.zero_grad()
                    loss.backward()
                    if CLIPPING:
                        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIPPING).item()
                    else:
                        total_norm = 0
                    optimizer.step()

                    if use_scheduler:
                        scheduler.step()

                    if rank == 0:
                        # LOGS
                        LS['loss'] += [loss.item()]
                        LS['lr'] += [optimizer.param_groups[0]['lr']]
                        if LOG_ALL_GRADS:
                            [LS['grads'][k].append(torch.norm(v.grad).item()) for k,v in model.named_parameters() if v.requires_grad]
                        pairs = {n:l.item() for l,n in zip(losses,ls_names)}
                        writer.add_scalars(f'Train/loss_all', pairs, i_step)
                        writer.add_scalar(f'Train/loss', loss.item(), i_step)
                        writer.add_scalar(f'Train/lr', optimizer.param_groups[0]['lr'], i_step)
                        writer.add_scalar(f'TrainNorms/embedding_weight_norm', np.mean([torch.norm(x.weight).item() for x in model.module.embedding.embeddings]), i_step)
                        writer.add_scalar(f'TrainNorms/embedding_grad_norm', np.mean([torch.norm(x.weight.grad).item() for x in model.module.embedding.embeddings]), i_step)
                        writer.add_scalar(f'TrainNorms/output_weight_norm', np.mean([torch.norm(x.weight).item() for x in model.module.head.out_proj3]), i_step)
                        writer.add_scalar(f'TrainNorms/output_grad_norm', np.mean([torch.norm(x.weight.grad).item() for x in model.module.head.out_proj3]), i_step)
                        if LOG_TOTAL_NORM:
                            total_norm = 0.
                            for p in model.parameters():
                                if not p.requires_grad:
                                    continue
                                param_norm = p.grad.detach().data.norm(2)
                                total_norm += param_norm.item() ** 2
                            total_norm = total_norm ** 0.5
                            LS['total_grad'].append(total_norm)
                            writer.add_scalar(f'TrainNorms/total_grad_norm', total_norm, i_step)
                        bar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'], norm=total_norm, ep=ep, step=i_step)

                    if i_step % val_every == val_every-1:
                        val_ce, val_acc = validate(model, val_loader)
                        if world_size > 1 and ddp:
                            ce_all, acc_all = [[torch.zeros(1,device=device) for i in range(world_size)] for _ in range(2)]
                            [torch.distributed.all_gather(a, torch.tensor(x, dtype=torch.float32, device=device)) for a,x in zip([ce_all,acc_all], [val_ce,val_acc])]
                            val_ce, val_acc = [torch.cat(a).mean().item() for a in [ce_all,acc_all]]
                        if rank == 0:
                            # VAL LOGS
                            LS['val_ce'] += [val_ce]
                            LS['val_acc'] += [val_acc]
                            writer.add_scalar(f'Val/ce', val_ce, i_step+1)
                            writer.add_scalar(f'Val/acc', val_acc, i_step+1)
                            if val_ce < best_ce:
                                patience = 0
                                best_ce = val_ce
                            else:
                                patience += 1
                            print(f'{ep}: val_ce={val_ce}, val_acc={val_acc}, patience={patience}')

                    # CHECKPOINT
                    if (i_step % save_every == save_every-1) and rank == 0:
                        LS['grads'] = dict(LS['grads'])
                        torch.save({'history':LS,'epoch':ep,'params':params}, f'{save_dir}/hist_{save_name}.pt')
                        torch.save(model.module.state_dict(), f'{save_dir}/model_{save_name}_{(i_step+1)//1000}k.pt')
                        if save_optimizer:
                            torch.save(optimizer.state_dict(), f'{save_dir}/optimizer_{save_name}_{(i_step+1)//1000}k.pt')
        except KeyboardInterrupt:
            print('EXEC MODE: type /run for continue training, /stop for exit')
            while True:
                cmd = input()
                if cmd in ['/run','/stop']:
                    break
                try:
                    exec(cmd)
                except Exception as e:
                    print(e)
                

if __name__ == "__main__":
    print(NAME, SEED)
    world_size = torch.cuda.device_count()
    if DDP:
        torch.multiprocessing.spawn(train_distributed, args=(world_size,), nprocs=world_size, join=True)
    else:
        train_distributed(0, 1, ddp_=False)

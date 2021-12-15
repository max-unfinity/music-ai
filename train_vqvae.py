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

from encoders.Midi2NumpyEncoder import Midi2NumpyEncoder
from encoders.ImgEncoder import *
from utils.schedulers import InversePowerWithWarmupLRScheduler


params = dict(
    SEED = 0,
    NAME = 'lakh_4t_19k_gumbel_vq_test',
#     DS_DIR = 'data/youtube_23k_final_CNNEncoder/',
    num_epochs = 300,
    batch_size = 16,
    num_workers = 3,
    val_every = 2000,
    save_every = 2000,
    lr = 1e-4,
    use_scheduler = False,
    peak_lr = 1e-4,
    warmup_steps = 1500,
    power = 1,
    shift = 51500,
    LOG_TOTAL_NORM = True,
    LOG_ALL_GRADS = False,
    CLIPPING = False,
    gpus = [0],
    DDP = False,
    save_optimizer = False,
)

globals().update(params)
continue_params = dict(
    continue_train = False,
    model_checkpoint = f'output/{NAME}/model_{NAME}_175k.pt',
    optimizer_checkpoint = f'output/{NAME}/optimizer_{NAME}_175k.pt',
    i_step = 175513,
    ep = 11,
)
params['continue_params'] = continue_params

class DummyWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.module = model
        
    def forward(self, *a, **kw):
        return self.module(*a, **kw)


def create_dataloaders(batch_size, num_workers=0):
    print('loading data...')

    train_dataset = CustomDataset(encoder)
    val_dataset = CustomDataset(encoder)
    
    np.random.seed(0)
    idxs = np.random.permutation(len(train_dataset))
    vl, tr = np.split(idxs, [2000])
    
    train_dataset = Subset(train_dataset, tr)
    val_dataset = Subset(val_dataset, vl)
    
    sampler = DistributedSampler(train_dataset, world_size, rank, True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, pin_memory=False, num_workers=num_workers)
    sampler = DistributedSampler(val_dataset, world_size, rank, False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size*5, sampler=sampler, pin_memory=False, num_workers=num_workers)
    
    return train_loader, val_loader

def init_model(lr, seed=0):
    print('loading model...')
    from structures.vq_vae_taming import GumbelVQ, SoftQuantize
    from utils.schedulers import LambdaWarmUpCosineScheduler
    
    torch.manual_seed(seed)
    
    ddconfig = {'double_z': False, 'z_channels': 256, 'resolution': 256, 'in_channels': 1, 'out_ch': 1, 'ch': 128, 'ch_mult': [1, 2, 4], 'num_res_blocks': 2, 'attn_resolutions': [16], 'dropout': 0.0}
    model_params = dict(
        ddconfig = ddconfig,
        n_embed = 512,
        embed_dim = 256,
        kl_weight = 1e-6,
    )

    model = GumbelVQ(**model_params).to(device)
#     model.quantize = SoftQuantize(512, 512).to(device)
    model.out_act = nn.Identity()
    
    locals().update(model_params)
    params['config'] = model_params
    
    temp_scheduler = LambdaWarmUpCosineScheduler(0, 1e-6, 0.9, 0.9, 300000)    

    print(sum((torch.numel(x) for x in model.parameters()))/1e6, 'M parameters')
    optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-5)
    
    # continue
    if continue_params['continue_train']:
        print(f'continue from {continue_params["model_checkpoint"]}')
        model.load_state_dict(torch.load(continue_params['model_checkpoint'], map_location=device))
        if continue_params.get('optimizer_checkpoint'):
            optimizer.load_state_dict(torch.load(continue_params['optimizer_checkpoint'], map_location=device))
    
    if ddp:
        model = DistributedDataParallel(model, device_ids=[rank])
    else:
        model = DummyWrapper(model)
        
    return model, optimizer, temp_scheduler


def validate(model, val_loader):
    n, CE, ACC = 0, 0, 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_loader):
            x0, mask = process_batch(batch, device)
            x = x0
            rec, vq_loss = model(x)
            loss = F.mse_loss(rec[mask], x[mask], reduction='sum')
            
            n += mask.sum().item()
            CE += loss.item()
            
    model.train()
    print(metrics(x.detach().cpu().numpy(), rec.detach().cpu().numpy()))
    return CE/n, 0


def train_distributed(rank_, world_size_, ddp_=True):
    global device, NAME, SEED, rank, world_size, ddp, encoder, PAD_TOKEN
    rank, world_size, ddp = rank_, world_size_, ddp_
    
    if ddp:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        torch.distributed.init_process_group('nccl', rank=rank, world_size=world_size)
    
    device = torch.device(f'cuda:{rank}')
    print(device, rank)
    
    encoder = ImgEncoderSingle()
    
    model, optimizer, temp_scheduler = init_model(lr, SEED)
    
    if use_scheduler:
        scheduler = InversePowerWithWarmupLRScheduler(optimizer, peak_lr=peak_lr, warmup_steps=warmup_steps, power=power, shift=shift)
    
    if rank == 0:
        save_dir = f'output/{NAME}'
        save_name = f'{NAME}'
        if os.path.exists(save_dir):
            print(f'WARNING: {save_dir} exists! It may rewrite useful files.\nClean this dir? [1/0]')
            if int(input()):
                shutil.rmtree(save_dir, True)
        os.makedirs(save_dir, exist_ok=True)
        writer = SummaryWriter(f'output/{NAME}/tensorboard')


    # TRAIN
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    LS = {'rec_loss':[], 'lr':[], 'val_rec_loss':[], 'val_acc':[], 'grads':defaultdict(lambda:[]), 'total_grad':[]}

    i_step = -1# if continue_params.get('i_step') is None else continue_params['i_step']-1
    best_ce = float('inf')
    patience = 0
    ep = -1 #if continue_params.get('ep') is None else continue_params['ep']-1
    
    rec_w = 1.0
    assert hasattr(model.module.quantize, 'temperature')
    
    

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
                    model.module.quantize.temperature = temp_scheduler(i_step)
                    x0, mask = process_batch(batch, device)
                    x = x0 + (torch.rand_like(x0)-0.5)*0.05
                    
                    rec, vq_loss = model(x+(torch.rand_like(x)-0.5)*0.05)
                    rec_loss = F.mse_loss(rec[mask], x[mask])
#                     k = torch.topk(torch.abs(rec[~mask]), 1024*rec.shape[0])[0]
                    k = rec[~mask]
                    pad_loss = F.mse_loss(k, torch.zeros_like(k)*torch.randn_like(k)*0.05)
                    loss = rec_w*rec_loss + pad_loss*0.1 
                    loss1 = loss + vq_loss*0.01

                    loss = loss1 #+ loss2

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
                        LS['rec_loss'] += [rec_loss.item()]
                        LS['lr'] += [optimizer.param_groups[0]['lr']]
                        if LOG_ALL_GRADS:
                            [LS['grads'][k].append(torch.norm(v.grad).item()) for k,v in model.named_parameters() if v.requires_grad]
                        writer.add_scalar(f'Train/loss', loss.item(), i_step)
                        writer.add_scalar(f'Train/rec_loss', rec_loss.item(), i_step)
                        writer.add_scalar(f'Train/pad_loss', pad_loss.item(), i_step)
                        writer.add_scalar(f'Train/vq_loss', vq_loss.item(), i_step)
                        
                        writer.add_scalar(f'Train/vq_temp', model.module.quantize.temperature, i_step)
                        writer.add_scalar(f'Train/lr', optimizer.param_groups[0]['lr'], i_step)
                        
                        writer.add_scalar(f'TrainNorms/embedding_weight_norm', torch.norm(model.module.quantize.embed.weight).item(), i_step)
                        writer.add_scalar(f'TrainNorms/embedding_grad_norm', torch.norm(model.module.quantize.embed.weight.grad).item(), i_step)
                        writer.add_scalar(f'TrainNorms/output_weight_norm', torch.norm(model.module.decoder.conv_out.weight).item(), i_step)
                        writer.add_scalar(f'TrainNorms/output_grad_norm', torch.norm(model.module.decoder.conv_out.weight.grad).item(), i_step)
                        writer.add_scalar(f'TrainNorms/input_weight_norm', torch.norm(model.module.encoder.conv_in.weight).item(), i_step)
                        writer.add_scalar(f'TrainNorms/input_grad_norm', torch.norm(model.module.encoder.conv_in.weight.grad).item(), i_step)
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
                        bar.set_postfix(loss=loss.item(), rec_loss=rec_loss.item(), lr=optimizer.param_groups[0]['lr'], norm=total_norm, ep=ep, step=i_step, avq_temp=model.module.quantize.temperature, rec_w=rec_w)

                    if i_step % val_every == val_every-1:
                        writer.add_image('rec', rec[0].detach().cpu().numpy(), i_step+1)
                        val_ce, val_acc = validate(model, val_loader)
                        if world_size > 1 and ddp:
                            ce_all, acc_all = [[torch.zeros(1,device=device) for i in range(world_size)] for _ in range(2)]
                            [torch.distributed.all_gather(a, torch.tensor(x, dtype=torch.float32, device=device)) for a,x in zip([ce_all,acc_all], [val_ce,val_acc])]
                            val_ce, val_acc = [torch.cat(a).mean().item() for a in [ce_all,acc_all]]
                        if rank == 0:
                            # VAL LOGS
                            LS['val_rec_loss'] += [val_ce]
#                             LS['val_acc'] += [val_acc]
                            writer.add_scalar(f'Val/rec_loss', val_ce, i_step+1)
#                             writer.add_scalar(f'Val/acc', val_acc, i_step+1)
                            if val_ce < best_ce:
                                patience = 0
                                best_ce = val_ce
                            else:
                                patience += 1
                            print(f'{ep}: val_rec_loss={val_ce}, patience={patience}')

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

#!/usr/bin/env python3
#
# Solving Statistical Mechanics using Variational Autoregressive Networks
# 2d classical Ising model

import time

import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
torch.manual_seed(12) 
from numpy import sqrt
from torch import nn

from args_ import args
from bernoulli import BernoulliMixture
from made_ import MADE
from pixelcnn import PixelCNN
from utils import (
    clear_checkpoint,
    clear_log,
    get_last_checkpoint_step,
    ignore_param,
    init_out_dir,
    my_log,
    print_args,
)


def main():
    
    
    N = args.N
    M = args.M
    K = args.K 
    p_prior = args.p_prior
    p = args.p
    
    
    num_samples = 10
    num_codes = 1       
    
    
    Ns = str(N)
    Ms = str(M)
    Ks = str(K) 
    pps = str(p_prior).replace('.', '')
    ps = str(p).replace('.', '')
       
    
    path = '/home/rodsveiga/Dropbox/DOC/van_error_codes/runs_files/'
    
    message = torch.load(path + 'message_N_%s_M_%s_K_%s_p_%s_p_prior_%s.pt' % (Ns, Ms, Ks,ps, pps))
    
    
    ### User chooses what type o NN will provide the parametrization of 
    ### the autoregressive neural network
    if args.net == 'made':
        net = MADE(**vars(args))
    elif args.net == 'pixelcnn':
        net = PixelCNN(**vars(args))
    elif args.net == 'bernoulli':
        net = BernoulliMixture(**vars(args))
    else:
        raise ValueError('Unknown net: {}'.format(args.net))
        
    
    for k in range(num_codes):
        
        ks = str(k)
                        
        PATH = path + 'model_N_%s_M_%s_K_%s_p_%s_p_prior_%s_code_%s.pt' % (Ns, Ms, Ks,ps, pps, ks)
        
                
        net = MADE(**vars(args))
        net.load_state_dict(torch.load(PATH))

    
        print('Running code %d' % k)
    
        for j in range(num_samples):
            
            print('j= ', j)
        
               
            sample, _ = net.sample(args.num_messages)  
            
            overlap = (sample == message).sum().item() / (message.shape[0]*message.shape[1])
            
            print('sample= %d -- overlap= %.5f' % (j, overlap))

    
    
if __name__ == '__main__':
    main()

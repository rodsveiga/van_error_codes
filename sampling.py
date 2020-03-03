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
    
    
    num_samples = 15
    num_codes = 1
    code = str(0)

    # String
    ps = '{:.3f}'.format(p).replace('.', '')
    pps = '{:.3f}'.format(p_prior).replace('.', '')
    Ns = str(N)
    Ms = str(M)
    Ks = str(K)
    steps = str(args.max_step)
    
        
    # Select path for retrieving model and messages
    if args.laptop:
        path_ = '/home/rodrigo/Dropbox/DOC/van_error_codes/models/'
    else:
        path_ = '/home/rodsveiga/Dropbox/DOC/van_error_codes/models/'

    # Path for given choices of N, M, K, and so on

    path__ = path_ + '_N_%s_M_%s_K_%s_p_prior_%s_steps_%s_code_%s/' % (Ns, Ms, Ks, pps, steps, code)

 
    ##############

    path = path__ + 'model_N_%s_M_%s_K_%s_p_%s_p_prior_%s_steps_%s/' % (Ns, Ms, Ks, ps, pps, steps)




    
    
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
                        
        PATH = path + 'model_N_%s_M_%s_K_%s_p_%s_p_prior_%s_code_%s_step_%s.pt' % (Ns, Ms, Ks,ps, pps, ks, str(steps))
        
                
        net = MADE(**vars(args))
        net.load_state_dict(torch.load(PATH))

        sample_ = torch.zeros(message.shape)

    
        print('Running code %d' % k)
    
        for j in range(num_samples):
            
            #print('j= ', j)
                       
            sample, _ = net.sample(args.num_messages)

            #print('sample', sample)  
            
            overlap = (sample == message).sum().item() / (message.shape[0]*message.shape[1])
            
            print('sample= %d -- overlap= %.5f' % (j, overlap))

            #print('av_over_samples=', torch.sign(torch.sum(sample, dim=1)/100).sum().item())
            #print('av_over_mess=', torch.sign(torch.sum(sample, dim=1)/100).sum().item())

            sample_ = sample_ + sample

        print('-----Final sample', torch.sign(sample_))

        overlap_av = (torch.sign(sample_) == message).sum().item() / (message.shape[0]*message.shape[1])

        print('-----overplap_av=', overlap_av )

        torch.save(torch.sign(sample_), path + 'av_sample_N_%s_M_%s_K_%s_p_%s_p_prior_%s.pt' % (Ns, Ms, Ks,ps, pps))

            

    
    
if __name__ == '__main__':
    main()

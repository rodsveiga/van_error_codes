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

import ising
from args import args
from bernoulli import BernoulliMixture
from made import MADE
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
        
    #path = '/home/rodsveiga/stat-mech-van/src_ising/sourlas_test_loop/sample_message_beta_p_00_beta_10.pt'
    path = '/home/rodsveiga/stat-mech-van/src_ising/sourlas_test_loop/sample_message_beta_p_345_beta_10.pt'
    
    #path = '/home/rodsveiga/Dropbox/DOC/stat-mech-van/src_ising/sourlas_test_loop_NOV_21/sample_message_beta_p_00_beta_10.pt'
    
    message = torch.load(path)
    
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
        
    samp__cod = torch.zeros(message.shape)
    
    num_samples = 10
    num_codes = 1       
    
    for k in range(num_codes):
        
        
        PATH_ = '/home/rodsveiga/stat-mech-van/src_ising/sourlas_test_loop/saved_model_beta_p_00_beta_10_code_%s.pt' % str(k)
        PATH_2 = '/home/rodsveiga/stat-mech-van/src_ising/sourlas_test_loop/saved_model_beta_p_345_beta_10_code_%s.pt' % str(k)
        
                
        net = MADE(**vars(args))
        net.load_state_dict(torch.load(PATH_))
        
        net2 = MADE(**vars(args))
        net2.load_state_dict(torch.load(PATH_2))
        
        
     
        samp_ = torch.zeros(samp__cod.shape)
    
        print('Running code %d' % k)
    
        for j in range(num_samples):
            
            print('j= ', j)
        
               
            sample, _ = net.sample(args.batch_size)  
            #log_prob = net.log_prob(sample)
            #print('sample: ', sample)
            #print('sample_shape: ', sample.shape)
            #print('log_prob: ', log_prob)
            #print('log_prob_shape: ', log_prob.shape)
            
            sample = sample.view(sample.shape[0], sample.shape[2]*sample.shape[3])
            
            print('sample: ', sample)
            
                      
            
            #print(sample.shape[0])
            
            sample2, _ = net2.sample(args.batch_size)  
            sample2 = sample2.view(sample2.shape[0], sample2.shape[2]*sample2.shape[3])
            
            print('sample2: ', sample2)
            #print('sample2_shape: ', sample2.shape)
            
            
                        
            print('COMPARING THE SAMPLES:')
            print(sample == sample2)
            
            print('Over hole set:')          
            print( (sample == sample2).sum().item() / (sample.shape[0]*sample.shape[1]) )
            
            print('Over sample_0:')          
            print( (sample[0] == sample2[0]).sum().item() / sample[0].shape[0] )
            
            
            #print('overlap_1 =',  sample_.mean().item()  )
            
            #del sample
        
            samp_ = samp_ + (sample/num_samples)
            
            del sample
            
            #print(samp_[0])
            
        del net
                
        samp__cod = samp__cod + torch.sign(samp_)
        
        del samp_
        
        
        #####################
        
    #print(message)
    
       
    #print(samp__cod)
    
    #print(samp__cod[0])
    
    #print(torch.sign(samp__cod))
    
    #####################print('OUTPUT_0:')
    
    #####################print(torch.sign(samp__cod)[0])
    
    #####################print('boolean= ', torch.sign(samp__cod)[0] > 0)
    
    #####################print('MESSAGE_0: ')
    
    #####################print(message[0])
    
    #####################print('boolean= ', message[0] > 0 )
    
    #####################print('PERFORMANCE_0:')
    
    #####################print('pho_0= ', torch.dot(message[0], torch.sign(samp__cod)[0] ).item() / samp__cod.shape[1]  )
        
    #output = torch.sign(samp__cod)
    
    #####################rho_batch_ = torch.mul(message, torch.sign(samp__cod)).sum(dim=1) / samp__cod.shape[1] 
    
    #####################rho_batch = rho_batch_.mean()
        
    #print('output =', output)
    
    #####################print('PERFORMANCE - AVERAGE OVER THE SET OF MESSAGES:')
        
        
    #####################print('pho =', rho_batch.item()  )

    #del output    
    
        #overlap = torch.zeros(message.shape[0])
        
        #for i in range(message.shape[0]):
        #    overlap[i] = torch.dot(output, message[i]).item()
            
        #print('overlap = ', overlap)
            
        #print('overlap_ = ', overlap.mean().item())    
         
        #overlap_av = overlap.mean().item()/ message.shape[1]
        
        #print('overlap_av= ', overlap_av)
        
        
        #####################
        
        #torch.empty_cache()
    
    #rho_batch = torch.mul(message, torch.sign(samp__cod)).sum(dim=1)
    
    del samp__cod
    
    #print(rho_batch)
    #print(rho_batch.shape)
    
    #print('Average performance over the set of messages')
    #print(rho_batch.mean().item() / message.shape[1])
    
    #del rho_batch
      
    
        
        
        #out = torch.sign(sample_.mean(dim=1))
        
        #rho = torch.mul(message.t(), out).sum(dim=0).mean().item()
        
        #rho_.append(rho)
        
        #print('j = %d -- rho = %f' % (j, rho))
        
    #print('rho_av = %f' % np.array(rho_).mean())

    
    

   ################################################################################
   ##### Average over spins j dependent
   
        
        ### For each code
        #sample_ = sample.view(sample.shape[0], sample.shape[2]*sample.shape[3])
        #sign_tensor = sign_tensor + sample_
        
        ### 
        #print('Sample for the trained network for Code %d and beta %.3f' % (j, args.beta))
        #print(sample_)
        #print('--------------------------------------------')
        
    ## Results
    #print('Performance for a given set of messages:')
    #rho_batch = torch.mul(sample_in, torch.sign(sign_tensor)).sum(dim=0)
    #print(rho_batch / N)
    
    #print('Average performance over the set of messages')
    #print(rho_batch.mean().item() / N)
    
   ################################################################################
   ##### Average over spins j independent
        
        ### For each code
        

        
        
    ### Now the average over the codes
    
    #sign_tensor_A = torch.sign(sign_tensor)
        
    ## Results
    #print('Performance for a given set of messages:')
    #rho_batch = torch.mul(sample_in, sign_tensor_A).sum(dim=0)
    #print(rho_batch / N)
    
    #print('Average performance over the set of messages')
    #print(rho_batch.mean().item() / N)
    
    #out1 = rho_batch.mean().item() / N
    #out2 = args.beta
    
    #file = open('output_test.txt', 'a')
    #file.write(str(out1) + '  ' + str(out2) + '\n') 
    #file.close() 
        
    
    
    
    
if __name__ == '__main__':
    main()

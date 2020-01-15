# 2D classical Ising model

import torch
import numpy as np


def sourlas(sample, sample_in, C, p, p_prior):
    
    #sample_ = sample.view(sample.shape[0], sample.shape[2]*sample.shape[3])
          
    en_tensor = torch.empty([sample.shape[0]])
      
    #M = C.shape[0]
    
    beta_prior = 0.5*np.log( (1. - p_prior) / p_prior)
    beta_p =  0.5*np.log( (1. - p) / p)
    

    # Loop in j: over the hole batch
    
    en = 0
    
    for j in range(sample.shape[0]):     
        #en = 0
        #for k in range(M):
            
            #spinK = torch.prod(torch.take(sample_[j], C[k])).item()
            
        spinK = torch.take(sample[j], C).prod(dim=1)
        
        ###
        J0 = torch.take(sample_in[j], C).prod(dim=1)
        ###
            
        ###################################################################
            
        random = torch.rand(J0.shape)
                      
        for k in range(J0.shape[0]):
            
            if random[k] <= p:
                J0[k] = -J0[k]
                
        #####################################################################
         
        en = - beta_p*(torch.dot(J0, spinK).item()) - beta_prior*(sample[j].sum().item())
        ########################    

            #en = en - beta_p*spinK - beta_prior*sample_[j].sum().item()
            
        en_tensor[j] = en
        
    return en_tensor


            
    
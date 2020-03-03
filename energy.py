# LDPC: Sourlas code

import torch
import numpy as np


def sourlas(sample, J, C, beta_p, beta_prior, device):
    
         
    en_tensor = torch.empty([sample.shape[0]], 
                            device= device)
          
    #beta_prior = 0.5*np.log( (1. - p_prior) / p_prior)
    #beta_p =  0.5*np.log( (1. - p) / p)
    

    # Loop in j: over the hole set of messages
    
    en = 0
    
    for j in range(sample.shape[0]):     
            
        spinK = torch.take(sample[j], C).prod(dim=1)
        
        ###
        #J = torch.take(sample_in[j], C).prod(dim=1)
        ###
            
        ###################################################################

        ### Corrupted version of the message
        ### Observed that this is calculated at each iteration
        ### Is that correct?
            
        #random = torch.rand(J0.shape)
                      
        #for k in range(J0.shape[0]):
            
        #    if random[k] <= p:
        #        J0[k] = -J0[k]
                
        #####################################################################
         
        en = - beta_p*(torch.dot(J[j], spinK).item()) - beta_prior*(sample[j].sum().item())
        #en = - beta_p*(torch.dot(J0[j], spinK).item()) - beta_prior*(sample[j].sum().item())
        ########################    

            #en = en - beta_p*spinK - beta_prior*sample_[j].sum().item()
            
        en_tensor[j] = en
        
    return en_tensor


            
    
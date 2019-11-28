# 2D classical Ising model

import torch
import numpy as np


def energy(sample, ham, lattice, boundary):
    term = sample[:, :, 1:, :] * sample[:, :, :-1, :]
    term = term.sum(dim=(1, 2, 3))
    output = term
    term = sample[:, :, :, 1:] * sample[:, :, :, :-1]
    term = term.sum(dim=(1, 2, 3))
    output += term
    if lattice == 'tri':
        term = sample[:, :, 1:, 1:] * sample[:, :, :-1, :-1]
        term = term.sum(dim=(1, 2, 3))
        output += term

    if boundary == 'periodic':
        term = sample[:, :, 0, :] * sample[:, :, -1, :]
        term = term.sum(dim=(1, 2))
        output += term
        term = sample[:, :, :, 0] * sample[:, :, :, -1]
        term = term.sum(dim=(1, 2))
        output += term
        if lattice == 'tri':
            term = sample[:, :, 0, 1:] * sample[:, :, -1, :-1]
            term = term.sum(dim=(1, 2))
            output += term
            term = sample[:, :, 1:, 0] * sample[:, :, :-1, -1]
            term = term.sum(dim=(1, 2))
            output += term
            term = sample[:, :, 0, 0] * sample[:, :, -1, -1]
            term = term.sum(dim=1)
            output += term

    if ham == 'fm':
        output *= -1

    return output


def energy_sourlas(sample, sample_in, C, beta_p, p_prior):
    
    sample_ = sample.view(sample.shape[0], sample.shape[2]*sample.shape[3])
          
    en_tensor = torch.empty([sample_.shape[0]])
      
    #M = C.shape[0]
    
    beta_prior = 0.5*np.log( (1. - p_prior) / p_prior)
    
    p = 1. / (1. + np.exp(2*beta_p))
    
    # Loop in j: over the hole batch
    
    en = 0
    
    for j in range(sample_.shape[0]):     
        #en = 0
        #for k in range(M):
            
            #spinK = torch.prod(torch.take(sample_[j], C[k])).item()
            
        spinK = torch.take(sample_[j], C).prod(dim=1)
        
        ###
        J0 = torch.take(sample_in[j], C).prod(dim=1)
        ###
            
        ###################################################################
            
        random = torch.rand(J0.shape)
                      
        for k in range(J0.shape[0]):
            
            if random[k] <= p:
                J0[k] = -J0[k]
                
        #####################################################################
         
        en = - beta_p*(torch.dot(J0, spinK).item()) - beta_prior*(sample_[j].sum().item())
        ########################    

            #en = en - beta_p*spinK - beta_prior*sample_[j].sum().item()
            
        en_tensor[j] = en
        
    return en_tensor


###########################################################################################################
    
def energy_gallager(sample, C, gamma, p_noise, noise_in):
    
    sample_ = sample.view(sample.shape[0], sample.shape[2]*sample.shape[3])
          
    en_tensor = torch.empty([sample_.shape[0]])
        
    F = np.atanh( 1.0 - 2.0*p_noise )
       
    # Loop in j: over the hole batch
    
    en = 0
    
    for j in range(sample_.shape[0]):     
       
        
        J = torch.take(noise_in[j], C).prod(dim=1)
        
        tau_prod = torch.take(sample_[j], C).prod(dim=1)
            
        en = - gamma*( (torch.dot(J, tau_prod) - J.shape[0]).item() ) - F*(sample_[j].sum().item())
            
        en_tensor[j] = en
        
    return en_tensor
        
            
    
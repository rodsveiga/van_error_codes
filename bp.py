import torch
import numpy as np

def m_to_mhat(N, M, J_, m, beta, encoding):
    
    m_hat = torch.empty(M, N)
    
    for mu in range(M):
        for j in range(N):
                     
            # Keep only L(mu) which a are different of j
            index_no_j = torch.nonzero(encoding[mu] != j).squeeze()
            L_no_j = encoding[mu][index_no_j]
        
            # Message update        
            m_hat[mu, j] = torch.tanh( beta* J_[mu])*torch.take(m[mu], L_no_j).prod(dim=0)
            
    return m_hat


#####################################################################


def mhat_to_m(N, M, m_hat, beta_prior, encoding):
    
    m = torch.empty(M, N)
    
    for j in range(N):
        
        for mu in range(M):
            
            M_set = torch.where(encoding == j)[0]
        
            M_set_no_mu = M_set[torch.nonzero(M_set != mu).squeeze()]
            
            m[mu, j] = torch.tanh(torch.take(np.arctanh(m_hat[:, j]), M_set_no_mu).sum() + beta_prior)
            
    return m


#####################################################################


def m_bayes(N, m_hat_final, beta_prior, encoding):
    
    m_bayes = []

    for j in range(N):
           
        M_set = torch.where(encoding == j)[0]
  
        m_j = torch.tanh(torch.take(np.arctanh(m_hat_final[:, j]), M_set).sum() + beta_prior)
    
        m_bayes.append(m_j.item())
        
    return m_bayes

#####################################################################


def BP_LDPC(N, M, J_, beta, beta_prior, encoding, message, num_it, verbose):
     
    # Initializing
    m = torch.ones([M, N])*np.tanh(beta_prior)
    m_hat = torch.ones([M, N])*0.0
    
    for j in range(num_it):
        
        print('--it = %d' % j)

        m_hat = m_to_mhat(N, M, J_, m, beta, encoding)      
        m = mhat_to_m(N, M, m_hat, beta_prior, encoding)
                
        ## Monitoring performace
        if (j % verbose) == 0:
            m_ = m_bayes(N, m_hat, beta_prior, encoding)
            print('overlap_BP = ', torch.dot(message, torch.Tensor(np.sign(m_))).item() / N)
        
    m_final = m_bayes(N, m_hat, beta_prior, encoding)
    
    return np.sign(m_final)
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

import ising_
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
    
    N = 10*10
    R_inv = 2
    M = R_inv*N
    K = 4
    num_codes = 1  
    p_prior = 0.1
    

    
    start_time = time.time()
    
    
    

    #init_out_dir()
    if args.clear_checkpoint:
        clear_checkpoint()
    last_step = get_last_checkpoint_step()
    if last_step >= 0:
        my_log('\nCheckpoint found: {}\n'.format(last_step))
    else:
        clear_log()
    print_args()

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
    net.to(args.device)
    my_log('{}\n'.format(net))

    params = list(net.parameters())
    params = list(filter(lambda p: p.requires_grad, params))
    nparams = int(sum([np.prod(p.shape) for p in params]))
    my_log('Total number of trainable parameters: {}'.format(nparams))
    named_params = list(net.named_parameters())
    
    
    

    ### User chooses the optimizer 
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr)
    elif args.optimizer == 'sgdm':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(params, lr=args.lr, alpha=0.99)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999))
    elif args.optimizer == 'adam0.5':
        optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.5, 0.999))
    else:
        raise ValueError('Unknown optimizer: {}'.format(args.optimizer))

    if args.lr_schedule:
        # 0.92**80 ~ 1e-3
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.92, patience=100, threshold=1e-4, min_lr=1e-6)

    if last_step >= 0:
        state = torch.load('{}_save/{}.state'.format(args.out_filename,
                                                     last_step))
        ignore_param(state['net'], net)
        net.load_state_dict(state['net'])
        if state.get('optimizer'):
            optimizer.load_state_dict(state['optimizer'])
        if args.lr_schedule and state.get('scheduler'):
            scheduler.load_state_dict(state['scheduler'])

    init_time = time.time() - start_time
    my_log('init_time = {:.3f}'.format(init_time))


    print('-----------------------------')
    my_log('Training...')
    sample_time = 0
    train_time = 0
    start_time = time.time()
    
    ### Fix a message. To start we will considered this message sampled from 
    ### the autoregressive network
    sample_start_time = time.time()
    with torch.no_grad():
          ### VAN: samples from the autoregressive network
          ### Possible options for `net`
          ### MADE, PixelCNN and BernoulliMixture
          sample, x_hat = net.sample(args.num_messages)
    assert not sample.requires_grad
    assert not x_hat.requires_grad
    ################################
       
    ##sample_in = sample.view(sample.shape[0], sample.shape[2]*sample.shape[3])
    #sample_in = 2*torch.randint(0, 2, [args.num_messages, N], dtype=torch.float)-1.
    
    print('sample: ', sample)
    print('shape: ', sample.shape)
    
    
       
    
    ########################################################################
    ## With message prior
    random = torch.rand([args.num_messages, N])
    
    sample_in = torch.zeros([args.num_messages, N])
                    
    for j in range(random.shape[0]):
            
        for k in range(random.shape[1]):
                
            #### -1 with probability p_prior
            if random[j,k] <= p_prior:
                sample_in[j,k] = -1.
            #### +1 with probability 1 - p_prior
            else:
                sample_in[j,k] = +1.
                
    ########################################################################
    
    
    # Saving the initial message
    bp_s = str(args.beta_p).replace('.', '')
    bs = str(args.beta).replace('.', '')
    
    path = '/home/rodsveiga/stat-mech-van/src_ising/sourlas_test_loop/'
    
    
    ##torch.save(sample_in, path + 'sample_message_beta_p_%s_beta_%s.pt' % (bp_s, bs))
    
    # Generate the codes
    C_list = []
    for j in range(num_codes):
        C = torch.randint(0, N, [M, K])
        ##torch.save(C, path + 'codes_beta_p_%s_beta_%s_code_%d.pt' % (bp_s, bs, j))
        C_list.append(C)
        
      
    ### Given a temperature (args.beta) and a set of messages(sample_) we make
    ### a loop over codes
    
    ### Tensor to store the decoding 'signals'
    #sign_tensor = torch.zeros(sample_in.shape)
    #sign_tensor_ = torch.zeros(sample_in.shape[0])
    
    for j in range(num_codes):
        
        print('Code number= %d' % j)
        
        C = C_list[j]
        
        ### TensorBoard
        ##writer = SummaryWriter()
        
        
        #########################################################################
        ## Without Message prior temperature
        #random = torch.rand(M)
            
        #J = torch.zerostake(sample_in, C).prod(dim=1)
        #torch.prod(torch.take(sample_in[j], C[k])).item()
                    
        #for k in range(N*R_inv):
            
            #### Flip with probability p
        #    if random[k] <= p:
        #        J[k] = -J[k]
            
        #########################################################################
        
        #########################################################################
        ## With Message prior temperature (random J)
        #random = torch.rand(M)
                
        #J = torch.zeros(M)
                    
        #for k in range(M):
            
            #### -1 with probability 0.5
        #    if random[k] <= p_beta:
        #        J[k] = -1.
            #### +1 with probability 0.5
        #    else:
        #        J[k] = +1.
            
            
        ##################
        
        for step in range(last_step + 1, args.max_step + 1):
                                               
            ### The user can choose the optimizer
            optimizer.zero_grad()
                
            sample_time += time.time() - sample_start_time
        
            train_start_time = time.time()
            
    
            # Log-prob is calculated from `sample`
            log_prob = net.log_prob(sample)
            # 0.998**9000 ~ 1e-8
            beta = args.beta * (1 - args.beta_anneal**step)
                        
            with torch.no_grad():
                ### Only moment where the Hamiltonian is taken into account
                ### It is calculated for each `sample`
                ### The GOAL is to understand the sampling
                #energy = ising.energy(sample, args.ham, args.lattice,
                #                      args.boundary)
                
                ## What do I choose here? args.beta or beta?
                energy = ising_.energy_sourlas(sample, sample_in, C, args.beta_p, p_prior)
                
                ### Beta*FE is the loss function
                loss = log_prob + beta * energy
                        
            assert not energy.requires_grad
            assert not loss.requires_grad
            ### Average of Beta*FE: ~ Eq.(3)
            loss_reinforce = torch.mean((loss - loss.mean()) * log_prob)
            ### Gradient of Beta*FE: ~ Eq.(5)
            loss_reinforce.backward()
    
            if args.clip_grad:
                nn.utils.clip_grad_norm_(params, args.clip_grad)
    
            ### Update of the network weights
            optimizer.step()
    
            if args.lr_schedule:
                scheduler.step(loss.mean())
    
            train_time += time.time() - train_start_time
            
            
    
            with torch.no_grad():
                ### VAN: samples from the autoregressive network
                ### Possible options for `net`
                ### MADE, PixelCNN and BernoulliMixture
                #sample, x_hat = net.sample(args.num_messages)
                sample, _ = net.sample(args.num_messages)
            assert not sample.requires_grad
            #assert not x_hat.requires_grad
        
            ### For a given Beta, we can compute:
            ### FE: from `loss`
            ### Entropy: from `log_prob`
            ### Energy: from `energy`, calculated from the Hamiltonian
            ### Magnetization: from `sample`
            
   
            if args.print_step and step % args.print_step == 0:
                
                #print('loss:')
                #print(loss)
                #print('loss shape= %d' % loss.shape)
                #print('free_energy_mean:')
                #print(loss.mean())
            
                free_energy_mean = loss.mean() / args.beta / args.L**2
                free_energy_std = loss.std() / args.beta / args.L**2
                #entropy_mean = -log_prob.mean() / args.L**2
                #energy_mean = energy.mean() / args.L**2
                #mag = sample.mean(dim=0)
                #mag_mean = mag.mean()
                #mag_sqr_mean = (mag**2).mean()
                #if step % 10 == 0:
                if step > 0:                        
                    sample_time /= args.print_step
                    train_time /= args.print_step
                    used_time = time.time() - start_time
                    FE_mean = free_energy_mean.item()
                    FE_std = free_energy_std.item()
                    #FE_r_mean_std = np.abs(FE_std / FE_mean) 
                    
                    my_log(
                            #'step = {}, F = {:.8g}, F_std = {:.8g}, S = {:.8g}, E = {:.8g}, M = {:.8g}, Q = {:.8g}, lr = {:.3g}, beta = {:.8g}, sample_time = {:.3f}, train_time = {:.3f}, used_time = {:.3f}'
                            'step = {}, F = {:.8g}, F_std = {:.8g}, beta = {:.8g}, sample_time = {:.3f}, train_time = {:.3f}, used_time = {:.3f}'
                            .format(
                                    step,
                                    FE_mean,
                                    FE_std,
                                    #FE_r_mean_std,
                                    #energy_mean.item(),
                                    #mag_mean.item(),
                                    #mag_sqr_mean.item(),
                                    #optimizer.param_groups[0]['lr'],
                                    beta,
                                    sample_time,
                                    train_time,
                                    used_time,
                                    ))
                    
                    
                    
                    ##writer.add_scalar('Free_Energy/mean', FE_mean, step)
                    ##writer.add_scalar('Free_Energy/std', FE_std, step)
                    ##writer.add_scalar('Free_Energy/beta_step', beta, step)
                    sample_time = 0
                    train_time = 0
        
                    if args.save_sample:
                        state = {
                                'sample': sample,
                                'x_hat': x_hat,
                                'log_prob': log_prob,
                                'energy': energy,
                                'loss': loss,
                                }
                        torch.save(state, '{}_save/{}.sample'.format(
                                args.out_filename, step))
                        
                        if (args.out_filename and args.save_step
                            and step % args.save_step == 0):
                            state = {
                                    'net': net.state_dict(),
                                    'optimizer': optimizer.state_dict(),
                                    }
                            if args.lr_schedule:
                                state['scheduler'] = scheduler.state_dict()
                                torch.save(state, '{}_save/{}.state'.format(
                                        args.out_filename, step))
                                
                                if (args.out_filename and args.visual_step
                                    and step % args.visual_step == 0):
                                    torchvision.utils.save_image(
                                            sample,
                                            '{}_img/{}.png'.format(args.out_filename, step),
                                            nrow=int(sqrt(sample.shape[0])),
                                            padding=0,
                                            normalize=True)
                                    
                
                if args.print_sample:
                    x_hat_np = x_hat.view(x_hat.shape[0], -1).cpu().numpy()
                    x_hat_std = np.std(x_hat_np, axis=0).reshape([args.L] * 2)
    
                    x_hat_cov = np.cov(x_hat_np.T)
                    x_hat_cov_diag = np.diag(x_hat_cov)
                    x_hat_corr = x_hat_cov / (
                            sqrt(x_hat_cov_diag[:, None] * x_hat_cov_diag[None, :]) +
                            args.epsilon)
                    x_hat_corr = np.tril(x_hat_corr, -1)
                    x_hat_corr = np.max(np.abs(x_hat_corr), axis=1)
                    x_hat_corr = x_hat_corr.reshape([args.L] * 2)
                    
                    energy_np = energy.cpu().numpy()
                    energy_count = np.stack(
                            np.unique(energy_np, return_counts=True)).T
                            
                    my_log(
                                    '\nsample\n{}\nx_hat\n{}\nlog_prob\n{}\nenergy\n{}\nloss\n{}\nx_hat_std\n{}\nx_hat_corr\n{}\nenergy_count\n{}\n'
                                    .format(
                                            sample[:args.print_sample, 0],
                                            x_hat[:args.print_sample, 0],
                                            log_prob[:args.print_sample],
                                            energy[:args.print_sample],
                                            loss[:args.print_sample],
                                            x_hat_std,
                                            x_hat_corr,
                                            energy_count,
                                            ))
    
                if args.print_grad:
                    my_log('grad max_abs min_abs mean std')
                    for name, param in named_params:
                        if param.grad is not None:
                            grad = param.grad
                            grad_abs = torch.abs(grad)
                            
                            max_norm = torch.max(grad_abs).item() 
                            min_norm = torch.min(grad_abs).item()
                            mean_norm = torch.mean(grad).item()
                            std_norm = torch.std(grad).item()
                            
                            my_log('{} {:.3g} {:.3g} {:.3g} {:.3g}'.format(
                                    name,
                                    max_norm,
                                    min_norm,
                                    mean_norm,
                                    std_norm,
                                    ))
                            
                            ##writer.add_scalar('Grad_Norm/max', max_norm, step)
                            ##writer.add_scalar('Grad_Norm/min', min_norm, step)
                            ##writer.add_scalar('Grad_Norm/mean', mean_norm, step)
                            ##writer.add_scalar('Grad_Norm/std', std_norm, step)
                            
                            
                        else:
                            my_log('{} None'.format(name))
                    my_log('')
                    
                    
                    
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
        
        #sample_ = sample.view(sample.shape[0], sample.shape[2]*sample.shape[3])
        #torch.save(sample_, path + 'sample_out_beta_p_%s_beta_%s_code_%d.pt' % (bp_s, bs, j))
        
        
        
        
        
        
        
        ## Saving the model
        ##torch.save(net.state_dict(), path + 'saved_model_beta_p_%s_beta_%s_code_%d.pt' % (bp_s, bs, j))
        
        
        
        
        
        
        
        
        
        
        
        #sign_tensor_ = sign_tensor_ + torch.sign(sample_.sum(dim=1))
        
        ### Sampling from the network to construct < S_j >
        #sample_trained, _ = net.sample(args.num_messages)
        
        #for j in range(num_samples - 1):
        #    samp, _ = net.sample(args.num_messages)
        #    sample_trained = sample_trained + samp
            
        #sample_trained = sample_trained.view(sample_trained.shape[0], 
        #                                     sample_trained.shape[2]*sample_trained.shape[3]) / num_samples
                                             
        #signSj = torch.sign(sample_trained)
        
        #sign_tensor = sign_tensor + signSj
        
        
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

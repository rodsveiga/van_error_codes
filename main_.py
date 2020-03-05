#!/usr/bin/env python3
#
# Solving Statistical Mechanics using Variational Autoregressive Networks
# 2d classical Ising model

import time

import numpy as np
import torch
#import torchvision
from torch.utils.tensorboard import SummaryWriter
torch.manual_seed(12) 
from numpy import sqrt
from torch import nn
import os

import energy
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
from bp import BP_LDPC


def main():
    
    N = args.N
    M = args.M
    K = args.K
    num_codes = args.num_codes 
    net_depth = args.net_depth
    net_width = args.net_width
       
    p = args.p
    beta_p =  0.5*np.log( (1. - p) / p)

    p_prior = args.p_prior
    beta_prior = 0.5*np.log( (1. - p_prior) / p_prior)
    
    if args.save_step == 0:
        save_step = args.max_step
    else:
        save_step = args.save_step
        
    # String for saving
    ps = '{:.3f}'.format(p).replace('.', '')
    pps = '{:.3f}'.format(p_prior).replace('.', '')
    Ns = str(N)
    Ms = str(M)
    Ks = str(K)
    steps = str(args.max_step)
    
    # Select path for saving
    if args.laptop:
        path_ = '/home/rodrigo/Dropbox/DOC/van_error_codes/models'
    else:
        path_ = '/home/rodsveiga/Dropbox/DOC/van_error_codes/models'

    # Creating the directory to save important information
    PATH_ = path_ + '/N_%s_M_%s_K_%s_p_prior_%s_steps_%s' % (Ns, Ms, Ks, pps, steps)

    try:
        os.mkdir(PATH_)
    except OSError:
        print ("Creation of the directory %s failed" % PATH_)
    else:
        print ("Successfully created the directory %s " % PATH_)


    path = PATH_ + '/model_N_%s_M_%s_K_%s_p_%s_p_prior_%s_net_depth_%s_net_width_%s_steps_%s' % (Ns, Ms, Ks, ps, pps, str(net_depth), str(net_width), steps)

    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)
    
   
    
    start_time = time.time()
               

    #init_out_dir()
    #if args.clear_checkpoint:
    #    clear_checkpoint()
    #last_step = get_last_checkpoint_step()    
    #if last_step >= 0:
    #    my_log('\nCheckpoint found: {}\n'.format(last_step))
    #else:
    #    clear_log()
    

        
    last_step = -1
        
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
    
    
    #####################################
    print('{}\n'.format(net))
    ##############################################
   
    
    params = list(net.parameters())
    params = list(filter(lambda p: p.requires_grad, params))
    nparams = int(sum([np.prod(p.shape) for p in params]))
    
    print('Total number of trainable parameters: {}'.format(nparams))
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
            optimizer, factor=0.92, patience=100, threshold=0.001, min_lr=1e-6, verbose=True)

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
    print('init_time = {:.3f}'.format(init_time))
    
    
    ##############################################        
    #if args.log:
                
    #    file = open(path + '/log_N_%s_M_%s_K_%s_p_%s_p_prior_%s.txt' % (Ns, Ms, Ks,ps, pps), 'a')
        
    #    for k, v in args._get_kwargs():
    #        file.write('{} = {}'.format(k, v) + '\n')
        
    #    file.write('\n' + '{}\n'.format(net) + '\n' + 'Total number of trainable parameters: {}'.format(nparams) + '\n')
    #    file.write('init_time = {:.3f}'.format(init_time) + '\n' + '-----------------------------' + '\n' + 'Training...')
    ##############################################
        
            

    print('-----------------------------')
    print('Training...')
    sample_time = 0
    train_time = 0
    start_time = time.time()
    

    #sample_start_time = time.time()
       
    
    ########################################################################
    ## With message prior
    random = torch.rand([args.num_messages, N],
                        device=args.device)

    # -1 with probability p_prior
    # +1 with probability 1 - p_prior
    sample_in = 2* (random > p_prior).float() - 1

    if args.save_model:
        torch.save(sample_in, path + '/message_N_%s_M_%s_K_%s_p_%s_p_prior_%s.pt' % (Ns, Ms, Ks,ps, pps))

    ########################################################################
    
    #sample_in = torch.zeros([args.num_messages, N],
    #                        device=args.device)
    
                       
    #for j in range(random.shape[0]):
            
    #    for k in range(random.shape[1]):
                
            #### -1 with probability p_prior
    #        if random[j,k] <= p_prior:
    #            sample_in[j,k] = -1.
            #### +1 with probability 1 - p_prior
    #        else:
    #            sample_in[j,k] = +1.


    #           random = torch.rand([n, N])

                
    ########################################################################
    
    
    # Generate the codes
    C_list = []

    for k in range(num_codes):
        C = torch.randint(0, N, [M, K], device=args.device)
        
        if args.save_model:
            torch.save(C, path + '/codes_N_%s_M_%s_K_%s_p_%s_p_prior_%s_code_%s.pt' % (Ns, Ms, Ks,ps, pps, str(k)))
        
        C_list.append(C)
        
      
    
    for j in range(num_codes):


        ##############################################        
        if args.log:
                
            file = open(path + '/log_N_%s_M_%s_K_%s_p_%s_p_prior_%s_code_%s.txt' % (Ns, Ms, Ks, ps, pps, str(j)), 'a')
        
            for k, v in args._get_kwargs():
                file.write('{} = {}'.format(k, v) + '\n')
        
            file.write('\n' + '{}\n'.format(net) + '\n' + 'Total number of trainable parameters: {}'.format(nparams) + '\n')
            file.write('init_time = {:.3f}'.format(init_time) + '\n' + '-----------------------------' + '\n' + 'Training...' + '\n')
        ##############################################
        
        print('Code number= %d' % j)
        
        C = C_list[j]
        
        ### Tensorboard
        if args.tensorboard:
            writer = SummaryWriter()
        
        sample = sample_in


        ##################################################################
        ##### Encoding

        # Initializing
        J0 = torch.take(sample_in[0], C).prod(dim=1)
        J0 = J0.unsqueeze(0)

        # Loop over all messages
        for l in range(1, sample_in.shape[0]):
    
            J0_ = torch.take(sample_in[l], C).prod(dim=1)
            J0_ = J0_.unsqueeze(0)
    
            J0 = torch.cat((J0, J0_), dim= 0)

        if args.save_model:
            torch.save(J0, path + '/encoded_J_N_%s_M_%s_K_%s_p_%s_p_prior_%s_code_%s.pt' % (Ns, Ms, Ks,ps, pps, str(j)))

        #####  Corrupted version (if we work like this, the overlap decreases. It is curious. The machine is better when we corrupt)

        random = torch.rand(J0.shape)
        ## Flip tensor: -1 with probability p
        flip = 2*(random > p).float() - 1

        ## J corrupted version: element wise multiplication
        J = torch.mul(J0, flip)

        if args.save_model:
            torch.save(J, path + '/corrupted_J_N_%s_M_%s_K_%s_p_%s_p_prior_%s_code_%s.pt' % (Ns, Ms, Ks,ps, pps, str(j)))

        ##################################################################


        
        for step in range(last_step + 1, args.max_step + 1):
                                               
            ### The user can choose the optimizer
            optimizer.zero_grad()
                
            #sample_time += time.time() - sample_start_time
        
            t0 = time.time()
            
    
            # Log-prob is calculated from `sample`
            log_prob = net.log_prob(sample)
            # 0.998**9000 ~ 1e-8
            beta = beta_p * (1 - args.beta_anneal**step)
                        
            with torch.no_grad():
                ### Only moment where the Hamiltonian is taken into account
                ### It is calculated for each `sample`
                ### The GOAL is to understand the sampling
                #energy = ising.energy(sample, args.ham, args.lattice,
                #                      args.boundary)
                
                ## What do I choose here? args.beta or beta?
                energy_ = energy.sourlas(sample, J, C, beta_p, beta_prior, args.device)
                
                ### Beta*FE is the loss function
                loss = log_prob + beta * energy_
                
                        
            assert not energy_.requires_grad
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

            t1 = time.time()

        
   
            if args.print_step and step % args.print_step == 0:
                
                if step > 0:                        
                    sample_time /= args.print_step
                    train_time /= args.print_step
                    used_time = time.time() - start_time
                    FE_mean = ( loss.mean() / (beta_p * N) ).item()
                    FE_std = ( loss.std() / (beta_p * N) ).item()
                    E_mean = ( energy_.mean() / N ).item()
                    E_std = ( energy_.std() / N ).item()
                    S_mean = ( -log_prob.mean() / N ).item()
                    S_std = ( -log_prob.std() / N ).item()
                    overlap_local = (sample == sample_in).sum().item() / (sample_in.shape[0]*sample_in.shape[1])

                    
                    print('step= {}, F= {:.6f}, F_std= {:.4f}, E= {:.6f}, E_std= {:.6f}, S= {:.6f}, S_std= {:.6f}, Ov_proxy= {:.4f}, p= {:.4f}, p_prior= {:.4f}, beta:prior/p= {:.4f}, time= {:.3f}, total_time= {:.3f}'
                            .format(
                                    step,
                                    FE_mean,
                                    FE_std,
                                    E_mean,
                                    E_std,
                                    S_mean,
                                    S_std,
                                    overlap_local,
                                    p,
                                    p_prior,
                                    beta_p / beta_prior,
                                    t1-t0,
                                    t1-start_time,
                                    ))
                    
                    if args.log:
                        
                         file.write('step= {}, F= {:.6f}, F_std= {:.4f}, E= {:.6f}, E_std= {:.6f}, S= {:.6f}, S_std= {:.6f}, Ov_proxy= {:.4f}, p= {:.4f}, p_prior= {:.4f}, beta:prior/p= {:.4f}, time= {:.3f}, total_time= {:.3f}'
                                    .format(
                                            step,
                                            FE_mean,
                                            FE_std,
                                            E_mean,
                                            E_std,
                                            S_mean,
                                            S_std,
                                            overlap_local,
                                            p,
                                            p_prior,
                                            beta_p / beta_prior,
                                            t1-t0,
                                            t1-start_time,
                                            ) + '\n')

                    if args.monitor_ov:

                        if step % args.monitor_freq == 0:

                            print('--Monitoring overlap until now')

                            if args.log:
                                file.write('--Monitoring overlap until now'+ '\n')

                            sample__ = torch.zeros(sample_in.shape, requires_grad = False).float()

                            for r in range(args.monitor_num_samples):

                                with torch.no_grad():
                                    sample_, _ = net.sample(args.num_messages)
                                assert not sample_.requires_grad

                                sample__.add_(sample_)

                                ov__ = (torch.sign(sample__) == sample_in).sum().item() / (sample_in.shape[0]*sample_in.shape[1])
                                print('sample= %d -- overlap= %.5f' % (r, ov__))

                                if args.log:
                                    file.write(('sample: {} -- overlap= {:.5f}'
                                        .format(
                                                r,
                                                ov__,
                                                ) + '\n'))


                                del sample_
                        
                            monitor_overlap = (torch.sign(sample__) == sample_in).sum().item() / (sample_in.shape[0]*sample_in.shape[1])
                            del sample__

                            print('Samples from the trained model: %d. Overlap= %.6f' % (args.monitor_num_samples, monitor_overlap))

                            if args.log:
                                file.write(('Samples from the trained model: {}. Overlap= {:.6f}'
                                    .format(
                                            args.monitor_num_samples,
                                            monitor_overlap,
                                            ) + '\n'))


                            #print('Samples from the trained model: {}. Overlap= {:.6g}'
                            # .format(
                            #           args.monitor_num_samples,
                            #          monitor_overlap,
                            #          ))

                            #if args.log:
                            #    file.write('--Monitoring overlap until now')

                            #    file.write(('Samples from the trained model: {}. Overlap= {:.6g}'
                            #            .format(
                            #                    args.monitor_num_samples,
                            #                    monitor_overlap,
                            #                    ) + '\n'))


                    
                if args.tensorboard:
                    writer.add_scalar('Model/Free_Energy_mean', FE_mean, step)
                    writer.add_scalar('Model/Free_Energy_std', FE_std, step)
                    writer.add_scalar('Model/Energy_mean', E_mean, step)
                    writer.add_scalar('Model/Entropy_mean', S_mean, step)
                    writer.add_scalar('Model/Ov_local', overlap_local, step)
                    writer.add_scalar('Model/Beta_step', beta, step)
                    if args.monitor_ov:
                        writer.add_scalar('Model/Monitor_ov', monitor_overlap, step)

        
        
                if args.print_grad:
                    
                    for name, param in named_params:
                        if param.grad is not None:
                            grad = param.grad
                            grad_abs = torch.abs(grad)
                            
                            max_norm = torch.max(grad_abs).item() 
                            min_norm = torch.min(grad_abs).item()
                            mean_norm = torch.mean(grad).item()
                            std_norm = torch.std(grad).item()
  
                            
                            if args.tensorboard:
                                writer.add_scalar('Grad_Norm/max', max_norm, step)
                                writer.add_scalar('Grad_Norm/min', min_norm, step)
                                writer.add_scalar('Grad_Norm/mean', mean_norm, step)
                                writer.add_scalar('Grad_Norm/std', std_norm, step)
                            
                            
                        else:
                            my_log('{} None'.format(name))
                            
                            
            if args.save_model and step % save_step == 0:
                if step > 0:
                    torch.save(net.state_dict(), path + '/model_N_%s_M_%s_K_%s_p_%s_p_prior_%s_code_%s_step_%s.pt' % (Ns, Ms, Ks,ps, pps, str(j), str(step)))
                
                    #my_log('')
        
        if args.log:
            file.close()

        ps_ = '{:.3f}'.format(p)
        #overlap_ = '{:.6f}'.format(overlap_local)


        #PATH_ = path_ + '/p_versus_overlap_N_%s_M_%s_K_%s_p_prior_%s_net_depth_%s_net_width_%s_steps_%s_code_%s' % (Ns, Ms, Ks, pps, str(net_depth), str(net_width), steps, str(j))
        #file_ = open(PATH_, 'a')
        #file_.write(ps_ + '  ' + overlap_ + '\n') 
        #file_.close()

        #print('overlap = ', overlap_local)



        ##### BELIEF PROPAGATION #### 


        if args.BP:

            overlap_BP_ = []

            #args.BP_it

            for k in range(int(J.shape[0]/10)):

                print('----BP message %d' % k)

                t0 = time.time()
       
                opt_dec_Bayes_BP = BP_LDPC(N, M, J[k], beta, beta_prior, C, sample_in[k], num_it= args.BP_it, verbose= 1)

                t1 = time.time()

                print('time = {:.3f}'.format(t1 - t0))
    
                overlap_BP_.append(opt_dec_Bayes_BP)

            overlap_BP = np.mean(np.array(overlap_BP_))

            num_it = str(args.BP_it)
            overlap_BP_ = '{:.6f}'.format(overlap_BP)


            PATH_ = path_ + '/p_versus_overlap_BP_N_%s_M_%s_K_%s_p_prior_%s_code_%s_it_%s' % (Ns, Ms, Ks, pps, str(j), num_it)
            file_ = open(PATH_, 'a')
            file_.write(ps_ + '  ' + overlap_BP_ + '\n') 
            file_.close()



        #######################################
            
                    
                    
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
        #if args.save_model and:
        #    torch.save(net.state_dict(), path + 'model_N_%s_M_%s_K_%s_p_%s_p_prior_%s_code_%s.pt' % (Ns, Ms, Ks,ps, pps, js))
        
        
        #torch.save(net.state_dict(), path + 'saved_model_beta_p_%s_beta_%s_code_%d.pt' % (bp_s, bs, j))
        
        
        
        
        
        
        
        
        
        
        
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

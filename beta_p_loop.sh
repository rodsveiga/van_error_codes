#!/bin/sh

# Fig. 2(a)
#for beta_p in 3.45 1.47 1.10 0.87 0.69 0.55; do
#    python main_.py --ham fm --lattice sqr --L 10 --beta 1.0 --beta_p $beta_p --net made --net_depth 3 --net_width 4 --beta_anneal 0.998 --clip_grad 1  --save_step 0 --print_sample 0 --print_grad --max_step 3000 --batch_size 2000 --bias
#done


#for beta_p in 0.00 0.10 0.20 0.31 0.42 0.55 0.69 0.87 1.10 1.47 3.45; do
#    python main_.py --ham fm --lattice sqr --L 10 --beta 1.0 --beta_p $beta_p --net made --net_depth 3 --net_width 4 --beta_anneal 0.998 --clip_grad 1  --save_step 0 --print_sample 0 --print_grad --max_step 2000 --batch_size 2000 --bias
#done


#for beta_p in 3.45 1.47 1.10 0.87 0.69 0.42 0.20 0.00; do
#    python main_.py --ham fm --lattice sqr --L 10 --beta 1.0 --beta_p $beta_p --net made --net_depth 3 --net_width 4 --beta_anneal 0.998 --clip_grad 1  --save_step 0 --print_sample 0 --print_grad --max_step 2000 --batch_size 2000 --bias
#done

#python sampling.py --ham fm --lattice sqr --L 10 --beta 1.0 --beta_p 0.00 --net made --net_depth 3 --net_width 4 --beta_anneal 0.998 --clip_grad 1  --save_step 0 --print_sample 0 --print_grad --max_step 3000 --batch_size 2000 --bias

for beta_p in 3.45 0.00; do
    python main_.py --ham fm --lattice sqr --L 10 --beta 1.0 --beta_p $beta_p --net made --net_depth 3 --net_width 4 --beta_anneal 0.998 --clip_grad 1  --save_step 0 --print_sample 0 --print_grad --max_step 2000 --batch_size 2000 --bias
done


#python main_.py --N 100 --M 200 --p 0.1 --p_prior 0.1 --net made --net_depth 3 --net_width 4 --beta_anneal 0.998 --clip_grad 1  --save_step 0 --print_sample 0 --print_grad --max_step 10 --num_messages 500 --bias
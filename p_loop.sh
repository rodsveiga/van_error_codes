for p in 0.1 0.2 0.225 0.25 0.275 0.3 0.325 0.35 0.375 0.4 0.45 0.495; do
    python main_.py --N 250 --M 500 --K 4 --p $p --p_prior 0.01 --num_messages 100 --max_step 1000 --save_model --net made --net_depth 2 --net_width 3  --beta_anneal 0.998 --clip_grad 1 --print_grad --log
done


#for p in 0.1 0.2 0.225 0.25 0.275 0.3 0.325 0.35 0.375 0.4 0.45 0.495; do
#    python main_.py --N 250 --M 500 --K 4 --p $p --p_prior 0.25 --num_messages 100 --max_step 1000 --net made --net_depth 2 --net_width 3  --beta_anneal 0.998 --clip_grad 1 --cuda 0 --laptop
#done


#for p in 0.1 0.2 0.225 0.25 0.275 0.3 0.325 0.35 0.375 0.4 0.45 0.495; do
#    python main_.py --N 250 --M 1000 --K 5 --p $p --p_prior 0.1 --num_messages 100 --max_step 1000 --save_model --net made --net_depth 2 --net_width 3  --beta_anneal 0.998 --clip_grad 1 --print_grad --log
#done



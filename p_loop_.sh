for p in 0.495 0.45 0.4 0.375 0.35 0.325 0.3 0.275 0.25 0.225 0.2 0.1; do
    python main_.py --N 250 --M 1000 --K 5 --p $p --p_prior 0.1 --num_messages 100 --max_step 1000 --save_model --net made --net_depth 2 --net_width 10  --beta_anneal 0.998 --clip_grad 1 --print_grad --log
done

#0.1 0.2 0.225 0.25 0.275 0.3 0.325 0.35 0.375 0.4 0.45 0.495

#python sampling.py --N 250 --M 1000 --K 5 --p 0.495 --p_prior 0.1 --num_messages 1000 --max_step 5000 --net made --net_depth 2 --net_width 1  --beta_anneal 0.0 --clip_grad 1 --print_grad --log --laptop
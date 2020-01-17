for p in 0.2 0.225 0.25 0.275 0.3 0.325 0.35 0.375 0.4 0.45 0.495; do
    python main_.py --N 100 --M 80 --K 4 --p $p --p_prior 0.05 --num_messages 100 --max_step 1000 --save_model --net made --net_depth 2 --net_width 3  --beta_anneal 0.998 --clip_grad 1 --save_step 100 --save_model --print_grad --tensorboard --log
done

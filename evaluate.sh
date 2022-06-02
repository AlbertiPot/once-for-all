# evaluate OFA Networks
# nohup \
/home/gbc/.conda/envs/rookie/bin/python \
eval_ofa_net.py \
--path 'data' \
--net ofa_mbv3_d234_e346_k357_w1.0 \
--gpu 1 \
--batch-size 100 \
--workers 8

# evaluate OFA Specialized Networks
# nohup \
# /home/gbc/.conda/envs/rookie/bin/python \
# eval_specialized_net.py \
# --path 'data' \
# --net resnet50D_MAC@3.0B_top1@79.3 \
# --gpu 0 \
# --batch-size 100 \
# --workers 8
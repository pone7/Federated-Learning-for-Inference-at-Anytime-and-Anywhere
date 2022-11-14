#!/bin/bash

lr=5e-3
dataset='cifar100'
num_classes=100
base_model='deit_small_patch16' # deit
data_path='/usr/data/cifar100'
mode='accumulator'

for alpha in 0.1 1 1000
do
(python main.py \
--num_client_cpus 2 \
--num_gpus 1. \
--fit_frac 0.1 \
--output_folder "results/${dataset}/anyfed/alpha_${alpha}/exp3/${base_model}_${mode}/" \
--clip_grad 0.0 \
--lr $lr \
--base_model $base_model \
--alpha $alpha \
--dataset $dataset \
--num_classes $num_classes \
--data_path $data_path \
--user_num 100 \
--local_bs 10 \
--num_rounds 500 \
--input_size 224 \
--mode $mode \
--adpffn \
--freeze_base \
--replace \
--anytime
) &
done

# anytime: replace, adpffn, freeze_base
# This will allow you to use CTRL+C to stop all background processes
# trap: SIGINT: bad trap 
# remove SIG the prefix from 
# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
trap "trap - TERM && kill -- -$$" INT TERM
# Wait for all background processes to complete
wait

# if you can't kill the processes use the following command
# ps ax | grep 'python x' | grep -v grep | awk '{kill -9 $1}'   
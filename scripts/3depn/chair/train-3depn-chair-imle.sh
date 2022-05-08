#!/usr/bin/env bash
python train_imle.py --proj_dir /scratch/haa/proj_log \
                --exp_name mpc-3depn-chair2 \
                --module imle_gan \
                --dataset_name 3depn \
                --category chair \
                --data_root /scratch/haa/data/cdata \
                --data_raw_root /scratch/haa/data/pdata \
                --pretrain_ae_path /scratch/haa/proj_log/mpc-3depn-chair/ae/model/ckpt_epoch750.pth \
                --pretrain_vae_path /scratch/haa/proj_log/ckpt_epoch1500.pth \
                --batch_size 5 \
                --lr 5e-4 \
                --save_frequency 25 \
                --nr_epochs 300 \
                -g 0 \
                --vis 
                
                
                

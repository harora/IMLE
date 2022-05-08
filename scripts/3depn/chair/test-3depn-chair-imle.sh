#!/usr/bin/env bash
python test.py --proj_dir /scratch/haa/proj_log \
                --exp_name mpc-3depn-chair6 \
                --module imle_gan \
                --dataset_name 3depn \
                --category chair \
                --data_root /scratch/haa/data/cdata \
                --data_raw_root /scratch/haa/data/pdata \
                --pretrain_ae_path /scratch/haa/proj_log/mpc-3depn-chair/ae/model/ckpt_epoch2000.pth \
                --pretrain_vae_path /scratch/haa/proj_log/mpc-3depn-chair/vae/model/ckpt_epoch1500.pth \
                --num_sample -1 \
                --num_z 10 \
                --ckpt 25 \
                -g 0

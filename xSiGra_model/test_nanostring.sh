#!/bin/bash
python3 train_nanostring.py --test_only 1 --dataset lung5-rep1 --root ../dataset/nanostring/lung5-rep1 --save_path ../checkpoint/nanostring_train_lung5_rep1 --seed 1234 --epochs 200 --lr 1e-3 --num_fov 30 --device cuda:0
python3 train_nanostring.py --test_only 1 --dataset lung5-rep2 --root ../dataset/nanostring/lung5-rep2 --save_path ../checkpoint/nanostring_train_lung5_rep2 --seed 1234 --epochs 200 --lr 1e-3 --num_fov 30 --device cuda:0
python3 train_nanostring.py --test_only 1 --dataset lung5-rep3 --root ../dataset/nanostring/lung5-rep3 --save_path ../checkpoint/nanostring_train_lung5_rep3 --seed 1234 --epochs 200 --lr 1e-3 --num_fov 30 --device cuda:0
python3 train_nanostring.py --test_only 1 --dataset lung6 --ncluster 4 --root ../dataset/nanostring/lung6 --save_path ../checkpoint/nanostring_train_lung6 --seed 1234 --epochs 200 --lr 1e-3 --num_fov 30 --device cuda:0
python3 train_nanostring.py --test_only 1 --dataset lung9-rep1  --root ../dataset/nanostring/lung9-rep1 --save_path ../checkpoint/nanostring_train_lung9_rep1 --seed 1234 --epochs 200 --lr 1e-3 --num_fov 20 --device cuda:0
python3 train_nanostring.py --test_only 1 --dataset lung9-rep2 --ncluster 4 --root ../dataset/nanostring/lung9-rep2 --save_path ../checkpoint/nanostring_train_lung9_rep2 --seed 1234 --epochs 200 --lr 1e-3 --num_fov 45 --device cuda:0
python3 train_nanostring.py --test_only 1 --dataset lung12 --root ../dataset/nanostring/lung12 --save_path ../checkpoint/nanostring_train_lung12 --seed 1234 --epochs 200 --lr 1e-3 --num_fov 28 --device cuda:0
python3 train_nanostring.py --test_only 1 --dataset lung13 --root ../dataset/nanostring/lung13 --save_path ../checkpoint/nanostring_train_lung13 --seed 1234 --epochs 200 --lr 1e-3 --num_fov 20 --device cuda:0
python3 train_nanostring.py --test_only 1 --save_path ../checkpoint/nanostring_train_lung13

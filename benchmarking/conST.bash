#!/bin/bash
python3 run_conST.py --root '../dataset/nanostring/lung13' --n_clusters 8 --num_fov 20 --name "lung13"
python3 run_conST.py --root '../dataset/nanostring/lung12' --n_clusters 8 --num_fov 28 --name "lung12"
python3 run_conST.py --root '../dataset/nanostring/lung9-rep1' --n_clusters 8 --num_fov 20 --name "lung9-1"
python3 run_conST.py --root '../dataset/nanostring/lung9-rep2'  --n_clusters 4 --num_fov 45 --name "lung9-2"
python3 run_conST.py --root '../dataset/nanostring/lung6' --n_clusters 4 --num_fov 30 --name "lung6"
python3 run_conST.py --root '../dataset/nanostring/lung5-rep1' --n_clusters 8 --num_fov 30 --name "lung5-1"
python3 run_conST.py --root '../dataset/nanostring/lung5-rep2' --n_clusters 8 --num_fov 30 --name "lung5-2"
python3 run_conST.py --root '../dataset/nanostring/lung5-rep3' --n_clusters 8 --num_fov 30 --name "lung5-3"

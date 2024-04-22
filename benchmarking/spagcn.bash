#!/bin/bash
python3 run_spagcn.py --root ../dataset/nanostring/lung5-rep1/ --num_fov 30 --save_path ./cluster_results_spagcn/lung5-1/ --ncluster 8
python3 run_spagcn.py --root ../dataset/nanostring/lung5-rep2/ --num_fov 30 --save_path ./cluster_results_spagcn/lung5-2/ --ncluster 8
python3 run_spagcn.py --root ../dataset/nanostring/lung5-rep3/ --num_fov 30 --save_path ./cluster_results_spagcn/lung5-3/ --ncluster 8
python3 run_spagcn.py --root ../dataset/nanostring/lung6/ --num_fov 30 --save_path ./cluster_results_spagcn/lung6/ --ncluster 4
python3 run_spagcn.py --root ../dataset/nanostring/lung9-rep1/ --num_fov 20 --save_path ./cluster_results_spagcn/lung9-1/ --ncluster 8
python3 run_spagcn.py --root ../dataset/nanostring/lung9-rep2/ --num_fov 45 --save_path ./cluster_results_spagcn/lung9-2/ --ncluster 4
python3 run_spagcn.py --root ../dataset/nanostring/lung12/ --num_fov 28 --save_path ./cluster_results_spagcn/lung12/ --ncluster 8
python3 run_spagcn.py --root ../dataset/nanostring/lung13/ --num_fov 20 --save_path ./cluster_results_spagcn/lung13/ --ncluster 8

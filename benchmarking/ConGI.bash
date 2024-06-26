#!/bin/bash
python3 run_ConGI.py --path ../dataset/nanostring/lung13 --root ../dataset/nanostring/lung13 --num_fov 20 --name lung13
python3 run_ConGI.py --path ../dataset/nanostring/lung12 --root ../dataset/nanostring/lung12 --num_fov 28 --name lung12
python3 run_ConGI.py --path ../dataset/nanostring/lung9-rep1 --root ../dataset/nanostring/lung9-rep1 --num_fov 20 --name lung9-1
python3 run_ConGI.py --path ../dataset/nanostring/lung9-rep2 --root ../dataset/nanostring/lung9-rep2 --num_fov 20 --name lung9-2
python3 run_ConGI.py --path ../dataset/nanostring/lung6 --root ../dataset/nanostring/lung6 --num_fov 30 --name lung6
python3 run_ConGI.py --path ../dataset/nanostring/lung5-rep1 --root ../dataset/nanostring/lung5-rep1 --num_fov 30 --name lung5-1 
python3 run_ConGI.py --path ../dataset/nanostring/lung5-rep2 --root ../dataset/nanostring/lung5-rep2 --num_fov 30 --name lung5-2
python3 run_ConGI.py --root ../dataset/nanostring/lung5-rep3 --path ../dataset/nanostring/lung5-rep3 --num_fov 30 --name lung5-3

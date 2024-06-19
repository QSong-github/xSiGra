#!/bin/bash
#SBATCH --nodes=1
#SBATCH -p gpu
#SBATCH --gpus-per-node 1
#SBATCH -A r00618
#SBATCH --ntasks-per-node=1
#SBATCH --time=20:00:00
#SBATCH --output=out/const_visium_%j.out
#SBATCH --mem=150GB
#SBATCH -J const_visium
#SBATCH --mail-type=START,END,FAIL

python3 run_conST.py --ncluster 7 --id 151676 --ncluster 7
python3 run_conST.py --ncluster 7 --id 151675 --ncluster 7
python3 run_conST.py --ncluster 7 --id 151674 --ncluster 7
python3 run_conST.py --ncluster 7 --id 151673 --ncluster 7 
python3 run_conST.py --ncluster 5 --id 151672 --ncluster 5
python3 run_conST.py --ncluster 5 --id 151671 --ncluster 5
python3 run_conST.py --ncluster 5 --id 151670 --ncluster 5
python3 run_conST.py --ncluster 5 --id 151669 --ncluster 5
python3 run_conST.py --ncluster 7 --id 151507 --ncluster 7
python3 run_conST.py --ncluster 7 --id 151508 --ncluster 7
python3 run_conST.py --ncluster 7 --id 151509 --ncluster 7
python3 run_conST.py --ncluster 7 --id 151510 --ncluster 7
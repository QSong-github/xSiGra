#!/bin/bash
#SBATCH --nodes=1
#SBATCH -p gpu
#SBATCH --gpus-per-node 1
#SBATCH -A r00618
#SBATCH --ntasks-per-node=1
#SBATCH --time=3:00:00
#SBATCH --output=out/congi_visium_%j.out
#SBATCH --mem=150GB
#SBATCH -J congi_visium
#SBATCH --mail-type=START,END,FAIL

python3 run_ConGI.py --name 151676
python3 run_ConGI.py --name 151675
python3 run_ConGI.py --name 151674
python3 run_ConGI.py --name 151673
python3 run_ConGI.py --name 151672
python3 run_ConGI.py --name 151671
python3 run_ConGI.py --name 151670
python3 run_ConGI.py --name 151669
python3 run_ConGI.py --name 151507
python3 run_ConGI.py --name 151508
python3 run_ConGI.py --name 151509
python3 run_ConGI.py --name 151510
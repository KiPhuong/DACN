#!/bin/bash

#SBATCH --job-name=sqli
#SBATCH -o output/result2.out
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=3-00:00:00

module load python39

#pip3 install networkx==2.8.8
#pip3 install fastdtw==0.3.4
#pip3 install  dgl -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
python3 train_model_on_gpu.py

#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=test
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=10-00:00:00
#SBATCH --nodelist=node1
#SBATCH --output=outputs/sim_result.%J.out

export OMP_NUM_THREADS=32

module load conda
module load cuda/11.8
module load gcc/9

cd /scratch/karthick/pretrain/param-7b
source /scratch/karthick/cotrain/cotrain-env/bin/activate


# torchrun --nproc_per_node=2 hf_train.py
python hf_train.py
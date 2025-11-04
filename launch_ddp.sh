#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=pretrain
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --time=10-00:00:00
#SBATCH --nodelist=node2
#SBATCH --output=outputs/sim_result.%J.out

export OMP_NUM_THREADS=32

module load conda
module load cuda/11.8
module load gcc/9

cd /scratch/karthick/pretrain/param-7b
source /scratch/karthick/cotrain/cotrain-env/bin/activate

# torchrun --nproc_per_node=1 ddp_trainer.py --resume_checkpoint /scratch/karthick/pretrain/param-7b/param-checkpoints/20250220_173103_096d4b770e/checkpoint_step_20000.pt
# python test_model.py

torchrun --nproc_per_node=2 --nnodes=1 hf_train.py

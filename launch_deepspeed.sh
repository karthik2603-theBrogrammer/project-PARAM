#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=pretrain
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=10-00:00:00
#SBATCH --nodelist=node8
#SBATCH --output=outputs/sim_result.%J.out

module load conda
module load cuda/11.8
module load gcc/9

cd /scratch/karthick/pretrain/param-7b
source /scratch/karthick/cotrain/cotrain-env/bin/activate

deepspeed --num_gpus=4 deepspeed_trainer.py --deepspeed --deepspeed_config ds_config.json

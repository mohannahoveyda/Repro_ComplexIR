#!/bin/bash
# ----------  job resources  ----------
#SBATCH --account=rusei12394
#SBATCH --partition=gpu_h100
#SBATCH --job-name=check_LLMs
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mohanna.hoveyda@ru.nl

source /home/mhoveyda1/anaconda3/etc/profile.d/conda.sh
conda activate eap-ig


ROOT=/home/mhoveyda1/REASON

cd $ROOT
echo "Start run"
python check_models.py 
echo "Done checking!"
deactivate

#!/bin/bash
#SBATCH --job-name=e5_build_index
#SBATCH --output=outputs/logs/e5_build_index_%j.out
#SBATCH --error=outputs/logs/e5_build_index_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Load modules (adjust based on Snellius setup)
# Uncomment and modify as needed for your Snellius environment
# module load Python/3.9.6-GCCcore-11.2.0
# module load CUDA/11.7.0
# module load cuDNN/8.4.1.50-CUDA-11.7.0

# Activate conda environment
# Adjust path to your conda installation on Snellius
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate sigir26_repro_py39

# Set environment variables for optimal GPU usage
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Print GPU information
echo "GPU Information:"
nvidia-smi

# Navigate to project directory (adjust path as needed)
# Assuming you're running from the project root
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Run the index building script
# Using larger batch size for A100 (40GB memory can handle more)
# Adjust batch_size based on your specific A100 model (40GB vs 80GB)
python code/baselines/e5-mistral-7b-instruct/build_quest_e5_index.py \
    --corpus data/QUEST/documents.jsonl \
    --output_dir outputs/quest_e5_mistral_index \
    --batch_size 32 \
    --device cuda

echo "=========================================="
echo "End Time: $(date)"
echo "Job completed"
echo "=========================================="


sbatch --partition=gpu_a100 --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:15:00 \
  --output=outputs/logs/nvcc-check-%j.out --error=outputs/logs/nvcc-check-%j.err \
  --wrap="bash -lc '
    set -e
    mkdir -p outputs/logs
    module purge
    module spider cuda | head -n 120 || true
    # module load <THE_RIGHT_ONE>
    which nvcc || true
    nvcc --version || true
  '"

# sbatch --partition=gpu_a100 --gres=gpu:1 --cpus-per-task=4 --mem=8G --time=00:05:00 \
#   --output=outputs/logs/gpu-check-%j.out --error=outputs/logs/gpu-check-%j.err \
#   --wrap="bash -lc '
#     set -e
#     mkdir -p outputs/logs
#     source ~/anaconda3/etc/profile.d/conda.sh
#     conda activate reasonir
#     nvidia-smi
#     python -c '\''import torch; print(\"torch\", torch.__version__); print(\"cuda\", torch.version.cuda); print(\"cuda_available\", torch.cuda.is_available()); print(\"device\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"no-gpu\")'\''
#   '"
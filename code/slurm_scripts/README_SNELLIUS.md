# Running E5-Mistral on Snellius

This directory contains SLURM scripts for running E5-Mistral-7B-Instruct on Snellius with A100 GPUs.

## Prerequisites

1. **Conda environment**: Make sure you have the `sigir26_repro_py39` conda environment set up on Snellius with all required packages:
   - `sentence-transformers`
   - `faiss-cpu` or `faiss-gpu` (GPU version recommended for A100)
   - `torch` with CUDA support
   - `numpy`
   - `tqdm`

2. **Data**: Ensure your QUEST data is available at:
   - `data/QUEST/documents.jsonl`
   - `data/QUEST/test_id_added.jsonl`

3. **Output directories**: The scripts will create:
   - `outputs/quest_e5_mistral_index/` (for the index)
   - `outputs/runs/` (for TREC run files)
   - `outputs/logs/` (for SLURM logs)

## Configuration

Before running, you may need to adjust the SLURM scripts:

1. **Conda path**: Update the conda source path in both scripts:
   ```bash
   source $HOME/miniconda3/etc/profile.d/conda.sh
   ```
   Or if using a different conda installation:
   ```bash
   source /path/to/your/conda/etc/profile.d/conda.sh
   ```

2. **Module loading**: Uncomment and adjust module loading if needed:
   ```bash
   module load Python/3.9.6-GCCcore-11.2.0
   module load CUDA/11.7.0
   module load cuDNN/8.4.1.50-CUDA-11.7.0
   ```

3. **Partition name**: Verify the partition name. Snellius may use:
   - `gpu` (default in scripts)
   - `gpu_shared`
   - `gpu_t4rtx`
   - Or check with: `sinfo -p gpu`

4. **GPU resource**: The scripts request `gpu:a100:1`. If your partition uses different syntax, adjust:
   - `--gres=gpu:1` (generic)
   - `--gres=gpu:a100:1` (specific A100)
   - Check available: `sinfo -o "%P %G"`

5. **Batch sizes**: The scripts use optimized batch sizes for A100:
   - Index building: `--batch_size 32` (can be increased to 64 for 80GB A100)
   - Retrieval: `--batch_size 64` (can be increased to 128 for 80GB A100)

## Usage

### Step 1: Build the Index

Submit the index building job:

```bash
cd /path/to/SIGIR_Repro_temp
sbatch code/slurm_scripts/build_e5_index_snellius.sh
```

This will:
- Request 1 A100 GPU for up to 24 hours
- Build embeddings for all documents
- Create FAISS index
- Save to `outputs/quest_e5_mistral_index/`

**Expected time**: 2-6 hours depending on corpus size and GPU model (40GB vs 80GB A100)

**Monitor progress**:
```bash
# Check job status
squeue -u $USER

# View output
tail -f outputs/logs/e5_build_index_<JOB_ID>.out

# Check GPU usage
squeue -j <JOB_ID> -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R %b"
```

### Step 2: Run Retrieval

After the index is built, submit the retrieval job:

```bash
sbatch code/slurm_scripts/retrieve_e5_snellius.sh
```

This will:
- Request 1 A100 GPU for up to 4 hours
- Load the pre-built index
- Encode queries and retrieve documents
- Save TREC run file to `outputs/runs/e5_mistral_quest.trec`

**Expected time**: 10-30 minutes depending on number of queries

## Optimizing for Speed

### For 40GB A100:
- Index building: `--batch_size 32` (default in script)
- Retrieval: `--batch_size 64` (default in script)

### For 80GB A100:
You can increase batch sizes for faster processing:
- Index building: `--batch_size 64` or `--batch_size 128`
- Retrieval: `--batch_size 128` or `--batch_size 256`

To modify, edit the scripts and change the `--batch_size` parameter in the Python command.

## Troubleshooting

### Out of Memory (OOM) errors:
- Reduce `--batch_size` in the script
- Check GPU memory: `nvidia-smi` (should show during job execution)

### Module not found errors:
- Ensure conda environment is activated correctly
- Install missing packages: `pip install sentence-transformers faiss-gpu torch`

### Index not found:
- Make sure Step 1 completed successfully
- Check `outputs/quest_e5_mistral_index/` contains:
  - `doc_ids.txt`
  - `doc_embeddings.npy`
  - `faiss_index_ip.bin`

### Job fails immediately:
- Check SLURM error log: `outputs/logs/e5_*_<JOB_ID>.err`
- Verify paths are correct for your Snellius setup
- Ensure you're submitting from the project root directory

## Checking Results

After retrieval completes, verify the output:

```bash
# Check run file was created
ls -lh outputs/runs/e5_mistral_quest.trec

# View first few lines
head outputs/runs/e5_mistral_quest.trec

# Count lines (should be ~1000 * num_queries)
wc -l outputs/runs/e5_mistral_quest.trec
```

Then evaluate using the evaluation script:

```bash
python code/evaluation/evaluate.py \
    --run outputs/runs/e5_mistral_quest.trec \
    --qrels data/QUEST/test_qrels \
    --cutoffs 1 5 10
```


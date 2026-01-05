#!/bin/bash
# Quick setup checker for E5-Mistral on Snellius

echo "=========================================="
echo "E5-Mistral Snellius Setup Checker"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "code/baselines/e5-mistral-7b-instruct/build_quest_e5_index.py" ]; then
    echo "❌ ERROR: Not in project root directory"
    echo "   Please run from SIGIR_Repro_temp directory"
    exit 1
fi
echo "✓ Project structure found"

# Check data files
if [ ! -f "data/QUEST/documents.jsonl" ]; then
    echo "❌ WARNING: data/QUEST/documents.jsonl not found"
else
    echo "✓ Documents file found"
fi

if [ ! -f "data/QUEST/test_id_added.jsonl" ]; then
    echo "❌ WARNING: data/QUEST/test_id_added.jsonl not found"
else
    echo "✓ Test queries file found"
fi

# Check conda
if command -v conda &> /dev/null; then
    echo "✓ Conda found: $(which conda)"
    
    # Try to find conda environment
    if conda env list | grep -q "sigir26_repro_py39"; then
        echo "✓ Conda environment 'sigir26_repro_py39' exists"
    else
        echo "❌ WARNING: Conda environment 'sigir26_repro_py39' not found"
        echo "   Create it with: conda env create -f environment.yml"
    fi
else
    echo "❌ WARNING: Conda not found in PATH"
fi

# Check if index already exists
if [ -d "outputs/quest_e5_mistral_index" ]; then
    if [ -f "outputs/quest_e5_mistral_index/faiss_index_ip.bin" ]; then
        echo "✓ Index already exists (can skip Step 1)"
    else
        echo "⚠ Index directory exists but incomplete"
    fi
else
    echo "ℹ Index directory does not exist (will be created)"
fi

# Check output directories
mkdir -p outputs/logs outputs/runs
echo "✓ Output directories ready"

# Check GPU availability (if on compute node)
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
    echo "ℹ nvidia-smi not available (normal if not on GPU node)"
fi

# Check SLURM (if available)
if command -v sbatch &> /dev/null; then
    echo ""
    echo "SLURM Information:"
    echo "  User: $USER"
    echo "  Available partitions:"
    sinfo -o "%P %G" 2>/dev/null | head -5 || echo "    (could not query partitions)"
else
    echo "ℹ SLURM not available (normal if not on cluster)"
fi

echo ""
echo "=========================================="
echo "Setup check complete!"
echo ""
echo "Next steps:"
echo "1. If environment missing: conda env create -f environment.yml"
echo "2. Install GPU packages: pip install sentence-transformers faiss-gpu"
echo "3. Build index: sbatch code/slurm_scripts/build_e5_index_snellius.sh"
echo "4. Run retrieval: sbatch code/slurm_scripts/retrieve_e5_snellius.sh"
echo "=========================================="


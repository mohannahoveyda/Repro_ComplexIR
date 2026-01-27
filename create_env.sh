#!/bin/bash

# Script to create conda environment from environment.yml
# Usage: ./create_env.sh

set -e  # Exit on error

ENV_FILE="environment.yml"
ENV_NAME="sigir26_repro_py39"

echo "Creating conda environment '${ENV_NAME}' from ${ENV_FILE}..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    exit 1
fi

# Check if environment.yml exists
if [ ! -f "$ENV_FILE" ]; then
    echo "Error: ${ENV_FILE} not found in current directory"
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Warning: Environment '${ENV_NAME}' already exists."
    read -p "Do you want to remove it and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n "${ENV_NAME}" -y
    else
        echo "Aborting. Environment already exists."
        exit 1
    fi
fi

# Create environment from yml file
echo "Creating environment from ${ENV_FILE}..."
conda env create -f "${ENV_FILE}"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Environment '${ENV_NAME}' created successfully!"
    echo ""
    echo "Installing gensim (requires special handling with older setuptools)..."
    # Get conda environment path and use its pip directly
    ENV_PATH=$(conda env list | grep "^${ENV_NAME} " | awk '{print $NF}')
    if [ -z "$ENV_PATH" ]; then
        # Fallback: try common conda paths
        if [ -d "$CONDA_PREFIX/envs/${ENV_NAME}" ]; then
            ENV_PATH="$CONDA_PREFIX/envs/${ENV_NAME}"
        elif [ -d "$HOME/anaconda3/envs/${ENV_NAME}" ]; then
            ENV_PATH="$HOME/anaconda3/envs/${ENV_NAME}"
        elif [ -d "$HOME/miniconda3/envs/${ENV_NAME}" ]; then
            ENV_PATH="$HOME/miniconda3/envs/${ENV_NAME}"
        fi
    fi
    
    if [ -n "$ENV_PATH" ] && [ -f "${ENV_PATH}/bin/pip" ]; then
        "${ENV_PATH}/bin/pip" install gensim==3.8.3
        PIP_CMD="${ENV_PATH}/bin/pip"
        PYTHON_CMD="${ENV_PATH}/bin/python"
        USE_ACTIVATE=false
    else
        # Fallback: use conda activate
        eval "$(conda shell.bash hook)"
        conda activate "${ENV_NAME}"
        pip install gensim==3.8.3
        PIP_CMD="pip"
        PYTHON_CMD="python"
        USE_ACTIVATE=true
    fi
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "Downloading required NLTK data..."
        "${PYTHON_CMD}" -c "import nltk; nltk.download('punkt_tab', quiet=True)" 2>/dev/null || true
        
        if [ "$USE_ACTIVATE" = true ]; then
            conda deactivate
        fi
        
        echo ""
        echo "✓ All packages installed successfully!"
        echo ""
        echo "To activate the environment, run:"
        echo "  conda activate ${ENV_NAME}"
    else
        if [ "$USE_ACTIVATE" = true ]; then
            conda deactivate
        fi
        echo ""
        echo "⚠ Environment created but gensim installation failed."
        echo "You may need to install it manually:"
        echo "  conda activate ${ENV_NAME}"
        echo "  pip install gensim==3.8.3"
    fi
else
    echo ""
    echo "✗ Failed to create environment"
    exit 1
fi


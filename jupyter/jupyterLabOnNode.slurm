#!/bin/bash

RESET="\033[0m"
RED="\033[0;31m"
GREEN="\033[0;32m"
CYAN="\033[0;36m"

IP=$(hostname -I | awk '{print $1}')
USER=${SLURM_SUBMIT_USER}
INFO=$(scontrol show node ${SLURMD_NODENAME})
JUPYTER_PORT=8888
CONDA_ENV=$1

echo
echo "--- Job: ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) ---"
echo "Partition: ${SLURM_JOB_PARTITION}"
echo
echo "Node: ${SLURMD_NODENAME} - ${IP}"
echo "---"
echo "${INFO}"
echo "---"
echo
echo -e "${RED}If you use VS Code, disable its auto‑tunnel.${RESET}"
echo "On your laptop, run:"
echo -e "${GREEN}  ssh -L${JUPYTER_PORT}:${IP}:${JUPYTER_PORT} ${USER}@ada.middlebury.edu${RESET}"
echo "Then open in browser (replace <token>):"
echo -e "${GREEN}  http://127.0.0.1:${JUPYTER_PORT}/lab?token=<token>${RESET}"
echo
echo "Starting: $(date +"%D %T")"
echo

# Source conda and activate env
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$HOME/.conda/envs/${CONDA_ENV}"

# Prepend LD_LIBRARY_PATH for Conda CUDA/NCCL
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# Start JupyterLab
cd $HOME
jupyter lab --no-browser --ip=0.0.0.0 --port=${JUPYTER_PORT}

echo
echo "Ending: $(date +"%D %T")"
echo "--- ENDED ---"
#!/bin/bash
JOB_NAME="--job-name=jupyterlab"
PARTITION="--partition=standard"
TASKS="--ntasks 1"
CPUS="--cpus-per-task 1"
MEM="--mem=24GB"
TIME="--time=2-00:00:00"
USER=$(whoami)
LOG_FOLDER="/home/${USER}/logs"
LOG="${LOG_FOLDER}/%x_%j.log"
PART_INFO=$(sinfo | sed 's/\(.*\)/    \1/')

USAGE="Usage: $(basename $0) conda_env [-n JOB_NAME] [-p PARTITION] [-m MEMORY] [-t TIME] [-h]

  -n JOB_NAME (default: jupyterlab)
    Specifies the name of the job

  -p PARTITION (default: standard)
    Specifies the partition (queue) to use. Available partitions:
${PART_INFO}

  -m MEMORY (default: 24GB)
    Specifies the max memory

  -t TIME (default: 2-00:00:00)
    Specifies the time limit (e.g. 1-00:00:00 for 1 day)

  The logs will be saved to: ${LOG}"

if [[ $1 == -* ]]; then
    echo "${USAGE}"
    echo
    echo "options should be specified after the conda_env"
    exit 1
fi

if [[ -z "$1" ]]; then
    echo "${USAGE}"
    echo
    echo "conda_env should be specified"
    exit 1
fi

CONDA_ENV=$1
shift

while getopts "n:p:m:t:h" opt; do
    case "$opt" in
        n)
            JOB_NAME="--job-name=$OPTARG"
            ;;
        p)
            arg="$OPTARG"
            if [[ $arg == *"gpu"* ]]; then
                PARTITION="--partition=${arg} --gres=gpu:1"
                CPUS="--cpus-per-task 4"
            else
                PARTITION="--partition=${arg}"
            fi
            ;;
        m)
            MEM="--mem=$OPTARG"
            ;;
        t)
            TIME="--time=$OPTARG"
            ;;
        ?|h)
            echo "${USAGE}"
            exit 1
            ;;
    esac
done
shift $((OPTIND - 1))

echo "The following command will be run:"
echo "  sbatch ${JOB_NAME} ${PARTITION} ${TASKS} ${CPUS} ${MEM} ${TIME} --output=${LOG} jupyterLabOnNode.slurm ${CONDA_ENV}"
echo
echo "The script:"
echo "- will source your conda.sh from ~/.conda or ~/miniconda3"
echo "- will activate: ~/.conda/envs/${CONDA_ENV} or ~/miniconda3/envs/${CONDA_ENV}"
echo "- will start JupyterLab"
echo
read -p "Ok? " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sbatch $JOB_NAME $PARTITION $TASKS $CPUS $MEM $TIME --output=$LOG jupyterLabOnNode.slurm $CONDA_ENV
    echo "Waiting to parse log file..."
    sleep 5
    echo -ne "$(cat $(ls -Art ${LOG_FOLDER}/jupyterlab_*.log | tail -n 1) | sed 's/\(http:\/\/127.*$\)/\\033[0;36m\1\\033[0m/g')"
fi

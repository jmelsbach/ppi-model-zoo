#!/bin/bash
#SBATCH --job-name=step-training
#SBATCH --gres=gpu:2          # Request 1 GPU
#SBATCH --cpus-per-task=4     # Request 4 CPU cores
#SBATCH --mem=64G             # Request 16GB RAM
#SBATCH --time=48:00:00       # Maximum runtime of 10 hours
#SBATCH --output=logs/step_%j.log # Standard output and error log

## SCRIPT
export CUDA_VISIBLE_DEVICES=0,1
# Default values
DEBUG=false

# Function to print the usage of the script
usage() {
    echo "Usage: $0 [--debug]"
    exit 1
}

# Parse the command line arguments
while [ "$#" -gt 0 ]; do
    case "$1" in
        -e | --debug)
            DEBUG=true # [TODO] PTL -> trainer flag
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
    shift
done

# Example of using the DEBUG variable
if [ "$DEBUG" = true ]; then
    echo "Debug mode is ON"
else
    echo "Debug mode is OFF"
fi

args=()

# standard arguments
args+=( "--trainer.precision=16-mixed" )
args+=( "--trainer.logger=WandbLogger" )
args+=( "--trainer.logger.project=protein" )
args+=( "--trainer.logger.offline=false" )
args+=( "--trainer.strategy=ddp_find_unused_parameters_true" )
args+=( "--data.data_dir=.data" )
args+=( "--data.file_name=benchmarkingGS_v1-0_similarityMeasure_sequence_v3-1.csv" )
args+=( "--data.batch_size=2" )
args+=( "--data.tokenizer=Rostlab/prot_bert_bfd" )

# arguments effected by DEBUG
if [ "$DEBUG" = true ]; then
    args+=( '--data.limit=2000' )
fi
if [ "$DEBUG" = true ]; then
    args+=( "--data.max_len=2" )
fi
[[ $DEBUG = true ]] && EPOCHS=3 || EPOCHS=10
args+=( "--trainer.max_epochs=$EPOCHS" )

python scripts/step/model_cli.py fit "${args[@]}"

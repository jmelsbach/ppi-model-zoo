#!/bin/bash
#SBATCH --job-name=step-training
#SBATCH --gres=gpu:2          # Request 1 GPU
#SBATCH --cpus-per-task=4     # Request 4 CPU cores
#SBATCH --mem=64G             # Request 16GB RAM
#SBATCH --time=48:00:00       # Maximum runtime of 10 hours
#SBATCH --output=logs/step_%j.log # Standard output and error log

## SCRIPT
#export CUDA_VISIBLE_DEVICES=0,1
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
            DEBUG=true
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

args+=( "--trainer.precision=16-mixed" ) 
args+=( "--trainer.logger=WandbLogger" )
args+=( "--trainer.logger.project=protein-rapppid" )
args+=( "--trainer.logger.offline=false" )
args+=( "--trainer.callbacks+=StochasticWeightAveraging" )
args+=( "--trainer.callbacks.swa_lrs=0.01" )
args+=( "--data.batch_size=80" )
args+=( "--data.trunc_len=1500" )
args+=( "--data.train_path=.data/comparatives/string_c3/train_pairs.pkl.gz" )
args+=( "--data.val_path=.data/comparatives/string_c3/val_pairs.pkl.gz" )
args+=( "--data.test_path=.data/comparatives/string_c3/test_pairs.pkl.gz" )
args+=( "--data.seqs_path=.data/comparatives/string_c3/seqs.pkl.gz" )
args+=( "--data.vocab_size= 250" )
args+=( "--data.model_file=.data/spm.model" )
args+=( "--data.seed=5353456" )

# arguments effected by DEBUG
if [ "$DEBUG" = true ]; then
    args+=( '--data.limit=500' )
fi
if [ "$DEBUG" = true ]; then
    args+=( "--data.max_len=2" )
fi
[[ $DEBUG = true ]] && EPOCHS=3 || EPOCHS=100
args+=( "--trainer.max_epochs=$EPOCHS" )

python scripts/rapppid/original_rapppid_model_cli.py fit "${args[@]}"

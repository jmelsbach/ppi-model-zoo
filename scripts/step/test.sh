#!/bin/bash

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

# standard arguments
args+=( "--trainer.precision=16-mixed" )
args+=( "--trainer.strategy=ddp_find_unused_parameters_true" )
args+=( "--data.data_dir=.data" )
args+=( "--data.file_name=benchmarkingGS_v1-0_similarityMeasure_sequence_v3-1.csv" )
args+=( "--data.batch_size=16" )
args+=( "--data.tokenizer=Rostlab/prot_bert_bfd" )

args+=( "--ckpt_path=./protein/smxuznts/checkpoints/epoch=9-step=219840.ckpt")

# arguments effected by DEBUG
if [ "$DEBUG" = true ]; then
    args+=( '--data.limit=10000' )
fi
if [ "$DEBUG" = true ]; then
    args+=( "--data.max_len=2" )
fi

python scripts/step/model_cli.py test "${args[@]}"

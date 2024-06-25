python model_cli.py fit \
 --trainer.max_epochs=2 \
 --trainer.strategy=ddp_find_unused_parameters_true \
 --data.data_dir="../../.data/benchmarkingGS_v1-0_similarityMeasure_sequence_v3-1.csv" \
 --data.batch_size=2 \
 --data.tokenizer="Rostlab/prot_bert_bfd" \
 --data.max_len=1024
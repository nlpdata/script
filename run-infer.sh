#!/bin/bash

export OUTPUT_DIR=output
export BERT_DIR=roberta_wwm_ext_large
export SHARED_DATA=data/c3_release



cp -r $OUTPUT_DIR/roberta_wwm_ext_large_ibc3_1epoch_666 $OUTPUT_DIR/roberta_wwm_ext_large_ibc3_1epoch_666infer
cp -r $OUTPUT_DIR/roberta_wwm_ext_large_gbc3_1epoch_666 $OUTPUT_DIR/roberta_wwm_ext_large_gbc3_1epoch_666infer
cp -r $OUTPUT_DIR/roberta_wwm_ext_large_lbc3_1epoch_666 $OUTPUT_DIR/roberta_wwm_ext_large_lbc3_1epoch_666infer
cp -r $OUTPUT_DIR/roberta_wwm_ext_large_ctc3_1epoch_666 $OUTPUT_DIR/roberta_wwm_ext_large_ctc3_1epoch_666infer



export INPUT_DATA=data/cn/lb_infer

python3 run_classifier_teacher.py --task_name infer --data_dir $INPUT_DATA --train_batch_size 24 --eval_batch_size 64 --num_train_epochs 1.0 --gradient_accumulation_steps 6 --bert_config_file $BERT_DIR/config.json  --vocab_file $BERT_DIR/vocab.txt --shared_data_dir $SHARED_DATA --output_dir $OUTPUT_DIR/roberta_wwm_ext_large_lbc3_1epoch_666infer/ --init_checkpoint $BERT_DIR/pytorch_model.bin  --do_eval --seed 888 --gpu_ids 4,5,6,7 --resume



export INPUT_DATA=data/cn/ct_infer

python3 run_classifier_teacher.py --task_name infer --data_dir $INPUT_DATA --train_batch_size 24 --eval_batch_size 64 --num_train_epochs 1.0 --gradient_accumulation_steps 6 --bert_config_file $BERT_DIR/config.json  --vocab_file $BERT_DIR/vocab.txt --shared_data_dir $SHARED_DATA --output_dir $OUTPUT_DIR/roberta_wwm_ext_large_ctc3_1epoch_666infer/ --init_checkpoint $BERT_DIR/pytorch_model.bin  --do_eval --seed 888 --gpu_ids 4,5,6,7 --resume



export INPUT_DATA=data/cn/ib_infer

python3 run_classifier_teacher.py --task_name infer --data_dir $INPUT_DATA --train_batch_size 24 --eval_batch_size 64 --num_train_epochs 1.0 --gradient_accumulation_steps 6 --bert_config_file $BERT_DIR/config.json  --vocab_file $BERT_DIR/vocab.txt --shared_data_dir $SHARED_DATA --output_dir $OUTPUT_DIR/roberta_wwm_ext_large_ibc3_1epoch_666infer/ --init_checkpoint $BERT_DIR/pytorch_model.bin  --do_eval --seed 888 --gpu_ids 4,5,6,7 --resume



export INPUT_DATA=data/cn/gb_infer

python3 run_classifier_teacher.py --task_name infer --data_dir $INPUT_DATA --train_batch_size 24 --eval_batch_size 64 --num_train_epochs 1.0 --gradient_accumulation_steps 6 --bert_config_file $BERT_DIR/config.json  --vocab_file $BERT_DIR/vocab.txt --shared_data_dir $SHARED_DATA --output_dir $OUTPUT_DIR/roberta_wwm_ext_large_gbc3_1epoch_666infer/ --init_checkpoint $BERT_DIR/pytorch_model.bin  --do_eval --seed 334 --gpu_ids 4,5,6,7 --resume



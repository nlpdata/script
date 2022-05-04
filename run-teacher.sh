#!/bin/bash

export OUTPUT_DIR=output
export BERT_DIR=roberta_wwm_ext_large
export SHARED_DATA=data/c3_release

export INPUT_DATA=data/cn/ct
cp -r data/cn/ct data/cn/ct_infer

python3 run_classifier_teacher.py --task_name on --data_dir $INPUT_DATA --train_batch_size 24 --eval_batch_size 64 --num_train_epochs 1.0 --gradient_accumulation_steps 6 --bert_config_file $BERT_DIR/config.json  --vocab_file $BERT_DIR/vocab.txt  --shared_data_dir $SHARED_DATA --output_dir $OUTPUT_DIR/roberta_wwm_ext_large_ctc3_1epoch_666/ --init_checkpoint $BERT_DIR/pytorch_model.bin  --do_train --do_eval --seed 666 --gpu_ids 4,5,6,7
rm $OUTPUT_DIR/roberta_wwm_ext_large_ctc3_1epoch_666/model_best.pt




export INPUT_DATA=data/cn/lb
cp -r data/cn/lb data/cn/lb_infer

python3 run_classifier_teacher.py --task_name on --data_dir $INPUT_DATA --train_batch_size 24 --eval_batch_size 64 --num_train_epochs 1.0 --gradient_accumulation_steps 6 --bert_config_file $BERT_DIR/config.json  --vocab_file $BERT_DIR/vocab.txt  --shared_data_dir $SHARED_DATA --output_dir $OUTPUT_DIR/roberta_wwm_ext_large_lbc3_1epoch_666/ --init_checkpoint $BERT_DIR/pytorch_model.bin  --do_train --do_eval --seed 666 --gpu_ids 4,5,6,7
rm $OUTPUT_DIR/roberta_wwm_ext_large_lbc3_1epoch_666/model_best.pt




export INPUT_DATA=data/cn/gb
cp -r data/cn/gb data/cn/gb_infer

python3 run_classifier_teacher.py --task_name on --data_dir $INPUT_DATA --train_batch_size 24 --eval_batch_size 64 --num_train_epochs 1.0 --gradient_accumulation_steps 6 --bert_config_file $BERT_DIR/config.json  --vocab_file $BERT_DIR/vocab.txt  --shared_data_dir $SHARED_DATA --output_dir $OUTPUT_DIR/roberta_wwm_ext_large_gbc3_1epoch_666/ --init_checkpoint $BERT_DIR/pytorch_model.bin  --do_train --do_eval --seed 666 --gpu_ids 4,5,6,7
rm $OUTPUT_DIR/roberta_wwm_ext_large_gbc3_1epoch_666/model_best.pt



export INPUT_DATA=data/cn/ib
cp -r data/cn/ib data/cn/ib_infer

python3 run_classifier_teacher.py --task_name on --data_dir $INPUT_DATA --train_batch_size 24 --eval_batch_size 64 --num_train_epochs 1.0 --gradient_accumulation_steps 6 --bert_config_file $BERT_DIR/config.json  --vocab_file $BERT_DIR/vocab.txt  --shared_data_dir $SHARED_DATA --output_dir $OUTPUT_DIR/roberta_wwm_ext_large_ibc3_1epoch_666/ --init_checkpoint $BERT_DIR/pytorch_model.bin  --do_train --do_eval --seed 666 --gpu_ids 4,5,6,7
rm $OUTPUT_DIR/roberta_wwm_ext_large_ibc3_1epoch_666/model_best.pt



#!/bin/bash

export OUTPUT_DIR=output
export BERT_DIR=roberta_wwm_ext_large
export SHARED_DATA=data/c3_release


cp -r $OUTPUT_DIR/roberta_wwm_ext_large_3btc3_5-15-kl0.5_1epoch_666/ $OUTPUT_DIR/roberta_wwm_ext_large_3btc3_5-15-kl0.5_1epoch_666_8epoch_kl0.5-1s_666/


export INPUT_DATA=data/c3_soft_acl


python3 run_classifier_onkl_teacher_student.py --task_name on --data_dir $INPUT_DATA --train_batch_size 24 --eval_batch_size 64 --num_train_epochs 8.0 --gradient_accumulation_steps 6 --bert_config_file $BERT_DIR/config.json  --vocab_file $BERT_DIR/vocab.txt   --shared_data_dir $SHARED_DATA --output_dir $OUTPUT_DIR/roberta_wwm_ext_large_3btc3_5-15-kl0.5_1epoch_666_8epoch_kl0.5-1s_666/ --init_checkpoint $BERT_DIR/pytorch_model.bin  --do_train --do_eval --seed 666 --gpu_ids 4,5,6,7 --resume
rm $OUTPUT_DIR/roberta_wwm_ext_large_3btc3_5-15-kl0.5_1epoch_666_8epoch_kl0.5-1s_666/model_best.pt


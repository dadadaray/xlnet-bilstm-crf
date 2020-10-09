#!/usr/bin/env bash
  #CUDA_VISIBLE_DEVICES=0\
  python XLNET_NER_wnut17.py\
    --task_name="NER"  \
    --do_lower_case=False \
    --crf=False \
    --do_train=True   \
    --do_eval=True   \
    --do_predict=True \
    --do_export=False \
    --data_dir=data/UNUT17_data   \
    --vocab_file=xlnet_cased_L-12_H-768_A-12/spiece.model  \
    --xlnet_config_file=xlnet_cased_L-12_H-768_A-12/xlnet_config.json \
    --init_checkpoint=xlnet_cased_L-12_H-768_A-12/xlnet_model.ckpt   \
    --max_seq_length=128   \
    --train_batch_size=32   \
    --eval_batch_size=16 \
    --predict_batch_size=8 \
    --learning_rate=2e-5   \
    --num_train_epochs=80   \
    --use_bfloat16=False  \
    --train_steps=100000\
    --min_lr_ratio=0.0\
    --adam_epsilon=1e-8\
    --model_dir=/home/Extra/yanrongen/python_code/XLNET-BILSTM-master/output/model1\
    --export_dir=/home/Extra/yanrongen/python_code/XLNET-BILSTM-master/output/export1\
    --output_dir=/home/Extra/yanrongen/python_code/XLNET-BILSTM-master/output/result_dir1\
    --prefile=/home/Extra/yanrongen/python_code/XLNET-BILSTM-master/output/result_dir1\









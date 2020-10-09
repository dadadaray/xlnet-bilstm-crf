#!/usr/bin/env bash

  python XLNET_NER.py\
    --task_name="NER"  \
    --do_lower_case=False \
    --crf=False \
    --do_train=True   \
    --do_eval=True   \
    --do_predict=True \
    --do_export=False \
    --data_dir=data   \
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
    --model_dir=./output/model\
    --export_dir=./output/export\
    --output_dir=./output/result_dir\
    --prefile=./output/result_dir\








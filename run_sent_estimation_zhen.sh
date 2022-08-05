export CUDA_VISIBLE_DEVICES=0

LABEL_SUFFIX=score
accelerate launch --fp16 run_quality_estimation.py \
 --do_train \
 --data_dir ./data/zh-en/ \
 --model_type xlmr \
 --model_path ./xlm-roberta-base \
 --output_dir ./QE_outputs/zh-en \
 --batch_size 8 \
 --learning_rate 1e-5 \
 --max_epoch 20 \
 --valid_steps 500 \
 --train_type sent \
 --valid_type sent \
 --sentlab_suffix $LABEL_SUFFIX \
 --stop_criterion 10 \
 --best_metric spearmanr \
 --pad_to_max_length \
 --max_length 200 \
 --contrast_time 20 \
 --overwrite_output_dir \
 --overwrite_cache \
 --do_contrast \
 --only_contrast

# INFER_PREFIX=test20
# python run_quality_estimation.py \
#  --do_infer \
#  --do_partial_prediction \
#  --suffix_a mt \
#  --data_dir $DATA/original-data \
#  --infer_prefix $INFER_PREFIX \
#  --model_type bert \
#  --model_path ./QE_outputs/$DATA-sent-$SUFFIX/best_pearson \
#  --batch_size 16 \
#  --infer_type sent \

# python eval_sentence_level.py $DATA/original-data/$INFER_PREFIX.sent $DATA/original-data/$INFER_PREFIX.$LABEL_SUFFIX -v
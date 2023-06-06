export CUDA_VISIBLE_DEVICES=0

LABEL_SUFFIX=score
DATA_DIR=./data/TASK1-ende
OUTPUT_DIR=./QE_outputs/TASK1-ende
# accelerate launch --mixed_precision fp16 run_quality_estimation.py \
#  --do_train \
#  --data_dir $DATA_DIR \
#  --model_type xlmr \
#  --model_path ./xlm-roberta-base \
#  --output_dir $OUTPUT_DIR \
#  --batch_size 8 \
#  --learning_rate 1e-5 \
#  --max_epoch 20 \
#  --valid_steps 500 \
#  --train_type sent \
#  --valid_type sent \
#  --sentlab_suffix $LABEL_SUFFIX \
#  --stop_criterion 10 \
#  --best_metric pearson \
#  --pad_to_max_length \
#  --max_length 200 \
#  --contrast_time 20 \
#  --overwrite_output_dir \
#  --overwrite_cache \
#  --do_contrast

INFER_PREFIX=test21
python run_quality_estimation.py \
 --do_infer \
 --data_dir $DATA_DIR \
 --infer_prefix $INFER_PREFIX \
 --model_type xlmr \
 --model_path $OUTPUT_DIR/best_pearson \
 --batch_size 16 \
 --infer_type sent \
 --do_contrast

python eval_sentence_level.py $DATA_DIR/$INFER_PREFIX.sent $DATA_DIR/$INFER_PREFIX.$LABEL_SUFFIX -v
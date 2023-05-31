export CUDA_VISIBLE_DEVICES=1

LABEL_SUFFIX=score
accelerate launch --fp16 run_quality_estimation.py \
 --do_train \
 --data_dir ./data/en-de/ \
 --model_type xlmr \
 --model_path ./xlm-roberta-base \
 --output_dir ./QE_outputs/en-de \
 --batch_size 8 \
 --learning_rate 1e-5 \
 --max_epoch 20 \
 --valid_steps 500 \
 --train_type sent \
 --valid_type sent \
 --sentlab_suffix $LABEL_SUFFIX \
 --stop_criterion 10 \
 --best_metric pearson \
 --pad_to_max_length \
 --max_length 200 \
 --contrast_time 20 \
 --overwrite_output_dir \
 --overwrite_cache \
 --do_contrast \
 --only_contrast

INFER_PREFIX=dev
python run_quality_estimation.py \
 --do_infer \
 --data_dir ./data/en-de/ \
 --infer_prefix $INFER_PREFIX \
 --model_type xlmr \
 --model_path ./QE_outputs/en-de/best_pearson \
 --batch_size 16 \
 --infer_type sent \
 --do_contrast \
 --only_contrast

python eval_sentence_level.py ./data/en-de/$INFER_PREFIX.sent ./data/en-de/$INFER_PREFIX.$LABEL_SUFFIX -v
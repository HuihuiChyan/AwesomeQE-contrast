# AwesomeQE-contrast

This is the official repository for paper **Improving Translation Quality Estimation with Bias Mitigation**, accepted by ACL2023.

If you have any quesions, you can contact me with Wechat huanghui20200708.

# ⚡️ Usage
## Environment
Please refer to the following settings to prepare your environment:
- python 3.10.9
- transformers 4.28.1
- sacremoses 0.0.53

We recomment you to use conda to create a independent environment.

```shell
conda create -n awesomeqe python=3.10 -y
conda activate awesomeqe
```
## Negative Sample Generation
Please download the pre-trained model for negative sample generation in the current directory.

For example, you can download [bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased) and put it in the current directory to create negative samples:
```shell
export CUDA_VISIBLE_DEVICES=0

INPUT=train.src
INPUT_B=train.pe
MODEL_TYPE=mBERT
MODEL_PATH=bert-base-multilingual-cased
UPPER_MASK_RATIO=50
LOWER_MASK_RATIO=10
UPPER_INDE_RATIO=20
LOWER_INDE_RATIO=05
RANDOM_RATIO=50
NOISED=noised
OUTPUT=output

if [ ! -d $DATA_DIR ]; then
	mkdir $DATA_DIR
fi

python generate_synthetic_data.py \
	--model_path $MODEL_PATH \
	--model_type $MODEL_TYPE \
	--batch_size 32 \
	--input_file ./data/TASK1-enzh/$INPUT \
	--input_file_b ./data/TASK1-enzh/$INPUT_B \
	--noised_file ./data/TASK1-enzh/$INPUT_B.$NOISED \
	--output_file ./data/TASK1-enzh/$INPUT_B.$OUTPUT \
	--max_length 128 \
	--negative_time 30 \
	--upper_mask_ratio 0."$UPPER_MASK_RATIO" \
	--lower_mask_ratio 0."$LOWER_MASK_RATIO" \
	--upper_inde_ratio 0."$UPPER_INDE_RATIO" \
	--lower_inde_ratio 0."$LOWER_INDE_RATIO" \
	--random_ratio 0."$RANDOM_RATIO"
```

## Multi-task Contrastive Learning
With the previous created negative samples, you can perform contrastive learning with the following scripts:

```shell
export CUDA_VISIBLE_DEVICES=0

LABEL_SUFFIX=score
DATA_DIR=./data/TASK1-ende
OUTPUT_DIR=./QE_outputs/TASK1-ende
accelerate launch --mixed_precision fp16 run_quality_estimation.py \
 --do_train \
 --data_dir $DATA_DIR \
 --model_type xlmr \
 --model_path ./xlm-roberta-base \
 --output_dir $OUTPUT_DIR \
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
 --contrast_suffix $CONTRAST_SUFFIX

INFER_PREFIX=test20
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

```
# 💬 Citation
If you find our work is helpful, please cite as:

```
@inproceedings{bo2021xtune,
author = {Hui Huang, Shuangzhi Wu, Kehai Chen, Hui Di, Muyun Yang, Tiejun Zhao},
booktitle = {Proceedings of ACL 2023},
title = {{Improving Translation Quality Estimation with Bias Mitigation}},
year = {2023}
}
```
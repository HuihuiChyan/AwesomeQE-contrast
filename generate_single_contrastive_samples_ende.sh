export CUDA_VISIBLE_DEVICES=0

INPUT=train.pe
MODEL_TYPE=mBERT
MODEL_PATH=dbmdz-bert-base-german-cased
UPPER_MASK_RATIO=30
LOWER_MASK_RATIO=05
UPPER_INDE_RATIO=10
LOWER_INDE_RATIO=02
RANDOM_RATIO=50
NOISED=$MODEL_PATH"-single-"$UPPER_MASK_RATIO$LOWER_MASK_RATIO"-"$UPPER_INDE_RATIO$LOWER_INDE_RATIO"-"$RANDOM_RATIO".noised"
OUTPUT=$MODEL_PATH"-single-"$UPPER_MASK_RATIO$LOWER_MASK_RATIO"-"$UPPER_INDE_RATIO$LOWER_INDE_RATIO"-"$RANDOM_RATIO".output"

if [ ! -d $DATA_DIR ]; then
	mkdir $DATA_DIR
fi

# 后缀命名规则： model.type.noise_ratio
python generate_synthetic_data.py \
	--model_path $MODEL_PATH \
	--model_type $MODEL_TYPE \
	--batch_size 32 \
	--input_file ./data/TASK1-ende/$INPUT \
	--noised_file ./data/TASK1-ende/$INPUT.$NOISED \
	--output_file ./data/TASK1-ende/$INPUT.$OUTPUT \
	--max_length 128 \
	--negative_time 30 \
	--upper_mask_ratio 0."$UPPER_MASK_RATIO" \
	--lower_mask_ratio 0."$LOWER_MASK_RATIO" \
	--upper_inde_ratio 0."$UPPER_INDE_RATIO" \
	--lower_inde_ratio 0."$LOWER_INDE_RATIO" \
	--random_ratio 0."$RANDOM_RATIO"
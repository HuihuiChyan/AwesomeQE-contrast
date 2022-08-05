export CUDA_VISIBLE_DEVICES=0

INPUT=train.ref.dedup
MODEL_TYPE=BART
MODEL_PATH=bart-base
MASK_RATIO=0.35
INDE_RATIO=0.10
RANDOM_RATIO=0.50
NOISED=$MODEL_PATH"-single-"$MASK_RATIO"-"$INDE_RATIO"-"$RANDOM_RATIO".noised"
OUTPUT=$MODEL_PATH"-single-"$MASK_RATIO"-"$INDE_RATIO"-"$RANDOM_RATIO".output"

if [ ! -d $DATA_DIR ]; then
	mkdir $DATA_DIR
fi

# 后缀命名规则： model.type.noise_ratio
python generate_synthetic_data.py \
	--model_path $MODEL_PATH \
	--model_type $MODEL_TYPE \
	--batch_size 32 \
	--input_file ./data/zh-en/$INPUT \
	--noised_file ./data/zh-en/$INPUT.$NOISED \
	--output_file ./data/zh-en/$INPUT.$OUTPUT \
	--max_length 128 \
	--negative_time 30 \
	--vocab_lower_bound 4 \
	--mask_ratio $MASK_RATIO \
	--inde_ratio $INDE_RATIO \
	--random_ratio $RANDOM_RATIO
export MODEL_NAME="/root/autodl-tmp/stable-diffusion-v1-4"
export DATA_DIR="./dataset/elephant"

python ti2.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<R-ele>" \
  --initializer_token="elephant" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3600 \
  --save_steps 1200 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="textual_inversion_rele_plain" \
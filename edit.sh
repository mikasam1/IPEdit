# clean GPU resources
echo "Checking and killing python GPU processes..."
PIDS=$(nvidia-smi | grep python | awk '{print $5}' | grep -E '^[0-9]+$')
if [ -n "$PIDS" ]; then
  echo "Killing: $PIDS"
  kill -9 $PIDS
else
  echo "No GPU python processes to kill."
fi

# clear CUDA memory cache
python -c "import torch; torch.cuda.empty_cache()"

# avoid memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
 
# python edit.py \
#   --prompts "a smiling dog on the sofa" "a smiling dog on the sofa" \
#   --mode reweight \
#   --eq_words "smiling" \
#   --equalizer_vals -3.0 \
#   --blend_words "dog" "dog" \
#   --cross_steps 0.4 \
#   --self_steps 0.4 \
#   --seed 88488 \

# python edit.py \
#   --prompts "a cat in the yard" "a <round-bird> in the yard" \
#   --mode replace \
#   --eq_words "<round-bird>" \
#   --equalizer_vals 1.0 \
#   --blend_words "cat" "<round-bird>" \
#   --cross_steps 0.4 \
#   --self_steps 0.4 \
#   --seed 2222 \
#   --embeds "obj_bird3000.safetensors" \

# python ptpedit.py \
#   --prompts "a cat in the yard" "a <round-bird> in the yard" \
#   --mode replace \
#   --eq_words "<round-bird>" \
#   --equalizer_vals 1.0 \
#   --blend_words "cat" "<round-bird>" \
#   --cross_steps 0.4 \
#   --self_steps 0.4 \
#   --seed 2222 \
#   --embeds "obj_bird3000.safetensors" \

python ntiedit.py \
--inv_prompt "a cat sitting next to a mirror" \
--image_path "NTIdataset/gnochi_mirror.jpeg" \
--prompts "a cat sitting next to a mirror" "a dog sitting next to a mirror" \
--mode replace \
--eq_words "dog" \
--equalizer_vals 2.0 \
--blend_words "cat" "dog" \
--cross_steps 0.5 \
--self_steps 0.4 \
--embeds "obj_bird3000.safetensors" \


# CUDA_VISIBLE_DEVICES=2 python inference_dreambooth.py \
#   --model_path "dreambooth-kl-outputs/monster_toy_kl_weight_4/checkpoints/00800" \
#   --prompt_file_list eval_prompt_list/dreambooth-toy.yaml \
#   --output_path "dreambooth-kl-outputs/monster_toy_kl_weight_4/checkpoints/00800/eval" \
#   --seed_list 0 1 2 \
#   # --overwrite`


# CUDA_VISIBLE_DEVICES=2 python inference_dreambooth.py \
#   --model_path "dreambooth-kl-outputs/dog5" \
#   --prompt_file_list eval_prompt_list/dreambooth-dog5.yaml \
#   --output_path "dreambooth-kl-outputs/dog5/eval" \
#   --seed_list 0 1 2 \
#   # --overwrite`


# CUDA_VISIBLE_DEVICES=2 python inference_dreambooth.py \
#   --model_path "dreambooth-kl-outputs/grey_sloth_plushie_non_prompt_weight_10" \
#   --prompt_file_list eval_prompt_list/dreambooth-toy.yaml \
#   --output_path "dreambooth-kl-outputs/grey_sloth_plushie_non_prompt_weight_10/eval" \
#   --seed_list 0 1 2 \
#   # --overwrite`

# CUDA_VISIBLE_DEVICES=2 python inference_dreambooth.py \
#   --model_path "dreambooth-kl-outputs/dog_non_prompt_weight_10" \
#   --prompt_file_list eval_prompt_list/dreambooth-dog5.yaml \
#   --output_path "dreambooth-kl-outputs/dog_non_prompt_weight_10/eval" \
#   --seed_list 0 1 2 \
#   # --overwrite`


CUDA_VISIBLE_DEVICES=2 python inference_dreambooth.py \
  --model_path "dreambooth-kl-outputs/lannie_null_prompt_weight_5" \
  --prompt_file_list eval_prompt_list/dreambooth-cat.yaml \
  --output_path "dreambooth-kl-outputs/lannie_null_prompt_weight_5/eval" \
  --seed_list 0 1 2 \
  # --overwrite`





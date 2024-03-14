

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


# CUDA_VISIBLE_DEVICES=6 python inference_dreambooth.py \
#   --model_path "dreambooth-kl-outputs/monster_toy_non_prompt_weight_10_ref_as_loss" \
#   --prompt_file_list eval_prompt_list/dreambooth-toy.yaml \
#   --output_path "dreambooth-kl-outputs/monster_toy_non_prompt_weight_10_ref_as_loss/eval" \
#   --seed_list 0 1 2 \
#   # --overwrite`




# CUDA_VISIBLE_DEVICES=2 python inference_dreambooth.py \
#   --model_path "dreambooth-kl-outputs/grey_sloth_plushie_non_prompt_weight_10" \
#   --prompt_file_list eval_prompt_list/dreambooth-toy.yaml \
#   --output_path "dreambooth-kl-outputs/grey_sloth_plushie_non_prompt_weight_10/eval" \
#   --seed_list 0 1 2 \
#   # --overwrite`



CUDA_VISIBLE_DEVICES=3 python inference_dreambooth.py \
    --model_path "dreambooth-kl-outputs/grey_sloth_plushie_non_prompt_weight_10_non_class_suffix" \
    --prompt_file_list eval_prompt_list/dreambooth-toy.yaml \
    --output_path "dreambooth-kl-outputs/grey_sloth_plushie_non_prompt_weight_10_non_class_suffix/eval" \
    --seed_list 0 1 2 \
    # --overwrite`


# CUDA_VISIBLE_DEVICES=2 python inference_dreambooth.py \
#     --model_path "dreambooth-kl-outputs/grey_sloth_plushie_non_prompt_weight_10_importance_sampling_non_class_suffix_steer_weight_1" \
#     --prompt_file_list eval_prompt_list/dreambooth-toy.yaml \
#     --output_path "dreambooth-kl-outputs/grey_sloth_plushie_non_prompt_weight_10_importance_sampling_non_class_suffix_steer_weight_1/eval" \
#     --seed_list 0 1 2 \
#     # --overwrite`
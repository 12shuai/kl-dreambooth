# CUDA_VISIBLE_DEVICES=1 python inference.py \
#   --model_path "outputs/watch-no-asset" \
#   --prompt_file_list eval_prompt_list/watch.yaml \
#   --output_path "outputs/watch-no-asset/eval" \
#   --seed_list 0 1 2 \
#   # --overwrite


# CUDA_VISIBLE_DEVICES=1 python inference.py \
#   --model_path "outputs/watch-no-asset-prior-preserve" \
#   --prompt_file_list eval_prompt_list/watch.yaml \
#   --output_path "outputs/watch-no-asset-prior-preserve/eval" \
#   --seed_list 0 1 2 \
#   # --overwrite

# CUDA_VISIBLE_DEVICES=1 python inference.py \
#   --model_path "outputs/watch-init-prior-preserve" \
#   --prompt_file_list eval_prompt_list/watch.yaml \
#   --output_path "outputs/watch-init-prior-preserve/eval" \
#   --seed_list 0 1 2 \
#   # --overwrite



# CUDA_VISIBLE_DEVICES=1 python inference_dreambooth.py \
#   --model_path "dreambooth-self-play-outputs/dog6_interval50_gen_num_1_scale_dpo_1e7_diff2weight_0.1" \
#   --prompt_file_list eval_prompt_list/dreambooth-dog5.yaml \
#   --output_path "dreambooth-self-play-outputs/dog6_interval50_gen_num_1_scale_dpo_1e7_diff2weight_0.1/eval" \
#   --seed_list 0 1 2 \
#   # --overwrite




# CUDA_VISIBLE_DEVICES=2 python inference_dreambooth.py \
#   --model_path "dreambooth-self-play-outputs/dog5_scale_max_debug4_use_ppo_interval10_use_approximate" \
#   --prompt_file_list eval_prompt_list/dreambooth-dog5.yaml \
#   --output_path "dreambooth-self-play-outputs/dog5_scale_max_debug4_use_ppo_interval10_use_approximate/eval" \
#   --seed_list 0 1 2 \
#   # --overwrite


# CUDA_VISIBLE_DEVICES=1 python inference_dreambooth.py \
#   --model_path "dreambooth-self-play-outputs/cat2_interval50_gen_num_1_scale_dpo_1e7_diff2weight_0.1" \
#   --prompt_file_list eval_prompt_list/dreambooth-cat.yaml \
#   --output_path "dreambooth-self-play-outputs/cat2_interval50_gen_num_1_scale_dpo_1e7_diff2weight_0.1/eval" \
#   --seed_list 0 1 2 \
#   # --overwrite

CUDA_VISIBLE_DEVICES=5 python inference_dreambooth.py \
  --model_path "dreambooth-self-play-outputs/monster_toy_scale_max_debug4_use_ppo_1e-2_interval10_use_approximate_null_prompt" \
  --prompt_file_list eval_prompt_list/dreambooth-toy.yaml \
  --output_path "dreambooth-self-play-outputs/monster_toy_scale_max_debug4_use_ppo_1e-2_interval10_use_approximate_null_prompt/eval" \
  --seed_list 0 1 2 \
  # --overwrite

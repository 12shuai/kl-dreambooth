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


# CUDA_VISIBLE_DEVICES=2 python inference_dreambooth.py \
#   --model_path "dreambooth-outputs/dog5-non-pp-text-sofa" \
#   --prompt_file_list eval_prompt_list/dreambooth-dog5.yaml \
#   --output_path "dreambooth-outputs/dog5-non-pp-text-sofa/eval" \
#   --seed_list 0 1 2 \
#   # --overwrite


# CUDA_VISIBLE_DEVICES=2 python inference_dreambooth.py \
#   --model_path "dreambooth-outputs/dog5-pp-text-sofa" \
#   --prompt_file_list eval_prompt_list/dreambooth-dog5.yaml \
#   --output_path "dreambooth-outputs/dog5-pp-text-sofa/eval" \
#   --seed_list 0 1 2 \
#   # --overwrite

# CUDA_VISIBLE_DEVICES=3 python inference_dreambooth.py \
#   --model_path "dreambooth-outputs/dog" \
#   --prompt_file_list eval_prompt_list/dreambooth-dog5.yaml \
#   --output_path "dreambooth-outputs/dog/eval" \
#   --seed_list 0 1 2 \
#   # --overwrite




# CUDA_VISIBLE_DEVICES=2 python inference_dreambooth.py \
#   --model_path "dreambooth-outputs/cat2-pp" \
#   --prompt_file_list eval_prompt_list/dreambooth-cat.yaml \
#   --output_path "dreambooth-outputs/cat2-pp/eval" \
#   --seed_list 0 1 2 \
#   # --overwrite



# CUDA_VISIBLE_DEVICES=2 python inference_dreambooth.py \
#   --model_path "dreambooth-outputs/cat2-non-pp" \
#   --prompt_file_list eval_prompt_list/dreambooth-cat.yaml \
#   --output_path "dreambooth-outputs/cat2-non-pp/eval" \
#   --seed_list 0 1 2 \
#   # --overwrite


# CUDA_VISIBLE_DEVICES=2 python inference_dreambooth.py \
#   --model_path "dreambooth-outputs/dog-pp" \
#   --prompt_file_list eval_prompt_list/dreambooth-dog5.yaml \
#   --output_path "dreambooth-outputs/dog-pp/eval" \
#   --seed_list 0 1 2 \
#   # --overwrite



# CUDA_VISIBLE_DEVICES=2 python inference_dreambooth.py \
#   --model_path "dreambooth-outputs/dog-non-pp" \
#   --prompt_file_list eval_prompt_list/dreambooth-dog5.yaml \
#   --output_path "dreambooth-outputs/dog-non-pp/eval" \
#   --seed_list 0 1 2 \
#   # --overwrite



# CUDA_VISIBLE_DEVICES=2 python inference_dreambooth.py \
#   --model_path "dreambooth-outputs/grey_sloth_plushie-pp" \
#   --prompt_file_list eval_prompt_list/dreambooth-toy.yaml \
#   --output_path "dreambooth-outputs/grey_sloth_plushie-pp/eval" \
#   --seed_list 0 1 2 \
#   # --overwrite`



# # CUDA_VISIBLE_DEVICES=2 python inference_dreambooth.py \
# #   --model_path "dreambooth-outputs/grey_sloth_plushie-non-pp" \
# #   --prompt_file_list eval_prompt_list/dreambooth-toy.yaml \
# #   --output_path "dreambooth-outputs/grey_sloth_plushie-non-pp/eval" \
# #   --seed_list 0 1 2 \
# #   # --overwrite



# # CUDA_VISIBLE_DEVICES=2 python inference_dreambooth.py \
# #   --model_path "dreambooth-outputs/grobot_toy-pp" \
# #   --prompt_file_list eval_prompt_list/dreambooth-toy.yaml \
# #   --output_path "dreambooth-outputs/grobot_toy-pp/eval" \
# #   --seed_list 0 1 2 \
# #   # --overwrite`



# # CUDA_VISIBLE_DEVICES=2 python inference_dreambooth.py \
# #   --model_path "dreambooth-outputs/grobot_toy-non-pp" \
# #   --prompt_file_list eval_prompt_list/dreambooth-toy.yaml \
# #   --output_path "dreambooth-outputs/grobot_toy-non-pp/eval" \
# #   --seed_list 0 1 2 \
# #   # --overwrite




# CUDA_VISIBLE_DEVICES=2 python inference_dreambooth.py \
#   --model_path "dreambooth-outputs/robot_toy-pp" \
#   --prompt_file_list eval_prompt_list/dreambooth-toy.yaml \
#   --output_path "dreambooth-outputs/robot_toy-pp/eval" \
#   --seed_list 0 1 2 \
#   # --overwrite`



# # CUDA_VISIBLE_DEVICES=2 python inference_dreambooth.py \
# #   --model_path "dreambooth-outputs/robot_toy-non-pp" \
# #   --prompt_file_list eval_prompt_list/dreambooth-toy.yaml \
# #   --output_path "dreambooth-outputs/robot_toy-non-pp/eval" \
# #   --seed_list 0 1 2 \
# #   # --overwrite



# CUDA_VISIBLE_DEVICES=2 python inference_dreambooth.py \
#   --model_path "dreambooth-outputs/monster_toy-pp" \
#   --prompt_file_list eval_prompt_list/dreambooth-toy.yaml \
#   --output_path "dreambooth-outputs/monster_toy-pp/eval" \
#   --seed_list 0 1 2 \
#   # --overwrite`



# CUDA_VISIBLE_DEVICES=2 python inference_dreambooth.py \
#   --model_path "dreambooth-outputs/monster_toy-non-pp" \
#   --prompt_file_list eval_prompt_list/dreambooth-toy.yaml \
#   --output_path "dreambooth-outputs/monster_toy-non-pp/eval" \
#   --seed_list 0 1 2 \
#   # --overwrite



CUDA_VISIBLE_DEVICES=2 python inference_dreambooth.py \
  --model_path "dreambooth-kl-outputs/grey_sloth_plushie_non_prompt_weight_10" \
  --prompt_file_list eval_prompt_list/dreambooth-dog5.yaml \
  --output_path "dreambooth-kl-outputs/grey_sloth_plushie_non_prompt_weight_10/eval-dog" \
  --seed_list 0 1 2 \
  # --overwrite`



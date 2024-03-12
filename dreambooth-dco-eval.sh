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


# CUDA_VISIBLE_DEVICES=4 python inference_dreambooth.py \
#   --model_path "dreambooth-dco-outputs/dog" \
#   --prompt_file_list eval_prompt_list/dreambooth-dog5.yaml \
#   --output_path "dreambooth-dco-outputs/dog/eval" \
#   --seed_list 0 1 2 \
#   # --overwrite
  

# CUDA_VISIBLE_DEVICES=4 python inference_dreambooth.py \
#   --model_path "dreambooth-dco-outputs/dog5-text-sofa" \
#   --prompt_file_list eval_prompt_list/dreambooth-dog5.yaml \
#   --output_path "dreambooth-dco-outputs/dog5-text-sofa/eval" \
#   --seed_list 0 1 2 \
#   # --overwrite



CUDA_VISIBLE_DEVICES=4 python inference_dreambooth.py \
  --model_path "dreambooth-dco-outputs/cat2-text" \
  --prompt_file_list eval_prompt_list/dreambooth-cat.yaml \
  --output_path "dreambooth-dco-outputs/cat2-text/eval" \
  --seed_list 0 1 2 \
  # --overwrite
  

  
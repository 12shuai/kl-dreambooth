
CUDA_VISIBLE_DEVICES=2 python inference_dreambooth.py \
  --model_path "dreambooth-single-img-outputs/cake2/checkpoints/00600" \
  --prompt_file_list eval_prompt_list/dreambooth-cake.yaml \
  --output_path "dreambooth-single-img-outputs/cake2/checkpoints/00600/eval" \
  --seed_list 0 1 2 \





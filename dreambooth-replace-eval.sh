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



# CUDA_VISIBLE_DEVICES=2 python inference_dreambooth_replace.py \
#   --model_path "dreambooth-kl-outputs/grey_sloth_plushie_non_prompt_weight_10_non_class_suffix" \
#   --prompt_file_list eval_prompt_list/dreambooth-toy-non-suffix.yaml \
#   --output_path "dreambooth-kl-outputs/grey_sloth_plushie_non_prompt_weight_10_non_class_suffix/eval-replace-0.3" \
#   --seed_list 0 1 2 \
#   --placeholder_token "sbs" \
#   --instance_class "toy" \
#   --alpha 0.3
#   # --overwrite



# CUDA_VISIBLE_DEVICES=2 python inference_dreambooth_replace.py \
#   --model_path "dreambooth-kl-outputs/grey_sloth_plushie_non_prompt_weight_10_non_class_suffix" \
#   --prompt_file_list eval_prompt_list/dreambooth-toy-non-suffix.yaml \
#   --output_path "dreambooth-kl-outputs/grey_sloth_plushie_non_prompt_weight_10_non_class_suffix/eval-replace-0.6" \
#   --seed_list 0 1 2 \
#   --placeholder_token "sbs" \
#   --instance_class "toy" \
#   --alpha 0.6
#   # --overwrite
  


  
# CUDA_VISIBLE_DEVICES=2 python inference_dreambooth_replace.py \
#   --model_path "dreambooth-kl-outputs/grey_sloth_plushie_non_prompt_weight_10_non_class_suffix" \
#   --prompt_file_list eval_prompt_list/dreambooth-toy-non-suffix.yaml \
#   --output_path "dreambooth-kl-outputs/grey_sloth_plushie_non_prompt_weight_10_non_class_suffix/eval-replace-0.1" \
#   --seed_list 0 1 2 \
#   --placeholder_token "sbs" \
#   --instance_class "toy" \
#   --alpha 0.1
#   # --overwrite
  

CUDA_VISIBLE_DEVICES=2 python inference_dreambooth_replace.py \
  --model_path "dreambooth-kl-outputs/grey_sloth_plushie_non_prompt_weight_10_importance_sampling_non_class_suffix_steer_weight_1" \
  --prompt_file_list eval_prompt_list/dreambooth-toy-non-suffix.yaml \
  --output_path "dreambooth-kl-outputs/grey_sloth_plushie_non_prompt_weight_10_importance_sampling_non_class_suffix_steer_weight_1/eval-replace-0.3" \
  --seed_list 0 1 2 \
  --placeholder_token "sbs" \
  --instance_class "toy" \
  --alpha 0.3
  # --overwrite



CUDA_VISIBLE_DEVICES=2 python inference_dreambooth_replace.py \
  --model_path "dreambooth-kl-outputs/grey_sloth_plushie_non_prompt_weight_10_importance_sampling_non_class_suffix_steer_weight_1" \
  --prompt_file_list eval_prompt_list/dreambooth-toy-non-suffix.yaml \
  --output_path "dreambooth-kl-outputs/grey_sloth_plushie_non_prompt_weight_10_importance_sampling_non_class_suffix_steer_weight_1/eval-replace-0.6" \
  --seed_list 0 1 2 \
  --placeholder_token "sbs" \
  --instance_class "toy" \
  --alpha 0.6
  # --overwrite
  


  
CUDA_VISIBLE_DEVICES=2 python inference_dreambooth_replace.py \
  --model_path "dreambooth-kl-outputs/grey_sloth_plushie_non_prompt_weight_10_importance_sampling_non_class_suffix_steer_weight_1" \
  --prompt_file_list eval_prompt_list/dreambooth-toy-non-suffix.yaml \
  --output_path "dreambooth-kl-outputs/grey_sloth_plushie_non_prompt_weight_10_importance_sampling_non_class_suffix_steer_weight_1/eval-replace-0.1" \
  --seed_list 0 1 2 \
  --placeholder_token "sbs" \
  --instance_class "toy" \
  --alpha 0.1
  # --overwrite
  


CUDA_VISIBLE_DEVICES=2 python inference_dreambooth_replace.py \
  --model_path "dreambooth-outputs/grey_sloth_plushie-pp" \
  --prompt_file_list eval_prompt_list/dreambooth-toy.yaml \
  --output_path "dreambooth-outputs/grey_sloth_plushie-pp/eval-replace-0.3" \
  --seed_list 0 1 2 \
  --placeholder_token "sbs" \
  --instance_class "toy" \
  --alpha 0.3
  # --overwrite



CUDA_VISIBLE_DEVICES=2 python inference_dreambooth_replace.py \
  --model_path "dreambooth-outputs/grey_sloth_plushie-pp" \
  --prompt_file_list eval_prompt_list/dreambooth-toy.yaml \
  --output_path "dreambooth-outputs/grey_sloth_plushie-pp/eval-replace-0.6" \
  --seed_list 0 1 2 \
  --placeholder_token "sbs" \
  --instance_class "toy" \
  --alpha 0.6
  # --overwrite
  


  
CUDA_VISIBLE_DEVICES=2 python inference_dreambooth_replace.py \
  --model_path "dreambooth-outputs/grey_sloth_plushie-pp" \
  --prompt_file_list eval_prompt_list/dreambooth-toy.yaml \
  --output_path "dreambooth-outputs/grey_sloth_plushie-pp/eval-replace-0.1" \
  --seed_list 0 1 2 \
  --placeholder_token "sbs" \
  --instance_class "toy" \
  --alpha 0.1
  # --overwrite
  




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



# CUDA_VISIBLE_DEVICES=4 python inference_dreambooth_dco.py \
#   --model_path "dreambooth-dco-outputs/cat2-text" \
#   --prompt_file_list eval_prompt_list/dreambooth-cat.yaml \
#   --output_path "dreambooth-dco-outputs/cat2-text/eval" \
#   --seed_list 0 1 2 \
#   # --overwrite
  


# CUDA_VISIBLE_DEVICES=2 python inference_dreambooth_dco.py \
#   --ref_model_path "dreambooth-outputs/grey_sloth_plushie-non-pp" \
#   --prompt_file_list eval_prompt_list/dreambooth-toy.yaml \
#   --output_path "dreambooth-outputs/grey_sloth_plushie-non-pp/eval_dco_test" \
#   --w_rg 7.5\
#   --w_g 7.5\
#   --seed_list 0 1 2 \
#   # --overwrite


# CUDA_VISIBLE_DEVICES=1 python inference_dreambooth_dco.py \
#   --ref_model_path "dreambooth-outputs/dog-non-pp" \
#   --prompt_file_list eval_prompt_list/dreambooth-dog5.yaml \
#   --output_path "dreambooth-outputs/dog-non-pp/eval_dco_w_7.5" \
#   --w_rg 7.5\
#   --w_g 7.5\
#   --seed_list 0 1 2 \
#   # --overwrite

# CUDA_VISIBLE_DEVICES=2 python inference_dreambooth_dco.py \
#   --ref_model_path "dreambooth-outputs/cat2-non-pp" \
#   --prompt_file_list eval_prompt_list/dreambooth-cat.yaml \
#   --output_path "dreambooth-outputs/cat2-non-pp/eval_dco_w_7.5" \
#   --w_rg 7.5\
#   --w_g 7.5\
#   --seed_list 0 1 2 \
#   # --overwrite


CUDA_VISIBLE_DEVICES=2 python inference_dreambooth_dco.py \
  --ref_model_path "dreambooth-dco-outputs/monster_toy" \
  --prompt_file_list eval_prompt_list/dreambooth-toy.yaml \
  --output_path "dreambooth-dco-outputs/monster_toy/eval_dco_test" \
  --w_rg 7.5\
  --w_g 7.5\
  --seed_list 0 1 2 \
  # --overwrite

# CUDA_VISIBLE_DEVICES=3 python inference_dreambooth_dco.py \
#   --ref_model_path "dreambooth-dco-outputs/cat2-text" \
#   --prompt_file_list eval_prompt_list/dreambooth-cat.yaml \
#   --output_path "dreambooth-dco-outputs/cat2-text/eval_dco" \
#   --w_rg 7.5\
#   --w_g 7.5\
#   --seed_list 0 1 2 \
#   # --overwrite


# CUDA_VISIBLE_DEVICES=3 python inference_dreambooth_dco.py \
#   --ref_model_path "dreambooth-self-play-outputs/cat2_interval50_gen_num_1_scale_dpo_1e7_diff2weight_0.1" \
#   --prompt_file_list eval_prompt_list/dreambooth-cat.yaml \
#   --output_path "dreambooth-self-play-outputs/cat2_interval50_gen_num_1_scale_dpo_1e7_diff2weight_0.1/eval_dco" \
#   --w_rg 7.5\
#   --w_g 7.5\
#   --seed_list 0 1 2 \
#   # --overwrite

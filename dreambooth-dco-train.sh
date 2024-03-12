# ##1.singe-image
# python3 train.py \
#   --instance_data_dir examples/creature-test  \
#   --num_of_assets 1 \
#   --initializer_tokens creature \
#   --class_data_dir inputs/data_dir \
#   --phase1_train_steps 400 \
#   --phase2_train_steps 400 \
#   --output_dir outputs/creature-test \
#   --img_log_steps 200 \
#   --log_checkpoints \

# python3 train.py \
#   --instance_data_dir examples/creature-test  \
#   --num_of_assets 1 \
#   --initializer_tokens creature \
#   --class_data_dir inputs/data_dir \
#   --phase1_train_steps 400 \
#   --phase2_train_steps 400 \
#   --output_dir outputs/creature-test-non-mask-loss \
#   --img_log_steps 200 \
#   --log_checkpoints \
#   --do_not_apply_masked_loss



# python3 train.py \
#   --instance_data_dir examples/creature-test  \
#   --num_of_assets 1 \
#   --initializer_tokens creature \
#   --class_data_dir inputs/data_dir \
#   --phase1_train_steps 400 \
#   --phase2_train_steps 400 \
#   --output_dir outputs/creature-test-non-prior \
#   --img_log_steps 200 \
#   --log_checkpoints \
#   --do_not_apply_masked_loss \
#   --no_prior_preservation \
#   --lambda_attention 0


# python3 train.py \
#   --instance_data_dir examples/creature-test  \
#   --num_of_assets 1 \
#   --initializer_tokens creature \
#   --class_data_dir inputs/data_dir \
#   --phase1_train_steps 800 \
#   --phase2_train_steps 0 \
#   --output_dir outputs/creature-test-non-text-encoder-have-mask \
#   --img_log_steps 200 \
#   --log_checkpoints \
#   --no_prior_preservation \
#   --lambda_attention 0 \
#   # --do_not_apply_masked_loss \


#--initializer_tokens "man,<man0>" "watch,<watch0>"
# CUDA_VISIBLE_DEVICES=2 python3 train_prompt.py \
#   --instance_img_path_list examples/watch/man_with_watch/img.jpg  \
#   --customized_prompt_list "a man with watch"\
#   --class_data_dir inputs/data_dir \
#   --phase1_train_steps 400 \
#   --phase2_train_steps 400 \
#   --output_dir outputs/watch-no-asset-prior-preserve\
#   --img_log_steps 200 \
#   --log_checkpoints \
#   --flip_p 1\
  # --no_prior_preservation \
  # --initializer_tokens creature \



# CUDA_VISIBLE_DEVICES=2 python3 train_prompt.py \
#   --instance_img_path_list examples/watch/man_with_watch/img.jpg  \
#   --customized_prompt_list "a <man0> with <watch0>"\
#   --class_data_dir inputs/data_dir \
#   --phase1_train_steps 400 \
#   --phase2_train_steps 400 \
#   --output_dir outputs/watch-init-prior-preserve\
#   --img_log_steps 200 \
#   --log_checkpoints \
#   --flip_p 1\
#   --initializer_tokens "<man0>,man" "<watch0>,watch" \
#   # --no_prior_preservation \
#   # --initializer_tokens creature \



# CUDA_VISIBLE_DEVICES=2 python3 train_prompt.py \
#   --instance_img_path_list   examples/watch/woman_with_watch/women_with_watch.jpg \
#   --customized_prompt_list "a <woman0> with <watch0>"\
#   --class_data_dir inputs/data_dir \
#   --phase1_train_steps 400 \
#   --phase2_train_steps 400 \
#   --output_dir outputs/women-watch-init-prior-preserve\
#   --img_log_steps 200 \
#   --log_checkpoints \
#   --flip_p 1\
#   --initializer_tokens "<woman0>,woman" "<watch0>,watch" \


###########################3dog
# CUDA_VISIBLE_DEVICES=2 python3 train_dreambooth_dco.py \
#   --instance_data_dir "/mnt/CV_teamz/users/shuaixincheng/datasets/customized/dreambooth/dog5" \
#   --class_prompt "a photo of a dog" \
#   --class_data_dir inputs/dreambooth_data_dir \
#   --phase1_train_steps 0 \
#   --phase2_train_steps 1000 \
#   --output_dir dreambooth-dco-outputs/dog5-text-sofa\
#   --img_log_steps 200 \
#   --log_checkpoints \
#   --initial_learning_rate 5e-5 \
#   --train_batch_size 2 \
#   --instance_class "dog" \
#   --beta_dpo 1000\
#   --no_prior_preservation \
#   --train_text_encoder \
#   # --no_prior_preservation \

#   # --train_token \
#   # --flip_p 1\





# #############################cat
# CUDA_VISIBLE_DEVICES=2 python3 train_dreambooth_dco.py \
#   --instance_data_dir "/mnt/CV_teamz/users/shuaixincheng/datasets/customized/dreambooth/cat2" \
#   --class_prompt "a photo of a cat" \
#   --class_data_dir inputs/dreambooth_data_dir \
#   --phase1_train_steps 0 \
#   --phase2_train_steps 1000 \
#   --output_dir dreambooth-dco-outputs/cat2-text\
#   --img_log_steps 200 \
#   --log_checkpoints \
#   --initial_learning_rate 5e-5 \
#   --train_batch_size 2 \
#   --instance_class "cat" \
#   --beta_dpo 1000\
#   --no_prior_preservation \
#   --train_text_encoder \
#   # --no_prior_preservation \

#   # --train_token \
#   # --flip_p 1\








#############################cat
CUDA_VISIBLE_DEVICES=2 python3 train_dreambooth_dco.py \
  --instance_data_dir "/mnt/CV_teamz/users/shuaixincheng/datasets/customized/dreambooth/dog" \
  --class_prompt "a photo of a dog" \
  --class_data_dir inputs/dreambooth_data_dir \
  --phase1_train_steps 0 \
  --phase2_train_steps 1000 \
  --output_dir dreambooth-dco-outputs/dog\
  --img_log_steps 200 \
  --log_checkpoints \
  --initial_learning_rate 5e-5 \
  --train_batch_size 2 \
  --instance_class "dog" \
  --beta_dpo 1000\
  --no_prior_preservation \
  --train_text_encoder \
  # --no_prior_preservation \

  # --train_token \
  # --flip_p 1\


#############################cat
CUDA_VISIBLE_DEVICES=2 python3 train_dreambooth_dco.py \
  --instance_data_dir "/mnt/CV_teamz/users/shuaixincheng/datasets/customized/dreambooth/grey_sloth_plushie" \
  --class_prompt "a photo of a toy" \
  --class_data_dir inputs/dreambooth_data_dir \
  --phase1_train_steps 0 \
  --phase2_train_steps 1000 \
  --output_dir dreambooth-dco-outputs/grey_sloth_plushie\
  --img_log_steps 200 \
  --log_checkpoints \
  --initial_learning_rate 5e-5 \
  --train_batch_size 2 \
  --instance_class "toy" \
  --beta_dpo 1000\
  --no_prior_preservation \
  --train_text_encoder \
  # --no_prior_preservation \

  # --train_token \
  # --flip_p 1\



#############################cat
CUDA_VISIBLE_DEVICES=2 python3 train_dreambooth_dco.py \
  --instance_data_dir "/mnt/CV_teamz/users/shuaixincheng/datasets/customized/dreambooth/robot_toy" \
  --class_prompt "a photo of a toy" \
  --class_data_dir inputs/dreambooth_data_dir \
  --phase1_train_steps 0 \
  --phase2_train_steps 1000 \
  --output_dir dreambooth-dco-outputs/robot_toy\
  --img_log_steps 200 \
  --log_checkpoints \
  --initial_learning_rate 5e-5 \
  --train_batch_size 2 \
  --instance_class "toy" \
  --beta_dpo 1000\
  --no_prior_preservation \
  --train_text_encoder \
  # --no_prior_preservation \

  # --train_token \
  # --flip_p 1\


#############################cat
CUDA_VISIBLE_DEVICES=2 python3 train_dreambooth_dco.py \
  --instance_data_dir "/mnt/CV_teamz/users/shuaixincheng/datasets/customized/dreambooth/monster_toy" \
  --class_prompt "a photo of a toy" \
  --class_data_dir inputs/dreambooth_data_dir \
  --phase1_train_steps 0 \
  --phase2_train_steps 1000 \
  --output_dir dreambooth-dco-outputs/monster_toy\
  --img_log_steps 200 \
  --log_checkpoints \
  --initial_learning_rate 5e-5 \
  --train_batch_size 2 \
  --instance_class "toy" \
  --beta_dpo 1000\
  --no_prior_preservation \
  --train_text_encoder \
  # --no_prior_preservation \

  # --train_token \
  # --flip_p 1\


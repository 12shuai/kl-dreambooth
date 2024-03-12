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



# CUDA_VISIBLE_DEVICES=3 python3 train_prompt.py \
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



# CUDA_VISIBLE_DEVICES=3 python3 train_prompt.py \
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

# CUDA_VISIBLE_DEVICES=6 python3 train_dreambooth_self_play.py \
#   --instance_data_dir "/mnt/CV_teamz/users/shuaixincheng/datasets/customized/dreambooth/dog6" \
#   --instance_class "dog"\
#   --phase1_train_steps 0\
#   --phase2_train_steps 1000 \
#   --output_dir dreambooth-self-play-outputs\
#   --img_log_steps 200 \
#   --log_checkpoints \
#   --initial_learning_rate 5e-5 \
#   --train_batch_size 2 \
#   --train_text_encoder \
#   --no_prior_preservation \
#   --gen_data_dir inputs/dreambooth_self_play_data_dir\
#   --exchange_interval 50 \
#   --gen_num 2 \
#   --proj_name "dog6_interval50_gen_num_2"\
#   --beta_dpo 1e10\
#   --loss_type "logsigmoid"
#   # --loss_type "one_minus"
  
#   # --dataset_length 2 \
#   # --train_text_encoder \


#   # --train_token \
#   # --flip_p 1\

# #######################dog
# CUDA_VISIBLE_DEVICES=3 python3 train_dreambooth_self_play.py \
#   --instance_data_dir "/mnt/CV_teamz/users/shuaixincheng/datasets/customized/dreambooth/dog5" \
#   --instance_class "dog"\
#   --phase1_train_steps 0\
#   --phase2_train_steps 1000 \
#   --output_dir dreambooth-self-play-outputs\
#   --img_log_steps 200 \
#   --log_checkpoints \
#   --initial_learning_rate 5e-5 \
#   --train_batch_size 1 \
#   --no_prior_preservation \
#   --gen_data_dir inputs/dreambooth_self_play_data_dir\
#   --exchange_interval 10 \
#   --gen_num 1 \
#   --proj_name "dog5_interval10_prior_num_50_scale_dpo_1e7_diff2weight_0.1_use_prior_data"\
#   --beta_dpo 1e7\
#   --loss_type "logsigmoid" \
#   --train_text_encoder \
#   --use_prior_data \
#   --prior_num 50\
#   # --loss_type "one_minus"
  
#   # --dataset_length 2 \
#   # --train_text_encoder \


#   # --train_token \
#   # --flip_p 1\



# CUDA_VISIBLE_DEVICES=3 python3 train_dreambooth_self_play.py \
#   --instance_data_dir "/mnt/CV_teamz/users/shuaixincheng/datasets/customized/dreambooth/dog5" \
#   --instance_class "dog"\
#   --phase1_train_steps 0\
#   --phase2_train_steps 1000 \
#   --output_dir dreambooth-self-play-outputs\
#   --img_log_steps 200 \
#   --log_checkpoints \
#   --initial_learning_rate 5e-6 \
#   --train_batch_size 2 \
#   --no_prior_preservation \
#   --gen_data_dir inputs/dreambooth_self_play_data_dir\
#   --exchange_interval 50 \
#   --gen_num 1 \
#   --proj_name "dog5_new"\
#   --beta_dpo 5000\
#   --loss_type "logsigmoid" \
#   --train_text_encoder \
#   --diff_2_weights 1 \
  # --use_prior_data \
  # --prior_num 10\
  # --loss_type "one_minus"
  
  # --dataset_length 2 \
  # --train_text_encoder \


  # --train_token \
  # --flip_p 1\













# #######################cat
# CUDA_VISIBLE_DEVICES=6 python3 train_dreambooth_self_play.py \
#   --instance_data_dir "/mnt/CV_teamz/users/shuaixincheng/datasets/customized/dreambooth/cat2" \
#   --instance_class "cat"\
#   --phase1_train_steps 0\
#   --phase2_train_steps 1000 \
#   --output_dir dreambooth-self-play-outputs\
#   --img_log_steps 200 \
#   --log_checkpoints \
#   --initial_learning_rate 5e-5 \
#   --train_batch_size 1 \
#   --no_prior_preservation \
#   --gen_data_dir inputs/dreambooth_self_play_data_dir\
#   --exchange_interval 50 \
#   --gen_num 1 \
#   --proj_name "cat2_interval50_gen_num_1_scale_dpo_1e7_diff2weight_0.1"\
#   --beta_dpo 1e7\
#   --loss_type "logsigmoid" \
#   --train_text_encoder \
#   # --loss_type "one_minus"
  
#   # --dataset_length 2 \
#   # --train_text_encoder \














#################PPO#######################################################
# CUDA_VISIBLE_DEVICES=3 python3 train_dreambooth_self_play.py \
#   --instance_data_dir "/mnt/CV_teamz/users/shuaixincheng/datasets/customized/dreambooth/dog5" \
#   --instance_class "dog"\
#   --phase1_train_steps 0\
#   --phase2_train_steps 1000 \
#   --output_dir dreambooth-self-play-outputs\
#   --img_log_steps 200 \
#   --log_checkpoints \
#   --initial_learning_rate 5e-6 \
#   --train_batch_size 2 \
#   --no_prior_preservation \
#   --gen_data_dir inputs/dreambooth_self_play_data_dir\
#   --exchange_interval 10 \
#   --gen_num 1 \
#   --proj_name "dog5_use_ppo_scale_max_debug4_interval10"\
#   --beta_dpo 5000\
#   --loss_type "logsigmoid" \
#   --train_text_encoder \
#   --diff_2_weights 1 \
#   --use_ppo \
#   --ppo_episilon 1e-1 \
#   # --use_prior_data \
#   # --prior_num 10\
#   #   --ppo_episilon 1e-6 \



# CUDA_VISIBLE_DEVICES=3 python3 train_dreambooth_self_play.py \
#   --instance_data_dir "/mnt/CV_teamz/users/shuaixincheng/datasets/customized/dreambooth/dog5" \
#   --instance_class "dog"\
#   --phase1_train_steps 0\
#   --phase2_train_steps 1000 \
#   --output_dir dreambooth-self-play-outputs\
#   --img_log_steps 200 \
#   --log_checkpoints \
#   --initial_learning_rate 5e-6 \
#   --train_batch_size 2 \
#   --no_prior_preservation \
#   --gen_data_dir inputs/dreambooth_self_play_data_dir\
#   --exchange_interval 10 \
#   --gen_num 1 \
#   --proj_name "dog5_scale_max_debug4_interval10_use_approximate"\
#   --beta_dpo 5000\
#   --loss_type "logsigmoid" \
#   --train_text_encoder \
#   --diff_2_weights 1 \
#   --use_approximate 
#   # --use_ppo \
#   # --ppo_episilon 1e-8 \

#   # --use_prior_data \
#   # --prior_num 10\
#   #   --ppo_episilon 1e-6 \





# CUDA_VISIBLE_DEVICES=3 python3 train_dreambooth_self_play.py \
#   --instance_data_dir "/mnt/CV_teamz/users/shuaixincheng/datasets/customized/dreambooth/dog5" \
#   --instance_class "dog"\
#   --phase1_train_steps 0\
#   --phase2_train_steps 1000 \
#   --output_dir dreambooth-self-play-outputs\
#   --img_log_steps 200 \
#   --log_checkpoints \
#   --initial_learning_rate 5e-6 \
#   --train_batch_size 2 \
#   --no_prior_preservation \
#   --gen_data_dir inputs/dreambooth_self_play_data_dir\
#   --exchange_interval 10 \
#   --gen_num 1 \
#   --proj_name "dog5_scale_max_debug4_use_ppo_interval10_use_approximate"\
#   --beta_dpo 5000\
#   --loss_type "logsigmoid" \
#   --train_text_encoder \
#   --diff_2_weights 1 \
#   --use_approximate \
#   --use_ppo \
#   --ppo_episilon 1e-2 \

  # --use_prior_data \
  # --prior_num 10\
  #   --ppo_episilon 1e-6 \


CUDA_VISIBLE_DEVICES=3 python3 train_dreambooth_self_play.py \
  --instance_data_dir "/mnt/CV_teamz/users/shuaixincheng/datasets/customized/dreambooth/monster_toy" \
  --instance_class "toy"\
  --phase1_train_steps 0\
  --phase2_train_steps 1000 \
  --output_dir dreambooth-self-play-outputs\
  --img_log_steps 200 \
  --log_checkpoints \
  --initial_learning_rate 5e-6 \
  --train_batch_size 2 \
  --no_prior_preservation \
  --gen_data_dir inputs/dreambooth_self_play_data_dir\
  --exchange_interval 10 \
  --gen_num 1 \
  --proj_name "monster_toy_scale_max_debug4_use_ppo_1e-2_interval10_use_approximate_null_prompt"\
  --beta_dpo 5000\
  --loss_type "logsigmoid" \
  --train_text_encoder \
  --diff_2_weights 1 \
  --use_approximate \
  --use_ppo \
  --ppo_episilon 1e-2 \
  --class_prompt "" \
  # --class_prompt "a photo of a toy" \



# CUDA_VISIBLE_DEVICES=3 python3 train_dreambooth_self_play.py \
#   --instance_data_dir "/mnt/CV_teamz/users/shuaixincheng/datasets/customized/dreambooth/dog5" \
#   --instance_class "dog"\
#   --phase1_train_steps 0\
#   --phase2_train_steps 1000 \
#   --output_dir dreambooth-self-play-outputs\
#   --img_log_steps 200 \
#   --log_checkpoints \
#   --initial_learning_rate 5e-6 \
#   --train_batch_size 2 \
#   --no_prior_preservation \
#   --gen_data_dir inputs/dreambooth_self_play_data_dir\
#   --exchange_interval 50 \
#   --gen_num 1 \
#   --proj_name "dog5_scale_max_debug4_interval50_use_approximate"\
#   --beta_dpo 5000\
#   --loss_type "logsigmoid" \
#   --train_text_encoder \
#   --diff_2_weights 1 \
#   --use_approximate \
#   # --use_ppo \
#   # --ppo_episilon 1e-8 \


# CUDA_VISIBLE_DEVICES=3 python3 train_dreambooth_self_play.py \
#   --instance_data_dir "/mnt/CV_teamz/users/shuaixincheng/datasets/customized/dreambooth/dog5" \
#   --instance_class "dog"\
#   --phase1_train_steps 0\
#   --phase2_train_steps 1000 \
#   --output_dir dreambooth-self-play-outputs\
#   --img_log_steps 200 \
#   --log_checkpoints \
#   --initial_learning_rate 5e-6 \
#   --train_batch_size 2 \
#   --no_prior_preservation \
#   --gen_data_dir inputs/dreambooth_self_play_data_dir\
#   --exchange_interval 50 \
#   --gen_num 1 \
#   --proj_name "dog5_scale_max_debug4_use_ppo_interval50_use_approximate"\
#   --beta_dpo 5000\
#   --loss_type "logsigmoid" \
#   --train_text_encoder \
#   --diff_2_weights 1 \
#   --use_approximate \
#   --use_ppo \
#   --ppo_episilon 1e-1 \



# CUDA_VISIBLE_DEVICES=3 python3 train_dreambooth_self_play.py \
#   --instance_data_dir "/mnt/CV_teamz/users/shuaixincheng/datasets/customized/dreambooth/dog5" \
#   --instance_class "dog"\
#   --phase1_train_steps 0\
#   --phase2_train_steps 1000 \
#   --output_dir dreambooth-self-play-outputs\
#   --img_log_steps 200 \
#   --log_checkpoints \
#   --initial_learning_rate 5e-6 \
#   --train_batch_size 2 \
#   --no_prior_preservation \
#   --gen_data_dir inputs/dreambooth_self_play_data_dir\
#   --exchange_interval 1 \
#   --gen_num 1 \
#   --proj_name "dog5_scale_max_debug4_interval1_use_approximate"\
#   --beta_dpo 5000\
#   --loss_type "logsigmoid" \
#   --train_text_encoder \
#   --diff_2_weights 1 \
#   --use_approximate \
#   # --use_ppo \
#   # --ppo_episilon 1e-8 \


# CUDA_VISIBLE_DEVICES=3 python3 train_dreambooth_self_play.py \
#   --instance_data_dir "/mnt/CV_teamz/users/shuaixincheng/datasets/customized/dreambooth/dog5" \
#   --instance_class "dog"\
#   --phase1_train_steps 0\
#   --phase2_train_steps 1000 \
#   --output_dir dreambooth-self-play-outputs\
#   --img_log_steps 200 \
#   --log_checkpoints \
#   --initial_learning_rate 5e-6 \
#   --train_batch_size 2 \
#   --no_prior_preservation \
#   --gen_data_dir inputs/dreambooth_self_play_data_dir\
#   --exchange_interval 1 \
#   --gen_num 1 \
#   --proj_name "dog5_scale_max_debug4_use_ppo_interval1_use_approximate"\
#   --beta_dpo 5000\
#   --loss_type "logsigmoid" \
#   --train_text_encoder \
#   --diff_2_weights 1 \
#   --use_approximate \
#   --use_ppo \
#   --ppo_episilon 1e-12 \
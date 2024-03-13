# #############################cat
CUDA_VISIBLE_DEVICES=4 python3 train_dreambooth_KL_V2.py \
  --instance_data_dir "/mnt/CV_teamz/users/shuaixincheng/datasets/customized/dreambooth/monster_toy" \
  --class_prompt "" \
  --class_data_dir inputs/dreambooth_kl_v2_data_dir \
  --phase1_train_steps 0 \
  --phase2_train_steps 1000 \
  --output_dir dreambooth-kl-v2-outputs/monster_toy_non_prompt_weight_10\
  --img_log_steps 200 \
  --log_checkpoints \
  --initial_learning_rate 5e-5 \
  --train_batch_size 2 \
  --instance_class "toy" \
  --beta_dpo 1000\
  --no_prior_preservation \
  --train_text_encoder \
  --with_kl \
  --kl_loss_weight 10 \
  # --kl_batch 8


  # --train_token \
  # --flip_p 1\
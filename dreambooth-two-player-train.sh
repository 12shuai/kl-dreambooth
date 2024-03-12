

CUDA_VISIBLE_DEVICES=5 python3 train_dreambooth_two_player.py \
  --instance_data_dir "/mnt/CV_teamz/users/shuaixincheng/datasets/customized/dreambooth/dog5" \
  --class_prompt "a photo of a dog" \
  --class_data_dir inputs/dreambooth_two_player_data_dir\
  --phase1_train_steps 0\
  --phase2_train_steps 1000 \
  --output_dir dreambooth-two-player-outputs\
  --img_log_steps 200 \
  --log_checkpoints \
  --initial_learning_rate 5e-5 \
  --train_batch_size 2 \
  --instance_class "dog" \
  --train_text_encoder \
  --proj_name dog5-20\
  --exchange_interval 20
  # --no_prior_preservation \
  # --dataset_length 2 \
  # --train_text_encoder \

  # --train_token \
  # --flip_p 1\



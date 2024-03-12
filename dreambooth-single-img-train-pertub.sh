CUDA_VISIBLE_DEVICES=5 python3 train_dreambooth_single_img_pertub.py \
  --instance_data_dir "examples/cake2" \
  --instance_prompt "a round cake with orange frosting on a plate" \
  --class_prompt "a round cake with orange frosting on a plate" \
  --class_data_dir inputs/dreambooth_single_img_pertub_data_dir \
  --phase1_train_steps 0\
  --phase2_train_steps 1000 \
  --output_dir dreambooth-single-img-pertub-outputs/cake2-no-regular\
  --img_log_steps 200 \
  --log_checkpoints \
  --initial_learning_rate 5e-5 \
  --train_batch_size 1 \
  --instance_class "cake" \
  --no_prior_preservation \
  --regular_term 0\


CUDA_VISIBLE_DEVICES=5 python3 train_dreambooth_single_img_pertub.py \
  --instance_data_dir "examples/cake2" \
  --instance_prompt "a round cake with orange frosting on a plate" \
  --class_prompt "a round cake with orange frosting on a plate" \
  --class_data_dir inputs/dreambooth_single_img_pertub_data_dir \
  --phase1_train_steps 0\
  --phase2_train_steps 1000 \
  --output_dir dreambooth-single-img-pertub-outputs/cake2-1-regular\
  --img_log_steps 200 \
  --log_checkpoints \
  --initial_learning_rate 5e-5 \
  --train_batch_size 1 \
  --instance_class "cake" \
  --no_prior_preservation \
  --regular_term 1\

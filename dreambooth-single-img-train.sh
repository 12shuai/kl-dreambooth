CUDA_VISIBLE_DEVICES=5 python3 train_dreambooth_single_img.py \
  --instance_data_dir "examples/cake2" \
  --instance_prompt "a round cake with orange frosting on a plate" \
  --class_prompt "a round cake with orange frosting on a plate" \
  --class_data_dir inputs/dreambooth_single_img_data_dir \
  --phase1_train_steps 0\
  --phase2_train_steps 1000 \
  --output_dir dreambooth-single-img-outputs/cake2\
  --img_log_steps 200 \
  --log_checkpoints \
  --initial_learning_rate 5e-5 \
  --train_batch_size 2 \
  --instance_class "cake" \
  --train_text_encoder \
  --no_prior_preservation \


!accelerate config default

!accelerate launch src/slimface/training/accelerate_train.py \
    --batch_size 32 --algorithm yolo \
    --learning_rate 1e-4 --max_lr_factor 4\
    --warmup_steps 0.05 \
    --num_epochs 100 \
    --dataset_dir "./data/processed_ds" \
    --classification_model_name regnet_y_800mf
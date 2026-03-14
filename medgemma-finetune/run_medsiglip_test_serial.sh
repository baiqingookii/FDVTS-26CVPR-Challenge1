#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

SCRIPT=/remote-home/share/25-jianfabai/medgemma-finetune/test_medsiglip_covid_val_and_infer.py
# LOG_ROOT=/remote-home/share/25-jianfabai/medgemma-finetune/log/predict_log/iccv
LOG_ROOT=/remote-home/share/25-jianfabai/medgemma-finetune/log/predict_log/cvpr

model_dirs=(
/remote-home/share/25-jianfabai/medgemma-finetune/save_model/3d_1_dataset_mean_lr5e-5_scale0.7-1_ori_and_1b/checkpoint-972
/remote-home/share/25-jianfabai/medgemma-finetune/save_model/3d/checkpoint-1312
# /remote-home/share/25-jianfabai/medgemma-finetune/save_model/3d_1_dataset/checkpoint-1080
/remote-home/share/25-jianfabai/medgemma-finetune/save_model/3d_1_dataset1_lse_lr5e-5_scale0.7-1_64_448_448_12_32/checkpoint-779
# /remote-home/share/25-jianfabai/medgemma-finetune/save_model/3d_1_dataset_lse_lr5e-5_scale0.7-1_24_448_448_12_24/checkpoint-1944
# /remote-home/share/25-jianfabai/medgemma-finetune/save_model/3d_1_dataset_lse_lr5e-5_scale0.7-1_64_448_448_12_32/checkpoint-1350
# /remote-home/share/25-jianfabai/medgemma-finetune/save_model/3d_1_dataset_mean_lr5e-5_scale0.7-1/checkpoint-1728
# /remote-home/share/25-jianfabai/medgemma-finetune/save_model/3d_1_dataset_mean_lr5e-5_scale0.7-1_64_448_448_12_32/checkpoint-287
/remote-home/share/25-jianfabai/medgemma-finetune/save_model/3d_1_dataset_mean_lr5e-5_scale0.7-1_area/checkpoint-2132
# /remote-home/share/25-jianfabai/medgemma-finetune/save_model/3d_1_ori-and-1b_dataset_lse_lr5e-5_scale0.7-1/checkpoint-3240
/remote-home/share/25-jianfabai/medgemma-finetune/save_model/3d_continue/checkpoint-1566
/remote-home/share/25-jianfabai/medgemma-finetune/save_model/real_24*448*448_3d_1_ori-and-1b_dataset1_mean_lr5e-5_scale0.7-1/checkpoint-1105
# /remote-home/share/25-jianfabai/medgemma-finetune/save_model/64*448*448_3d_1_ori-and-1b_dataset1_mean_lr5e-5_scale0.7-1/checkpoint-528
/remote-home/share/25-jianfabai/medgemma-finetune/save_model/128*256*256_3d_1_ori-and-1b_dataset1_mean_lr5e-5_scale0.7-1/checkpoint-1836
)

mkdir -p "$LOG_ROOT"

for model_dir in "${model_dirs[@]}"; do
    exp_name=$(basename "$(dirname "$model_dir")")
    ckpt_name=$(basename "$model_dir")
    run_name="${exp_name}-${ckpt_name}"

    echo "=============================="
    echo "Running: $run_name"
    echo "MODEL  : $model_dir"
    echo "=============================="

    python "$SCRIPT" \
      --model_dir "$model_dir" \
      > "$LOG_ROOT/${run_name}.log" 2>&1

    echo "Finished: $run_name"
done

# nohup /remote-home/share/25-jianfabai/medgemma-finetune/run_medsiglip_test_serial.sh > /remote-home/share/25-jianfabai/medgemma-finetune/log/predict_log/cvpr/run_medsiglip_test_serial_launcher.log 2>&1 &
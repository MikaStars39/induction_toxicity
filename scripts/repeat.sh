export XDG_CACHE_HOME=/mnt/weka/hw_workspace/qy_workspace/lightning/.cache
CUDA_VISIBLE_DEVICES=1 python3 run.py \
    --task_name repeat \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --seed 31 \
    --config_path config/repeat.yaml \
    --token "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"
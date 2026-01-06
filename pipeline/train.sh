export CUDA_VISIBLE_DEVICES=7
# python pretrain/pre_train.py
# echo "Pre-training completed. Starting training in 100 seconds..."
# sleep 100
# while true; do
#     echo "Starting training..."
    python scripts/train.py
    
#     echo "Training stopped unexpectedly. Restarting in 100 second..."
#     sleep 100
# done
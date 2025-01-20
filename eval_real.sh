export PYTHONPATH=/data_local/commit/LaPose

python evaluation/evaluate.py \
--resume_model="./checkpoints/model_single.pth" \
--dataset=Real --use_scale_net \
--sn_path='./checkpoints/model_scale_net_Real.pth'
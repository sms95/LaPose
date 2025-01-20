
export PYTHONPATH=/data_local/commit/LaPose

python evaluation/evaluate.py \
--resume_model="./checkpoints/model_single.pth" \
--dataset=CAMERA --use_scale_net \
--sn_path='./checkpoints/model_scale_net_CAMERA.pth' \
--dataset_dir="./NOCS"
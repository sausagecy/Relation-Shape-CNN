#!/usr/bin/env sh
mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")
log_name="Face_BU3DFE_ID_LOG_"$now""
export CUDA_VISIBLE_DEVICES=1
python -u train_face_id.py \
--config cfgs/config_ssn_face_id.yaml \
2>&1|tee log/$log_name.log 

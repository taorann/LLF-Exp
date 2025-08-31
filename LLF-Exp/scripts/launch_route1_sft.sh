#!/usr/bin/env bash
set -e
PROJ=/map-vepfs/taoran/LLF-Exp
export PYTHONPATH=$PROJ:$PYTHONPATH

# —— Route-1 资产（环境变量） ——
export ROUTE_TYPE=route1
export ROUTE1_PHI=$PROJ/ckpts/route1/phi/phi_h.v1.npy
export ROUTE1_CWH=$PROJ/ckpts/route1/stats/cwh_scale.v1.npy
export ROUTE1_BAND=$PROJ/ckpts/route1/stats/band_stats.v1.json
export ROUTE1_TAU1=0.5
export ROUTE1_TAU2=0.5
# export ROUTE1_LAMBDA=1e-4   # 预留：若在 loss 里使用可打开

ACC_CFG=$PROJ/llama-factory/examples/accelerate/fsdp_config.yaml
LF_CFG=$PROJ/route1_ext/config.yml

accelerate launch --config_file "$ACC_CFG" \
  $PROJ/route1_ext/loss_patch.py \
  --cfg "$LF_CFG" \
  --report_to tensorboard \
  --logging_dir $PROJ/logs/tb/route1

#!/usr/bin/env bash
set -e
PROJ=/map-vepfs/taoran/LLF-Exp
BASE=$PROJ/ckpts/base/Qwen2.5-Math-7B

python $PROJ/route1_ext/precompute_witness.py \
  --base_model $BASE \
  --train_json $PROJ/data/json/train.json \
  --witness_list $PROJ/data/splits/witness_ids.txt \
  --phi_out   $PROJ/ckpts/route1/phi/phi_h.v1.npy \
  --cwh_out   $PROJ/ckpts/route1/stats/cwh_scale.v1.npy \
  --band_stats $PROJ/ckpts/route1/stats/band_stats.v1.json \
  --ema_momentum 0.99 --damp 1e-4 \
  --band_low_q 0.2 --band_high_q 0.8 --tau1 0.5 --tau2 0.5 \
  --load_in_4bit True

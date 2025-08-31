#!/usr/bin/env bash
set -e
PROJ=/map-vepfs/taoran/LLF-Exp

# 转换原始数据 -> LLaMA-Factory JSON
python $PROJ/scripts/make_math_json.py \
  --src $PROJ/data/raw/gsm8k \
  --out_train $PROJ/data/json/train.json \
  --out_eval  $PROJ/data/json/dev.json

# 抽样 witness 子集
python $PROJ/scripts/sample_witness_list.py \
  --train_json $PROJ/data/json/train.json \
  --out_list   $PROJ/data/splits/witness_ids.txt \
  --size 20000

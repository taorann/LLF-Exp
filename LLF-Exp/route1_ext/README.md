# Route-1 Extension: Global Witness

本目录包含 **Route-1 (Global Witness)** 的实现代码，用于在 LLaMA-Factory 中集成带符号权重的损失函数。

## 文件说明

- `__init__.py`  
  使目录成为 Python 包。

- `config.yml`  
  LLaMA-Factory 的训练配置（数据路径、LoRA 参数等）。可直接传给 `--cfg`。

- `precompute_witness.py`  
  离线脚本：在训练前计算全局见证向量 Φglob、对角白化因子 Cwh，以及样本难度分位数。  
  输出到 `ckpts/route1/{phi,stats}/`。

- `loss_patch.py`  
  Route-1 的核心损失替换逻辑。  
  - 在训练时 hook 最后一层 hidden states  
  - 计算白化余弦对齐度  
  - 结合带通窗权重，得到可正可负的 token 权重  
  - 替换 Trainer 的 `compute_loss`。

- `README.md`  
  当前说明文档。

## 使用步骤

1. **预计算 Witness**
   ```bash
    python route1_ext/precompute_witness.py \
     --base_model ckpts/base/Qwen2.5-Math-7B \
     --train_json data/json/train.json \
     --witness_list data/splits/witness_ids.txt \
     --phi_out ckpts/route1/phi/phi_h.v1.npy \
     --cwh_out ckpts/route1/stats/cwh_scale.v1.npy \
     --band_stats ckpts/route1/stats/band_stats.v1.json

2. **设置环境变量**
   ```bash
    export ROUTE_TYPE=route1
    export ROUTE1_PHI=ckpts/route1/phi/phi_h.v1.npy
    export ROUTE1_CWH=ckpts/route1/stats/cwh_scale.v1.npy
    export ROUTE1_BAND=ckpts/route1/stats/band_stats.v1.json

3. **启动训练**
   ```bash
    accelerate launch \
    --config_file llama-factory/examples/accelerate/zero2.yaml \
    route1_ext/patch_entry.py \
    --cfg route1_ext/config.yml \
    --report_to tensorboard \
    --logging_dir logs/tb/route1

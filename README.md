# LLF-Exp: LLaMA-Factory with Route-1 & Route-2 Extensions

本项目基于 **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)**，扩展实现了论文中提出的两条带符号权重训练路线：

- **Route-1 (Global Witness)**  
  在全局采样子集上计算见证向量 Φglob，结合 Fisher 白化与带通窗权重，在 SFT 中实现符合模型几何的更新。

- **Route-2 (Mini-batch Filtering)**  
  在 mini-batch 内通过岭回归近似得到符号权重，轻量稳定，后续可与 Route-1 对比。

目标：在监督微调 (SFT) 中引入类似 RL 的两类信号 ——  
**负梯度** & **policy-dependent 几何**，提升收敛方向与稳定性。

---

## 目录结构
```text
LLF-Exp/
├─ llama-factory/ # LLaMA-Factory 源码 (外部 clone)
├─ ckpts/ # 权重与统计
│ ├─ base/ # 基座模型 (e.g., Qwen2.5-Math-7B)
│ ├─ adapters/ # LoRA/QLoRA 输出
│ ├─ route1/{phi,stats}/ # Route-1 Witness 向量与统计
│ └─ route2/{phi,stats}/ # Route-2 统计 (预留)
├─ data/
│ ├─ raw/ # 原始 GSM8K / MATH 数据
│ ├─ json/ # 转换后 LF JSON
│ └─ splits/ # witness 抽样列表
├─ shared/ # 公共组件
│ ├─ whitening.py # 对角白化/阻尼
│ ├─ bandpass.py # 带通窗权重
│ ├─ hooks.py # 读出层 hook
│ ├─ io_utils.py # I/O 工具
│ ├─ metrics.py # 诊断日志与绘图
│ └─ init.py
├─ route1_ext/ # Route-1 实现
│ ├─ precompute_witness.py # 离线计算 Φglob
│ ├─ loss_patch.py # Route-1 损失函数
│ ├─ config.yml # Route-1 训练配置
│ ├─ init.py
│ └─ README.md
├─ route2_ext/ # Route-2 实现 (占位)
│ ├─ loss_patch.py
│ ├─ config.yml
│ └─ init.py
├─ scripts/ # 数据处理与启动脚本
│ ├─ make_math_json.py # 原始数据 → LF JSON
│ ├─ sample_witness_list.py # 抽样 witness 子集
│ ├─ launch_route1_sft.sh # 启动 Route-1 训练
│ ├─ launch_route2_sft.sh # 启动 Route-2 训练 (预留)
│ └─ eval_math.sh # 简单评测
├─ logs/ # 日志与诊断输出
│ ├─ tb/ # TensorBoard
│ └─ runs/ # JSON/PNG 直方图
└─ README.md # 顶层说明文档


---

##  环境安装

```bash

# 基础依赖
pip install torch --extra-index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate datasets bitsandbytes peft trl deepspeed tensorboard matplotlib pyyaml

# LLaMA-Factory
git clone https://github.com/hiyouga/LLaMA-Factory.git llama-factory
pip install -e llama-factory
# LLF-Exp

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练诊断指标与绘图
"""
import os, json
import torch
import matplotlib.pyplot as plt

class MetricsLogger:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def log_hist(self, name, values: torch.Tensor, step: int, bins=50):
        """保存直方图 (numpy) 和绘图 (png)"""
        arr = values.detach().cpu().numpy()
        hist, edges = torch.histogram(torch.tensor(arr), bins=bins)
        out_json = {
            "step": step,
            "hist": hist.tolist(),
            "edges": edges.tolist()
        }
        path = os.path.join(self.out_dir, f"{name}_{step}.json")
        with open(path, "w") as f: json.dump(out_json, f)

        # 绘图
        fig, ax = plt.subplots()
        ax.hist(arr, bins=bins, alpha=0.7)
        ax.set_title(f"{name} step={step}")
        fig.savefig(os.path.join(self.out_dir, f"{name}_{step}.png"))
        plt.close(fig)

    def log_scalar(self, name, value, step:int):
        """简单标量日志"""
        path = os.path.join(self.out_dir, f"{name}.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps({"step": step, "value": float(value)}) + "\n")

# ==== 使用示例 ====
# logger = MetricsLogger("logs/runs/route1")
# logger.log_hist("s_values", s, step)
# logger.log_scalar("grad_norm", gnorm, step)

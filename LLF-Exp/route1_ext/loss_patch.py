#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, json, torch
import numpy as np
from torch.nn import functional as F
from shared.whitening import load_scale, whiten_h
from shared.bandpass import load_band_stats, window_weight
from shared.hooks import register_readout_hook
import transformers

# --------- 读取 Route-1 资产（来自环境变量） ---------
def _load_assets():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    phi_path = os.environ.get("ROUTE1_PHI")
    cwh_path = os.environ.get("ROUTE1_CWH")
    band_path= os.environ.get("ROUTE1_BAND")
    if not all([phi_path, cwh_path, band_path]):
        return None
    # 保持 float32，后续再按模型 dtype 适配
    phi = torch.from_numpy(np.load(phi_path)).float().to(dev)
    phi = phi / (torch.norm(phi) + 1e-12)
    cwh = load_scale(cwh_path, device=dev)      # 确保内部张量在 dev 上
    dmin, dmax = load_band_stats(band_path)
    tau1 = float(os.environ.get("ROUTE1_TAU1","0.5"))
    tau2 = float(os.environ.get("ROUTE1_TAU2","0.5"))
    lam  = float(os.environ.get("ROUTE1_LAMBDA","1e-4"))
    return {"phi": phi, "cwh": cwh, "dmin": dmin, "dmax": dmax, "tau1": tau1, "tau2": tau2, "lam": lam}

def _install(model):
    cache, handle = register_readout_hook(model, readout_attr="lm_head")
    return cache, handle

# --------- Route-1 的 compute_loss ---------
def _route1_compute_loss(self, model, inputs, return_outputs=False):
    labels = inputs.get("labels")
    outputs = model(**inputs)
    logits = outputs.get("logits")  # [B, T, V]
    if labels is None:
        loss = outputs["loss"] if "loss" in outputs else None
        return (loss, outputs) if return_outputs else loss

    vocab = logits.size(-1)
    # 参考未加权 CE（用于日志，不参与反传）
    with torch.no_grad():
        ce_ref = F.cross_entropy(
            logits.view(-1, vocab),
            labels.view(-1),
            ignore_index=-100,
            reduction="mean"
        )

    # 按需把 ce 在 fp32 里算（可选）
    ce = F.cross_entropy(
        logits.float().view(-1, vocab),
        labels.view(-1),
        ignore_index=-100,
        reduction="none"
    ).view(labels.shape).to(logits.dtype)  # [B, T] -> 回到原 dtype 以便和 w 相乘

    with torch.no_grad():
        mask = (labels != -100)  # [B, T]
        if mask.sum() == 0:
            loss = (ce * 0).sum()
            if return_outputs:
                outputs["loss_ref_ce"] = ce_ref.detach()
            return (loss, outputs) if return_outputs else loss

        # 取被监督位置的向量
        h_all = self._route1_cache.pop() if getattr(self, "_route1_cache", None) else None
        if h_all is None:
            # 极端情况下 hook 未生效 —— 回退为普通 SFT，避免中断
            loss = ce[mask].mean()
            if return_outputs:
                outputs["loss_ref_ce"] = ce_ref.detach()
            return (loss, outputs) if return_outputs else loss

        h_t = h_all[mask]  # [Ntok, d]

        # 困难度 d = -log p[y]
        p = torch.softmax(logits, dim=-1)
        # 只在 mask 位置 gather，避免无意义索引
        y_masked = labels[mask]
        py = p[mask].gather(-1, y_masked.unsqueeze(-1)).squeeze(-1)  # [Ntok]
        d  = (-torch.log(py.clamp_min(1e-9)))                        # [Ntok]

        # 带通窗基础权重
        bp = window_weight(
            d,
            self._route1_assets["dmin"],
            self._route1_assets["dmax"],
            self._route1_assets["tau1"],
            self._route1_assets["tau2"]
        )  # [Ntok]

        # 白化 + 余弦对齐（Global Witness）
        hwh = whiten_h(h_t, self._route1_assets["cwh"])              # [Ntok, d]
        # dtype 对齐：让 Phi 与 hwh 一致（AMP 稳定）
        Phi = self._route1_assets["phi"].to(hwh.dtype).to(hwh.device)
        H   = hwh / (hwh.norm(dim=1, keepdim=True) + 1e-12)
        s   = (H @ Phi).clamp(-1, 1)                                  # [-1, 1]

        w = (bp * s).detach()                                         # [Ntok]
        w_full = torch.zeros_like(ce)
        w_full[mask] = w

        # 稳定归一化（方案A：按 |w| 归一）
        denom = (w_full.abs() * mask).sum().clamp_min(1.0)

    # 加权 CE
    loss = (w_full * ce).sum() / denom

    if return_outputs:
        outputs["loss_ref_ce"] = ce_ref.detach()
    return (loss, outputs) if return_outputs else loss


# --------- 打补丁 & 作为训练入口 ---------
def enable_route1_patch():
    if os.environ.get("ROUTE_TYPE", "") != "route1":
        return
    assets = _load_assets()
    if assets is None:
        raise RuntimeError(
            "ROUTE1 assets not set. Please export ROUTE1_PHI, ROUTE1_CWH, ROUTE1_BAND (and optional ROUTE1_TAU1/TAU2)."
        )
    transformers.Trainer._orig_compute_loss = transformers.Trainer.compute_loss

    def compute_loss(self, model, inputs, return_outputs=False):
        if not getattr(self, "_route1_ready", False):
            self._route1_assets = assets
            cache, handle = _install(model)
            self._route1_cache = cache
            self._route1_handle = handle
            self._route1_ready = True
            # 在 trainer 对象销毁时移除 hook（简易防护）
            if not hasattr(self, "_route1_finalize"):
                def _finalize(_self=self):
                    try:
                        _self._route1_handle.remove()
                    except Exception:
                        pass
                self._route1_finalize = _finalize

        out = _route1_compute_loss(self, model, inputs, return_outputs)
        return out

    transformers.Trainer.compute_loss = compute_loss


def _dispatch_llf():
    """调用 LLaMA-Factory 的 SFT 入口（不同版本路径略有差异，做兼容）"""
    try:
        from llama_factory.src.train_sft import main as sft_main
    except Exception:
        # 某些版本包名拼写不同
        from llamafactory.src.train_sft import main as sft_main
    sft_main()

if __name__ == "__main__":
    # 作为脚本启动时：先启用 Route-1 补丁，再把 CLI 交给 LLaMA-Factory
    enable_route1_patch()
    _dispatch_llf()

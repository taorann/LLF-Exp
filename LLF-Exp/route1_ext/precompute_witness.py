import os, json, argparse, torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from shared.hooks import register_readout_hook
from shared.whitening import DiagCEMA, whiten_h, save_scale
from shared.bandpass import batch_quantiles, window_weight, save_band_stats

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--train_json", required=True)
    ap.add_argument("--witness_list", required=True) # 每行一个样本索引或id
    ap.add_argument("--phi_out", required=True)
    ap.add_argument("--cwh_out", required=True)
    ap.add_argument("--band_stats", required=True)
    ap.add_argument("--ema_momentum", type=float, default=0.99)
    ap.add_argument("--damp", type=float, default=1e-4)
    ap.add_argument("--band_low_q", type=float, default=0.2)
    ap.add_argument("--band_high_q", type=float, default=0.8)
    ap.add_argument("--tau1", type=float, default=0.5)
    ap.add_argument("--tau2", type=float, default=0.5)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--load_in_4bit", type=bool, default=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    tok.pad_token = tok.eos_token

    quant_kwargs = dict()
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, **quant_kwargs).to(device)
    model.eval()

    cache, handle = register_readout_hook(model, readout_attr="lm_head")

    # 加载数据（LF 的简单 JSON：每行一条）
    with open(args.train_json, "r") as f:
        lines = [json.loads(x) for x in f]
    # witness 列表（索引）
    with open(args.witness_list, "r") as f:
        idxs = [int(x.strip()) for x in f if x.strip()]

    # 先跑一小批拿到 d 估计分位数
    d_all = []
    with torch.no_grad():
        for i in idxs[:min(512, len(idxs))]:
            ex = lines[i]
            text = (ex.get("instruction","") + "\n" + ex.get("input","")).strip() or ex.get("input","")
            labels = ex.get("output","")
            prompt = text + ("\nAnswer:")
            ids = tok(prompt, return_tensors="pt", truncation=True, max_length=args.max_tokens).to(device)
            with torch.inference_mode():
                out = model(**ids)
            logits = out.logits[:, -1, :]  # 只用最后 token 近似难度
            tgt = tok(labels, return_tensors="pt", truncation=True, max_length=64).to(device)["input_ids"][0,0].item()
            p = torch.softmax(logits, dim=-1)
            d = -torch.log(p[0, tgt].clamp_min(1e-9))
            d_all.append(d.item())
    d_tensor = torch.tensor(d_all, device=device)
    dmin, dmax = batch_quantiles(d_tensor, args.band_low_q, args.band_high_q)

    # 主循环：累计 C 的 EMA 与 Φ
    cema = None
    phi = None
    cnt = 0

    for i in idxs:
        ex = lines[i]
        text = (ex.get("instruction","") + "\n" + ex.get("input","")).strip() or ex.get("input","")
        labels = ex.get("output","")
        prompt = text + ("\nAnswer:")
        batch = tok(prompt, return_tensors="pt", truncation=True, max_length=args.max_tokens).to(device)

        out = model(**batch)
        logits = out.logits  # [1, T, V]
        h_all = cache.pop()  # [1, T, d]
        if h_all is None: continue

        # 只在最后一个监督 token 上取难度做近似（足够稳定）
        tgt_ids = tok(labels, return_tensors="pt", truncation=True, max_length=64).to(device)["input_ids"][0]
        tgt = tgt_ids[0].item()
        p_last = torch.softmax(logits[:, -1, :], dim=-1)[0]  # [V]
        d = (-torch.log(p_last[tgt].clamp_min(1e-9))).detach()

        h = h_all[0]  # [T, d]
        # 展平采样几个 token（避免只看最后一位）：这里简单取中间 + 最后若干
        pick = torch.linspace(0, h.size(0)-1, steps=min(8, h.size(0)), device=device).long()
        h_tok = h[pick]  # [m, d]

        if cema is None: cema = DiagCEMA(d=h_tok.size(1), momentum=args.ema_momentum, damp=args.damp, device=device)
        cema.update(h_tok)

        scale = cema.scale()
        hwh = (h_tok * scale)              # [m, d]
        # 这里走 Minimal witness：ψ ≈ Cwh h
        psi = hwh.mean(dim=0)              # [d]
        wwin = window_weight(d.unsqueeze(0), dmin, dmax, args.tau1, args.tau2)[0]
        contrib = wwin * psi               # [d]
        phi = contrib if phi is None else (phi + contrib)
        cnt += 1

        if (cnt % 200) == 0:
            print(f"[witness] processed {cnt} items")

    # 归一化保存
    phi = phi / (torch.norm(phi) + 1e-12)
    scale = cema.scale()
    os.makedirs(os.path.dirname(args.phi_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.cwh_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.band_stats), exist_ok=True)

    np.save(args.phi_out, phi.detach().float().cpu().numpy())
    save_scale(args.cwh_out, scale)
    save_band_stats(args.band_stats, dmin, dmax)
    handle.remove()
    print("[witness] saved:", args.phi_out, args.cwh_out, args.band_stats)

if __name__ == "__main__":
    main()

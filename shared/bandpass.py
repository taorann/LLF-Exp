import torch, json

def window_weight(d: torch.Tensor, dmin: float, dmax: float, tau1: float=0.5, tau2: float=0.5):
    # d: [Ntok]，wwin = σ((d-dmin)/τ1)*σ((dmax-d)/τ2)
    w1 = torch.sigmoid((d - dmin)/tau1)
    w2 = torch.sigmoid((dmax - d)/tau2)
    return w1 * w2

def batch_quantiles(d: torch.Tensor, low_q=0.2, high_q=0.8):
    # 返回两个标量分位数
    q = torch.quantile(d, torch.tensor([low_q, high_q], device=d.device))
    return float(q[0].item()), float(q[1].item())

def save_band_stats(path:str, dmin:float, dmax:float):
    with open(path, "w") as f:
        json.dump({"dmin": dmin, "dmax": dmax}, f, indent=2)

def load_band_stats(path:str):
    with open(path, "r") as f:
        obj = json.load(f); return obj["dmin"], obj["dmax"]

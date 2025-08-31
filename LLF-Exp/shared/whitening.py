import torch, json
import numpy as np

class DiagCEMA:
    def __init__(self, d:int, momentum:float=0.99, damp:float=1e-4, device="cuda"):
        self.m = momentum; self.damp = damp
        self.r = torch.zeros(d, device=device)

    @torch.no_grad()
    def update(self, h: torch.Tensor):
        # h: [Ntok, d]
        mom = (h*h).mean(dim=0)
        self.r.mul_(self.m).add_(mom, alpha=1-self.m)

    def scale(self)->torch.Tensor:
        # (EMA[h^2]+damp)^(-1/2)
        return torch.rsqrt(self.r + self.damp)

@torch.no_grad()
def whiten_h(h: torch.Tensor, scale: torch.Tensor):
    # elementwise scale
    return h * scale  # [Ntok, d]

def save_scale(path:str, scale: torch.Tensor):
    np.save(path, scale.detach().float().cpu().numpy())

def load_scale(path:str, device="cuda")->torch.Tensor:
    arr = np.load(path)
    return torch.from_numpy(arr).to(device)

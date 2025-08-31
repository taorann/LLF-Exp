import torch

class ReadoutCache:
    def __init__(self): self.buf = []
    def __call__(self, module, inputs, output):
        # inputs[0]: [B, T, d]
        self.buf.append(inputs[0].detach())
    def pop(self):
        if not self.buf: return None
        h = torch.cat(self.buf, dim=0); self.buf.clear(); return h

def register_readout_hook(model, readout_attr="lm_head"):
    readout = getattr(model, readout_attr, None)
    if readout is None:
        raise ValueError(f"Cannot find readout layer: {readout_attr}")
    cache = ReadoutCache()
    handle = readout.register_forward_hook(cache)
    return cache, handle

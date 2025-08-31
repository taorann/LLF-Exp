#!/usr/bin/env bash
set -e
PROJ=/map-vepfs/taoran/LLF-Exp
MODEL_OUT=$PROJ/ckpts/adapters/route1_qllora    # 按你LF config里的 output_dir 调整
BASE=$PROJ/ckpts/base/Qwen2.5-Math-7B

python - <<'PY'
import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

proj="/map-vepfs/taoran/LLF-Exp"
base=f"{proj}/ckpts/base/Qwen2.5-Math-7B"
adpt=f"{proj}/ckpts/adapters/route1_qllora"

tok=AutoTokenizer.from_pretrained(base, use_fast=True)
bnb=BitsAndBytesConfig(load_in_4bit=True)
model=AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb, torch_dtype=torch.bfloat16)
try:
    from peft import PeftModel
    model=PeftModel.from_pretrained(model, adpt)
except Exception as e:
    print("warning: no peft adapter loaded:", e)

device="cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()

dev=f"{proj}/data/json/dev.json"
with open(dev) as f: ds=[json.loads(x) for x in f][:64]  # 先测 64 条
ok=0
for ex in ds:
    prompt=(ex.get("instruction","")+"\n"+ex.get("input","")).strip() or ex.get("input","")
    ids=tok(prompt, return_tensors="pt").to(device)
    out=model.generate(**ids, max_new_tokens=128, do_sample=False)
    text=tok.decode(out[0], skip_special_tokens=True)
    ans=ex.get("output","")
    ok += int(ans.split()[0] in text)  # 最粗略命中
print("rough_acc:", ok/len(ds))
PY

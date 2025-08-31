#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, json, argparse, random
from pathlib import Path

def load_lines(path):
    p = Path(path)
    if p.suffix == ".jsonl":
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if line: yield json.loads(line)
    elif p.suffix == ".json":
        with open(p, "r", encoding="utf-8") as f:
            data=json.load(f)
            if isinstance(data, dict) and "data" in data: data=data["data"]
            for x in data: yield x
    else:
        raise ValueError(f"Unsupported file: {path}")

def norm_gsm8k_item(ex):
    q = ex.get("question") or ex.get("prompt") or ""
    a = ex.get("answer") or ex.get("final_answer") or ""
    # GSM8K 的答案常带 "#### 42"；保留推理 + 最终数值都可以
    return {
        "instruction": "Solve the math word problem and give the final numeric answer.",
        "input": q.strip(),
        "output": a.strip()
    }

def norm_math_item(ex):
    q = ex.get("problem") or ex.get("question") or ""
    a = ex.get("solution") or ex.get("answer") or ""
    return {
        "instruction": "Solve the following math problem and provide a clear solution.",
        "input": q.strip(),
        "output": a.strip()
    }

def detect_format(first_ex):
    k = set(first_ex.keys())
    if {"question","answer"} & k or "question" in k:
        return "gsm8k"
    if {"problem","solution"} & k or "problem" in k:
        return "math"
    # 兜底：看有没有明显的键
    if "answer" in k: return "gsm8k"
    if "solution" in k: return "math"
    raise ValueError(f"Cannot detect format from keys: {k}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="path to raw dataset file or folder")
    ap.add_argument("--out_train", required=True)
    ap.add_argument("--out_eval", required=True)
    ap.add_argument("--train_ratio", type=float, default=0.98)  # 大多数数据训练集更大
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    # 收集数据（支持传目录，自动合并其中的 json/jsonl）
    paths=[]
    if os.path.isdir(args.src):
        for fn in os.listdir(args.src):
            if fn.endswith(".json") or fn.endswith(".jsonl"):
                paths.append(os.path.join(args.src, fn))
    else:
        paths.append(args.src)

    examples=[]
    for p in paths:
        for ex in load_lines(p):
            examples.append(ex)
    if not examples:
        raise RuntimeError(f"No data found under {args.src}")

    fmt = detect_format(examples[0])

    def to_lf(ex):
        return norm_gsm8k_item(ex) if fmt=="gsm8k" else norm_math_item(ex)

    mapped=[to_lf(x) for x in examples if (to_lf(x)["input"] and to_lf(x)["output"])]
    # 简单清洗：去掉过短/空答案
    clean=[]
    for ex in mapped:
        if len(ex["input"])<3 or len(ex["output"])<1: continue
        clean.append(ex)

    # 切分
    random.shuffle(clean)
    n=len(clean)
    n_train=int(n*args.train_ratio)
    train, dev = clean[:n_train], clean[n_train:]

    # 保存
    os.makedirs(os.path.dirname(args.out_train), exist_ok=True)
    with open(args.out_train, "w", encoding="utf-8") as f:
        for ex in train: f.write(json.dumps(ex, ensure_ascii=False)+"\n")
    with open(args.out_eval, "w", encoding="utf-8") as f:
        for ex in dev: f.write(json.dumps(ex, ensure_ascii=False)+"\n")

    print(f"[make_math_json] format={fmt} total={n} train={len(train)} dev={len(dev)}")
    # 打印一点样例方便肉眼确认
    for i in range(min(2,len(train))):
        print(f"sample[{i}]:", train[i]["instruction"][:60], "|Q|", len(train[i]["input"]), "|A|", len(train[i]["output"]))

if __name__ == "__main__":
    main()

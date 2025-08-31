#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json, argparse, random
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", required=True)
    ap.add_argument("--out_list", required=True)
    ap.add_argument("--size", type=int, default=20000)
    ap.add_argument("--bins", type=int, default=8)  # 按输入长度分成若干桶
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    # 读入并粗略估计长度
    items=[]
    with open(args.train_json, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            ex=json.loads(line)
            q=(ex.get("instruction","")+"\n"+ex.get("input","")).strip() or ex.get("input","")
            items.append((i, len(q)))

    if not items: raise RuntimeError("empty train.json")

    # 分桶
    items.sort(key=lambda x:x[1])
    n=len(items)
    per_bin=args.size//args.bins
    sel=[]
    for b in range(args.bins):
        l=int(n*b/args.bins)
        r=int(n*(b+1)/args.bins)
        bucket=items[l:r]
        if not bucket: continue
        take=min(per_bin, len(bucket))
        sel.extend(random.sample(bucket, take))

    # 不足则补齐
    if len(sel)<args.size:
        left=[x for x in items if x not in sel]
        need=args.size-len(sel)
        sel.extend(random.sample(left, min(need, len(left))))

    sel=sorted(set([i for i,_ in sel]))
    with open(args.out_list, "w", encoding="utf-8") as f:
        for i in sel: f.write(str(i)+"\n")
    print(f"[sample_witness_list] wrote {len(sel)} ids to {args.out_list}")

if __name__ == "__main__":
    main()

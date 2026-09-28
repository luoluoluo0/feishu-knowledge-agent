"""汇总一份评测结果文件：各类别通过率 + 检索命中/MRR。供回归对比用。"""
import json, sys
from collections import defaultdict
from pathlib import Path

def summarize(path: str) -> dict:
    cats = defaultdict(lambda: {"n": 0, "pass": 0, "chunk_hit": 0, "item_hit": 0, "rr": 0.0})
    total_lat = []
    for line in Path(path).read_text(encoding="utf-8").strip().splitlines():
        o = json.loads(line)
        c = cats[o["category"]]
        c["n"] += 1
        if o.get("passed"):
            c["pass"] += 1
        ret = (o.get("checks") or {}).get("retrieval")
        if ret:
            if ret.get("chunk_hit"):
                c["chunk_hit"] += 1
                rank = ret.get("hit_rank") or 0
                if rank > 0:
                    c["rr"] += 1.0 / rank
            if ret.get("item_hit"):
                c["item_hit"] += 1
        lat = o.get("latency_ms")
        if lat:
            total_lat.append(lat)
    return cats, (sum(total_lat) / len(total_lat) / 1000 if total_lat else 0)

if __name__ == "__main__":
    cats, avg_lat = summarize(sys.argv[1])
    print(f"== {sys.argv[1]}  （平均时延 {avg_lat:.1f}s）")
    for cat, c in cats.items():
        mrr = c["rr"] / c["n"] if c["n"] else 0
        print(f"  {cat:<16} {c['n']:>3} 条  pass {c['pass']/c['n']:>6.1%}  "
              f"item_hit {c['item_hit']/c['n']:>6.1%}  chunk_hit {c['chunk_hit']/c['n']:>6.1%}  MRR {mrr:.3f}")

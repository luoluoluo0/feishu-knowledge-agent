"""并发压测：吞吐 / 延迟分位 / 首 token 延迟（TTFT）/ 显存采样。

部署报告「性能表」的数据来源。对任何 OpenAI 兼容端点通用
（vLLM / Ollama / llama-server），用真实测试集问题当负载——
压的是生产形状的流量，不是 hello world。

设计：
- 每档并发先发 5 条预热（vLLM 首请求要编译 CUDA graph，不预热
  会把冷启动算进数字，虚低吞吐）；
- 流式请求拿 TTFT：第一个 SSE data 块到达的时刻；
- 显存由后台线程每 0.5s 采样 nvidia-smi（找不到命令就跳过）；
- 每档并发 R 条请求、c 个 worker，吞吐 = R / 墙钟时间。

用法：
  python bench_serving.py --endpoint http://127.0.0.1:8001 --model intent-router
  python bench_serving.py --endpoint http://127.0.0.1:8001 --model intent-router \
      --concurrency 1,4,8,16 --requests 96 --tag vllm_fp16
"""

import argparse
import json
import shutil
import subprocess
import threading
import time
from collections import defaultdict
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"

SYSTEM_PROMPT = """你是论文资料问答系统的意图分类模块。判断用户问题属于哪一类意图，只输出 JSON。

输出格式：{"intent": "六选一"}"""


def percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = min(int(len(sorted_vals) * q), len(sorted_vals) - 1)
    return sorted_vals[idx]


class GpuSampler:
    """后台每 0.5s 采一次显存占用（MB）。nvidia-smi 不可用时静默跳过。"""

    def __init__(self):
        self.available = shutil.which("nvidia-smi") is not None
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if not self.available:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                ).stdout.strip()
                self.samples.append(float(out.splitlines()[0]))
            except Exception:
                pass
            self._stop.wait(0.5)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def summary(self) -> dict:
        if not self.samples:
            return {"available": False}
        s = sorted(self.samples)
        return {
            "available": True,
            "used_mb_avg": round(sum(s) / len(s)),
            "used_mb_max": round(s[-1]),
        }


def bench_level(url: str, model: str, questions: list[str], concurrency: int,
                total: int, timeout: int) -> dict:
    latencies: list[float] = []
    ttfts: list[float] = []
    errors = 0
    lock = threading.Lock()

    def one(idx: int) -> None:
        nonlocal errors
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": questions[idx % len(questions)]},
            ],
            "temperature": 0,
            "max_tokens": 64,
            "stream": True,
        }
        t0 = time.perf_counter()
        ttft = None
        try:
            with requests.post(url, json=payload, stream=True, timeout=timeout) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines(decode_unicode=True):
                    if ttft is None and line and line.startswith("data:") and "[DONE]" not in line:
                        ttft = (time.perf_counter() - t0) * 1000
                    if line and line.startswith("data:") and "[DONE]" in line:
                        break
        except Exception:
            errors += 1
            return
        total_ms = (time.perf_counter() - t0) * 1000
        with lock:
            if ttft is not None:
                ttfts.append(ttft)
            latencies.append(total_ms)

    t0 = time.time()
    threads = []
    for i in range(total):
        t = threading.Thread(target=one, args=(i,))
        # 简单信号量控制并发，比线程池更省线程
        threads.append(t)
    sem = threading.Semaphore(concurrency)
    results_lock = threading.Lock()

    def guarded(idx: int):
        with sem:
            one(idx)

    threads = [threading.Thread(target=guarded, args=(i,)) for i in range(total)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0

    latencies.sort()
    ttfts.sort()
    return {
        "concurrency": concurrency,
        "requests": total,
        "errors": errors,
        "wall_s": round(wall, 2),
        "throughput_rps": round(total / wall, 2) if wall else None,
        "latency_ms_p50": round(percentile(latencies, 0.5)) if latencies else None,
        "latency_ms_p90": round(percentile(latencies, 0.9)) if latencies else None,
        "latency_ms_avg": round(sum(latencies) / len(latencies)) if latencies else None,
        "ttft_ms_p50": round(percentile(ttfts, 0.5)) if ttfts else None,
        "ttft_ms_p90": round(percentile(ttfts, 0.9)) if ttfts else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tag", default="bench")
    parser.add_argument("--concurrency", default="1,4,8,16")
    parser.add_argument("--requests", type=int, default=96, help="每档并发发多少条请求")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    tests = [json.loads(l) for l in open(DATA_DIR / "splits" / "test.jsonl", encoding="utf-8") if l.strip()]
    questions = [t["question"] for t in tests]
    url = f"{args.endpoint.rstrip('/')}/v1/chat/completions"

    print(f"预热 5 条...")
    bench_level(url, args.model, questions, concurrency=1, total=5, timeout=args.timeout)

    gpu = GpuSampler()
    gpu.start()
    results = []
    try:
        for c in [int(x) for x in args.concurrency.split(",")]:
            print(f"压测并发={c}，{args.requests} 条请求...")
            r = bench_level(url, args.model, questions, concurrency=c,
                            total=args.requests, timeout=args.timeout)
            results.append(r)
            print(f"  吞吐 {r['throughput_rps']} req/s | 延迟 p50={r['latency_ms_p50']}ms "
                  f"p90={r['latency_ms_p90']}ms | TTFT p50={r['ttft_ms_p50']}ms | 错误 {r['errors']}")
    finally:
        gpu.stop()

    report = {
        "endpoint": args.endpoint,
        "model": args.model,
        "tag": args.tag,
        "levels": results,
        "gpu_memory": gpu.summary(),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"bench_{args.tag}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告: {path}")


if __name__ == "__main__":
    main()

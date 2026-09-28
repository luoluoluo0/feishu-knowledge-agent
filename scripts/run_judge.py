import argparse
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

try:
    from app.llm_judge import EVAL_DIR, run_judge
except ModuleNotFoundError as exc:
    print(f"缺少依赖：{exc.name}")
    print("先 conda activate base 再跑。")
    raise SystemExit(1)


# 离线 LLM 评判：不重新提问，只判 data/eval 里最新一份 e2e 结果的已存答案。
#
#   python scripts/run_judge.py                 # 全量
#   python scripts/run_judge.py --limit 5       # 先跑几条看判定质量
#   python scripts/run_judge.py --result result_e2e_xxx.jsonl
#
# 结果落盘 data/eval/judged_<时间戳>.jsonl，/admin/eval-report 读最新一份。


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM-as-Judge 离线评判 e2e 答案")
    parser.add_argument("--result", default=None, help="e2e 结果文件名，默认取最新一份")
    parser.add_argument("--limit", type=int, default=None, help="只评判前 N 条")
    parser.add_argument("--concurrency", type=int, default=6, help="并发数，默认 6")
    args = parser.parse_args()

    if args.result:
        result_path = EVAL_DIR / args.result
        if not result_path.exists():
            print(f"找不到 {result_path}")
            return 1
    else:
        candidates = sorted(EVAL_DIR.glob("result_e2e_*.jsonl"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            print("data/eval 下没有 e2e 结果文件。")
            return 1
        result_path = candidates[-1]

    print(f"评判对象：{result_path.name}")
    last_line = [""]

    def on_progress(done: int, total: int) -> None:
        line = f"进度 {done}/{total}"
        print(line + " " * max(0, len(last_line[0]) - len(line)) + "\r", end="", flush=True)
        last_line[0] = line

    summary = run_judge(
        result_path,
        limit=args.limit,
        concurrency=args.concurrency,
        on_progress=on_progress,
    )
    print()
    print(f"完成：{summary['total']} 条，通过率 {summary['ok_rate']}，"
          f"编造 {summary['fabricated']} 条")
    for cat, s in sorted(summary["by_category"].items()):
        print(f"  {cat:16s} n={s['n']:3d}  ok={s['ok_rate']}  编造={s['fabricated']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

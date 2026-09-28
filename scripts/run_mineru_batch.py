import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

PAPER_DIR = PROJECT_DIR / "data" / "raw" / "papers"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed" / "mineru_json"
FAILED_LOG = PROJECT_DIR / "data" / "processed" / "mineru_failed.txt"

# mineru-kit 可能不在 PATH 上（没激活 conda 环境）。按这些位置依次找。
MINERU_SEARCH_ROOTS = [
    Path("D:/anaconda3/envs"),
    Path("C:/ProgramData/anaconda3/envs"),
    Path.home() / "anaconda3/envs",
    Path.home() / "miniconda3/envs",
]

# 优先用这个环境：它的 torch 是 cu128，原生支持 RTX 5060（sm_120）。
PREFERRED_ENV = "torch5060"


def find_mineru() -> str | None:
    """定位 mineru-kit 可执行文件。"""

    on_path = shutil.which("mineru-kit")
    if on_path:
        return on_path

    found: list[Path] = []
    for root in MINERU_SEARCH_ROOTS:
        if not root.exists():
            continue
        for env in sorted(root.iterdir()):
            for name in ("mineru-kit.exe", "mineru-kit"):
                candidate = env / "Scripts" / name
                if candidate.exists():
                    found.append(candidate)

    if not found:
        return None

    # 优先用 torch5060，其余按路径顺序取第一个。
    for candidate in found:
        if PREFERRED_ENV in str(candidate):
            return str(candidate)
    return str(found[0])


# 逐篇调用 MinerU 解析 PDF。
#
# 为什么不直接用 `mineru-kit parse <目录>`：
# 目录模式下一篇卡住会让整批停住。实测有一篇（8 页、14 张图，完全正常的文件）
# 在所有计算阶段都跑到 100% 之后卡死在输出环节，CPU 满载但再也不出结果。
#
# 逐篇调用之后，单篇卡住只影响它自己：超时就杀掉，继续下一篇。
# 已完成的会被跳过，所以可以随时中断、随时续跑。


def load_done() -> set[str]:
    """已产出 json 的文件名（不含扩展名）。"""

    return {p.stem for p in OUTPUT_DIR.glob("*.json")}


def list_papers(done: set[str]) -> list[Path]:
    papers = sorted(PAPER_DIR.glob("*.pdf"))
    return [p for p in papers if p.stem not in done]


def parse_one(mineru_cmd: str, pdf: Path, timeout: int) -> tuple[bool, str]:
    """解析单篇。返回 (是否成功, 说明)。"""

    cmd = [
        mineru_cmd,
        "parse",
        str(pdf),
        "-o",
        str(OUTPUT_DIR),
        "--tier",
        "basic",
        "--ocr-mode",
        "txt",
        "-f",
        "middle_json",
    ]

    started = time.perf_counter()

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
    except FileNotFoundError:
        return False, f"找不到可执行文件：{mineru_cmd}"

    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        # 超时：连同子进程一起杀掉，否则渲染 worker 会残留。
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        elapsed = time.perf_counter() - started
        return False, f"超时 {elapsed:.0f}s（上限 {timeout}s），已终止"

    elapsed = time.perf_counter() - started
    produced = OUTPUT_DIR / f"{pdf.stem}.json"

    if process.returncode != 0:
        return False, f"退出码 {process.returncode}（{elapsed:.0f}s）"
    if not produced.exists():
        return False, f"进程正常退出但没有产出（{elapsed:.0f}s）"

    return True, f"{elapsed:.0f}s  {produced.stat().st_size / 1024:.0f} KB"


def main() -> int:
    parser = argparse.ArgumentParser(description="逐篇调用 MinerU 解析 PDF")
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="单篇超时秒数，默认 300。正常一篇 15-30 秒，超过太多说明卡住了。",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="本次最多处理几篇，默认全部"
    )
    parser.add_argument(
        "--mineru",
        default="",
        help="mineru-kit 的完整路径。默认自动在 conda 环境里找。",
    )
    args = parser.parse_args()

    mineru_cmd = args.mineru or find_mineru()
    if not mineru_cmd:
        print("找不到 mineru-kit。")
        print("请用 --mineru 指定，例如：")
        print("  python scripts/run_mineru_batch.py --mineru D:/anaconda3/envs/torch5060/Scripts/mineru-kit.exe")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    done = load_done()
    pending = list_papers(done)

    if args.limit > 0:
        pending = pending[: args.limit]

    print(f"mineru-kit：{mineru_cmd}")
    print(f"PDF 目录  ：{PAPER_DIR}")
    print(f"输出目录  ：{OUTPUT_DIR}")
    print(f"已完成    ：{len(done)} 篇")
    print(f"待处理    ：{len(pending)} 篇")
    print(f"单篇超时  ：{args.timeout} 秒")
    print("=" * 66)

    if not pending:
        print("没有待处理的文件。")
        return 0

    ok = failed = 0
    failures: list[tuple[str, str]] = []

    for index, pdf in enumerate(pending, start=1):
        name = pdf.stem
        print(f"[{index:>3}/{len(pending)}] {name[:52]}", end="  ", flush=True)

        success, note = parse_one(mineru_cmd, pdf, args.timeout)

        if success:
            ok += 1
            print(f"✓  {note}")
        else:
            failed += 1
            failures.append((name, note))
            print(f"✗  {note}")

    print("=" * 66)
    print(f"成功 {ok}   失败 {failed}")

    if failures:
        print()
        print("失败清单：")
        for name, note in failures:
            print(f"  {name}")
            print(f"      {note}")

        FAILED_LOG.write_text(
            "\n".join(f"{name}\t{note}" for name, note in failures),
            encoding="utf-8",
        )
        print()
        print(f"已写入：{FAILED_LOG}")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

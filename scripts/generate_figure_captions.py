"""批量为论文图表生成 VLM 结构化 caption（方案 A 环节 2）。

数据源：data/processed/mineru_blocks/*.jsonl 里 block_type=figure 的块
（自带 item_id/page/caption/image_path 关联，不需要额外建索引）。
输出：data/processed/figure_captions.jsonl，断点续跑（已生成的跳过）。

VLM 走硅基流动 OpenAI 兼容接口，图片以 base64 data URL 内联。
设计要点见 docs/project2-multimodal-plan.md：只述可见内容、读不准标
uncertain、输出带 confidence——幻觉防线。

用法：
  python scripts/generate_figure_captions.py             # 全量（自动跳过已完成）
  python scripts/generate_figure_captions.py --limit 5   # 先试 5 张
"""

import argparse
import base64
import json
import logging
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from app.config import get_settings  # noqa: E402

BLOCKS_DIR = PROJECT_DIR / "data" / "processed" / "mineru_blocks"
OUT_PATH = PROJECT_DIR / "data" / "processed" / "figure_captions.jsonl"
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
# 硅基流动 2026-09 在架的 VL 系列（Qwen2.5-VL 已下线，用旧名会 403）。
# 可用档位：Qwen3-VL-8B（便宜）/ 32B（默认，质量优先）/ 30B-A3B（MoE 快）。
DEFAULT_MODEL = "Qwen/Qwen3-VL-32B-Instruct"

CAPTION_PROMPT = """你是学术论文图表分析助手。只根据图片中可见的内容输出 JSON，不要编造图中不存在的信息。

上下文：论文《{title}》第 {page} 页的图表。原文图注：{caption}

输出 JSON，字段如下：
- "chart_type": 图表类型，取值 line_chart / bar_chart / pie_chart / flowchart / diagram / table_image / map / other
- "axes": {{"x": "横轴含义(含单位)", "y": "纵轴含义(含单位)"}}，非坐标类图表填 null
- "key_claims": [5~8 条可直接被问答引用的事实。要求穷举式覆盖：地图类写出各区域/颜色的空间分布（哪里颜色深、哪里是热点/冷点、具体地名）；流程图按顺序写出每个环节及其连接关系（含分支走向）；关系图写出核心节点及其全部子项；统计图写出关键数值和对比结论。图中有可读数值的要写出数值；读不准的数值用"约"表述或干脆不写]
- "summary": 一句话概括这张图展示了什么
- "confidence": "high" 或 "medium" 或 "low"——对你整体解读把握的自评；图片模糊、内容看不懂时必须如实标 low

只输出 JSON，不要输出其他文字。"""

logger = logging.getLogger(__name__)


def load_figure_blocks() -> list[dict]:
    """从 mineru_blocks 收集全部 figure 块（按 image_path 去重）。"""

    seen: dict[str, dict] = {}
    for path in sorted(BLOCKS_DIR.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("block_type") != "figure":
                continue
            image_path = str(row.get("image_path") or "")
            if not image_path:
                continue
            # 不同块可能引用同一张图，保留信息最全的一条
            key = image_path
            if key not in seen or len(str(row.get("caption") or "")) > len(
                str(seen[key].get("caption") or "")
            ):
                seen[key] = row
    return list(seen.values())


def encode_image(image_path: Path) -> str:
    """图片转 base64 data URL。"""

    suffix = image_path.suffix.lower().lstrip(".") or "png"
    if suffix == "jpg":
        suffix = "jpeg"
    data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/{suffix};base64,{data}"


def call_vlm(
    base_url: str, model: str, api_key: str, data_url: str, prompt: str, timeout: int = 120
) -> str:
    """调硅基流动 OpenAI 兼容接口，返回模型文本输出。"""

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "temperature": 0.1,
        "max_tokens": 800,
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return str(body["choices"][0]["message"]["content"] or "")


def parse_caption_json(raw: str) -> dict | None:
    """从模型输出里抠出 JSON。失败返回 None（记录原因后跳过）。"""

    text = raw.strip()
    if text.startswith("```"):
        # 剥掉 markdown 代码栅栏
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj.get("key_claims"), list) or not obj.get("summary"):
        return None
    obj["key_claims"] = [str(c)[:200] for c in obj["key_claims"]][:4]
    obj["confidence"] = obj.get("confidence") if obj.get("confidence") in (
        "high",
        "medium",
        "low",
    ) else "medium"
    return obj


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 张（0=全量）")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--sleep", type=float, default=1.0, help="请求间隔秒")
    args = parser.parse_args()

    settings = get_settings()
    api_key = settings.silicon_api_key
    if not api_key:
        print("没有读到 SILICON_API_KEY，请检查 .env")
        return 1

    blocks = load_figure_blocks()
    if args.limit:
        blocks = blocks[: args.limit]

    done: set[str] = set()
    if OUT_PATH.exists():
        for line in OUT_PATH.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                done.add(json.loads(line).get("image_path"))
    todo = [b for b in blocks if b.get("image_path") not in done]
    print(f"figure 块 {len(blocks)} 张，已完成 {len(done)}，待处理 {len(todo)}")
    if not todo:
        return 0

    out_file = OUT_PATH.open("a", encoding="utf-8")
    ok = failed = 0
    for index, block in enumerate(todo, start=1):
        image_rel = str(block.get("image_path"))
        image_path = PROJECT_DIR / image_rel
        if not image_path.exists():
            print(f"  [{index}/{len(todo)}] 图片缺失，跳过：{image_rel}")
            failed += 1
            continue

        prompt = CAPTION_PROMPT.format(
            title=block.get("item_id", ""),
            page=block.get("page", "?"),
            caption=str(block.get("caption") or block.get("text") or "")[:300],
        )
        record = {
            "image_path": image_rel,
            "item_id": block.get("item_id"),
            "page": block.get("page"),
            "paper_caption": str(block.get("caption") or ""),
            "vlm": None,
            "error": None,
        }
        try:
            data_url = encode_image(image_path)
            raw = call_vlm(args.base_url, args.model, api_key, data_url, prompt)
            parsed = parse_caption_json(raw)
            if parsed is None:
                record["error"] = "输出无法解析为合法 JSON"
                failed += 1
            else:
                record["vlm"] = parsed
                ok += 1
        except Exception as exc:  # 网络与接口错误：记录后继续，断点续跑兜底
            record["error"] = f"{type(exc).__name__}: {exc}"[:200]
            failed += 1

        out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        out_file.flush()
        mark = "ok" if record["vlm"] else "ERR"
        print(f"  [{index}/{len(todo)}] {image_rel} {mark}", flush=True)
        time.sleep(args.sleep)

    out_file.close()
    print(f"\n完成：成功 {ok}，失败 {failed}。输出：{OUT_PATH.name}")
    print("下一步：python scripts/update_figure_chunks.py --dry-run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

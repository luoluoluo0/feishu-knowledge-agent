import json
import re


# 解析 LLM 输出的 JSON。
#
# 提示词里虽然写了「只输出 JSON」，模型偶尔还是会带 ```json 围栏或
# 一句前置说明。这类输出直接 json.loads 会失败，然后各调用方
# （planner、coreference、intent）就得走兜底——planner 兜底是退化为
# 全库检索，白白损失规划质量。所以在这里统一先清理再解析。


_FENCE_START = re.compile(r"^```[a-zA-Z0-9_-]*\s*")
_FENCE_END = re.compile(r"\s*```$")


def parse_llm_json(text: str) -> dict | list | None:
    """把模型输出解析成 JSON 对象，容忍围栏和前后缀说明文字。

    返回 None 表示解析失败，调用方走各自的降级路径。
    """

    if not isinstance(text, str):
        return None

    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = _FENCE_START.sub("", cleaned)
        cleaned = _FENCE_END.sub("", cleaned)

    # 模型偶尔在 JSON 前后加说明文字，取最外层花括号之间的部分。
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        cleaned = cleaned[start : end + 1]

    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return None

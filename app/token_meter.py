"""一次请求的 token 计量器。

挂在 build_run_config 的 callbacks 上：意图识别、查询改写、Agent 循环、
Planner 的每一步共用同一份 config，每次 LLM 调用结束都会触发 on_llm_end，
把用量累加起来。请求结束时由 /admin/logs 记录、问答详情展示——观测聚合
走 Langfuse（见 /admin/observe），这里补的是「单次视角」和本地持久层：
不依赖 Langfuse 容器活着。

为什么不事后从 state 里的消息求和：流式路径与 Planner 的调用不都留全量
消息，回调是 invoke / stream 两条路都会触发的那一个点。
"""

from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


class TokenMeter(BaseCallbackHandler):
    """累加本次请求所有 LLM 调用的 token。计量失败绝不影响问答。"""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.llm_calls = 0

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        try:
            usage = (getattr(response, "llm_output", None) or {}).get("token_usage") or {}
            if not usage:
                # 部分模型把用量放在 generation 的消息上而不是 llm_output。
                for generation_list in getattr(response, "generations", None) or []:
                    for generation in generation_list:
                        meta = getattr(getattr(generation, "message", None), "usage_metadata", None)
                        if meta:
                            usage = meta
                            break
                    if usage:
                        break
            if usage:
                self.input_tokens += int(
                    usage.get("prompt_tokens") or usage.get("input_tokens") or 0
                )
                self.output_tokens += int(
                    usage.get("completion_tokens") or usage.get("output_tokens") or 0
                )
                self.llm_calls += 1
        except Exception:  # noqa: BLE001 — 计量是旁路，坏了就当没这行数据
            pass

    def summary(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "llm_calls": self.llm_calls,
        }


def read_token_usage(config: Any) -> dict[str, int] | None:
    """从一次请求的 RunnableConfig 里读出 TokenMeter 的累计值。

    config 不是本次请求的（或计量器不存在）返回 None——老调用方没挂
    计量器时日志列留空，不报错。
    """

    meter = None
    try:
        meter = (config or {}).get("configurable", {}).get("token_meter")
    except AttributeError:
        return None
    return meter.summary() if meter is not None else None

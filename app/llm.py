from langchain_openai import ChatOpenAI

from app.config import Settings


def build_llm(settings: Settings) -> ChatOpenAI:
    """创建 DeepSeek 聊天模型。"""
    if not settings.deepseek_api_key:
        raise ValueError("没有读取到 DEEPSEEK_API_KEY，请检查 .env")

    return ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=0.2,
        # 不设超时的话 SDK 默认 600 秒：一次挂起的调用会占住工作线程
        # 十分钟，流式接口看起来就像永远不回复。embedding 一侧有重试，
        # 这里对齐（短暂抖动重试两次，超时交给上层降级路径）。
        timeout=settings.llm_timeout_seconds,
        max_retries=2,
    )

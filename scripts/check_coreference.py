import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from app.conversation_store import append_turn, clear_history, load_history
from app.coreference import Turn, rewrite_query


# 人工检查查询改写效果。
# 自动化测试不覆盖改写质量，因为模型输出不稳定。
# 这个脚本连问几轮，把每轮的“原问题 -> 改写后”打出来，供人工判断。

DEMO_THREAD_ID = "check-coreference-demo"

DEMO_QUESTIONS = [
    "第一篇讲了什么？",
    "它的研究方法是什么？",
    "那结论呢？",
    "第002篇用了什么数据？",
    "它的样本量是多少？",
]


def main() -> int:
    clear_history(DEMO_THREAD_ID)

    for question in DEMO_QUESTIONS:
        history = load_history(DEMO_THREAD_ID)
        result = rewrite_query(question, history)

        flag = "改写" if result.rewritten else "跳过"
        print(f"[{flag}] {result.original_question}")
        print(f"        -> {result.rewritten_query}")
        if result.item_id:
            print(f"        item_id={result.item_id}")
        print(f"        原因：{result.reason}")
        print()

        # 写入占位回答，让下一轮有历史可依。
        append_turn(
            DEMO_THREAD_ID,
            Turn(
                question=result.original_question,
                rewritten_query=result.rewritten_query,
                item_id=result.item_id,
                answer=f"（演示占位）围绕「{result.rewritten_query}」的检索结果。",
            ),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

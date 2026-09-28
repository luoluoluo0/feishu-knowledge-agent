import sys
from pathlib import Path


# 运行标准工具调用 Agent。
# 这和 run_planner_agent.py 不一样：
# - run_planner_agent.py 是“先规划，再由代码执行工具”。
# - run_tool_agent.py 是“大模型自己决定调用哪些工具”。
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from app.agent import build_agent, run_agent


def print_agent_trace(result: dict):
    """打印 Agent 的执行轨迹，方便看它有没有真的调用工具。"""

    messages = result.get("messages", [])

    print("\n========== Agent Trace ==========")
    for index, message in enumerate(messages, start=1):
        message_type = getattr(message, "type", message.__class__.__name__)
        content = getattr(message, "content", "")
        tool_calls = getattr(message, "tool_calls", None)
        name = getattr(message, "name", "")

        print(f"\n[{index}] type={message_type}")
        if name:
            print("name：", name)
        if tool_calls:
            print("tool_calls：", tool_calls)
        if content:
            preview = content[:1200]
            print("content：")
            print(preview)


def main():
    agent = build_agent()

    question = input("请输入你的问题：").strip()
    if not question:
        question = "帮我生成偏见同化这篇文献的组会汇报提纲。"

    result = run_agent(agent, question)

    print_agent_trace(result)

    print("\n========== Final Answer ==========")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()

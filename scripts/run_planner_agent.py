import sys
from pathlib import Path


# 运行可控 Planner Agent。

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from app.planner_agent import PlannerAgent, format_plan


def main():
    agent = PlannerAgent()

    question = input("请输入你的问题：").strip()
    if not question:
        question = "帮我生成偏见同化这篇文献的组会汇报提纲。"

    result = agent.answer(question)

    print("\n========== Plan ==========")
    print(format_plan(result["plan"]))

    print("\n========== Step Results Preview ==========")
    for step_result in result["step_results"]:
        print("-" * 80)
        print("步骤：", step_result["step_index"])
        print("工具：", step_result["tool"])
        print("检索问题：", step_result["query"])
        print("结果预览：")
        print(step_result["result"][:1000])

    print("\n========== Final Answer ==========")
    print(result["answer"])


if __name__ == "__main__":
    main()

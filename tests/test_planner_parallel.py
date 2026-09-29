from __future__ import annotations

import dataclasses
import time

from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.llm import build_llm
from app.planner import PlanResult, PlanStep
from app.planner_agent import PlannerAgent, _bind_output_cap


def _plan(step_count: int) -> PlanResult:
    return PlanResult(
        task_type="对比文献",
        steps=[
            PlanStep(
                tool="search_by_item",
                query=f"查询{i}",
                item_id=f"00{i}",
                reason=f"理由{i}",
            )
            for i in range(1, step_count + 1)
        ],
    )


class FakePlanner(PlannerAgent):
    """绕过 __init__（真实构造要拖起 379MB 检索栈），只测调度语义。"""

    def __init__(self, max_parallel: int, delays: dict[int, float]):
        self.settings = dataclasses.replace(
            get_settings(),
            planner_max_parallel_steps=max_parallel,
            planner_max_output_tokens=0,
        )
        self.delays = delays
        self.executed: list[int] = []

    def run_step(self, step: PlanStep) -> str:
        time.sleep(self.delays.get(int(step.item_id), 0.0))
        self.executed.append(int(step.item_id))
        return f"步骤{step.item_id}的结果"


def test_run_steps_parallel_keeps_step_order():
    planner = FakePlanner(4, delays={1: 0.25, 2: 0.05, 3: 0.05})

    started = time.perf_counter()
    results = planner.run_steps(_plan(3))
    elapsed = time.perf_counter() - started

    # 结果按步骤顺序排列——最终 prompt 里「步骤N」的编号依赖这个约定
    assert [r["step_index"] for r in results] == [1, 2, 3]
    assert [r["result"] for r in results] == [
        "步骤001的结果",
        "步骤002的结果",
        "步骤003的结果",
    ]
    # 三步串行至少 0.35s；并行下总时长由最慢的 0.25s 决定
    assert elapsed < 0.34, f"并行未生效，耗时 {elapsed:.2f}s"


def test_run_steps_fires_on_step_in_completion_order():
    planner = FakePlanner(4, delays={1: 0.25, 2: 0.02, 3: 0.02})

    completion_order: list[int] = []
    results = planner.run_steps(
        _plan(3), on_step=lambda sr: completion_order.append(sr["step_index"])
    )

    # 回调按实际完成顺序触发：慢的步骤 1 最后回调，但结果列表仍在第 1 位
    assert completion_order[-1] == 1
    assert len(completion_order) == 3
    assert results[0]["step_index"] == 1


def test_run_steps_serial_when_max_parallel_is_one():
    planner = FakePlanner(1, delays={})

    results = planner.run_steps(_plan(3))

    # 串行回退路径：执行顺序 = 步骤顺序
    assert planner.executed == [1, 2, 3]
    assert [r["step_index"] for r in results] == [1, 2, 3]


def test_parallel_steps_see_caller_contextvars():
    """池线程必须继承调用方的 ContextVar——意图和「资料N」引用注册表
    都存在 ContextVar 里，丢了对整个检索链路是静默降级。"""

    from app.citations import current_citation_registry, reset_citation_registry
    from app.query_context import get_current_intent, set_current_intent

    seen: dict = {}

    class ContextSpyPlanner(FakePlanner):
        def run_step(self, step):
            seen["intent"] = get_current_intent()
            seen["registry"] = current_citation_registry()
            return super().run_step(step)

    set_current_intent("compare_papers")
    reset_citation_registry()
    planner = ContextSpyPlanner(4, delays={})
    planner.run_steps(_plan(3))

    assert seen["intent"] == "compare_papers"
    assert seen["registry"] is not None


def test_output_cap_binds_when_positive():
    llm = _bind_output_cap(build_llm(get_settings()), 1500)

    assert getattr(llm, "kwargs", {}).get("max_tokens") == 1500


def test_output_cap_noop_when_disabled():
    llm = _bind_output_cap(build_llm(get_settings()), 0)

    assert isinstance(llm, ChatOpenAI)


def test_final_prompt_carries_concision_instruction():
    planner = FakePlanner(4, delays={})
    plan = _plan(2)
    step_results = [
        {
            "step_index": i,
            "tool": "search_by_item",
            "query": f"查询{i}",
            "item_id": f"00{i}",
            "reason": "测试",
            "result": f"结果{i}",
        }
        for i in (1, 2)
    ]

    prompt = planner.build_final_prompt("对比两篇的方法", plan, step_results)

    assert "不要复述资料原文段落" in prompt
    assert "步骤1" in prompt and "步骤2" in prompt

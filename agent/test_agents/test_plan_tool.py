"""TodoItem + PlanTool（方案 B）的离线断言。

运行方式：
    PYTHONPATH=agent python agent/test_agents/test_plan_tool.py
"""

from __future__ import annotations

import sys
import types
from datetime import datetime

from pydantic import ValidationError


# 与其它离线测试一致：先桩掉 openai，避免导入链需要真实 SDK。
openai_stub = types.ModuleType("openai")


class OfflineOpenAI:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


openai_stub.OpenAI = OfflineOpenAI
sys.modules.setdefault("openai", openai_stub)

from my_agents.core.state import AgentState, TaskState, TodoItem
from my_agents.tools.builtin.plan_tool import PlanTool


def _new_state() -> AgentState:
    return AgentState(session_id="s1", user_id="u1", namespace="n1")


# --------------------------------------------------------------------- TaskState
def test_status_state_machine():
    ts = TaskState()
    assert ts.status == "idle"

    ts.transition("in_progress")
    assert ts.status == "in_progress"

    ts.transition("waiting_user")
    ts.transition("in_progress")
    ts.transition("done")
    assert ts.status == "done"

    # 非法迁移：done -> waiting_user 不被允许
    try:
        ts.transition("waiting_user")
    except ValueError as exc:
        assert "非法状态转移" in str(exc)
    else:
        raise AssertionError("done -> waiting_user 应该被拒绝")

    # 非法字面量同样应被状态机拦下（不泄漏 pydantic 错误）
    try:
        ts.transition("flying")
    except ValueError:
        pass
    else:
        raise AssertionError("非法状态值应该被拒绝")


def test_todos_lifecycle_and_invariant():
    ts = TaskState()
    items = ts.set_todos(["定位慢查询", "  ", "加索引", "压测验证"])

    # 空白项被过滤，id 连续生成 t1/t2/t3
    assert [t.id for t in items] == ["t1", "t2", "t3"]
    assert all(t.status == "pending" for t in items)
    assert ts.progress == "0/3"

    # 按 id 开始
    ts.start_todo("t1")
    assert ts.todos[0].status == "in_progress"

    # 至多一个 in_progress：开始 t2 应把 t1 降回 pending
    ts.start_todo("2")  # 1 基序号引用
    assert ts.todos[1].status == "in_progress"
    in_progress = [t for t in ts.todos if t.status == "in_progress"]
    assert len(in_progress) == 1 and in_progress[0].id == "t2"
    assert ts.todos[0].status == "pending"

    # 完成与进度
    ts.complete_todo("t2")
    assert ts.todos[1].status == "completed"
    assert ts.progress == "1/3"

    # 引用不存在的待办应抛错
    try:
        ts.start_todo("t99")
    except ValueError as exc:
        assert "未找到待办项" in str(exc)
    else:
        raise AssertionError("不存在的待办应抛 ValueError")


def test_decisions_constraints_questions():
    ts = TaskState()
    assert ts.add_decision("采用增量索引") is True
    assert ts.add_decision("   ") is False           # 空内容被拒
    assert ts.add_constraint("不改表结构") is True
    assert ts.raise_question("是否允许停机窗口?") is True
    assert ts.raise_question("是否允许停机窗口?") is False  # 去重
    assert ts.resolve_question("是否允许停机窗口?") is True
    assert ts.resolve_question("不存在的问题") is False

    assert ts.decisions == ["采用增量索引"]
    assert ts.constraints == ["不改表结构"]
    assert ts.open_questions == []


def test_plan_tool_question_status_flow():
    state = _new_state()
    tool = PlanTool(state=state, expandable=False)

    assert state.task_state.status == "idle"
    out = tool.run({"action": "raise_question", "question": "请提供完整 traceback"})
    assert out.startswith("❓")
    assert state.task_state.status == "waiting_user"
    assert state.task_state.open_questions == ["请提供完整 traceback"]

    out = tool.run({"action": "resolve_question", "question": "请提供完整 traceback"})
    assert out.startswith("✅")
    assert state.task_state.open_questions == []
    assert state.task_state.status == "in_progress"


def test_render_compact():
    ts = TaskState()
    ts.transition("in_progress")
    ts.goal = "排查登录超时"
    ts.set_todos(["定位慢查询", "加缓存"])
    ts.start_todo("t1")
    ts.complete_todo("t2")
    ts.add_constraint("不停机")
    text = ts.render()

    assert "status=in_progress" in text
    assert "进度=1/2" in text
    assert "目标：排查登录超时" in text
    assert "▶t1" in text and "✓t2" in text
    assert "约束：不停机" in text


def test_todo_default_dict_backward_compatible():
    # 空 todos 时 TaskState.dict() 形状保持不变（与既有测试断言一致）
    assert TaskState().dict() == {
        "status": "idle",
        "goal": "",
        "todos": [],
        "constraints": [],
        "decisions": [],
        "open_questions": [],
    }
    # 非空时 todos 序列化为结构化字典
    ts = TaskState()
    ts.set_todos(["a"])
    assert ts.dict()["todos"][0] == {
        "id": "t1",
        "content": "a",
        "status": "pending",
    }


# ---------------------------------------------------------------------- PlanTool
def test_plan_tool_run_dispatch():
    state = _new_state()
    tool = PlanTool(state=state, expandable=False)

    assert tool.run({"action": "set_goal", "goal": "迁移数据库"}).startswith("✅")
    assert state.task_state.goal == "迁移数据库"

    out = tool.run({"action": "set_todos", "todos": ["导出", "转换", "导入"]})
    assert out.startswith("✅") and "3 条待办" in out

    assert tool.run({"action": "start_todo", "todo": "t1"}).startswith("✅")
    assert state.task_state.todos[0].status == "in_progress"

    assert tool.run({"action": "complete_todo", "todo": "1"}).startswith("✅")
    assert state.task_state.todos[0].status == "completed"

    assert tool.run({"action": "add_decision", "decision": "分批迁移"}).startswith("✅")

    # 状态机：idle 不能直达 done，需先进入 in_progress
    assert tool.run({"action": "set_status", "status": "done"}).startswith("❌")
    assert tool.run({"action": "set_status", "status": "in_progress"}).startswith("✅")
    assert tool.run({"action": "set_status", "status": "done"}).startswith("✅")
    assert state.task_state.status == "done"

    # 非法状态迁移经工具返回友好错误，而非抛异常
    assert tool.run({"action": "set_status", "status": "waiting_user"}).startswith("❌")
    # 未知 action
    assert tool.run({"action": "fly"}).startswith("❌")
    # 缺少必需的 action
    assert tool.run({}).startswith("❌")


def test_plan_tool_expanded_subtools():
    state = _new_state()
    tool = PlanTool(state=state, expandable=True)

    expanded = tool.get_expanded_tools() or []
    names = {t.name for t in expanded}
    expected = {
        "plan_set_goal", "plan_set_todos", "plan_start_todo", "plan_complete_todo",
        "plan_add_decision", "plan_add_constraint", "plan_raise_question",
        "plan_resolve_question", "plan_set_status", "plan_view",
    }
    assert expected <= names, f"缺少子工具: {expected - names}"

    by_name = {t.name: t for t in expanded}
    # set_todos 的子工具应把数组参数识别为 array
    todos_param = next(p for p in by_name["plan_set_todos"].get_parameters() if p.name == "todos")
    assert todos_param.type == "array"

    # 直接调用展开后的子工具，写入应落到同一个 state
    assert by_name["plan_set_todos"].run({"todos": ["调研", "编码"]}).startswith("✅")
    assert by_name["plan_start_todo"].run({"todo": "t1"}).startswith("✅")
    assert state.task_state.todos[0].status == "in_progress"
    assert "调研" in by_name["plan_view"].run({})


def test_plan_tool_survives_state_reset():
    state = _new_state()
    tool = PlanTool(state=state, expandable=False)
    tool.run({"action": "set_todos", "todos": ["a", "b"]})
    assert len(state.task_state.todos) == 2

    # reset() 会重建 task_state；工具持有 AgentState，应继续写入新的 task_state
    state.reset()
    assert state.task_state.todos == []
    tool.run({"action": "set_todos", "todos": ["x"]})
    assert len(state.task_state.todos) == 1
    assert tool.task_state is state.task_state


if __name__ == "__main__":
    test_status_state_machine()
    test_todos_lifecycle_and_invariant()
    test_decisions_constraints_questions()
    test_plan_tool_question_status_flow()
    test_render_compact()
    test_todo_default_dict_backward_compatible()
    test_plan_tool_run_dispatch()
    test_plan_tool_expanded_subtools()
    test_plan_tool_survives_state_reset()
    print(f"PlanTool/TodoItem 断言通过：{datetime.now().isoformat()}")

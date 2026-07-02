"""计划工具

LLM 显式计划工具更新状态
让 Agent 通过函数调用维护运行时 ``TaskState``：制定/更新待办、记录决策与约束、
提出/消解未决问题、迁移任务状态。所有写操作都作用在 Agent 持有的 ``AgentState``
上，属于短期运行时状态，不写入长期记忆。

与 MemoryTool 一致，支持两种使用模式：
- 普通模式：``run({"action": ...})`` 按 action 分发；
- 展开模式：每个 ``@tool_action`` 方法展开为独立子工具（``plan_set_todos`` 等）。
"""

from typing import Any, Dict, List

from ..base import Tool, ToolParameter, tool_action
from ...core.state import AgentState


class PlanTool(Tool):
    """运行时任务状态写入工具。

    持有 ``AgentState`` 而非 ``TaskState``：``AgentState.reset()`` 会重建
    ``task_state``，因此每次都通过 ``self.task_state`` 读取当前对象，避免持有过期引用。
    """

    def __init__(self, state: AgentState, expandable: bool = False):
        super().__init__(
            name="plan",
            description="计划工具 - 维护任务待办、决策、约束、未决问题与任务状态（短期运行时状态）",
            expandable=expandable,
        )
        self.state = state

    @property
    def task_state(self):
        """始终读取 AgentState 上的当前 task_state（reset 后也安全）。"""
        return self.state.task_state

    def run(self, parameters: Dict[str, Any]) -> str:
        """执行工具（非展开模式），按 action 分发。"""
        if not self.validate_parameters(parameters):
            return "❌ 参数验证失败：缺少必需的参数"

        action = parameters.get("action")
        if action == "set_goal":
            return self._set_goal(goal=parameters.get("goal", ""))
        elif action == "set_todos":
            return self._set_todos(todos=parameters.get("todos") or [])
        elif action == "start_todo":
            return self._start_todo(todo=parameters.get("todo", ""))
        elif action == "complete_todo":
            return self._complete_todo(todo=parameters.get("todo", ""))
        elif action == "add_decision":
            return self._add_decision(decision=parameters.get("decision", ""))
        elif action == "add_constraint":
            return self._add_constraint(constraint=parameters.get("constraint", ""))
        elif action == "raise_question":
            return self._raise_question(question=parameters.get("question", ""))
        elif action == "resolve_question":
            return self._resolve_question(question=parameters.get("question", ""))
        elif action == "set_status":
            return self._set_status(status=parameters.get("status", ""))
        elif action == "view":
            return self._view()
        else:
            return f"❌ 不支持的操作: {action}"

    def get_parameters(self) -> List[ToolParameter]:
        """获取工具参数定义 - Tool 基类要求的接口。"""
        return [
            ToolParameter(
                name="action",
                type="string",
                description=(
                    "要执行的操作："
                    "set_goal(设目标), set_todos(重写待办), start_todo(开始某条待办), "
                    "complete_todo(完成某条待办), add_decision(记录决策), add_constraint(记录约束), "
                    "raise_question(提出未决问题并自动进入 waiting_user), "
                    "resolve_question(消解未决问题，全部消解后恢复 in_progress), "
                    "set_status(迁移任务状态), view(查看当前计划)"
                ),
                required=True,
            ),
            ToolParameter(name="goal", type="string", description="任务目标的一句话描述（set_goal）", required=False),
            ToolParameter(name="todos", type="array", description="待办内容字符串数组，按执行顺序排列（set_todos）", required=False),
            ToolParameter(name="todo", type="string", description="待办引用：id（如 t2）或 1 基序号（如 2）（start_todo/complete_todo）", required=False),
            ToolParameter(name="decision", type="string", description="一条已确定的决策（add_decision）", required=False),
            ToolParameter(name="constraint", type="string", description="一条硬约束（add_constraint）", required=False),
            ToolParameter(name="question", type="string", description="未决问题（raise_question/resolve_question）", required=False),
            ToolParameter(name="status", type="string", description="目标任务状态：in_progress/waiting_user/blocked/done（set_status）", required=False),
        ]

    @tool_action("plan_set_goal", "设置或刷新当前任务目标")
    def _set_goal(self, goal: str = "") -> str:
        """设置任务目标

        Args:
            goal: 任务目标的一句话描述
        """
        text = (goal or "").strip()
        if not text:
            return "❌ 目标不能为空"
        self.task_state.goal = text
        return f"✅ 已设置目标：{text}"

    @tool_action("plan_set_todos", "用命令式描述重写整张待办列表（全部置为待办）")
    def _set_todos(self, todos: List[str] = None) -> str:
        """重写待办列表

        Args:
            todos: 待办内容字符串数组，按执行顺序排列，例如 ["定位慢查询","加索引","压测验证"]
        """
        todos = todos or []
        if not isinstance(todos, list):
            return "❌ todos 必须是字符串数组"
        items = self.task_state.set_todos([str(t) for t in todos])
        if not items:
            return "⚠️ 未写入任何待办（内容为空）"
        return f"✅ 已写入 {len(items)} 条待办\n{self.task_state.render()}"

    @tool_action("plan_start_todo", "把某条待办标记为进行中（自动保持至多一个进行中）")
    def _start_todo(self, todo: str = "") -> str:
        """开始某条待办

        Args:
            todo: 待办引用，id（如 t2）或 1 基序号（如 2）
        """
        try:
            item = self.task_state.start_todo(todo)
        except ValueError as exc:
            return f"❌ {exc}"
        return f"✅ 进行中：{item.id} {item.content}\n{self.task_state.render()}"

    @tool_action("plan_complete_todo", "把某条待办标记为已完成")
    def _complete_todo(self, todo: str = "") -> str:
        """完成某条待办

        Args:
            todo: 待办引用，id（如 t2）或 1 基序号（如 2）
        """
        try:
            item = self.task_state.complete_todo(todo)
        except ValueError as exc:
            return f"❌ {exc}"
        return f"✅ 已完成：{item.id} {item.content}（进度 {self.task_state.progress}）"

    @tool_action("plan_add_decision", "记录一条已确定的决策")
    def _add_decision(self, decision: str = "") -> str:
        """记录决策

        Args:
            decision: 一条已确定的决策
        """
        if not self.task_state.add_decision(decision or ""):
            return "❌ 决策内容不能为空"
        return f"✅ 已记录决策：{decision.strip()}"

    @tool_action("plan_add_constraint", "记录一条硬约束")
    def _add_constraint(self, constraint: str = "") -> str:
        """记录约束

        Args:
            constraint: 一条硬约束，例如“不改表结构”
        """
        if not self.task_state.add_constraint(constraint or ""):
            return "❌ 约束内容不能为空"
        return f"✅ 已记录约束：{constraint.strip()}"

    @tool_action("plan_raise_question", "提出一个需要用户澄清的未决问题，并自动进入 waiting_user")
    def _raise_question(self, question: str = "") -> str:
        """提出未决问题

        Args:
            question: 需要用户澄清的问题
        """
        if not self.task_state.raise_question(question or ""):
            return "⚠️ 问题为空或已存在"
        try:
            if self.task_state.status != "waiting_user":
                if self.task_state.status in {"idle", "blocked", "done"}:
                    self.task_state.transition("in_progress")
                self.task_state.transition("waiting_user")
        except ValueError as exc:
            return f"❌ 已记录未决问题，但状态切换失败：{exc}"
        return f"❓ 已记录未决问题并等待用户：{question.strip()}"

    @tool_action("plan_resolve_question", "消解一个已澄清的未决问题，全部消解后恢复 in_progress")
    def _resolve_question(self, question: str = "") -> str:
        """消解未决问题

        Args:
            question: 要消解的问题（需与提出时文本一致）
        """
        text = (question or "").strip()
        if not self.task_state.resolve_question(text):
            return f"⚠️ 未找到该未决问题：{text}"
        if self.task_state.status == "waiting_user" and not self.task_state.open_questions:
            try:
                self.task_state.transition("in_progress")
            except ValueError as exc:
                return f"❌ 已消解未决问题，但状态切换失败：{exc}"
            return f"✅ 已消解未决问题并继续执行：{text}"
        return f"✅ 已消解未决问题：{text}"

    @tool_action("plan_set_status", "迁移任务状态（受状态机约束）")
    def _set_status(self, status: str = "") -> str:
        """迁移任务状态

        Args:
            status: 目标状态：in_progress/waiting_user/blocked/done
        """
        text = (status or "").strip()
        try:
            self.task_state.transition(text)
        except ValueError as exc:
            return f"❌ {exc}"
        return f"✅ 任务状态 -> {text}"

    @tool_action("plan_view", "查看当前任务计划与状态")
    def _view(self) -> str:
        """查看当前计划"""
        return self.task_state.render()

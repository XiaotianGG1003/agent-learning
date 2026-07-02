"""有状态 Agent 的运行时状态模型。"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, validator


TaskStatus = Literal["idle", "in_progress", "waiting_user", "blocked", "done"]
TodoStatus = Literal["pending", "in_progress", "completed", "cancelled"]


# 任务状态机：每个状态允许迁移到的目标状态集合（含自身，便于幂等赋值）。
_ALLOWED_TRANSITIONS: Dict[str, set] = {
    "idle": {"idle", "in_progress"},
    "in_progress": {"in_progress", "waiting_user", "blocked", "done"},
    "waiting_user": {"waiting_user", "in_progress", "done"},
    "blocked": {"blocked", "in_progress", "done"},
    "done": {"done", "in_progress"},  # 新一轮可以重新开始
}

# 待办状态 -> 展示符号。
_TODO_MARK = {
    "pending": "○",
    "in_progress": "▶",
    "completed": "✓",
    "cancelled": "✗",
}


class TodoItem(BaseModel):
    """单条待办项，自带状态，用于多步任务的进度跟踪。"""

    id: str
    content: str
    status: TodoStatus = "pending"


class TaskState(BaseModel):
    """结构化短期任务状态，只保存在运行时，不写入长期记忆。

    写入约定：
    - ``status`` / ``goal`` 由 Agent 框架机械维护，也可由模型经 PlanTool 微调；
    - ``todos`` / ``decisions`` / ``constraints`` / ``open_questions`` 由模型经 PlanTool 写入；
    - ``raise_question`` 会进入 ``waiting_user``，未决问题全部消解后可恢复 ``in_progress``；
    - 所有写操作都应走本类提供的方法，以维持状态机与“至多一个 in_progress”不变式。
    """

    status: TaskStatus = "idle"
    goal: str = ""
    todos: List[TodoItem] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    decisions: List[str] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)

    # TODO 旧历史压缩摘要
    # summary: str | None

    class Config:
        validate_assignment = True

    # ----------------------------------------------------------------- 任务状态机
    def transition(self, to: TaskStatus) -> None:
        """按状态机迁移 ``status``，非法迁移抛 ValueError。"""
        allowed = _ALLOWED_TRANSITIONS.get(self.status, set())
        if to not in allowed:
            raise ValueError(f"非法状态转移：{self.status} -> {to}")
        self.status = to

    # --------------------------------------------------------------------- 待办
    def set_todos(self, contents: List[str]) -> List[TodoItem]:
        """用一批命令式描述重写整张待办列表，全部置为 pending。

        重写时丢弃旧列表（含其状态），id 按位置重新生成为 ``t1``、``t2`` …。
        """
        cleaned = [text for text in (str(c).strip() for c in contents) if text]
        todos = [TodoItem(id=f"t{i + 1}", content=text) for i, text in enumerate(cleaned)]
        self.todos = todos
        return todos

    def _find_todo(self, ref: str) -> Optional[TodoItem]:
        """按 id（如 ``t2``）或 1 基序号（如 ``2``）定位待办项。"""
        key = str(ref).strip()
        for todo in self.todos:
            if todo.id == key:
                return todo
        if key.isdigit():
            idx = int(key) - 1
            if 0 <= idx < len(self.todos):
                return self.todos[idx]
        return None

    def start_todo(self, ref: str) -> TodoItem:
        """把某条待办置为 in_progress，并把其它 in_progress 降回 pending。"""
        todo = self._find_todo(ref)
        if todo is None:
            raise ValueError(f"未找到待办项：{ref}")
        for other in self.todos:
            if other is not todo and other.status == "in_progress":
                other.status = "pending"
        todo.status = "in_progress"
        return todo

    def complete_todo(self, ref: str) -> TodoItem:
        """把某条待办置为 completed。"""
        todo = self._find_todo(ref)
        if todo is None:
            raise ValueError(f"未找到待办项：{ref}")
        todo.status = "completed"
        return todo

    # --------------------------------------------------------- 决策/约束/未决问题
    # TODO 添加更新、删除方法
    def add_decision(self, decision: str) -> bool:
        text = decision.strip()
        if not text:
            return False
        self.decisions.append(text)
        return True

    def add_constraint(self, constraint: str) -> bool:
        text = constraint.strip()
        if not text:
            return False
        self.constraints.append(text)
        return True

    def raise_question(self, question: str) -> bool:
        text = question.strip()
        if not text or text in self.open_questions:
            return False
        self.open_questions.append(text)
        return True

    def resolve_question(self, question: str) -> bool:
        text = question.strip()
        before = len(self.open_questions)
        self.open_questions = [q for q in self.open_questions if q != text]
        return len(self.open_questions) < before

    # --------------------------------------------------------------- 派生与渲染
    @property
    def progress(self) -> str:
        """已完成 / 总数，例如 ``2/4``。"""
        if not self.todos:
            return "0/0"
        done = sum(1 for t in self.todos if t.status == "completed")
        return f"{done}/{len(self.todos)}"

    def render(self) -> str:
        """渲染成紧凑中文文本，供注入上下文，比裸 JSON 更省 token。"""
        lines = [f"status={self.status} 进度={self.progress}"]
        if self.goal:
            lines.append(f"目标：{self.goal}")
        if self.todos:
            todo_str = "  ".join(
                f"{_TODO_MARK.get(t.status, '?')}{t.id} {t.content}" for t in self.todos
            )
            lines.append(f"待办：{todo_str}")
        if self.constraints:
            lines.append("约束：" + "；".join(self.constraints))
        if self.decisions:
            lines.append("决策：" + "；".join(self.decisions))
        if self.open_questions:
            lines.append("未决：" + "；".join(self.open_questions))
        return "\n".join(lines)


class ToolObservation(BaseModel):
    """单轮对话中一次工具调用的结构化观察记录。"""

    tool_call_id: str
    tool_name: str
    args: Dict[str, Any] = Field(default_factory=dict)
    status: str = "success"
    result: str = ""
    error: Optional[str] = None


class AgentState(BaseModel):
    """会话级运行时状态，负责替代原来的 WorkingMemory。"""

    session_id: str
    user_id: str
    namespace: str
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    task_state: TaskState = Field(default_factory=TaskState)
    tool_results: List[ToolObservation] = Field(default_factory=list)

    @validator("messages")
    def _validate_message_roles(cls, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        allowed = {"user", "assistant", "system", "tool"}
        for message in messages:
            role = message.get("role")
            if role not in allowed:
                raise ValueError(f"invalid message role: {role}")
        return messages

    def reset(self) -> None:
        """清空易变的会话状态，同时保留 session_id、user_id 和 namespace。"""
        self.messages.clear()
        self.tool_results.clear()
        self.task_state = TaskState()

    def append_message(self, role: str, content: str, **metadata: Any) -> None:
        """向运行时状态追加一条原始对话消息。"""
        if role not in {"user", "assistant", "system", "tool"}:
            raise ValueError(f"invalid message role: {role}")
        message = {"role": role, "content": content}
        if metadata:
            message["metadata"] = metadata
        self.messages.append(message)

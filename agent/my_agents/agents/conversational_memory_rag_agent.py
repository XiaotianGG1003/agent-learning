"""基于运行时 AgentState、长期记忆工具和 RAG 工具的连续对话 Agent。"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
from datetime import datetime
from typing import Any, Literal, Optional, Union

from .function_call_agent import FunctionCallAgent, ToolApprovalDecision
from ..core.llm import HelloAgentsLLM
from ..core.state import AgentState, ToolObservation
from ..memory import MemoryConfig
from ..tools.builtin.memory_tool import MemoryTool
from ..tools.builtin.plan_tool import PlanTool
from ..tools.builtin.rag_tool import RAGTool
from ..tools.builtin.terminal_tool import TerminalTool


PermissionMode = Literal["read_only", "restricted_access", "full_access", "approval_all"]


class ConversationalMemoryRAGAgent(FunctionCallAgent):
    """使用内存态 AgentState 管理短期会话的连续对话 Agent。
    """

    DEFAULT_SYSTEM_PROMPT = (
        "你是一个严谨、简洁的中文编程助手，负责解释代码、修改代码、设计提示词、排查错误和回答知识库问题。\n\n"
        "运行状态：系统会在消息中提供 [AgentState]，其中 task_state 是本轮任务的短期控制面板。"
        "只把它当作当前任务状态使用，不要把 AgentState、完整对话或原始工具结果写入长期记忆。\n\n"
        "任务规划：简单问答、概念解释或单步检查可以直接回答，不要为了形式调用 plan。"
        "多步骤任务、代码修改、排错、需要多次工具调用或用户明确要求跟踪进度时，先使用 plan_set_todos 制定少量可执行待办；"
        "执行前用 plan_start_todo，完成后用 plan_complete_todo；关键约束用 plan_add_constraint，关键取舍用 plan_add_decision。"
        "信息不足且会阻塞任务时，调用 plan_raise_question；进入 waiting_user 后不要继续调用其它工具，等待用户补充。"
        "任务完成时调用 plan_set_status(status=\"done\")。\n\n"
        "工具选择：代码库事实必须来自工具结果，不要凭记忆编造文件、函数或行号。"
        "查文件名或目录结构用 glob_files；搜代码内容用 grep_code；读取文件内容用 read_file。"
        "修改已有文件用 edit_file，新建或覆盖文件用 write_file。运行测试、构建、脚本或环境检查时才用 bash；"
        "不要用 bash 代替 read_file/glob_files/grep_code 做常规代码阅读。\n\n"
        "长期信息：只有需要用户长期偏好、历史经验或项目事实时才调用 memory_search/memory_summary。"
        "只有需要知识库文档证据时才调用 rag_search 或 rag_ask。"
        "只有用户明确要求“记住/保存到记忆”或内容明显是长期稳定偏好/事实时，才考虑 memory_add；不要自动保存普通回答。\n\n"
        "权限审批：只读检索工具通常会自动执行；edit_file、write_file、bash 以及写入/删除记忆或知识库的工具可能需要用户审批。"
        "如果工具调用被拒绝，接受拒绝结果，不要反复尝试同一高风险动作；改为说明影响，并选择安全替代方案或询问用户。\n\n"
        "回答策略：如果使用了工具或检索结果，优先依据工具返回的事实回答。"
        "不要把工具输出的原始 JSON 直接暴露给用户，要提炼为自然语言；涉及代码时给出文件路径、符号名和必要行号。"
        "若工具结果与用户描述冲突，明确指出冲突和依据。若没有执行测试或命令，说明未验证。"
        "最终回答保持中文、结构清晰、结论优先。"
    )

    SAFE_TOOLS = {
        "read_file",
        "glob_files",
        "grep_code",
        "memory_search",
        "memory_summary",
        "memory_stats",
        "rag_search",
        "rag_ask",
        "rag_stats",
        "lsp",
    }
    SAFE_PREFIXES = ("plan_", "lsp_")
    PERMISSION_MODES = {"read_only", "restricted_access", "full_access", "approval_all"}

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        *,
        user_id: str = "default_user",
        session_id: Optional[str] = None,
        namespace: str = "default",
        enable_memory: bool = True,
        memory_config: Optional[MemoryConfig] = None,
        memory_types: Optional[list[str]] = None,
        enable_rag: bool = True,
        rag_knowledge_base_path: str = "./knowledge_base",
        enable_terminal: bool = False,
        terminal_workspace: str = ".",
        terminal_config: Optional[dict[str, Any]] = None,
        enable_plan: bool = True,
        max_tool_iterations: int = 3,
        permission_mode: PermissionMode = "restricted_access",
        trace_enabled: bool = False,
        trace_reasoning: bool = True,
        trace_tool: bool = True,
    ):
        if permission_mode not in self.PERMISSION_MODES:
            raise ValueError(f"不支持的权限模式: {permission_mode}")

        self.user_id = user_id
        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.namespace = namespace
        self.turn_count = 0
        self.permission_mode: PermissionMode = permission_mode
        self.trace_enabled = trace_enabled
        self.trace_reasoning = trace_reasoning
        self.trace_tool = trace_tool

        self.state = AgentState(
            session_id=self.session_id,
            user_id=user_id,
            namespace=namespace,
        )

        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt or self.DEFAULT_SYSTEM_PROMPT,
            enable_tool_calling=True,
            default_tool_choice="auto",
            max_tool_iterations=max_tool_iterations,
        )

        self.plan_tool: Optional[PlanTool] = None
        if enable_plan:
            self.plan_tool = PlanTool(state=self.state, expandable=True)
            self.add_tool(self.plan_tool)

        self.memory_tool: Optional[MemoryTool] = None
        if enable_memory:
            self.memory_tool = MemoryTool(
                user_id=user_id,
                session_id=self.session_id,
                memory_config=memory_config,
                memory_types=memory_types,
                expandable=True,
            )
            self.add_tool(self.memory_tool)

        self.rag_tool: Optional[RAGTool] = None
        if enable_rag:
            self.rag_tool = RAGTool(
                knowledge_base_path=rag_knowledge_base_path,
                namespace=namespace,
                expandable=True,
            )
            self.add_tool(self.rag_tool)

        self.terminal_tool: Optional[TerminalTool] = None
        if enable_terminal:
            self.terminal_tool = TerminalTool(
                workspace=terminal_workspace,
                expandable=True,
                **(terminal_config or {}),
            )
            self.add_tool(self.terminal_tool)

    def run(
        self,
        input_text: str,
        *,
        max_tool_iterations: Optional[int] = None,
        tool_choice: Optional[Union[str, dict]] = None,
        **kwargs: Any,
    ) -> str:
        """使用 AgentState 执行一轮对话。"""
        self.state.tool_results.clear()
        self.state.task_state.transition("in_progress")
        self.state.task_state.goal = input_text.strip()

        response = super().run(
            input_text,
            max_tool_iterations=max_tool_iterations,
            tool_choice=tool_choice,
            **kwargs,
        )


        self.state.append_message("user", input_text)
        self.state.append_message("assistant", response)
        self.turn_count += 1
        return response

    def _build_model_messages(self, input_text: str) -> list[dict[str, Any]]:
        """构造模型消息：system prompt + AgentState + 全部历史 + 当前输入。"""
        base_prompt = self.DEFAULT_SYSTEM_PROMPT
        if self.system_prompt:
            base_prompt = f"{base_prompt}\n\n补充指令：\n{self.system_prompt}"

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": f"{base_prompt}\n\n[AgentState]\n{self.state.task_state.render()}",
            }
        ]

        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": input_text})
        return messages

    def on_llm_response(
        self,
        step: int,
        reasoning: str,
        content: str,
        has_tool_calls: bool,
    ) -> None:
        """按需输出一次 LLM 响应中的推理和内容。"""
        if not self.trace_enabled:
            return

        print(f"[TRACE][LLM step {step}]")
        if self.trace_reasoning:
            reasoning_text = (reasoning or "").strip() or "<empty>"
            content_text = (content or "").strip() or "<empty>"
            print(f"reasoning: {reasoning_text}")
            print(f"content: {content_text}")
        if self.trace_tool and has_tool_calls:
            print("tool_calls:")

    def before_tool_call(
        self,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolApprovalDecision:
        """按权限模式在工具执行前决定是否审批、拒绝或修改参数。"""
        risk = self._tool_risk(tool_name)

        if self.permission_mode == "full_access":
            return ToolApprovalDecision(action="approve", arguments=arguments)

        if self.permission_mode == "read_only":
            if risk == "low":
                return ToolApprovalDecision(action="approve", arguments=arguments)
            return ToolApprovalDecision(
                action="reject",
                arguments=arguments,
                result=self._rejected_tool_result(
                    tool_name,
                    "tool call rejected by read_only permission mode",
                ),
                reason="tool call rejected by read_only permission mode",
            )

        if self.permission_mode == "restricted_access" and risk == "low":
            return ToolApprovalDecision(action="approve", arguments=arguments)

        return self._console_approval_handler(
            tool_call_id,
            tool_name,
            dict(arguments),
            self.permission_mode,
            risk,
        )

    def _tool_risk(self, tool_name: str) -> str:
        if tool_name in self.SAFE_TOOLS or tool_name.startswith(self.SAFE_PREFIXES):
            return "low"
        return "high"

    def _console_approval_handler(
        self,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        permission_mode: PermissionMode,
        risk: str,
    ) -> ToolApprovalDecision:
        print(f"[APPROVAL] mode={permission_mode} risk={risk}")
        print(f"tool_call_id: {tool_call_id}")
        print(f"tool_name: {tool_name}")
        print("arguments:")
        print(json.dumps(arguments, ensure_ascii=False, indent=2, default=str))
        print()

        choice = input("approve / reject / edit > ").strip().lower()
        if choice in {"approve", "a"}:
            return ToolApprovalDecision(action="approve", arguments=arguments)
        if choice in {"edit", "e"}:
            try:
                edited_text = self._edit_arguments_json_text(arguments)
                edited_arguments = json.loads(edited_text)
            except json.JSONDecodeError as exc:
                reason = f"tool call edit failed: invalid JSON ({exc})"
                return ToolApprovalDecision(
                    action="reject",
                    arguments=arguments,
                    result=self._rejected_tool_result(tool_name, reason),
                    reason=reason,
                )
            except (OSError, RuntimeError) as exc:
                reason = f"tool call edit failed: {exc}"
                return ToolApprovalDecision(
                    action="reject",
                    arguments=arguments,
                    result=self._rejected_tool_result(tool_name, reason),
                    reason=reason,
                )
            if not isinstance(edited_arguments, dict):
                reason = "tool call edit failed: edited arguments must be a JSON object"
                return ToolApprovalDecision(
                    action="reject",
                    arguments=arguments,
                    result=self._rejected_tool_result(tool_name, reason),
                    reason=reason,
                )
            return ToolApprovalDecision(action="edit", arguments=edited_arguments)

        return ToolApprovalDecision(
            action="reject",
            arguments=arguments,
            result=self._rejected_tool_result(tool_name, "tool call rejected by user"),
            reason="tool call rejected by user",
        )

    def _edit_arguments_json_text(self, arguments: dict[str, Any]) -> str:
        """把原始工具参数写入临时 JSON 文件，让用户直接编辑后读回。"""
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            delete=False,
        ) as temp_file:
            temp_file.write(json.dumps(arguments, ensure_ascii=False, indent=2, default=str))
            temp_file.write("\n")
            temp_path = temp_file.name

        try:
            print(f"已打开参数 JSON 文件，请修改、保存并关闭编辑器: {temp_path}")
            self._open_json_editor(temp_path)
            with open(temp_path, encoding="utf-8") as temp_file:
                return temp_file.read().strip()
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

    @staticmethod
    def _open_json_editor(path: str) -> None:
        """调用用户配置的编辑器；未配置时 Windows 使用 notepad，其它系统使用 nano。"""
        editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
        command = (
            shlex.split(editor, posix=(os.name != "nt")) + [path]
            if editor
            else (["notepad", path] if os.name == "nt" else ["nano", path])
        )
        result = subprocess.run(command)
        if result.returncode != 0:
            raise RuntimeError(f"editor exited with code {result.returncode}")

    @staticmethod
    def _rejected_tool_result(tool_name: str, reason: str) -> str:
        return json.dumps(
            {
                "status": "error",
                "error": reason,
                "tool_name": tool_name,
            },
            ensure_ascii=False,
        )

    def on_tool_complete(
        self,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        result: str
    ) -> Optional[str]:
        """工具执行完成后的回调：记录结果到 AgentState，必要时中断 LLM 调用。"""
        # 1. 判断状态：根据结果前缀判断工具执行是否失败
        status = self._tool_result_status(result)

        # 2. 截断结果：避免过长的工具输出占用过多上下文
        stored_result = self._truncate_tool_result(result)

        # 3. 构建观察记录：创建结构化的 ToolObservation 并追加到 AgentState
        observation = ToolObservation(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            args=arguments,
            status=status,
            result=stored_result,
            error=stored_result if status == "error" else None,
        )
        self.state.tool_results.append(observation)

        # 调试输出：启用 trace 时打印工具调用详情
        if self.trace_enabled and self.trace_tool:
            args_text = json.dumps(arguments, ensure_ascii=False, default=str)
            result_text = (stored_result or "").strip() or "<empty>"
            print(
                f"  - tool_name: {tool_name} "
                f"tool_args: {args_text} "
                f"tool_result: {result_text}"
            )

        # 4. 中断判断：若任务状态为 waiting_user，返回回复以中断后续 LLM 调用
        if self.state.task_state.status != "waiting_user":
            return None

        # 收集未决问题并生成提示回复
        questions = [q.strip() for q in self.state.task_state.open_questions if q.strip()]
        if not questions:
            return "我需要你补充信息后才能继续。"

        question_lines = "\n".join(f"{idx}. {question}" for idx, question in enumerate(questions, 1))
        return f"我需要你补充以下信息后才能继续：\n{question_lines}"

    @staticmethod
    def _truncate_tool_result(result: str, max_chars: int = 800) -> str:
        """截断工具结果后写入 AgentState，不使用 LLM 生成摘要。"""
        # TODO LLM生成摘要
        text = (result or "").strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + "\n...（工具结果已截断）"

    @staticmethod
    def _tool_result_status(result: str) -> str:
        """从工具返回值判断执行状态，兼容前缀文本和结构化 JSON。"""
        text = (result or "").strip()
        if text.startswith("❌"):
            return "error"
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return "success"
        if isinstance(payload, dict) and payload.get("status") == "error":
            return "error"
        return "success"

    def get_status(self) -> dict[str, Any]:
        """返回轻量运行状态，便于调试和诊断。"""
        return {
            "name": self.name,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "namespace": self.namespace,
            "turn_count": self.turn_count,
            "tools": self.list_tools(),
            "memory_available": self.memory_tool is not None,
            "rag_available": self.rag_tool is not None,
            "plan_available": self.plan_tool is not None,
            "terminal_available": self.terminal_tool is not None,
            "permission_mode": self.permission_mode,
            "trace_enabled": self.trace_enabled,
            "trace_reasoning": self.trace_reasoning,
            "trace_tool": self.trace_tool,
            "task_state": self.state.task_state.dict(),
            "tool_results_count": len(self.state.tool_results),
        }

    def clear_conversation(self) -> None:
        """清空进程内对话历史和运行时 AgentState。"""
        self.clear_history()
        self.state.reset()
        self.turn_count = 0

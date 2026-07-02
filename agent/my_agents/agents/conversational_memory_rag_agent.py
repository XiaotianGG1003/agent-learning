"""基于运行时 AgentState、长期记忆工具和 RAG 工具的连续对话 Agent。"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional, Union

from .function_call_agent import FunctionCallAgent
from ..core.llm import HelloAgentsLLM
from ..core.state import AgentState, ToolObservation
from ..memory import MemoryConfig
from ..tools.builtin.memory_tool import MemoryTool
from ..tools.builtin.plan_tool import PlanTool
from ..tools.builtin.rag_tool import RAGTool


class ConversationalMemoryRAGAgent(FunctionCallAgent):
    """使用内存态 AgentState 管理短期会话的连续对话 Agent。
    """

    DEFAULT_SYSTEM_PROMPT = (
        "你是一个严谨、简洁的中文编程助手。你的任务是帮助用户解释代码、修改代码、设计提示词和排查错误。\n\n"
        "短期状态策略：当前会话状态由系统提供的 AgentState 表示。不要把 AgentState 写入长期记忆。\n"
        "计划策略：面对多步骤或需跟踪进度的任务，先用 plan 工具写下待办（set_todos），执行时用 "
        "start_todo/complete_todo 更新进度；定下关键决策或约束时用 add_decision/add_constraint 记录；"
        "需要用户澄清时用 raise_question，它会自动把状态切到 waiting_user；收到用户补充信息后先用 "
        "resolve_question 消解问题，未决问题清空后状态会恢复 in_progress；进入 waiting_user 后本轮会直接结束并等待用户，"
        "不要继续规划或调用其它工具；任务完成时切到 done。简单的一问一答无需使用 plan。\n"
        "长期检索策略：需要历史经验、用户长期偏好或项目事实时，调用 memory_search 检索 episodic/semantic/perceptual 记忆。"
        "需要文档知识时，调用 rag_search 或 rag_ask。\n\n"
        "工具结果策略：如果使用了工具或外部检索结果，优先依据工具返回的事实。"
        "不要把工具输出的原始 JSON 直接暴露给用户，而是用自然语言总结。若工具结果与用户描述冲突，要指出冲突并说明依据。"
    )

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
        enable_plan: bool = True,
        expandable: bool = True,
        max_tool_iterations: int = 3,
        trace_enabled: bool = False,
        trace_reasoning: bool = True,
        trace_tool: bool = True,
    ):
        self.user_id = user_id
        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.namespace = namespace
        self.turn_count = 0
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
            self.plan_tool = PlanTool(state=self.state, expandable=expandable)
            self.add_tool(self.plan_tool)

        self.memory_tool: Optional[MemoryTool] = None
        if enable_memory:
            self.memory_tool = MemoryTool(
                user_id=user_id,
                session_id=self.session_id,
                memory_config=memory_config,
                memory_types=memory_types,
                expandable=expandable,
            )
            self.add_tool(self.memory_tool)

        self.rag_tool: Optional[RAGTool] = None
        if enable_rag:
            self.rag_tool = RAGTool(
                knowledge_base_path=rag_knowledge_base_path,
                namespace=namespace,
                expandable=expandable,
            )
            self.add_tool(self.rag_tool)

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
        _ = input_text
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": f"{self.DEFAULT_SYSTEM_PROMPT}\n\n[AgentState]\n{self.state.task_state.render()}",
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

    def on_tool_complete(
        self,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        result: str
    ) -> Optional[str]:
        """工具执行完成后的回调：记录结果到 AgentState，必要时中断 LLM 调用。"""
        # 1. 判断状态：根据结果前缀判断工具执行是否失败
        status = "error" if result.startswith("❌") else "success"

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

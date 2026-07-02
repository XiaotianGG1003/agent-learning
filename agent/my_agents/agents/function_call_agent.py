"""FunctionCallAgent - 使用OpenAI函数调用范式的Agent实现"""

from __future__ import annotations

import json
import logging
from typing import Iterator, Optional, Union, TYPE_CHECKING, Any, Dict

from ..core.agent import Agent
from ..core.llm import HelloAgentsLLM
from ..core.message import Message

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..tools.registry import ToolRegistry


class FunctionCallAgent(Agent):
    """基于OpenAI原生函数调用机制的Agent"""

    DEFAULT_SYSTEM_PROMPT = "你是一个可靠的AI助理，能够在需要时调用工具完成任务。\n当你判断需要外部信息或执行动作时，可以通过函数调用使用已注册工具。"

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        tool_registry: Optional["ToolRegistry"] = None,
        enable_tool_calling: bool = True,
        default_tool_choice: Union[str, dict] = "auto",
        max_tool_iterations: int = 3,
    ):
        super().__init__(name, llm, system_prompt)
        self.tool_registry = tool_registry
        self.enable_tool_calling = enable_tool_calling and tool_registry is not None
        self.default_tool_choice = default_tool_choice
        self.max_tool_iterations = max_tool_iterations
        self.last_reasoning: str = ""  # 最近一次思维链（如 MiMo reasoning_content）

    def _build_tool_schemas(self) -> list[dict[str, Any]]:
        """构建传递给 LLM 的工具 schemas"""
        if not self.enable_tool_calling or not self.tool_registry:
            return []

        schemas: list[dict[str, Any]] = []

        # Tool 对象：直接使用 to_openai_schema()
        for tool in self.tool_registry.get_all_tools():
            try:
                schemas.append(tool.to_openai_schema())
            except Exception as e:
                logger.warning(f"工具 {tool.name} schema 构建失败: {e}")

        # register_function 注册的工具（直接访问内部结构）
        function_map = getattr(self.tool_registry, "_functions", {})
        for name, info in function_map.items():
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": info.get("description", ""),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "input": {
                                    "type": "string",
                                    "description": "输入文本"
                                }
                            },
                            "required": ["input"]
                        }
                    }
                }
            )

        return schemas

    @staticmethod
    def _extract_message_content(raw_content: Any) -> str:
        """从OpenAI响应的message.content中安全提取文本"""
        if raw_content is None:
            return ""
        if isinstance(raw_content, str):
            return raw_content
        if isinstance(raw_content, list):
            parts: list[str] = []
            for item in raw_content:
                text = getattr(item, "text", None)
                if text is None and isinstance(item, dict):
                    text = item.get("text")
                if text:
                    parts.append(text)
            return "".join(parts)
        return str(raw_content)

    @staticmethod
    def _parse_function_call_arguments(arguments: Optional[str]) -> dict[str, Any]:
        """解析模型返回的JSON字符串参数"""
        if not arguments:
            return {}

        try:
            parsed = json.loads(arguments)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _convert_parameter_types(self, tool_name: str, param_dict: dict[str, Any]) -> dict[str, Any]:
        """根据工具定义尽可能转换参数类型"""
        if not self.tool_registry:
            return param_dict

        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            return param_dict

        try:
            tool_params = tool.get_parameters()
        except Exception:
            return param_dict

        type_mapping = {param.name: param.type for param in tool_params}
        converted: dict[str, Any] = {}

        for key, value in param_dict.items():
            param_type = type_mapping.get(key)
            if not param_type:
                converted[key] = value
                continue

            try:
                normalized = param_type.lower()
                if normalized in {"number", "float"}:
                    converted[key] = float(value)
                elif normalized in {"integer", "int"}:
                    converted[key] = int(value)
                elif normalized in {"boolean", "bool"}:
                    if isinstance(value, bool):
                        converted[key] = value
                    elif isinstance(value, (int, float)):
                        converted[key] = bool(value)
                    elif isinstance(value, str):
                        converted[key] = value.lower() in {"true", "1", "yes"}
                    else:
                        converted[key] = bool(value)
                else:
                    converted[key] = value
            except (TypeError, ValueError):
                converted[key] = value

        return converted

    def _execute_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """执行工具调用并返回字符串结果"""
        if not self.tool_registry:
            return "❌ 错误：未配置工具注册表"

        tool = self.tool_registry.get_tool(tool_name)
        if tool:
            try:
                typed_arguments = self._convert_parameter_types(tool_name, arguments)
                result = tool.run(typed_arguments)
                if isinstance(result, str):
                    return result
                return json.dumps(result, ensure_ascii=False, default=str)
                # return tool.run(typed_arguments)
            except Exception as exc:
                return f"❌ 工具调用失败：{exc}"

        func = self.tool_registry.get_function(tool_name)
        if func:
            try:
                input_text = arguments.get("input", "")
                return func(input_text)
            except Exception as exc:
                return f"❌ 工具调用失败：{exc}"

        return f"❌ 错误：未找到工具 '{tool_name}'"

    def _build_model_messages(self, input_text: str) -> list[dict[str, Any]]:
        """构造单轮模型输入消息。

        子类可以覆盖这个 hook 注入运行时状态，同时复用父类的函数调用循环。
        """
        messages: list[dict[str, Any]] = []
        messages.append({"role": "system", "content": self.system_prompt or self.DEFAULT_SYSTEM_PROMPT})

        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": input_text})
        return messages

    def on_tool_complete(
        self,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        result: str
    ) -> Optional[str]:
        """工具执行完成后的回调 hook。

        返回字符串时，该字符串会作为本轮最终回复，并中断后续 LLM 调用。
        子类可覆盖此方法，实现工具结果的持久化或状态更新。
        """
        # 基类默认不做处理，直接返回 None 表示不中断后续 LLM 调用
        return None

    def on_llm_response(
        self,
        step: int,
        reasoning: str,
        content: str,
        has_tool_calls: bool,
    ) -> None:
        """LLM 响应完成后的回调 hook，默认不做额外处理。"""
        return None

    def _invoke_with_tools(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], tool_choice: Union[str, dict], **kwargs):
        """调用底层OpenAI客户端执行函数调用

        自动适配 max_tokens / max_completion_tokens（部分服务商如 MiMo 使用后者）。
        """
        client = getattr(self.llm, "_client", None)
        if client is None:
            raise RuntimeError("未正确初始化客户端，无法执行函数调用。")

        client_kwargs = dict(kwargs)
        client_kwargs.setdefault("temperature", self.llm.temperature)
        if self.llm.max_tokens is not None:
            # 优先使用 max_completion_tokens（兼容 MiMo 等服务商）
            if "max_completion_tokens" not in client_kwargs:
                client_kwargs.setdefault("max_tokens", self.llm.max_tokens)

        return client.chat.completions.create(
            model=self.llm.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            **client_kwargs,
        )

    def run(
        self,
        input_text: str,
        *,
        max_tool_iterations: Optional[int] = None,
        tool_choice: Optional[Union[str, dict]] = None,
        **kwargs,
    ) -> str:
        """
        执行函数调用范式的对话流程
        """
        messages = self._build_model_messages(input_text)

        tool_schemas = self._build_tool_schemas()
        if not tool_schemas:
            response_text = self.llm.invoke(messages, **kwargs)
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(response_text, "assistant"))
            return response_text


        iterations_limit = max_tool_iterations if max_tool_iterations is not None else self.max_tool_iterations
        effective_tool_choice: Union[str, dict] = tool_choice if tool_choice is not None else self.default_tool_choice

        current_iteration = 0
        final_response = ""

        while current_iteration < iterations_limit:
            response = self._invoke_with_tools(
                messages,
                tools=tool_schemas,
                tool_choice=effective_tool_choice,
                **kwargs,
            )

            choice = response.choices[0]
            assistant_message = choice.message
            content = self._extract_message_content(assistant_message.content)
            reasoning_content = getattr(assistant_message, "reasoning_content", "")
            self.last_reasoning = reasoning_content or ""
            tool_calls = list(assistant_message.tool_calls or [])
            self.on_llm_response(
                current_iteration + 1,
                reasoning_content or "",
                content or "",
                bool(tool_calls),
            )

            if tool_calls:
                assistant_payload: dict[str, Any] = {"role": "assistant", "reasoning_content": reasoning_content, "content": content, "tool_calls": []}

                for tool_call in tool_calls:
                    assistant_payload["tool_calls"].append(
                        {
                            "id": tool_call.id,
                            "type": tool_call.type,
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                    )
                messages.append(assistant_payload)

                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    arguments = self._parse_function_call_arguments(tool_call.function.arguments)
                    result = self._execute_tool_call(tool_name, arguments)
                    interrupt_response = self.on_tool_complete(tool_call.id, tool_name, arguments, result)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_name,
                            "content": result,
                        }
                    )
                    if interrupt_response is not None:
                        final_response = interrupt_response
                        break

                if final_response:
                    messages.append({"role": "assistant", "content": final_response})
                    break

                current_iteration += 1
                continue

            final_response = content
            messages.append({"role": "assistant", "reasoning_content": reasoning_content, "content": final_response})
            break

        if current_iteration >= iterations_limit and not final_response:
            try:
                final_choice = self._invoke_with_tools(
                    messages,
                    tools=tool_schemas,
                    tool_choice="none",
                    **kwargs,
                )
                final_msg = final_choice.choices[0].message
                final_response = self._extract_message_content(final_msg.content)
                self.on_llm_response(current_iteration + 1, "", final_response, False)
            except Exception as exc:
                final_response = ""
            if not final_response:
                final_response = "（已达到工具调用次数上限，且未能生成最终回答）"
            messages.append({"role": "assistant", "content": final_response})

        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_response, "assistant"))

        return final_response

    def add_tool(self, tool) -> None:
        """便捷方法：将工具注册到当前Agent"""
        if not self.tool_registry:
            from ..tools.registry import ToolRegistry

            self.tool_registry = ToolRegistry()
            self.enable_tool_calling = True

        if hasattr(tool, "auto_expand") and getattr(tool, "auto_expand"):
            expanded_tools = tool.get_expanded_tools()
            if expanded_tools:
                for expanded_tool in expanded_tools:
                    self.tool_registry.register_tool(expanded_tool)
                print(f"✅ MCP工具 '{tool.name}' 已展开为 {len(expanded_tools)} 个独立工具")
                return

        self.tool_registry.register_tool(tool)

    def remove_tool(self, tool_name: str) -> bool:
        if self.tool_registry:
            before = set(self.tool_registry.list_tools())
            self.tool_registry.unregister(tool_name)
            after = set(self.tool_registry.list_tools())
            return tool_name in before and tool_name not in after
        return False

    def list_tools(self) -> list[str]:
        if self.tool_registry:
            return self.tool_registry.list_tools()
        return []

    def has_tools(self) -> bool:
        return self.enable_tool_calling and self.tool_registry is not None

    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
        """流式调用暂未实现，直接回退到一次性调用"""
        result = self.run(input_text, **kwargs)
        yield result

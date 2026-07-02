# 配置第三方库的日志级别，减少噪音
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("qdrant_client").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("neo4j").setLevel(logging.WARNING)
logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

# 核心组件
from .core.llm import HelloAgentsLLM
from .core.message import Message
from .core.exceptions import HelloAgentsException

# Agent实现
from .agents.simple_agent import SimpleAgent
from .agents.function_call_agent import FunctionCallAgent
from .agents.react_agent import ReActAgent
from .agents.reflection_agent import ReflectionAgent
from .agents.plan_solve_agent import PlanAndSolveAgent
from .agents.tool_aware_agent import ToolAwareSimpleAgent
from .agents.conversational_memory_rag_agent import ConversationalMemoryRAGAgent

# 工具系统
from .tools.registry import ToolRegistry, global_registry
from .tools.builtin.search_tool import SearchTool, search
from .tools.builtin.calculator import CalculatorTool, calculate
from .tools.chain import ToolChain, ToolChainManager
from .tools.async_executor import AsyncToolExecutor

__all__ = [
    # 核心组件
    "HelloAgentsLLM",
    "Message",
    "HelloAgentsException",

    # Agent范式
    "SimpleAgent",
    "FunctionCallAgent",
    "ReActAgent",
    "ReflectionAgent",
    "PlanAndSolveAgent",
    "ToolAwareSimpleAgent",
    "ConversationalMemoryRAGAgent",

    # 工具系统
    "ToolRegistry",
    "global_registry",
    "SearchTool",
    "search",
    "CalculatorTool",
    "calculate",
    "ToolChain",
    "ToolChainManager",
    "AsyncToolExecutor",
]


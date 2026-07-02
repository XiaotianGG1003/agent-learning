from dotenv import load_dotenv
load_dotenv()

from my_agents import SimpleAgent, HelloAgentsLLM
from my_agents.context import ContextBuilder, ContextConfig
from my_agents.tools import MemoryTool, RAGTool

class ContextAwareAgent(SimpleAgent):
    """具有上下文感知能力的 Agent"""

    def __init__(self, name: str, llm: HelloAgentsLLM, **kwargs):
        super().__init__(name=name, llm=llm, system_prompt=kwargs.get("system_prompt", ""))

        # 初始化上下文构建器
        self.memory_tool = MemoryTool(
            user_id=kwargs.get("user_id", "default"),
            memory_types=["episodic", "semantic"]
        )
        self.rag_tool = RAGTool(knowledge_base_path=kwargs.get("knowledge_base_path", "./knowledge_base"),
                                namespace=kwargs.get("namespace", "default"))

        self.context_builder = ContextBuilder(
            memory_tool=self.memory_tool,
            rag_tool=self.rag_tool,
            config=ContextConfig(max_tokens=4000),
            namespace=kwargs.get("namespace", "default")
        )

        self.conversation_history = []

    def run(self, user_input: str) -> str:
        """运行 Agent,自动构建优化的上下文"""

        # 1. 使用 ContextBuilder 构建优化的上下文
        optimized_context = self.context_builder.build(
            user_query=user_input,
            conversation_history=self.conversation_history,
            system_instructions=self.system_prompt
        )
        print("构建的上下文" + "=" * 80)
        print(optimized_context)
        print("=" * 80)
        # 2. 使用优化后的上下文调用 LLM
        messages = [
            {"role": "system", "content": optimized_context},
            {"role": "user", "content": user_input}
        ]
        response = self.llm.invoke(messages)

        # 3. 更新对话历史
        from my_agents.core.message import Message
        from datetime import datetime

        self.conversation_history.append(
            Message(content=user_input, role="user", timestamp=datetime.now())
        )
        self.conversation_history.append(
            Message(content=response, role="assistant", timestamp=datetime.now())
        )

        return response





# 使用示例
agent = ContextAwareAgent(
    name="ContextAwareAgent",
    llm=HelloAgentsLLM(),
    system_prompt="你是一位资深的Python机器学习顾问。你的回答需要:1) 提供具体可行的建议 2) 解释技术原理 3) 给出代码示例",
    user_id="test_context_user",
    namespace="test_context",
)

# 在 agent.run() 之前，检查知识库状态
stats = agent.rag_tool.run({"action": "stats","namespace": "test_context"})
print(stats)

result = agent.rag_tool.add_document(
    file_path="./knowledge_base/demo.md",
    chunk_size=100,
    chunk_overlap=10
)

# # 直接搜索测试
# search_result = agent.rag_tool.run({
#   "action": "search",
#   "query": "无监督学习",
# })
# print(search_result)

# state_results = agent.memory_tool.run({
#     "action": "search",
#     "query" : "(任务状态 OR 子目标 OR 结论 OR 阻塞)",
#     "min_importance": 0.7,
#     "limit": 5
# })
# print(f"state_results: {state_results}")

response = agent.run("什么是无监督学习?")
print(response)

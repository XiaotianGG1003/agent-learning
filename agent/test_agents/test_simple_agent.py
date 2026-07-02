from dotenv import load_dotenv


# 加载环境变量
load_dotenv()


from my_agents import SimpleAgent, HelloAgentsLLM, ToolRegistry, SearchTool


llm = HelloAgentsLLM()

tool_registry = ToolRegistry()
search_tool = SearchTool()
tool_registry.register_tool(search_tool)

my_simple_agent = SimpleAgent(
    name="SimpleAgent",
    llm=llm,
    system_prompt="你是一个智能助手，可以使用工具来帮助用户。",
    tool_registry=tool_registry,
    enable_tool_calling=True
)

response = my_simple_agent.run("请帮我搜索一下关于人工智能的最新新闻。")
print(f"响应: {response}\n")


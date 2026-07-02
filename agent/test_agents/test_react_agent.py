from dotenv import load_dotenv

from my_agents import HelloAgentsLLM, ReActAgent, ToolRegistry, SearchTool

load_dotenv()

llm = HelloAgentsLLM()

tool_registry = ToolRegistry()
search_tool = SearchTool()
tool_registry.register_tool(search_tool)

my_react_agent = ReActAgent(
    name="ReActAgent",
    llm=llm,
    tool_registry=tool_registry,
)

response = my_react_agent.run("明天武汉天气怎么样")
print(response)
from dotenv import load_dotenv

from my_agents import HelloAgentsLLM, ReActAgent, ToolRegistry, SearchTool, ReflectionAgent

load_dotenv()

llm = HelloAgentsLLM()

my_reflection_agent = ReflectionAgent(
    name="ReflectionAgent",
    llm=llm,
)

my_reflection_agent.run("编写一个Python函数，找出1到n之间所有的素数 (prime numbers)。")

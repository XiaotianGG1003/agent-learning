from dotenv import load_dotenv
load_dotenv()

from my_agents.tools import RAGTool

rag_tool = RAGTool(
    knowledge_base_path="./knowledge_base",
    namespace="test_rag_tool",

)

result1 = rag_tool.add_document(
    file_path="./knowledge_base/demo.md",
    chunk_size=100,
    chunk_overlap=10
)

# result2 = rag_tool.run({
#     "action": "add_document",
#     "file_path": "./knowledge_base/demo.md",
#     "namespace": "test_rag_tool",
#     "chunk_size": 100,
#     "chunk_overlap": 10
# })
# print(result2)

query = "监督学习"

result3 = rag_tool.run({
    "action": "search",
    "query": query,
    "limit": 3,
    "enable_advanced_search": True,
})

print(result3)

result4 = rag_tool.run({
    "action": "stats",
    "namespace": "test_rag_tool",
})

print(result4)

result5 = rag_tool.ask(question="什么是监督学习？")
print(result5)
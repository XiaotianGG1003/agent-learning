from dotenv import load_dotenv
load_dotenv()

from my_agents.tools import RAGTool

rag_tool = RAGTool(
    knowledge_base_path="./knowledge_base",
    namespace="test_rag_search_rerank",

)

result1 = rag_tool.add_document(
    file_path="knowledge_base/python-basic.pdf",
    chunk_size=300,
    chunk_overlap=50
)



print(result1)


from dotenv import load_dotenv

load_dotenv()

from my_agents.tools import MemoryTool


memory_tool = MemoryTool(
    user_id="test_memory_tool",
)

result1 = memory_tool.run({
    "action": "add",
    "content": "这是一个测试记忆项，用于验证MemoryTool的基本功能。",
    "memory_type": "episodic",
    "importance": 0.7
})

description1 = """2024年3月，华星科技与清华大学人工智能研究院在北京市签署战略合作协议。根据协议，双方将共同建设智能计算实验室，并开展大模型技术研究。
华星科技董事长李明表示，公司计划未来三年投入5亿元人民币用于人工智能基础设施建设。研究院院长王建国认为，此次合作将促进科研成果向产业应用转化。
与此同时，远航资本宣布向华星科技投资2亿元人民币，获得其10%的股份。该投资项目由远航资本合伙人张伟负责推进。
2024年6月，华星科技在上海市设立新的研发中心，并聘请曾任职于腾讯的陈晓担任首席技术官。陈晓带领团队开发的新一代语言模型已在金融、医疗和教育领域展开试点应用。
此外，华星科技与云海数据建立长期合作关系，由云海数据为其提供训练数据管理服务。双方预计将在2025年底前共同推出面向企业客户的数据治理平台。"""

description2 = "清华大学、北京大学都有研究院"

result2 = memory_tool.run({
    "action": "add",
    "content": description1,
    "memory_type": "semantic",
    "importance": 0.7
})

result3 = memory_tool.run({
    "action": "add",
    "content": description2,
    "memory_type": "semantic",
    "importance": 0.7
})



print(result3)

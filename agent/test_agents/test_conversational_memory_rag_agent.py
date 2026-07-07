"""ConversationalMemoryRAGAgent 使用示例

演示功能：
1. Agent 内部自动创建 MemoryTool 和 RAGTool
2. 向 RAG 知识库添加文档
3. 多轮对话（AgentState 维护短期状态）
4. 通过工具按需检索长期记忆和 RAG 文档
5. 通过 TerminalTool 读取/搜索/编辑工作区与执行受限命令
6. restricted_access 默认权限模式下，高风险工具触发控制台审批
7. 状态诊断
"""
import json
import sys

from dotenv import load_dotenv
load_dotenv()

from my_agents import ConversationalMemoryRAGAgent, HelloAgentsLLM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


EXIT_COMMANDS = {"q", "quit", "exit", "退出"}
PERMISSION_MODES = {"read_only", "restricted_access", "full_access", "approval_all"}


def build_agent(permission_mode: str = "restricted_access") -> ConversationalMemoryRAGAgent:
    return ConversationalMemoryRAGAgent(
        name="AI助手",
        llm=HelloAgentsLLM(),
        user_id="test_conversational_memory_rag_user",
        namespace="test_conversational_memory_rag_agent",
        # 记忆配置
        enable_memory=True,
        memory_types=["episodic"],
        # RAG 配置
        enable_rag=True,
        rag_knowledge_base_path="./knowledge_base",
        # TerminalTool 配置
        enable_terminal=True,
        terminal_workspace="../",
        terminal_config={"restricted_bash": True},
        # 权限/审批配置
        permission_mode=permission_mode,
        max_tool_iterations=1000,
        trace_enabled=True,
        trace_reasoning=True,
        trace_tool=True,
    )


def print_status(agent: ConversationalMemoryRAGAgent) -> None:
    status = agent.get_status()
    print("📊 Agent 状态:")
    for key, value in status.items():
        print(f"   {key}: {value}")
    print()


def read_question(turn: int) -> str | None:
    print(f"{'─' * 50}")
    print(f"👤 用户 [{turn}] 输入问题")
    print(
        "   命令：q/quit/exit 退出，/status 查看状态，/clear 清空会话，"
        "/mode <read_only|restricted_access|full_access|approval_all> 切换权限，"
        "/multi 多行输入（单独一行 . 结束）"
    )

    while True:
        try:
            first_line = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return None

        command = first_line.strip()
        if not command:
            continue
        if command.lower() in EXIT_COMMANDS:
            return None
        if command == "/multi":
            lines: list[str] = []
            print("   请输入多行内容，单独一行 . 结束：")
            while True:
                try:
                    line = input("... ")
                except (EOFError, KeyboardInterrupt):
                    print()
                    return None
                if line.strip() == ".":
                    break
                lines.append(line)
            question = "\n".join(lines).strip()
            if question:
                return question
            continue
        return first_line


def set_permission_mode(agent: ConversationalMemoryRAGAgent, command: str) -> None:
    parts = command.split(maxsplit=1)
    if len(parts) != 2 or parts[1].strip() not in PERMISSION_MODES:
        print("可用权限模式：read_only / restricted_access / full_access / approval_all")
        return

    mode = parts[1].strip()
    agent.permission_mode = mode
    print(f"🔐 权限模式已切换为: {mode}")


def main() -> None:
    agent = build_agent()

    schemas = agent._build_tool_schemas()
    print(f"共 {len(schemas)} 个工具:{json.dumps(schemas, ensure_ascii=False)}")
    print(f"🔐 当前权限模式: {agent.permission_mode}")
    print("   高风险工具（edit_file/write_file/bash/写入记忆或知识库）会按权限模式审批。")
    print()

    result = agent.rag_tool.add_document(
        file_path="./knowledge_base/demo.md",
        chunk_size=200,
        chunk_overlap=20,
        doc_id="demo_md",
        upsert_mode="REPLACE",
    )
    print(f"📚 添加文档: {result}")
    print("进入连续对话模式。每次看到助手回答后，可继续输入下一轮问题。")

    turn = 1
    while True:
        question = read_question(turn)
        if question is None:
            break
        if question.strip() == "/status":
            print_status(agent)
            continue
        if question.strip() == "/clear":
            agent.clear_conversation()
            print("🗑️ 已清空进程内 AgentState")
            continue
        if question.strip().startswith("/mode"):
            set_permission_mode(agent, question.strip())
            continue

        response = agent.run(question)
        print(f"🤖 助手 [{turn}]: {response}")
        print()
        turn += 1

    print_status(agent)


#
# agent.clear_conversation()
# print("🗑️ 已清空进程内 AgentState")


if __name__ == "__main__":
    main()

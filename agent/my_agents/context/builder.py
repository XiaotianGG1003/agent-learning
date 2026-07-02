"""ContextBuilder - GSSC流水线实现

实现 Gather-Select-Structure-Compress 上下文构建流程：
1. Gather: 从多源收集候选信息（历史、记忆、RAG、工具结果）
2. Select: 基于优先级、相关性、多样性筛选
3. Structure: 组织成结构化上下文模板
4. Compress: 在预算内压缩与规范化
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging
import tiktoken

logger = logging.getLogger(__name__)

from ..core.message import Message
from ..tools import MemoryTool, RAGTool


@dataclass
class ContextPacket:
    """上下文信息包"""
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: int = 0
    relevance_score: float = 0.0  # 0.0-1.0
    
    def __post_init__(self):
        """自动计算token数"""
        if self.token_count == 0:
            self.token_count = count_tokens(self.content)


@dataclass
class ContextConfig:
    """上下文构建配置"""
    max_tokens: int = 100000  # 总预算 (100K)
    reserve_ratio: float = 0.15  # 生成余量（10-20%）
    min_relevance: float = 0.3  # 源内检索的最小相关性阈值
    enable_mmr: bool = True  # 启用最大边际相关性（多样性）
    mmr_lambda: float = 0.7  # MMR平衡参数（0=纯多样性, 1=纯相关性）
    system_prompt_template: str = ""  # 系统提示模板
    enable_compression: bool = True  # 启用压缩
    instructions_cap: int = 2000  # 系统提示 token 上限（超出截断并 warn）

    # 分层预算：各类型 token 配比
    type_budget_ratios: Dict[str, float] = field(default_factory=lambda: {
        "task_state":      0.10,
        "knowledge_base":  0.40,
        "related_memory":  0.20,
        "history":         0.20,
    })

    # 溢出再分配优先级（从高到低）
    realloc_priority: List[str] = field(default_factory=lambda: [
        "knowledge_base", "related_memory", "history", "task_state"
    ])

    def get_available_tokens(self) -> int:
        """获取可用token预算（扣除余量）"""
        return int(self.max_tokens * (1 - self.reserve_ratio))


class ContextBuilder:
    """上下文构建器 - GSSC流水线
    
    用法示例：
    ```python
    builder = ContextBuilder(
        memory_tool=memory_tool,
        rag_tool=rag_tool,
        config=ContextConfig(max_tokens=8000)
    )
    
    context = builder.build(
        user_query="用户问题",
        conversation_history=[...],
        system_instructions="系统指令"
    )
    ```
    """
    
    def __init__(
        self,
        memory_tool: Optional[MemoryTool] = None,
        rag_tool: Optional[RAGTool] = None,
        config: Optional[ContextConfig] = None,
        namespace: str = "default"
    ):
        self.memory_tool = memory_tool
        self.rag_tool = rag_tool
        self.config = config or ContextConfig()
        self.namespace = namespace
        self._encoding = tiktoken.get_encoding("cl100k_base")
    
    def build(
        self,
        user_query: str,
        conversation_history: Optional[List[Message]] = None,
        system_instructions: Optional[str] = None,
        state: Optional[str] = None,
        additional_packets: Optional[List[ContextPacket]] = None,
        min_relevance: Optional[float] = None
    ) -> str:
        """构建完整上下文

        Args:
            user_query: 用户查询
            conversation_history: 对话历史
            system_instructions: 系统指令
            state: 当前任务状态（如已完成步骤、进行中任务、关键变量等）
            additional_packets: 额外的上下文包
            min_relevance: 最小相关性阈值（默认使用 config.min_relevance）

        Returns:
            结构化上下文字符串
        """
        # 1. Gather: 收集候选信息
        packets = self._gather(
            user_query=user_query,
            conversation_history=conversation_history or [],
            system_instructions=system_instructions,
            additional_packets=additional_packets or [],
            min_relevance=min_relevance
        )
        
        # 2. Select: 筛选与排序
        selected_packets = self._select(packets, user_query)

        # 3. Structure: 组织成结构化模板
        structured_context = self._structure(
            selected_packets=selected_packets,
            user_query=user_query,
            state=state
        )
        
        # 4. Compress: 压缩与规范化（如果超预算）
        final_context = self._compress(structured_context)
        
        return final_context
    
    def _gather(
        self,
        user_query: str,
        conversation_history: List[Message],
        system_instructions: Optional[str],
        additional_packets: List[ContextPacket],
        min_relevance: Optional[float] = None
    ) -> List[ContextPacket]:
        """Gather: 收集候选信息

        Args:
            min_relevance: 最小相关性阈值，用于过滤 related_memory（默认使用 config.min_relevance）
        """
        if min_relevance is None:
            min_relevance = self.config.min_relevance

        packets = []

        # P0: 系统指令（强约束）
        if system_instructions:
            packets.append(ContextPacket(
                content=system_instructions,
                metadata={"type": "instructions"}
            ))

        # P1: 从记忆中获取任务状态与关键结论（条目级 packet）
        if self.memory_tool:
            try:
                # 搜索任务状态相关记忆
                _state_text, state_items = self.memory_tool._search_memory(
                    query="(任务状态 OR 子目标 OR 结论 OR 阻塞)",
                    min_importance=0.7,
                    limit=5
                )
                # task_state 使用 importance（状态信息靠重要性留存，不靠相关性）
                for item in state_items:
                    packets.append(ContextPacket(
                        content=item["text"],
                        timestamp=item["timestamp"],
                        metadata={
                            "type": "task_state",
                            "memory_type": item["memory_type"]
                        },
                        relevance_score=item["importance"]
                    ))

                # 搜索与当前查询相关的记忆
                _related_text, related_items = self.memory_tool._search_memory(
                    query=user_query,
                    limit=5
                )
                # related_memory 使用 relevance_score（与 query 的相关性）
                # 按 min_relevance 过滤
                for item in related_items:
                    score = item.get("relevance_score", 0.0)
                    if score < min_relevance:
                        logger.debug(f"过滤低相关性记忆: score={score:.3f} < min_relevance={min_relevance}")
                        continue
                    packets.append(ContextPacket(
                        content=item["text"],
                        timestamp=item["timestamp"],
                        metadata={
                            "type": "related_memory",
                            "memory_type": item["memory_type"]
                        },
                        relevance_score=score
                    ))
            except Exception as e:
                logger.warning(f"记忆检索失败: {e}")

        # P2: 从RAG中获取事实证据（条目级 packet，citation.score 作为相关性分数）
        if self.rag_tool:
            try:
                rerank_result = self.rag_tool.search_rerank(
                    query=user_query,
                    limit=5,
                    min_score=min_relevance,
                    enable_mqe=True,
                    enable_hyde=True,
                    namespace=self.namespace,
                )
                text_list = rerank_result.get("text_list", [])
                citations = rerank_result.get("citations", [])
                for text, citation in zip(text_list, citations):
                    packets.append(ContextPacket(
                        content=text,
                        metadata={
                            "type": "knowledge_base",
                            "source_path": citation.get("source_path", ""),
                            "index": citation.get("index", 0)
                        },
                        relevance_score=citation.get("score", 0.0)
                    ))
            except Exception as e:
                logger.warning(f"RAG检索失败: {e}")
        
        # P3: 对话历史（辅助材料，继承最新消息的时间戳）
        if conversation_history:
            recent_history = conversation_history[-10:]
            history_text = "\n".join([
                f"[{msg.role}] {msg.content}"
                for msg in recent_history
            ])
            latest_ts = recent_history[-1].timestamp if recent_history else datetime.now()
            packets.append(ContextPacket(
                content=history_text,
                timestamp=latest_ts,
                metadata={"type": "history", "count": len(recent_history)}
            ))
        
        # 添加额外包
        packets.extend(additional_packets)
        
        return packets
    
    def _select(
        self,
        packets: List[ContextPacket],
        user_query: str
    ) -> List[ContextPacket]:
        """Select: 分层预算 + RFA 筛选

        1) 不补分、不归一化分数、不跨源比较分数
        2) 每个来源只使用 gather 返回的源内顺序作为排名
        3) instructions 完整保留（受 instructions_cap 保护）
        4) 先按 type 分配 token 配额，再用 RFA 按源内排名公平填充
        5) 未用预算继续用 RFA 再分配，仍然只看源内排名
        """
        _ = user_query  # 相关性过滤由各源在 gather 阶段完成。

        # --- 1) instructions 处理（完整保留，受 cap 保护）---
        instructions_packets = [p for p in packets if p.metadata.get("type") == "instructions"]
        selected: List[ContextPacket] = []
        instructions_tokens = 0
        cap = self.config.instructions_cap
        for p in instructions_packets:
            if instructions_tokens + p.token_count <= cap:
                selected.append(p)
                instructions_tokens += p.token_count
            else:
                logger.warning(
                    f"系统提示超出 instructions_cap({cap} tokens)，已截断"
                )

        # --- 2) 分层预算分配 ---
        available = self.config.get_available_tokens() - instructions_tokens
        if available < 200:
            logger.warning(f"可用预算极低({available} tokens)，仅保留 instructions")
            return selected

        ratios = self.config.type_budget_ratios
        total_ratio = sum(ratios.values())
        if total_ratio <= 0:
            logger.warning("type_budget_ratios 总和必须大于 0，仅保留 instructions")
            return selected

        # 只归一化预算比例，不归一化或比较候选分数。
        ratios = {t: r / total_ratio for t, r in ratios.items()}

        # 计算每类硬上限
        type_caps: Dict[str, int] = {}
        for ptype, ratio in ratios.items():
            type_caps[ptype] = int(available * ratio)

        # gather 已经按各源自身检索/排序逻辑返回结果，这里保留该源内顺序。
        ranked_by_type: Dict[str, List[ContextPacket]] = {t: [] for t in type_caps}
        for p in packets:
            ptype = p.metadata.get("type", "unknown")
            if ptype in ranked_by_type:
                ranked_by_type[ptype].append(p)

        type_used: Dict[str, int] = {t: 0 for t in type_caps}
        type_selected: Dict[str, List[ContextPacket]] = {t: [] for t in type_caps}
        cursors: Dict[str, int] = {t: 0 for t in type_caps}
        selected_ids = {id(p) for p in selected}

        allocation_order = [
            t for t in self.config.realloc_priority
            if t in type_caps
        ]
        allocation_order.extend([
            t for t in type_caps
            if t not in allocation_order
        ])

        def next_candidate(ptype: str) -> Optional[ContextPacket]:
            candidates = ranked_by_type[ptype]
            while cursors[ptype] < len(candidates):
                p = candidates[cursors[ptype]]
                if id(p) not in selected_ids:
                    return p
                cursors[ptype] += 1
            return None

        def select_packet(ptype: str, packet: ContextPacket) -> None:
            type_selected[ptype].append(packet)
            type_used[ptype] += packet.token_count
            selected_ids.add(id(packet))
            cursors[ptype] += 1

        def rfa_allocate(type_limits: Dict[str, int], total_remaining: int) -> int:
            """Ranked Fair Allocation: 每轮每个来源最多取一个源内最高未选候选。"""
            allocated = 0
            progress = True
            while total_remaining > 0 and progress:
                progress = False
                for ptype in allocation_order:
                    if total_remaining <= 0:
                        break
                    limit_remaining = type_limits[ptype] - type_used[ptype]
                    if limit_remaining <= 0:
                        continue
                    candidate = next_candidate(ptype)
                    if candidate is None:
                        continue
                    if candidate.token_count > limit_remaining:
                        continue
                    if candidate.token_count > total_remaining:
                        continue

                    select_packet(ptype, candidate)
                    total_remaining -= candidate.token_count
                    allocated += candidate.token_count
                    progress = True

            return allocated

        # 第一阶段：每个类型只能使用自己的分层配额。
        rfa_allocate(type_caps, available)

        # 第二阶段：把未用预算作为共享池继续 RFA，再分配时不跨源比分数。
        remaining_pool = available - sum(type_used.values())
        if remaining_pool > 0:
            overflow_limits = {
                ptype: type_used[ptype] + remaining_pool
                for ptype in type_caps
            }
            rfa_allocate(overflow_limits, remaining_pool)

        # 合并所有类型
        for ptype in type_caps:
            selected.extend(type_selected[ptype])

        return selected
    
    def _structure(
        self,
        selected_packets: List[ContextPacket],
        user_query: str,
        state: Optional[str] = None
    ) -> str:
        """Structure: 组织成结构化上下文模板

        六段式结构：
        [System Prompt]        系统提示（角色、规则、约束）
        [State]                任务状态（已完成步骤、进行中任务、关键变量）
        [Knowledge Base]       知识库内容（RAG 检索结果）
        [Memory]               记忆与偏好（用户偏好、历史事实）
        [Conversation History] 对话历史（近期交互）
        [Task]                 当前任务（本轮用户问题）

        注意：所有内容均来自 selected_packets（一切皆 packet），
        system_instructions 由 _gather 创建为 instructions packet，
        经 _select 筛选后在此处渲染。
        """
        sections = []

        # [System Prompt] - 系统提示（仅从 packet 渲染，不读参数）
        sys_packets = [p for p in selected_packets if p.metadata.get("type") == "instructions"]
        if sys_packets:
            sections.append("[System Prompt]\n" + "\n".join([p.content for p in sys_packets]))

        # [State] - 任务状态（已完成步骤、进行中任务、关键变量）
        task_state_packets = [p for p in selected_packets if p.metadata.get("type") == "task_state"]
        if state or task_state_packets:
            state_section = "[State]\n"
            if state:
                state_section += state
            if task_state_packets:
                if state:
                    state_section += "\n"
                state_section += "\n".join([p.content for p in task_state_packets])
            sections.append(state_section)

        # [Knowledge Base] - 知识库内容（RAG 条目级检索结果）
        kb_packets = [
            p for p in selected_packets
            if p.metadata.get("type") in {"knowledge_base", "retrieval", "tool_result"}
        ]
        if kb_packets:
            kb_section = "[Knowledge Base]\n"
            for i, p in enumerate(kb_packets, 1):
                source_path = p.metadata.get("source_path", "")
                index = p.metadata.get("index", i)
                header = f"## [{index}] {source_path}" if source_path else f"## [{index}]"
                kb_section += f"{header}\n{p.content}\n\n"
            sections.append(kb_section.rstrip())

        # [Memory] - 记忆与偏好（条目级记忆结果）
        memory_packets = [p for p in selected_packets if p.metadata.get("type") == "related_memory"]
        if memory_packets:
            mem_section = "[Memory]\n"
            mem_section += "\n".join([p.content for p in memory_packets])
            sections.append(mem_section)

        # [Conversation History] - 对话历史（近期交互记录）
        history_packets = [p for p in selected_packets if p.metadata.get("type") == "history"]
        if history_packets:
            hist_section = "[Conversation History]\n"
            hist_section += "\n".join([p.content for p in history_packets])
            sections.append(hist_section)

        # [Task] - 当前任务（本轮用户问题 + 输出格式约束）
        task_section = (
            f"[Task]\n"
            f"{user_query}\n"
            # f"请按以下格式回答：\n"
            # f"1. 结论（简洁明确）\n"
            # f"2. 依据（列出支撑证据及来源）\n"
            # f"3. 风险与假设（如有）\n"
            # f"4. 下一步行动建议（如适用）"
        )
        sections.append(task_section)

        return "\n\n".join(sections)
    
    def _compress(self, context: str) -> str:
        """Compress: 压缩与规范化"""
        if not self.config.enable_compression:
            return context
        
        current_tokens = count_tokens(context)
        available_tokens = self.config.get_available_tokens()
        
        if current_tokens <= available_tokens:
            return context
        
        # 简单截断策略（保留前N个token）
        # 实际应用中可用LLM做高保真摘要
        logger.warning(f"上下文超预算 ({current_tokens} > {available_tokens})，执行截断")
        
        # 按段落截断，保留结构
        lines = context.split("\n")
        compressed_lines = []
        used_tokens = 0
        
        for line in lines:
            line_tokens = count_tokens(line)
            if used_tokens + line_tokens > available_tokens:
                break
            compressed_lines.append(line)
            used_tokens += line_tokens
        
        return "\n".join(compressed_lines)


def count_tokens(text: str) -> int:
    """计算文本token数（使用tiktoken）"""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        # 降级方案：粗略估算（1 token ≈ 4 字符）
        return len(text) // 4

"""RAG工具 - 检索增强生成

为HelloAgents框架提供简洁易用的RAG能力：
- 🔄 数据流程：用户数据 → 文档解析 → 向量化存储 → 智能检索 → LLM增强问答
- 📚 多格式支持：PDF、Word、Excel、PPT、图片、音频、网页等
- 🧠 智能问答：自动检索相关内容，注入提示词，生成准确答案
- 🏷️ 命名空间：支持多项目隔离，便于管理不同知识库

使用示例：
```python
# 1. 初始化RAG工具
rag = RAGTool()

# 2. 添加文档
rag.run({"action": "add_document", "file_path": "document.pdf"})

# 3. 智能问答
answer = rag.run({"action": "ask", "question": "什么是机器学习？"})
```
"""

from typing import Dict, Any, List, Optional
import os
import time

from ..base import Tool, ToolParameter, tool_action
from ...memory.rag.pipeline import create_rag_pipeline
from ...core.llm import HelloAgentsLLM

class RAGTool(Tool):
    """RAG工具
    
    提供完整的 RAG 能力：
    - 添加多格式文档（PDF、Office、图片、音频等）
    - 智能检索与召回
    - LLM 增强问答
    - 知识库管理
    """
    
    def __init__(
        self,
        knowledge_base_path: str = "./knowledge_base",
        qdrant_url: str = None,
        qdrant_api_key: str = None,
        collection_name: str = "rag_knowledge_base",
        namespace: str = "default",
        expandable: bool = False
    ):
        super().__init__(
            name="rag",
            description="RAG工具 - 支持多格式文档检索增强生成，提供智能问答能力",
            expandable=expandable
        )

        self.knowledge_base_path = knowledge_base_path
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        self.collection_name = collection_name
        self.namespace = namespace
        self._pipelines: Dict[str, Dict[str, Any]] = {}

        # 确保知识库目录存在
        os.makedirs(knowledge_base_path, exist_ok=True)

        # 初始化组件
        self._init_components()

    def _init_components(self):
        """初始化RAG组件"""
        try:
            # 初始化默认命名空间的 RAG 管道
            default_pipeline = create_rag_pipeline(
                qdrant_url=self.qdrant_url,
                qdrant_api_key=self.qdrant_api_key,
                collection_name=self.collection_name,
                namespace=self.namespace
            )
            self._pipelines[self.namespace] = default_pipeline

            # 初始化 LLM 用于回答生成
            self.llm = HelloAgentsLLM()

            self.initialized = True
            print(f"✅ RAG工具初始化成功: namespace={self.namespace}, collection={self.collection_name}")

        except Exception as e:
            self.initialized = False
            self.init_error = str(e)
            print(f"❌ RAG工具初始化失败: {e}")

    def _get_pipeline(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """获取指定命名空间的 RAG 管道，若不存在则自动创建"""
        target_ns = namespace or self.namespace
        if target_ns in self._pipelines:
            return self._pipelines[target_ns]

        pipeline = create_rag_pipeline(
            qdrant_url=self.qdrant_url,
            qdrant_api_key=self.qdrant_api_key,
            collection_name=self.collection_name,
            namespace=target_ns
        )
        self._pipelines[target_ns] = pipeline
        return pipeline

    def run(self, parameters: Dict[str, Any]) -> str:
        """执行工具（非展开模式）

        Args:
            parameters: 工具参数字典，必须包含action参数

        Returns:
            执行结果字符串
        """
        if not self.validate_parameters(parameters):
            return "❌ 参数验证失败：缺少必需的参数"

        if not self.initialized:
            return f"❌ RAG工具未正确初始化，请检查配置: {getattr(self, 'init_error', '未知错误')}"

        action = parameters.get("action")

        # 根据action调用对应的方法，传入提取的参数
        try:
            if action == "add_document":
                return self._add_document(
                    file_path=parameters.get("file_path"),
                    document_id=parameters.get("document_id"),
                    namespace=parameters.get("namespace"),
                    chunk_size=parameters.get("chunk_size", 800),
                    chunk_overlap=parameters.get("chunk_overlap", 100)
                )
            elif action == "add_text":
                return self._add_text(
                    text=parameters.get("text"),
                    document_id=parameters.get("document_id"),
                    namespace=parameters.get("namespace"),
                    chunk_size=parameters.get("chunk_size", 800),
                    chunk_overlap=parameters.get("chunk_overlap", 100)
                )
            elif action == "ask":
                question = parameters.get("question") or parameters.get("query")
                return self._ask(
                    question=question,
                    limit=parameters.get("limit", 5),
                    min_score=parameters.get("min_score", 0.1),
                    enable_mqe=parameters.get("enable_mqe", False),
                    enable_hyde=parameters.get("enable_hyde", False),
                    max_context_chars=parameters.get("max_context_chars", 1200),
                    namespace=parameters.get("namespace")
                )
            elif action == "search":
                return self._search(
                    query=parameters.get("query") or parameters.get("question"),
                    limit=parameters.get("limit", 5),
                    min_score=parameters.get("min_score", 0.1),
                    enable_advanced_search=parameters.get("enable_advanced_search", True),
                    max_chars=parameters.get("max_chars", 1200),
                    namespace=parameters.get("namespace")
                )
            elif action == "search_rerank":
                query = parameters.get("query") or parameters.get("question")
                result = self.search_rerank(
                    query=query,
                    limit=parameters.get("limit", 5),
                    min_score=parameters.get("min_score", 0.1),
                    enable_mqe=parameters.get("enable_mqe", False),
                    enable_hyde=parameters.get("enable_hyde", False),
                    max_context_chars=parameters.get("max_context_chars", 1200),
                    namespace=parameters.get("namespace")
                )
                return result["text"]
            elif action == "stats":
                return self._get_stats(namespace=parameters.get("namespace"))
            elif action == "clear":
                return self._clear_knowledge_base(
                    confirm=parameters.get("confirm", False),
                    namespace=parameters.get("namespace")
                )
            else:
                return f"❌ 不支持的操作: {action}"
        except Exception as e:
            return f"❌ 执行操作 '{action}' 时发生错误: {str(e)}"

    def get_parameters(self) -> List[ToolParameter]:
        """获取工具参数定义 - Tool基类要求的接口"""
        return [
            # 核心操作参数
            ToolParameter(
                name="action",
                type="string",
                description="操作类型：add_document(添加文档), add_text(添加文本), ask(智能问答), search(搜索), search_rerank(搜索并重排序), stats(统计), clear(清空)",
                required=True
            ),
            
            # 内容参数
            ToolParameter(
                name="file_path",
                type="string",
                description="文档文件路径（支持PDF、Word、Excel、PPT、图片、音频等多种格式）",
                required=False
            ),
            ToolParameter(
                name="text",
                type="string",
                description="要添加的文本内容",
                required=False
            ),
            ToolParameter(
                name="question",
                type="string", 
                description="用户问题（用于智能问答）",
                required=False
            ),
            ToolParameter(
                name="query",
                type="string",
                description="搜索查询词（用于基础搜索）",
                required=False
            ),
            
            # 可选配置参数
            ToolParameter(
                name="namespace",
                type="string",
                description="知识库命名空间（用于隔离不同项目，默认使用初始化时的命名空间）",
                required=False,
                default=None
            ),
            ToolParameter(
                name="limit",
                type="integer",
                description="返回结果数量（默认：5）",
                required=False,
                default=5
            ),
            ToolParameter(
                name="min_score",
                type="number",
                description="最低相关度分数阈值，用于向量检索初筛（默认：0.1）",
                required=False,
                default=0.1
            ),
            ToolParameter(
                name="enable_mqe",
                type="boolean",
                description="是否启用多查询扩展 Multi-Query Expansion（默认：false）",
                required=False,
                default=False
            ),
            ToolParameter(
                name="enable_hyde",
                type="boolean",
                description="是否启用假设文档嵌入 Hypothetical Document Embeddings（默认：false）",
                required=False,
                default=False
            ),
            ToolParameter(
                name="max_context_chars",
                type="integer",
                description="上下文拼装的最大字符数（默认：1200）",
                required=False,
                default=1200
            )
        ]

    @tool_action("rag_add_document", "添加文档到知识库（支持PDF、Word、Excel、PPT、图片、音频等多种格式）")
    def _add_document(
        self,
        file_path: str,
        document_id: str = None,
        namespace: str = None,
        chunk_size: int = 800,
        chunk_overlap: int = 100
    ) -> str:
        """添加文档到知识库

        Args:
            file_path: 文档文件路径
            document_id: 文档ID（可选）
            namespace: 知识库命名空间（用于隔离不同项目，默认使用初始化时的命名空间）
            chunk_size: 分块大小
            chunk_overlap: 分块重叠大小

        Returns:
            执行结果
        """
        try:
            if not file_path or not os.path.exists(file_path):
                return f"❌ 文件不存在: {file_path}"

            pipeline = self._get_pipeline(namespace)
            t0 = time.time()

            chunks_added = pipeline["add_documents"](
                file_paths=[file_path],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            t1 = time.time()
            process_ms = int((t1 - t0) * 1000)

            if chunks_added == 0:
                return f"⚠️ 未能从文件解析内容: {os.path.basename(file_path)}"

            return (
                f"✅ 文档已添加到知识库: {os.path.basename(file_path)}\n"
                f"📊 分块数量: {chunks_added}\n"
                f"⏱️ 处理时间: {process_ms}ms\n"
                f"📝 命名空间: {pipeline.get('namespace', self.namespace)}"
            )

        except Exception as e:
            return f"❌ 添加文档失败: {str(e)}"
    
    @tool_action("rag_add_text", "添加文本到知识库")
    def _add_text(
        self,
        text: str,
        document_id: str = None,
        namespace: str = None,
        chunk_size: int = 800,
        chunk_overlap: int = 100
    ) -> str:
        """添加文本到知识库

        Args:
            text: 要添加的文本内容
            document_id: 文档ID（可选）
            namespace: 知识库命名空间（默认使用初始化时的命名空间）
            chunk_size: 分块大小
            chunk_overlap: 分块重叠大小

        Returns:
            执行结果
        """
        try:
            if not text or not text.strip():
                return "❌ 文本内容不能为空"

            # 创建临时文件
            document_id = document_id or f"text_{abs(hash(text)) % 100000}"
            tmp_path = os.path.join(self.knowledge_base_path, f"{document_id}.md")

            try:
                with open(tmp_path, 'w', encoding='utf-8') as f:
                    f.write(text)

                pipeline = self._get_pipeline(namespace)
                t0 = time.time()

                chunks_added = pipeline["add_documents"](
                    file_paths=[tmp_path],
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )

                t1 = time.time()
                process_ms = int((t1 - t0) * 1000)

                if chunks_added == 0:
                    return f"⚠️ 未能从文本生成有效分块"

                return (
                    f"✅ 文本已添加到知识库: {document_id}\n"
                    f"📊 分块数量: {chunks_added}\n"
                    f"⏱️ 处理时间: {process_ms}ms\n"
                    f"📝 命名空间: {pipeline.get('namespace', self.namespace)}"
                )

            finally:
                # 清理临时文件
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass

        except Exception as e:
            return f"❌ 添加文本失败: {str(e)}"
    
    @tool_action("rag_search", "搜索知识库中的相关内容")
    def _search(
        self,
        query: str,
        limit: int = 5,
        min_score: float = 0.1,
        enable_mqe: bool = False,
        enable_hyde: bool = False,
        max_context_chars: int = 1200,
        namespace: str = None
    ) -> str:
        """搜索知识库（使用精排）

        Args:
            query: 搜索查询词
            limit: 返回结果数量
            min_score: 最低相关度分数
            enable_mqe: 是否启用多查询扩展
            enable_hyde: 是否启用假设文档嵌入
            max_context_chars: 上下文最大字符数
            namespace: 知识库命名空间（默认使用初始化时的命名空间）

        Returns:
            搜索结果
        """
        try:
            if not query or not query.strip():
                return "❌ 搜索查询不能为空"

            # 使用 search_rerank 进行精排搜索
            result = self.search_rerank(
                query=query,
                limit=limit,
                min_score=min_score,
                enable_mqe=enable_mqe,
                enable_hyde=enable_hyde,
                max_context_chars=max_context_chars,
                namespace=namespace
            )

            return result.get("text", f"🔍 未找到与 '{query}' 相关的内容")

        except Exception as e:
            return f"❌ 搜索失败: {str(e)}"


    def search_rerank(
        self,
        query: str,
        limit: int = 5,
        min_score: float = 0.1,
        enable_mqe: bool = False,
        enable_hyde: bool = False,
        max_context_chars: int = 1200,
        namespace: str = None
    ) -> Dict[str, Any]:
        """搜索并重排序，返回结构化结果（供 ContextBuilder 等程序化调用）

        使用 pipeline 的 search_rerank 进行粗排 → 精排 → 压缩，
        返回带有 Cross-Encoder rerank 分数的结果。始终包含引用。

        Args:
            query: 搜索查询词
            limit: 返回结果数量
            min_score: 最低相关度分数（用于向量检索初筛）
            enable_mqe: 是否启用多查询扩展（Multi-Query Expansion）
            enable_hyde: 是否启用假设文档嵌入（Hypothetical Document Embeddings）
            max_context_chars: 上下文拼装的最大字符数
            namespace: 知识库命名空间

        Returns:
            dict: {
                "text": 格式化结果文本,
                "text_list": 原始文本列表,
                "citations": 引用列表（每个元素含 index、source_path、score 等）
            }
        """
        try:
            if not query or not query.strip():
                return {"text": "❌ 搜索查询不能为空", "text_list": [], "citations": []}

            pipeline = self._get_pipeline(namespace)
            rerank_output = pipeline["search_rerank"](
                query=query,
                top_k=limit,
                score_threshold=min_score if min_score > 0 else None,
                enable_mqe=enable_mqe,
                enable_hyde=enable_hyde,
                max_context_chars=max_context_chars
            )

            text_list = rerank_output.get("text_list", [])
            citations = rerank_output.get("citations", [])
            if not text_list:
                return {"text": f"🔍 未找到与 '{query}' 相关的内容", "text_list": [], "citations": []}

            # 格式化：文本 + 引用标注
            parts = []
            for t, c in zip(text_list, citations):
                parts.append(f"{t} [{c['index']}]")
            if citations:
                formatted = "\n\n".join(parts) + "\n\n引用来源:\n" + "\n".join(
                    f"[{c['index']}] {c.get('source_path') or c.get('doc_id') or '来源'}"
                    + (f" ({c['start']}-{c['end']})" if c.get('start') is not None and c.get('end') is not None else "")
                    for c in citations
                )
            else:
                formatted = "\n\n".join(text_list)

            return {
                "text": formatted,
                "text_list": text_list,
                "citations": citations
            }

        except Exception as e:
            return {"text": f"❌ 搜索重排序失败: {str(e)}", "text_list": [], "citations": []}

    @tool_action("rag_ask", "基于知识库进行智能问答")
    def _ask(
        self,
        question: str,
        limit: int = 5,
        min_score: float = 0.3,
        enable_mqe: bool = False,
        enable_hyde: bool = False,
        max_context_chars: int = 1200,
        namespace: str = None
    ) -> str:
        """智能问答：检索 → 精排 → 上下文注入 → LLM生成答案

        Args:
            question: 用户问题
            limit: 检索结果数量
            min_score: 最低相关度分数（用于向量检索初筛）
            enable_mqe: 是否启用多查询扩展（Multi-Query Expansion）
            enable_hyde: 是否启用假设文档嵌入（Hypothetical Document Embeddings）
            max_context_chars: 上下文拼装的最大字符数
            namespace: 知识库命名空间（默认使用初始化时的命名空间）

        Returns:
            智能问答结果

        核心流程:
        1. 使用 self.search_rerank 进行检索+精排
        2. 构建上下文和提示词
        3. LLM生成准确答案
        4. 添加引用来源
        """
        try:
            # 验证问题
            if not question or not question.strip():
                return "❌ 请提供要询问的问题"

            user_question = question.strip()
            print(f"🔍 智能问答: {user_question}")

            # 1. 使用 search_rerank 进行检索+精排
            search_start = time.time()
            rerank_result = self.search_rerank(
                query=user_question,
                limit=limit,
                min_score=min_score,
                enable_mqe=enable_mqe,
                enable_hyde=enable_hyde,
                max_context_chars=max_context_chars,
                namespace=namespace
            )
            search_time = int((time.time() - search_start) * 1000)

            # 获取格式化文本和引用
            context = rerank_result.get("text", "")
            citations = rerank_result.get("citations", [])
            if not context or "未找到" in context:
                return (
                    f"🤔 抱歉，我在知识库中没有找到与「{user_question}」相关的信息。\n\n"
                    f"💡 建议：\n"
                    f"• 尝试使用更简洁的关键词\n"
                    f"• 检查是否已添加相关文档\n"
                    f"• 使用 stats 操作查看知识库状态"
                )

            # 2. 构建增强提示词
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(user_question, context)
            print(f"user_prompt: {user_prompt}")
            enhanced_prompt = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            # 3. 调用 LLM 生成答案
            llm_start = time.time()
            answer = self.llm.invoke(enhanced_prompt)
            llm_time = int((time.time() - llm_start) * 1000)

            if not answer or not answer.strip():
                return "❌ LLM 未能生成有效答案，请稍后重试"

            # 4. 构建最终回答
            final_answer = self._format_final_answer(
                question=user_question,
                answer=answer.strip(),
                citations=citations,
                search_time=search_time,
                llm_time=llm_time,
            )

            return final_answer

        except Exception as e:
            return f"❌ 智能问答失败: {str(e)}\n💡 请检查知识库状态或稍后重试"

    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        return (
            "你是一个知识库问答助手。请严格基于提供的上下文回答问题。\n"
            "规则：\n"
            "1. 只使用上下文中的信息回答，不编造内容\n"
            "2. 如果上下文不足以回答，直接说明「根据现有知识库，暂无法回答该问题」\n"
            "3. 回答简洁准确，必要时使用列表或分点\n"
            "4. 引用原文时使用引号标注"
        )

    def _build_user_prompt(self, question: str, context: str) -> str:
        """构建用户提示词"""
        return (
            f"上下文：\n{context}\n\n"
            f"问题：{question}\n\n"
            f"请基于上述上下文回答问题。"
        )
    
    def _format_final_answer(self, question: str, answer: str, citations: Optional[List[Dict]] = None, search_time: int = 0, llm_time: int = 0) -> str:
        """格式化最终答案"""
        result = [answer]

        if citations:
            result.append("\n---")
            result.append("**参考来源：**")
            for citation in citations:
                source = citation.get("source_path") or citation.get("doc_id") or "未知来源"
                loc = ""
                if citation.get("start") is not None and citation.get("end") is not None:
                    loc = f" ({citation['start']}-{citation['end']})"
                result.append(f"[{citation['index']}] {source}{loc}")

        result.append(f"\n⏱️ 检索: {search_time}ms | 生成: {llm_time}ms")

        return "\n".join(result)

    @tool_action("rag_clear", "清空知识库（危险操作，请谨慎使用）")
    def _clear_knowledge_base(self, confirm: bool = False, namespace: str = None) -> str:
        """清空知识库

        Args:
            confirm: 确认执行（必须设置为True）
            namespace: 知识库命名空间（默认使用初始化时的命名空间）

        Returns:
            执行结果
        """
        try:
            if not confirm:
                return (
                    "⚠️ 危险操作：清空知识库将删除所有数据！\n"
                    "请使用 confirm=true 参数确认执行。"
                )

            pipeline = self._get_pipeline(namespace)
            store = pipeline.get("store")
            namespace_id = pipeline.get("namespace", self.namespace)
            clear_namespace = pipeline.get("clear_namespace")
            success = clear_namespace() if clear_namespace else False

            if success:
                # 重新初始化该命名空间
                self._pipelines[namespace_id] = create_rag_pipeline(
                    qdrant_url=self.qdrant_url,
                    qdrant_api_key=self.qdrant_api_key,
                    collection_name=self.collection_name,
                    namespace=namespace_id
                )
                return f"✅ 知识库命名空间已成功清空（命名空间：{namespace_id}）"
            else:
                return "❌ 清空知识库失败"

        except Exception as e:
            return f"❌ 清空知识库失败: {str(e)}"

    @tool_action("rag_stats", "获取知识库统计信息")
    def _get_stats(self, namespace: str = None) -> str:
        """获取知识库统计

        Args:
            namespace: 知识库命名空间（默认使用初始化时的命名空间）

        Returns:
            统计信息
        """
        try:
            pipeline = self._get_pipeline(namespace)
            stats = pipeline["get_stats"]()
            stats_info = [
                "📊 **RAG 知识库统计**",
                f"📝 命名空间: {pipeline.get('namespace', self.namespace)}",
                f"📋 集合名称: {self.collection_name}",
                f"📂 存储根路径: {self.knowledge_base_path}"
            ]
            
            # 添加存储统计
            if stats:
                store_type = stats.get("store_type", "unknown")
                total_vectors = (
                    stats.get("points_count") or 
                    stats.get("vectors_count") or 
                    stats.get("count") or 0
                )
                
                stats_info.extend([
                    f"📦 存储类型: {store_type}",
                    f"📊 当前命名空间文档分块数: {int(total_vectors)}",
                ])
                
                if "config" in stats:
                    config = stats["config"]
                    if isinstance(config, dict):
                        vector_size = config.get("vector_size", "unknown")
                        distance = config.get("distance", "unknown")
                        stats_info.extend([
                            f"🔢 向量维度: {vector_size}",
                            f"📎 距离度量: {distance}"
                        ])
            
            # 添加系统状态
            stats_info.extend([
                "",
                "🟢 **系统状态**",
                f"✅ RAG 管道: {'正常' if self.initialized else '异常'}",
                f"✅ LLM 连接: {'正常' if hasattr(self, 'llm') else '异常'}"
            ])
            
            return "\n".join(stats_info)
            
        except Exception as e:
            return f"❌ 获取统计信息失败: {str(e)}"

    def get_relevant_context(self, query: str, limit: int = 3, max_chars: int = 1200, namespace: Optional[str] = None) -> str:
        """为查询获取相关上下文
        
        这个方法可以被Agent调用来获取相关的知识库上下文
        """
        try:
            if not query:
                return ""
            
            # 使用统一 RAG 管道搜索
            pipeline = self._get_pipeline(namespace)
            results = pipeline["search"](
                query=query,
                top_k=limit
            )
            
            if not results:
                return ""
            
            # 合并上下文
            context_parts = []
            for result in results:
                content = result.get("metadata", {}).get("content", "")
                if content:
                    context_parts.append(content)
            
            merged_context = "\n\n".join(context_parts)
            
            # 限制长度
            if len(merged_context) > max_chars:
                merged_context = merged_context[:max_chars] + "..."
            
            return merged_context
            
        except Exception as e:
            return f"获取上下文失败: {str(e)}"

    def clear_all_namespaces(self) -> str:
        """清空当前工具管理的所有命名空间数据"""
        try:
            for ns, pipeline in self._pipelines.items():
                store = pipeline.get("store")
                if store:
                    store.clear_collection()
            self._pipelines.clear()
            # 重新初始化默认命名空间
            self._init_components()
            return "✅ 所有命名空间数据已清空并重新初始化"
        except Exception as e:
            return f"❌ 清空所有命名空间失败: {str(e)}"
    
    # ========================================
    # 便捷接口方法（简化用户调用）
    # ========================================
    
    def add_document(self, file_path: str, namespace: str = None, chunk_size: int = 800, chunk_overlap: int = 100) -> str:
        """便捷方法：添加单个文档"""
        return self.run({
            "action": "add_document",
            "file_path": file_path,
            "namespace": namespace,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap
        })

    def add_text(self, text: str, namespace: str = None, document_id: str = None, chunk_size: int = 800, chunk_overlap: int = 100) -> str:
        """便捷方法：添加文本内容"""
        return self.run({
            "action": "add_text",
            "text": text,
            "namespace": namespace,
            "document_id": document_id,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap
        })

    def ask(self, question: str, namespace: str = None, **kwargs) -> str:
        """便捷方法：智能问答"""
        params = {
            "action": "ask",
            "question": question,
            "namespace": namespace
        }
        params.update(kwargs)
        return self.run(params)

    def search(self, query: str, namespace: str = None, **kwargs) -> str:
        """便捷方法：搜索知识库"""
        params = {
            "action": "search",
            "query": query,
            "namespace": namespace
        }
        params.update(kwargs)
        return self.run(params)

    def add_documents_batch(self, file_paths: List[str], namespace: str = None) -> str:
        """批量添加多个文档"""
        if not file_paths:
            return "❌ 文件路径列表不能为空"
        
        results = []
        successful = 0
        failed = 0
        total_chunks = 0
        start_time = time.time()
        
        for i, file_path in enumerate(file_paths, 1):
            print(f"📄 处理文档 {i}/{len(file_paths)}: {os.path.basename(file_path)}")
            
            try:
                result = self.add_document(file_path, namespace)
                if "✅" in result:
                    successful += 1
                    # 提取分块数量
                    if "分块数量:" in result:
                        chunks = int(result.split("分块数量: ")[1].split("\n")[0])
                        total_chunks += chunks
                else:
                    failed += 1
                    results.append(f"❌ {os.path.basename(file_path)}: 处理失败")
            except Exception as e:
                failed += 1
                results.append(f"❌ {os.path.basename(file_path)}: {str(e)}")
        
        process_time = int((time.time() - start_time) * 1000)
        
        summary = [
            "📊 **批量处理完成**",
            f"✅ 成功: {successful}/{len(file_paths)} 个文档",
            f"📊 总分块数: {total_chunks}",
            f"⏱️ 总耗时: {process_time}ms",
            f"📝 命名空间: {namespace}"
        ]
        
        if failed > 0:
            summary.append(f"❌ 失败: {failed} 个文档")
            summary.append("\n**失败详情:**")
            summary.extend(results)
        
        return "\n".join(summary)
    
    def add_texts_batch(self, texts: List[str], namespace: str = None, document_ids: Optional[List[str]] = None) -> str:
        """批量添加多个文本"""
        if not texts:
            return "❌ 文本列表不能为空"
        
        if document_ids and len(document_ids) != len(texts):
            return "❌ 文本数量和文档ID数量不匹配"
        
        results = []
        successful = 0
        failed = 0
        total_chunks = 0
        start_time = time.time()
        
        for i, text in enumerate(texts):
            doc_id = document_ids[i] if document_ids else f"batch_text_{i+1}"
            print(f"📝 处理文本 {i+1}/{len(texts)}: {doc_id}")
            
            try:
                result = self.add_text(text, namespace, doc_id)
                if "✅" in result:
                    successful += 1
                    # 提取分块数量
                    if "分块数量:" in result:
                        chunks = int(result.split("分块数量: ")[1].split("\n")[0])
                        total_chunks += chunks
                else:
                    failed += 1
                    results.append(f"❌ {doc_id}: 处理失败")
            except Exception as e:
                failed += 1
                results.append(f"❌ {doc_id}: {str(e)}")
        
        process_time = int((time.time() - start_time) * 1000)
        
        summary = [
            "📊 **批量文本处理完成**",
            f"✅ 成功: {successful}/{len(texts)} 个文本",
            f"📊 总分块数: {total_chunks}",
            f"⏱️ 总耗时: {process_time}ms",
            f"📝 命名空间: {namespace}"
        ]
        
        if failed > 0:
            summary.append(f"❌ 失败: {failed} 个文本")
            summary.append("\n**失败详情:**")
            summary.extend(results)

        return "\n".join(summary)

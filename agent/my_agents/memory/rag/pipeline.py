from functools import lru_cache
from typing import List, Dict, Optional, Any, Tuple
import os
import hashlib
import sqlite3
import time
import json
import uuid
from ..embedding import get_text_embedder, get_dimension
from ..storage.qdrant_store import QdrantVectorStore
from ..storage.document_store import SQLiteDocumentStore
import logging
logger = logging.getLogger(__name__)


RAG_PARSER_VERSION = "markitdown_recursive_v1"
RAG_POINT_NAMESPACE_UUID = uuid.UUID("8c0fe9e8-5c45-4e34-a1d3-6f0c35531df2")
RAG_IMPORT_MODES = {"SKIP", "REPLACE", "APPEND", "FAIL"}


def _get_markitdown_instance():
    """获取配置好的 MarkItDown 实例"""
    try:
        from markitdown import MarkItDown
        return MarkItDown()
    except ImportError:
        print("[WARNING] MarkItDown 不可用，请安装: pip install markitdown")
        return None


def _convert_to_markdown(path: str) -> str:
    """使用 MarkItDown 将文档转换为 markdown 文本"""
    if not os.path.exists(path):
        return ""
    
    # 对PDF文件使用增强处理
    ext = (os.path.splitext(path)[1] or '').lower()
    if ext == '.pdf':
        return _enhanced_pdf_processing(path)
    
    # 其他格式使用原有MarkItDown
    md_instance = _get_markitdown_instance()
    if md_instance is None:
        return _fallback_text_reader(path)
    
    try:
        result = md_instance.convert(path)
        text = getattr(result, "text_content", None)
        if isinstance(text, str) and text.strip():
            return text
        return ""
    except Exception as e:
        print(f"[WARNING] MarkItDown 转换失败 {path}: {e}")
        return _fallback_text_reader(path)

def _enhanced_pdf_processing(path: str) -> str:
    """增强的 PDF 处理，带后处理清理"""
    print(f"[RAG] 使用增强 PDF 处理: {path}")

    # 使用原有MarkItDown提取
    md_instance = _get_markitdown_instance()
    if md_instance is None:
        return _fallback_text_reader(path)

    try:
        result = md_instance.convert(path)
        raw_text = getattr(result, "text_content", None)
        if not raw_text or not raw_text.strip():
            return ""

        # 后处理：清理和重组文本
        cleaned_text = _post_process_pdf_text(raw_text)
        print(f"[RAG] PDF 后处理完成: {len(raw_text)} -> {len(cleaned_text)} 字符")
        return cleaned_text

    except Exception as e:
        print(f"[WARNING] 增强 PDF 处理失败 {path}: {e}")
        return _fallback_text_reader(path)

def _post_process_pdf_text(text: str) -> str:
    """PDF 文本后处理，提升质量"""
    import re
    
    # 1. 按行分割并清理
    lines = text.splitlines()
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 移除单个字符的行（通常是噪音）
        if len(line) <= 2 and not line.isdigit():
            continue
            
        # 移除明显的页眉页脚噪音
        if re.match(r'^\d+$', line):  # 纯数字行（页码）
            continue
        if line.lower() in ['github', 'project', 'forks', 'stars', 'language']:
            continue
            
        cleaned_lines.append(line)
    
    # 2. 智能合并短行
    merged_lines = []
    i = 0
    
    while i < len(cleaned_lines):
        current_line = cleaned_lines[i]
        
        # 如果当前行很短，尝试与下一行合并
        if len(current_line) < 60 and i + 1 < len(cleaned_lines):
            next_line = cleaned_lines[i + 1]
            
            # 合并条件：都是内容，不是标题
            if (not current_line.endswith('：') and 
                not current_line.endswith(':') and
                not current_line.startswith('#') and
                not next_line.startswith('#') and
                len(next_line) < 120):
                
                merged_line = current_line + " " + next_line
                merged_lines.append(merged_line)
                i += 2  # 跳过下一行
                continue
        
        merged_lines.append(current_line)
        i += 1
    
    # 3. 重新组织段落
    paragraphs = []
    current_paragraph = []
    
    for line in merged_lines:
        # 检查是否是新段落的开始
        if (line.startswith('#') or  # 标题
            line.endswith('：') or   # 中文冒号结尾
            line.endswith(':') or    # 英文冒号结尾
            len(line) > 150 or       # 长句通常是段落开始
            not current_paragraph):  # 第一行
            
            # 保存当前段落
            if current_paragraph:
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
            
            paragraphs.append(line)
        else:
            current_paragraph.append(line)
    
    # 添加最后一个段落
    if current_paragraph:
        paragraphs.append(' '.join(current_paragraph))
    
    return '\n\n'.join(paragraphs)


def _fallback_text_reader(path: str) -> str:
    """
    Simple fallback reader for basic text files when MarkItDown is unavailable.
    """
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception:
        try:
            with open(path, 'r', encoding='latin-1', errors='ignore') as f:
                return f.read()
        except Exception:
            return ""


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF or
        0x3400 <= code <= 0x4DBF or
        0x20000 <= code <= 0x2A6DF or
        0x2A700 <= code <= 0x2B73F or
        0x2B740 <= code <= 0x2B81F or
        0x2B820 <= code <= 0x2CEAF or
        0xF900 <= code <= 0xFAFF
    )


def _approx_token_len(text: str) -> int:
    # 近似估计：CJK字符按1 token，其他按空白分词
    cjk = sum(1 for ch in text if _is_cjk(ch))
    non_cjk_tokens = len([t for t in text.split() if t])
    return cjk + non_cjk_tokens



# 默认分隔符，按优先级排列
DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "；", "，", ", ", " ", ""]


def _find_best_separator(text: str, separators: List[str]) -> str:
    """找到文本中存在的最高优先级分隔符"""
    for sep in separators:
        if sep == "" or sep in text:
            return sep
    return ""


def _recursive_split_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: Optional[List[str]] = None,
    length_function=None,
) -> List[str]:
    """
    递归分割文本（参考 LangChain RecursiveCharacterTextSplitter）。

    核心思想：优先按大边界切分，切不动再按小边界。
    分隔符优先级：\n\n → \n → 。 → ； → ， → 空格 → 强制切

    Args:
        text: 待分割文本
        chunk_size: 目标 chunk 大小（字符数）
        chunk_overlap: 重叠大小
        separators: 分隔符列表，按优先级排列
        length_function: 长度计算函数，默认为 len

    Returns:
        分割后的文本片段列表
    """
    if separators is None:
        separators = DEFAULT_SEPARATORS

    if length_function is None:
        length_function = len

    # 基础情况：文本已经足够小
    if length_function(text) <= chunk_size:
        return [text] if text.strip() else []

    # 找到当前可用的最高优先级分隔符
    separator = _find_best_separator(text, separators)

    # 用分隔符切分
    if separator == "":
        # 没有分隔符，强制按长度切
        return _force_split(text, chunk_size, chunk_overlap, length_function)

    splits = text.split(separator)

    # 合并小 chunk，递归处理大 chunk
    chunks = []
    current = ""

    for part in splits:
        part_len = length_function(part)

        # 如果单个 part 就超大，递归处理
        if part_len > chunk_size:
            # 先保存当前累积的内容
            if current:
                chunks.append(current.strip())
                current = ""
            # 用更低优先级的分隔符递归切分这个大 part
            sub_separators = separators[separators.index(separator) + 1:]
            sub_chunks = _recursive_split_text(
                part, chunk_size, chunk_overlap, sub_separators, length_function
            )
            chunks.extend(sub_chunks)
            continue

        # 尝试累积
        if current:
            candidate = current + separator + part
        else:
            candidate = part

        if length_function(candidate) <= chunk_size:
            # 还没超，继续累积
            current = candidate
        else:
            # 超了，保存 current，开始新的
            if current:
                chunks.append(current.strip())
            current = part

    # 保存最后一个
    if current and current.strip():
        chunks.append(current.strip())

    # 添加重叠
    if chunk_overlap > 0 and len(chunks) > 1:
        chunks = _add_overlap(chunks, chunk_overlap, separator, length_function)

    # 过滤空 chunk
    return [c for c in chunks if c.strip()]


def _force_split(
    text: str,
    chunk_size: int,
    overlap: int,
    length_function=None,
) -> List[str]:
    """没有分隔符时，强制按长度切分"""
    if length_function is None:
        length_function = len

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    return chunks


def _add_overlap(
    chunks: List[str],
    overlap: int,
    separator: str,
    length_function=None,
) -> List[str]:
    """为 chunk 添加重叠"""
    if length_function is None:
        length_function = len

    if len(chunks) <= 1:
        return chunks

    result = [chunks[0]]
    for i in range(1, len(chunks)):
        # 取前一个 chunk 的尾部作为重叠
        prev = chunks[i - 1]
        if length_function(prev) > overlap:
            overlap_text = prev[-overlap:]
        else:
            overlap_text = prev

        # 在分隔符边界截取重叠，避免切断单词
        if separator:
            sep_pos = overlap_text.find(separator)
            if sep_pos != -1:
                overlap_text = overlap_text[sep_pos + len(separator):]

        result.append(overlap_text + separator + chunks[i])

    return result




def _embedding_fingerprint(embedding_dimension: int) -> str:
    model_type = os.getenv("EMBED_MODEL_TYPE", "dashscope").strip() or "dashscope"
    model_name = os.getenv("EMBED_MODEL_NAME", "").strip() or "default"
    return f"{model_type}:{model_name}:{embedding_dimension}"


def _build_doc_version_hash(
    content_hash: str,
    chunk_size: int,
    chunk_overlap: int,
    parser_version: str,
    embedding_fingerprint: str,
) -> str:
    raw = "|".join([
        content_hash,
        str(chunk_size),
        str(chunk_overlap),
        parser_version,
        embedding_fingerprint,
    ])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _build_chunk_id(
    namespace: str,
    doc_id: str,
    doc_version_hash: str,
    chunk_index: int,
    chunk_hash: str,
) -> str:
    raw = f"{namespace}:{doc_id}:{doc_version_hash}:{chunk_index}:{chunk_hash}"
    return str(uuid.uuid5(RAG_POINT_NAMESPACE_UUID, raw))


def load_and_chunk_texts(
    file_path: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    namespace: Optional[str] = None,
    doc_id: Optional[str] = None,
    parser_version: str = RAG_PARSER_VERSION,
    embedding_dimension: Optional[int] = None,
) -> List[Dict]:
    """
    通用文档加载和分块器，使用递归分割。
    将所有支持的格式转换为 markdown，然后递归分块。

    Args:
        file_path: 单个文件路径
        chunk_size: 目标 chunk 大小（token 数）
        chunk_overlap: 重叠大小（token 数）
        namespace: RAG 命名空间

    Returns:
        chunk 列表，每个包含 id、content、metadata
    """
    normalized_namespace = namespace or "default"
    normalized_chunk_size = max(1, int(chunk_size))
    normalized_chunk_overlap = max(0, int(chunk_overlap))
    embedding_dimension = int(embedding_dimension or get_dimension(384))
    embedding_fingerprint = _embedding_fingerprint(embedding_dimension)

    print(f"[RAG] 开始加载: 文件={file_path} chunk大小={normalized_chunk_size} 重叠={normalized_chunk_overlap} 命名空间={normalized_namespace}")
    if not os.path.exists(file_path):
        print(f"[WARNING] 文件不存在: {file_path}")
        return []

    print(f"[RAG] 处理中: {file_path}")
    canonical_path = os.path.abspath(file_path).replace("\\", "/")
    resolved_doc_id = doc_id.strip() if doc_id and str(doc_id).strip() else f"doc_{hashlib.sha256(os.path.abspath(file_path).replace(chr(92), '/').encode('utf-8')).hexdigest()[:24]}"

    # 使用 MarkItDown 转换为 markdown
    markdown_text = _convert_to_markdown(file_path)
    if not markdown_text.strip():
        print(f"[WARNING] 未能提取内容: {file_path}")
        return []

    content_hash = hashlib.sha256(markdown_text.encode("utf-8")).hexdigest()
    doc_version_hash = _build_doc_version_hash(
        content_hash=content_hash,
        chunk_size=normalized_chunk_size,
        chunk_overlap=normalized_chunk_overlap,
        parser_version=parser_version,
        embedding_fingerprint=embedding_fingerprint,
    )

    # 纯递归分割（不追踪标题）
    print(f"[RAG] 使用递归分割")
    text_chunks = _recursive_split_text(
        markdown_text,
        chunk_size=normalized_chunk_size,
        chunk_overlap=normalized_chunk_overlap,
        separators=DEFAULT_SEPARATORS,
        length_function=_approx_token_len,
    )

    chunks: List[Dict] = []
    seen_hashes = set()
    offset = 0
    for chunk_text in text_chunks:
        norm = chunk_text.strip()
        if not norm:
            continue

        chunk_hash = hashlib.sha256(norm.encode("utf-8")).hexdigest()
        if chunk_hash in seen_hashes:
            continue
        seen_hashes.add(chunk_hash)

        # 计算位置
        start = markdown_text.find(chunk_text, offset)
        if start == -1:
            start = offset
        end = start + len(chunk_text)
        chunk_index = len(chunks)

        chunk_id = _build_chunk_id(
            namespace=normalized_namespace,
            doc_id=resolved_doc_id,
            doc_version_hash=doc_version_hash,
            chunk_index=chunk_index,
            chunk_hash=chunk_hash,
        )
        chunks.append({
            "id": chunk_id,
            "content": chunk_text,
            "metadata": {
                "source_path": canonical_path,
                "doc_id": resolved_doc_id,
                "doc_version_hash": doc_version_hash,
                "start": start,
                "end": end,
                "content_hash": content_hash,
                "chunk_hash": chunk_hash,
                "chunk_index": chunk_index,
                "chunk_size": normalized_chunk_size,
                "chunk_overlap": normalized_chunk_overlap,
                "parser_version": parser_version,
                "embedding_model": embedding_fingerprint,
                "embedding_dimension": embedding_dimension,
                "namespace": normalized_namespace,
            },
        })
        offset = end

    chunk_count = len(chunks)
    for chunk in chunks:
        chunk["metadata"]["chunk_count"] = chunk_count

    print(f"[RAG] 加载完成: 总chunk数={len(chunks)}")
    return chunks




def _preprocess_markdown_for_embedding(text: str) -> str:
    """预处理 markdown 文本，提升 embedding 质量"""
    import re

    # 移除 markdown 标题符号但保留文本
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

    # 移除 markdown 链接但保留文本
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    # 移除 markdown 强调标记
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # 粗体
    text = re.sub(r'\*([^*]+)\*', r'\1', text)      # 斜体
    text = re.sub(r'`([^`]+)`', r'\1', text)        # 行内代码

    # 移除 markdown 代码块但保留内容
    text = re.sub(r'```[^\n]*\n([\s\S]*?)```', r'\1', text)

    # 移除多余空白
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)

    return text.strip()


def _create_default_vector_store(dimension: int = None) -> QdrantVectorStore:
    """
    Create default Qdrant vector store with RAG-optimized settings.
    使用连接管理器避免重复连接。
    """
    if dimension is None:
        dimension = get_dimension(384)

    # 检查 Qdrant 配置
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    # 使用连接管理器
    from ..storage.qdrant_store import QdrantConnectionManager
    return QdrantConnectionManager.get_instance(
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name="hello_agents_rag_vectors",
        vector_size=dimension,
        distance="cosine"
    )


def _normalize_vectors(raw_vecs, expected_count: int) -> List[List[float]]:
    """
    将各种格式的向量输出统一转换为 List[List[float]]。

    支持的输入格式：
    - numpy 2D 数组 (shape: [n, dim])
    - numpy 1D 数组 (单个向量)
    - List[numpy.ndarray] (numpy 数组列表)
    - List[List[float]] (已经是目标格式)
    - List[float] (单个向量的扁平列表)
    """
    # 情况 1: numpy 2D 数组
    if hasattr(raw_vecs, "tolist") and hasattr(raw_vecs, "shape"):
        if len(raw_vecs.shape) == 2:
            return [list(row) for row in raw_vecs.tolist()]
        elif len(raw_vecs.shape) == 1:
            return [list(raw_vecs.tolist())]

    # 情况 2: 非列表（不应该发生，但防御性处理）
    if not isinstance(raw_vecs, (list, tuple)):
        if hasattr(raw_vecs, "tolist"):
            return [list(raw_vecs.tolist())]
        return [list(raw_vecs)]

    # 情况 3: 空列表
    if not raw_vecs:
        return []

    first = raw_vecs[0]

    # 情况 4: List[List[float]] - 已经是目标格式
    if isinstance(first, (list, tuple)):
        return [list(v) for v in raw_vecs]

    # 情况 5: List[numpy.ndarray] - numpy 数组列表
    if hasattr(first, "tolist"):
        return [list(v.tolist()) for v in raw_vecs]

    # 情况 6: List[float] - 单个向量的扁平列表（所有元素是标量）
    try:
        float(first)
        return [list(raw_vecs)]
    except (TypeError, ValueError):
        pass

    # 兜底: 尝试逐个转换
    result = []
    for v in raw_vecs:
        if hasattr(v, "tolist"):
            result.append(list(v.tolist()))
        elif isinstance(v, (list, tuple)):
            result.append(list(v))
        else:
            result.append([float(v)])
    return result


def index_chunks(
    store = None,
    chunks: List[Dict] = None,
    cache_db: Optional[str] = None,
    batch_size: int = 10,
    namespace: str = "default"
) -> int:
    """将 markdown chunks 索引到 Qdrant 向量存储"""
    if not chunks:
        print("[RAG] 没有需要索引的 chunks")
        return 0

    # 使用统一的 embedding 模块
    embedder = get_text_embedder()
    dimension = get_dimension(384)

    # 如果未提供 store，创建默认的 Qdrant store
    if store is None:
        store = _create_default_vector_store(dimension)
        print(f"[RAG] 创建默认 Qdrant store，维度: {dimension}")
    
    # 预处理 markdown 文本以获得更好的 embedding
    processed_texts = []
    for c in chunks:
        raw_content = c["content"]
        processed_content = _preprocess_markdown_for_embedding(raw_content)
        processed_texts.append(processed_content)

    print(f"[RAG] 开始 Embedding: 总文本数={len(processed_texts)} 批次大小={batch_size}")

    # 批量编码
    vecs: List[List[float]] = []
    valid_indices: List[int] = []  # 记录成功编码的 chunk 索引
    failed_chunks: List[Dict] = []  # 记录失败的 chunk
    for i in range(0, len(processed_texts), batch_size):
        part = processed_texts[i:i+batch_size]
        try:
            # 使用统一的 embedder
            part_vecs = embedder.encode(part)

            # 统一归一化为 List[List[float]]
            part_vecs = _normalize_vectors(part_vecs, len(part))

            for j, v in enumerate(part_vecs):
                try:
                    # 确保向量是float列表
                    if hasattr(v, "tolist"):
                        v = v.tolist()
                    v_norm = [float(x) for x in v]
                    # 检查维度，异常则跳过（不使用零向量）
                    if len(v_norm) != dimension:
                        failed_chunks.append({
                            "index": i + j,
                            "reason": f"向量维度异常: 期望{dimension}, 实际{len(v_norm)}",
                            "content_preview": part[j][:50] if part[j] else ""
                        })
                        continue
                    vecs.append(v_norm)
                    valid_indices.append(i + j)
                except Exception as e:
                    failed_chunks.append({
                        "index": i + j,
                        "reason": f"向量转换失败: {e}",
                        "content_preview": part[j][:50] if part[j] else ""
                    })
                    continue

        except Exception as e:
            # 整批失败
            for j in range(len(part)):
                failed_chunks.append({
                    "index": i + j,
                    "reason": f"批次编码失败: {e}",
                    "content_preview": part[j][:50] if part[j] else ""
                })

        print(f"[RAG] Embedding 进度: {min(i+batch_size, len(processed_texts))}/{len(processed_texts)}")

    # 打印失败统计
    if failed_chunks:
        print(f"[RAG] 失败的 chunks: {len(failed_chunks)}/{len(chunks)}")
        for fc in failed_chunks[:10]:  # 最多打印 10 条
            print(f"  - chunk[{fc['index']}]: {fc['reason']} | 内容: {fc['content_preview']}...")
        if len(failed_chunks) > 10:
            print(f"  - ... 还有 {len(failed_chunks) - 10} 条")

    # 准备元数据（只包含成功编码的 chunk）
    metas: List[Dict] = []
    ids: List[str] = []
    payload_metadata_keys = {
        "source_path",
        "doc_id",
        "doc_version_hash",
        "start",
        "end",
        "chunk_index",
        "namespace",
    }
    for idx in valid_indices:
        ch = chunks[idx]
        meta = {
            "id": ch["id"],
            "memory_type": "rag_chunk",
            "content": ch["content"],
            "namespace": namespace,
        }
        chunk_meta = ch.get("metadata", {})
        meta.update({
            key: chunk_meta[key]
            for key in payload_metadata_keys
            if key in chunk_meta
        })
        metas.append(meta)
        ids.append(ch["id"])

    print(f"[RAG] 开始写入 Qdrant: 向量数={len(vecs)}")
    if not vecs:
        print(f"[RAG] 没有有效的向量可以索引")
        return 0
    success = store.add_vectors(vectors=vecs, metadata=metas, ids=ids)
    if success:
        print(f"[RAG] Qdrant 写入完成: {len(vecs)} 个向量已索引")
        return len(vecs)
    else:
        print(f"[RAG] Qdrant 写入失败")
        raise RuntimeError("向量索引到 Qdrant 失败")


def embed_query(query: str) -> List[float]:
    """使用统一 embedding 对查询进行向量化"""
    embedder = get_text_embedder()
    dimension = get_dimension(384)
    try:
        vec = embedder.encode(query)

        # 归一化为 List[float]
        if hasattr(vec, "tolist"):
            vec = vec.tolist()

        # 处理嵌套列表情况
        if isinstance(vec, list) and vec and isinstance(vec[0], (list, tuple)):
            vec = vec[0]  # 提取第一个向量

        # 转换为float列表
        result = [float(x) for x in vec]

        # 检查维度
        if len(result) != dimension:
            print(f"[WARNING] 查询向量维度异常: 期望{dimension}, 实际{len(result)}")
            # 用零向量填充或截断
            if len(result) < dimension:
                result.extend([0.0] * (dimension - len(result)))
            else:
                result = result[:dimension]

        return result
    except Exception as e:
        print(f"[WARNING] 查询 Embedding 失败: {e}")
        # 返回零向量作为兜底
        return [0.0] * dimension


def _prompt_mqe(query: str, n: int) -> List[str]:
    """多查询扩展：生成语义等价的多样化查询"""
    try:
        from ...core.llm import HelloAgentsLLM
        llm = HelloAgentsLLM()
        prompt = [
            {"role": "system", "content": "你是检索查询扩展助手。生成语义等价或互补的多样化查询。使用中文，简短，避免标点。"},
            {"role": "user", "content": f"原始查询：{query}\n请给出{n}个不同表述的查询，每行一个。"}
        ]
        text = llm.invoke(prompt)
        lines = [ln.strip("- \t") for ln in (text or "").splitlines()]
        outs = [ln for ln in lines if ln]
        return outs[:n] or [query]
    except Exception:
        return [query]


def _prompt_hyde(query: str) -> Optional[str]:
    """假设文档扩展：生成假设的答案文档用于检索"""
    try:
        from ...core.llm import HelloAgentsLLM
        llm = HelloAgentsLLM()
        prompt = [
            {"role": "system", "content": "根据用户问题，先写一段可能的答案性段落，用于向量检索的查询文档（不要分析过程）。"},
            {"role": "user", "content": f"问题：{query}\n请直接写一段中等长度、客观、包含关键术语的段落。"}
        ]
        return llm.invoke(prompt)
    except Exception:
        return None


def search_vectors(
    store = None,
    query: str = "",
    top_k: int = 8,
    namespace: Optional[str] = None,
    score_threshold: Optional[float] = None,
    enable_mqe: bool = False,
    mqe_expansions: int = 2,
    enable_hyde: bool = False,
    candidate_pool_multiplier: int = 4,
) -> List[Dict]:
    """
    RAG 向量搜索，支持可选的查询扩展。

    Args:
        store: QdrantVectorStore 实例
        query: 查询文本
        top_k: 返回结果数量
        namespace: RAG 命名空间
        score_threshold: 最低分数阈值
        enable_mqe: 是否启用多查询扩展
        mqe_expansions: MQE 扩展数量
        enable_hyde: 是否启用假设文档扩展
        candidate_pool_multiplier: 候选池倍数（扩展时使用）

    Returns:
        搜索结果列表
    """
    if not query:
        return []

    # 如果未提供 store，创建默认的
    if store is None:
        store = _create_default_vector_store()

    # 构建 RAG 数据过滤条件
    where = {"memory_type": "rag_chunk"}
    if namespace:
        where["namespace"] = namespace

    # 无扩展：直接搜索
    if not enable_mqe and not enable_hyde:
        qv = embed_query(query)
        try:
            return store.search_similar(
                query_vector=qv,
                limit=top_k,
                score_threshold=score_threshold,
                where=where
            )
        except Exception as e:
            print(f"[WARNING] RAG 搜索失败: {e}")
            return []

    # 有扩展：多查询搜索
    expansions: List[str] = [query]

    if enable_mqe and mqe_expansions > 0:
        expansions.extend(_prompt_mqe(query, mqe_expansions))
    if enable_hyde:
        hyde_text = _prompt_hyde(query)
        if hyde_text:
            expansions.append(hyde_text)

    # 去重
    uniq: List[str] = []
    for e in expansions:
        if e and e not in uniq:
            uniq.append(e)
    expansions = uniq[: max(1, len(uniq))]
    # print(f"expansions: {expansions}")
    # 分配候选池
    pool = max(top_k * candidate_pool_multiplier, 20)
    per = max(1, pool // max(1, len(expansions)))

    # 收集所有扩展查询的结果
    agg: Dict[str, Dict] = {}
    for q in expansions:
        qv = embed_query(q)
        try:
            hits = store.search_similar(query_vector=qv, limit=per, score_threshold=score_threshold, where=where)
            for h in hits:
                mid = h.get("metadata", {}).get("id")
                if not mid:
                    continue
                s = float(h.get("score", 0.0))
                if mid not in agg or s > float(agg[mid].get("score", 0.0)):
                    agg[mid] = h
        except Exception as e:
            print(f"[WARNING] 扩展查询 '{q[:20]}...' 搜索失败: {e}")

    # 按分数排序返回
    merged = list(agg.values())
    merged.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return merged[:top_k]


@lru_cache(maxsize=4)
def _load_cross_encoder(model_name: str):
    """按模型名称缓存 Cross-Encoder，避免每轮检索重复从磁盘加载。"""
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name)
    logger.info(f"Cross-Encoder {model_name} 加载成功")
    return model


def _try_load_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    try:
        import os

        # 禁用 HuggingFace 的进度条和警告
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        return _load_cross_encoder(model_name)
    except Exception:
        return None


def rerank_with_cross_encoder(query: str, items: List[Dict], model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", top_k: int = 10) -> List[Dict]:
    ce = _try_load_cross_encoder(model_name)
    if ce is None or not items:
        return items[:top_k]
    pairs = [[query, it.get("metadata", {}).get("content", "")] for it in items]
    try:
        raw_scores = ce.predict(pairs, show_progress_bar=False)
        # 归一化到 [0, 1]
        min_s = float(min(raw_scores))
        max_s = float(max(raw_scores))
        if max_s - min_s < 1e-8:
            norm_scores = [1.0] * len(raw_scores)
        else:
            norm_scores = [(float(s) - min_s) / (max_s - min_s) for s in raw_scores]
        for it, ns in zip(items, norm_scores):
            it["rerank_score"] = ns
        items.sort(key=lambda x: x.get("rerank_score", x.get("score", 0.0)), reverse=True)
        return items[:top_k]
    except Exception:
        return items[:top_k]

def compress_ranked_items(
    ranked_items: List[Dict],
    enable_compression: bool = True,
    max_per_doc: int = 2,
    join_gap: int = 100,
) -> List[Dict]:
    """
    片段压缩：将同一文档中位置相邻的 chunk 合并，然后按分数降序输出。

    处理流程：
    1. 按文档分组
    2. 组内按 start 位置排序（不依赖输入顺序）
    3. 组内合并相邻 chunk（距离 ≤ join_gap 的连续 chunk 合并为一个）
    4. 合并后保留最高分数、最宽位置范围、拼接内容
    5. 每个文档最多保留 max_per_doc 个片段
    6. 所有结果按 rerank_score（优先）或 score 降序输出

    Args:
        ranked_items: 精排后的结果列表（位置顺序任意）
        enable_compression: 是否启用压缩
        max_per_doc: 每个文档最多保留的片段数
        join_gap: 相邻 chunk 合并的距离阈值（字符数）

    Returns:
        压缩后的结果列表，保留原始字段，按分数降序排列
    """
    if not enable_compression or not ranked_items:
        return ranked_items

    def _get_sort_score(item: Dict) -> float:
        """获取排序分数：优先使用 rerank_score，否则使用 score"""
        return float(item.get("rerank_score", item.get("score", 0.0)))

    # ── 第一步：按文档分组 ──────────────────────────────
    from collections import defaultdict
    grouped: Dict[str, List[Dict]] = defaultdict(list)

    for it in ranked_items:
        meta = it.get("metadata", {})
        did = meta.get("doc_id") or meta.get("source_path") or "unknown"
        grouped[did].append(it)

    # ── 第二步：组内按 start 排序 + 合并相邻 chunk ──────
    merged_items: List[Dict] = []

    def _new_segment(it: Dict, meta: Dict, content: str, start: int, end: int,
                     score: float, rerank_score: Optional[float], chunk_id: str) -> Dict:
        """创建新片段，只保留必要字段"""
        seg: Dict = {
            "id": it.get("id", ""),
            "content": content,
            "score": score,
            "metadata": {
                "doc_id": meta.get("doc_id", ""),
                "source_path": meta.get("source_path", ""),
                "start": start,
                "end": end,
                "merged_ids": [chunk_id] if chunk_id else [],
            },
        }
        if rerank_score is not None:
            seg["rerank_score"] = float(rerank_score)
        return seg

    for did, items in grouped.items():
        # 按 start 排序，确保位置递增
        items.sort(key=lambda x: int(x.get("metadata", {}).get("start", 0)))

        doc_segments: List[Dict] = []
        current = None

        for it in items:
            meta = it.get("metadata", {})
            content = (meta.get("content") or "").strip()
            start = int(meta.get("start") or 0)
            end = int(meta.get("end") or (start + len(content)))
            rerank_score = it.get("rerank_score")
            score = float(it.get("score", 0.0))
            chunk_id = meta.get("id")

            if current is None:
                current = _new_segment(it, meta, content, start, end, score, rerank_score, chunk_id)
                continue

            # 判断是否可以合并：当前位置起始 - 上一个结束 ≤ join_gap
            gap = start - current["metadata"]["end"]
            if gap <= join_gap:
                # 合并：通过位置计算处理重叠，拼接内容，扩展位置范围，保留最高分
                if content:
                    if start < current["metadata"]["end"]:
                        overlap = current["metadata"]["end"] - start
                        content = content[overlap:]
                    current["content"] += content
                current["metadata"]["end"] = max(current["metadata"]["end"], end)
                current["score"] = max(current["score"], score)
                if chunk_id:
                    current["metadata"]["merged_ids"].append(chunk_id)
                if rerank_score is not None:
                    prev = current.get("rerank_score", float("-inf"))
                    current["rerank_score"] = max(prev, float(rerank_score))
            else:
                doc_segments.append(current)
                current = _new_segment(it, meta, content, start, end, score, rerank_score, chunk_id)

        if current is not None:
            doc_segments.append(current)

        # 每个文档最多保留 max_per_doc 个片段
        doc_segments.sort(
            key=lambda x: x.get("rerank_score", x.get("score", 0.0)),
            reverse=True,
        )
        merged_items.extend(doc_segments[:max_per_doc])

    # ── 第三步：所有结果按分数降序输出 ──────────────────
    merged_items.sort(
        key=lambda x: x.get("rerank_score", x.get("score", 0.0)),
        reverse=True,
    )

    return merged_items


def assemble_context(ranked_items: List[Dict], max_chars: int = 1200) -> Tuple[List[str], List[Dict]]:
    """
    上下文拼装：按相关度分数从高到低依次拼接，直到达到 max_chars 上限。
    始终生成引用元数据，与文本列表一一对应。

    Args:
        ranked_items: 压缩后的结果列表（已按分数降序）
        max_chars: 最大字符数

    Returns:
        (文本内容列表, 引用列表) — 每个元素对应一个 chunk，两项列表长度一致
    """
    text_list: List[str] = []
    citations: List[Dict] = []
    total = 0
    cite_index = 1

    for it in ranked_items:
        text = (it.get("content") or "").strip()
        if not text:
            continue
        need = len(text)

        if total + need > max_chars:
            remain = max_chars - total
            if remain <= 0:
                break
            clipped = text[:remain]
            if clipped:
                text_list.append(clipped)
                m = it.get("metadata", {})
                citations.append({
                    "index": cite_index,
                    "source_path": m.get("source_path"),
                    "doc_id": m.get("doc_id"),
                    "start": m.get("start"),
                    "end": m.get("end"),
                    "score": it.get("rerank_score", it.get("score", 0.0)),
                })
            break

        text_list.append(text)
        total += need
        m = it.get("metadata", {})
        citations.append({
            "index": cite_index,
            "source_path": m.get("source_path"),
            "doc_id": m.get("doc_id"),
            "start": m.get("start"),
            "end": m.get("end"),
            "score": it.get("rerank_score", it.get("score", 0.0)),
        })
        cite_index += 1

    return text_list, citations

def search_and_rerank(
    store,
    query: str,
    top_k: int = 8,
    namespace: Optional[str] = None,
    score_threshold: Optional[float] = None,
    enable_mqe: bool = False,
    mqe_expansions: int = 2,
    enable_hyde: bool = False,
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    compress_max_per_doc: int = 2,
    compress_join_gap: int = 100,
    max_context_chars: int = 1200,
    coarse_top_k_multiplier: int = 4,
) -> Dict[str, Any]:
    """
    简化的检索排序流程：粗排 → 精排 → 压缩 → 上下文拼装。始终生成引用。

    Args:
        store: QdrantVectorStore 实例
        query: 查询文本
        top_k: 最终返回的结果数量
        namespace: RAG 命名空间
        score_threshold: 向量检索的最低分数阈值
        enable_mqe: 是否启用多查询扩展
        mqe_expansions: MQE 扩展数量
        enable_hyde: 是否启用假设文档扩展
        rerank_model: Cross-Encoder 模型名称
        compress_max_per_doc: 每个文档最多保留的片段数
        compress_join_gap: 同文档相邻片段合并的距离阈值（字符数）
        max_context_chars: 上下文拼装的最大字符数
        coarse_top_k_multiplier: 粗排候选池倍数（粗排数量 = top_k * multiplier）

    Returns:
        Dict 包含：
        - results: 最终排序结果列表
        - text_list: 文本内容列表
        - citations: 引用来源列表
        - stats: 各阶段统计信息
    """
    stats = {}
    query = query.strip()
    if not query:
        return {"results": [], "context": "", "citations": [], "stats": stats}

    # ── 阶段 1：向量粗排 ──────────────────────────────
    coarse_top_k = max(top_k * coarse_top_k_multiplier, 20)

    coarse_hits = search_vectors(
        store=store,
        query=query,
        top_k=coarse_top_k,
        namespace=namespace,
        score_threshold=score_threshold,
        enable_mqe=enable_mqe,
        mqe_expansions=mqe_expansions,
        enable_hyde=enable_hyde,
        candidate_pool_multiplier=1,
    )
    # print(f"粗排结果：\n {coarse_hits}")
    stats["coarse_count"] = len(coarse_hits)

    if not coarse_hits:
        return {"results": [], "context": "", "citations": [], "stats": stats}

    # ── 阶段 2：Cross-Encoder 精排 ────────────────────
    fine_hits = rerank_with_cross_encoder(
        query=query,
        items=coarse_hits,
        model_name=rerank_model,
        top_k=top_k * 2,  # 精排后多保留一些，供压缩和扩展使用
    )
    # print(f"精排结果：\n {fine_hits}")
    stats["rerank_count"] = len(fine_hits)

    # ── 阶段 3：片段压缩 ──────────────────────────────
    compressed = compress_ranked_items(
        ranked_items=fine_hits,
        enable_compression=True,
        max_per_doc=compress_max_per_doc,
        join_gap=compress_join_gap,
    )
    # print(f"压缩结果：\n {compressed}")
    stats["compressed_count"] = len(compressed)

    # ── 阶段 4：上下文拼装 + 引用溯源 ─────────────────
    text_list, citations = assemble_context(
        ranked_items=compressed[:top_k],
        max_chars=max_context_chars,
    )

    stats["context_chars"] = sum(len(t) for t in text_list)

    return {
        "results": compressed[:top_k],
        "text_list": text_list,
        "citations": citations,
        "stats": stats,
    }


# ==================
# 高层 RAG Pipeline API
# ==================


def _rag_chunk_filter(namespace: str, doc_id: Optional[str] = None, doc_version_hash: Optional[str] = None) -> Dict[str, Any]:
    where: Dict[str, Any] = {
        "memory_type": "rag_chunk",
        "namespace": namespace,
    }
    if doc_id:
        where["doc_id"] = doc_id
    if doc_version_hash:
        where["doc_version_hash"] = doc_version_hash
    return where


def create_rag_pipeline(
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    collection_name: str = "hello_agents_rag_vectors",
    namespace: str = "default",
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    创建完整的 RAG Pipeline，包含 Qdrant 和统一 embedding。

    Returns:
        包含 store、namespace 和辅助函数的字典
    """
    dimension = get_dimension(384)

    store = QdrantVectorStore(
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name=collection_name,
        vector_size=dimension,
        distance="cosine"
    )
    manifest_store = SQLiteDocumentStore(db_path=db_path)

    def add_documents(
        file_path: str,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        doc_id: Optional[str] = None,
        upsert_mode: str = "SKIP",
    ) -> Dict[str, Any]:
        """添加单个文档到 RAG Pipeline，并按 doc_id/doc_version_hash 做幂等判断。"""
        mode = (upsert_mode or "SKIP").strip().upper()
        if mode not in RAG_IMPORT_MODES:
            raise ValueError(f"不支持的 RAG 导入策略: {upsert_mode}，可选: SKIP/REPLACE/APPEND/FAIL")
        base_result = {
            "namespace": namespace,
            "mode": mode,
            "file_path": file_path,
            "chunks_added": 0,
        }

        chunks = load_and_chunk_texts(
            file_path=file_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            namespace=namespace,
            doc_id=doc_id,
            embedding_dimension=dimension,
        )

        if not chunks:
            return {
                **base_result,
                "status": "empty",
                "chunks_total": 0,
                "reason": "未能解析出可索引内容",
            }

        first_meta = chunks[0].get("metadata", {})
        doc_id = first_meta.get("doc_id")
        doc_version_hash = first_meta.get("doc_version_hash")
        content_hash = first_meta.get("content_hash")
        expected_count = len(chunks)

        same_version_where = _rag_chunk_filter(
            namespace=namespace,
            doc_id=doc_id,
            doc_version_hash=doc_version_hash,
        )
        active_manifests = manifest_store.get_active_rag_manifests(namespace, doc_id)
        same_version_manifest = next(
            (
                manifest
                for manifest in active_manifests
                if manifest.get("doc_version_hash") == doc_version_hash
            ),
            None,
        )
        old_version_manifests = [
            manifest
            for manifest in active_manifests
            if manifest.get("doc_version_hash") != doc_version_hash
        ]
        same_doc_count = sum(int(manifest.get("chunk_count") or 0) for manifest in active_manifests)
        same_version_count = int(same_version_manifest.get("chunk_count") or 0) if same_version_manifest else 0
        has_same_version = same_version_manifest is not None
        has_other_version = bool(old_version_manifests)

        base_report = {
            "source_path": first_meta.get("source_path"),
            "doc_id": doc_id,
            "doc_version_hash": doc_version_hash,
            "content_hash": content_hash,
            "chunks_total": expected_count,
            "chunks_existing": same_version_count,
            "chunks_existing_for_doc": same_doc_count,
        }

        if mode == "FAIL" and same_doc_count > 0:
            return {
                **base_result,
                "status": "failed",
                **base_report,
                "reason": "doc_id 已存在，FAIL 策略拒绝导入",
            }

        if has_same_version:
            return {
                **base_result,
                "status": "skipped",
                **base_report,
                "reason": "同版本已完整存在",
            }

        if mode == "SKIP" and has_other_version:
            return {
                **base_result,
                "status": "conflict",
                **base_report,
                "reason": "同 doc_id 已存在不同版本或不完整版本，SKIP 策略拒绝导入",
            }

        chunks_deleted = 0
        target_status = "imported"
        if mode == "REPLACE" and has_other_version:
            chunks_deleted = sum(int(manifest.get("chunk_count") or 0) for manifest in old_version_manifests)
            target_status = "replaced"
        elif mode == "APPEND" and has_other_version:
            target_status = "appended"

        try:
            manifest_store.create_rag_manifest_pending(
                namespace=namespace,
                doc_id=doc_id,
                doc_version_hash=doc_version_hash,
                source_path=first_meta.get("source_path") or file_path,
                content_hash=content_hash,
                chunk_count=expected_count,
                chunk_size=int(first_meta.get("chunk_size") or chunk_size),
                chunk_overlap=int(first_meta.get("chunk_overlap") or chunk_overlap),
                parser_version=first_meta.get("parser_version") or RAG_PARSER_VERSION,
                embedding_model=first_meta.get("embedding_model") or "",
                embedding_dimension=int(first_meta.get("embedding_dimension") or dimension),
                qdrant_collection=collection_name,
                metadata={"mode": mode, "target_status": target_status},
            )
        except Exception as e:
            return {
                **base_result,
                "status": "failed",
                **base_report,
                "reason": f"创建 RAG manifest 失败: {e}",
            }

        try:
            chunks_added = index_chunks(
                store=store,
                chunks=chunks,
                namespace=namespace
            )
        except Exception as e:
            manifest_store.mark_rag_manifest_failed(
                namespace,
                doc_id,
                doc_version_hash,
                error=str(e),
            )
            return {
                **base_result,
                "status": "failed",
                **base_report,
                "reason": f"写入 Qdrant 失败: {e}",
            }

        if target_status == "replaced":
            for old_manifest in old_version_manifests:
                old_where = _rag_chunk_filter(
                    namespace=namespace,
                    doc_id=doc_id,
                    doc_version_hash=old_manifest.get("doc_version_hash"),
                )
                if not store.delete_by_filter(old_where):
                    store.delete_by_filter(same_version_where)
                    manifest_store.mark_rag_manifest_failed(
                        namespace,
                        doc_id,
                        doc_version_hash,
                        error="REPLACE 策略删除旧版本失败",
                        chunks_added=int(chunks_added or 0),
                    )
                    return {
                        **base_result,
                        "status": "failed",
                        "chunks_added": int(chunks_added or 0),
                        **base_report,
                        "chunks_deleted": 0,
                        "reason": "REPLACE 策略删除旧版本失败",
                    }
            manifest_store.deactivate_rag_manifests(
                namespace,
                doc_id,
                exclude_version=doc_version_hash,
                reason="replaced",
            )

        imported = manifest_store.mark_rag_manifest_imported(
            namespace,
            doc_id,
            doc_version_hash,
            chunks_added=int(chunks_added or 0),
            chunks_deleted=chunks_deleted,
            metadata={"mode": mode, "target_status": target_status},
        )
        if not imported:
            store.delete_by_filter(same_version_where)
            return {
                **base_result,
                "status": "failed",
                **base_report,
                "reason": "更新 RAG manifest 导入状态失败",
            }

        return {
            **base_result,
            "status": target_status,
            "chunks_added": int(chunks_added or 0),
            **base_report,
            "chunks_deleted": chunks_deleted,
        }

    def search(
        query: str,
        top_k: int = 8,
        score_threshold: Optional[float] = None,
        enable_mqe: bool = False,
        mqe_expansions: int = 2,
        enable_hyde: bool = False,
    ):
        """搜索 RAG 知识库，支持可选的查询扩展"""
        return search_vectors(
            store=store,
            query=query,
            top_k=top_k,
            namespace=namespace,
            score_threshold=score_threshold,
            enable_mqe=enable_mqe,
            mqe_expansions=mqe_expansions,
            enable_hyde=enable_hyde,
        )

    def get_stats():
        """获取 Pipeline 统计信息"""
        where = {
            "memory_type": "rag_chunk",
            "namespace": namespace,
        }
        stats = store.get_collection_stats(where=where)
        manifest_stats = manifest_store.get_rag_manifest_stats(namespace=namespace)
        stats["manifest"] = manifest_stats
        stats["active_documents"] = manifest_stats.get("active_documents", 0)
        stats["active_versions"] = manifest_stats.get("active_versions", 0)
        stats["expected_chunk_count"] = manifest_stats.get("expected_chunk_count", 0)
        stats["failed_manifests"] = manifest_stats.get("failed_manifests", 0)
        return stats

    def clear_namespace():
        """清空当前 namespace 下的 RAG 数据，不影响同 collection 的其他 namespace。"""
        where = {
            "memory_type": "rag_chunk",
            "namespace": namespace,
        }
        deleted = store.delete_by_filter(where=where)
        if deleted:
            manifest_store.clear_rag_manifests(namespace)
        return deleted

    def search_rerank(
        query: str,
        top_k: int = 8,
        enable_mqe: bool = False,
        enable_hyde: bool = False,
        score_threshold: Optional[float] = None,
        max_context_chars: int = 1200,
    ) -> Dict[str, Any]:
        """检索排序：粗排 → 精排 → 压缩 → 上下文拼装，始终包含引用"""
        return search_and_rerank(
            store=store,
            query=query,
            top_k=top_k,
            namespace=namespace,
            score_threshold=score_threshold,
            enable_mqe=enable_mqe,
            enable_hyde=enable_hyde,
            max_context_chars=max_context_chars,
        )

    return {
        "store": store,
        "document_store": manifest_store,
        "namespace": namespace,
        "add_documents": add_documents,
        "search": search,
        "search_rerank": search_rerank,
        "get_stats": get_stats,
        "clear_namespace": clear_namespace,
    }

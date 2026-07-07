"""文档存储实现

支持多种文档数据库后端：
- SQLite: 轻量级关系型数据库
- PostgreSQL: 企业级关系型数据库（可扩展）
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import sqlite3
import json
import os
import threading


class DocumentStore(ABC):
    """文档存储基类"""
    
    @abstractmethod
    def add_memory(
        self,
        memory_id: str,
        user_id: str,
        content: str,
        memory_type: str,
        timestamp: int,
        importance: float,
        properties: Dict[str, Any] = None
    ) -> str:
        """添加记忆"""
        pass
    
    @abstractmethod
    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """获取单个记忆"""
        pass
    
    @abstractmethod
    def search_memories(
        self,
        user_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        importance_threshold: Optional[float] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """搜索记忆"""
        pass
    
    @abstractmethod
    def update_memory(
        self,
        memory_id: str,
        content: str = None,
        importance: float = None,
        properties: Dict[str, Any] = None
    ) -> bool:
        """更新记忆"""
        pass
    
    @abstractmethod
    def delete_memory(self, memory_id: str) -> bool:
        """删除记忆"""
        pass
    
    @abstractmethod
    def get_database_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        pass
    
    @abstractmethod
    def add_document(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """添加文档"""
        pass
    
    @abstractmethod
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """获取文档"""
        pass

class SQLiteDocumentStore(DocumentStore):
    """SQLite文档存储实现"""
    
    _instances = {}  # 存储已创建的实例
    _initialized_dbs = set()  # 存储已初始化的数据库路径
    
    def __new__(cls, db_path: str = "./memory.db"):
        """单例模式，同一路径只创建一个实例"""
        abs_path = os.path.abspath(db_path)
        if abs_path not in cls._instances:
            instance = super(SQLiteDocumentStore, cls).__new__(cls)
            cls._instances[abs_path] = instance
        return cls._instances[abs_path]
    
    def __init__(self, db_path: str = "./memory.db"):
        # 避免重复初始化
        if hasattr(self, '_initialized'):
            return
            
        self.db_path = db_path
        self.local = threading.local()
        
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        
        # 初始化数据库（只初始化一次）
        abs_path = os.path.abspath(db_path)
        if abs_path not in self._initialized_dbs:
            self._init_database()
            self._initialized_dbs.add(abs_path)
            print(f"[OK] SQLite 文档存储初始化完成: {db_path}")
        
        self._initialized = True
    
    def _get_connection(self):
        """获取线程本地连接"""
        if not hasattr(self.local, 'connection'):
            self.local.connection = sqlite3.connect(self.db_path)
            self.local.connection.row_factory = sqlite3.Row  # 使结果可以按列名访问
        return self.local.connection
    
    def _init_database(self):
        """初始化数据库表"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # 创建用户表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT,
                properties TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 创建记忆表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                importance REAL NOT NULL,
                properties TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        
        # 创建概念表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS concepts (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                properties TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 创建记忆-概念关联表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_concepts (
                memory_id TEXT NOT NULL,
                concept_id TEXT NOT NULL,
                relevance_score REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (memory_id, concept_id),
                FOREIGN KEY (memory_id) REFERENCES memories (id) ON DELETE CASCADE,
                FOREIGN KEY (concept_id) REFERENCES concepts (id) ON DELETE CASCADE
            )
        """)
        
        # 创建概念关系表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS concept_relationships (
                from_concept_id TEXT NOT NULL,
                to_concept_id TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                strength REAL DEFAULT 1.0,
                properties TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (from_concept_id, to_concept_id, relationship_type),
                FOREIGN KEY (from_concept_id) REFERENCES concepts (id) ON DELETE CASCADE,
                FOREIGN KEY (to_concept_id) REFERENCES concepts (id) ON DELETE CASCADE
            )
        """)

        # 创建 RAG 文档级 manifest 表。chunk 向量仍存 Qdrant；这里仅保存文档版本状态。
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rag_document_manifests (
                namespace TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                doc_version_hash TEXT NOT NULL,
                source_path TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                chunk_count INTEGER NOT NULL,
                chunk_size INTEGER NOT NULL,
                chunk_overlap INTEGER NOT NULL,
                parser_version TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                embedding_dimension INTEGER NOT NULL,
                qdrant_collection TEXT NOT NULL,
                status TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1,
                chunks_added INTEGER DEFAULT 0,
                chunks_deleted INTEGER DEFAULT 0,
                last_error TEXT,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                imported_at INTEGER,
                metadata_json TEXT,
                PRIMARY KEY (namespace, doc_id, doc_version_hash)
            )
        """)
        
        # 创建索引
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories (user_id)",
            "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories (memory_type)",
            "CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories (timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories (importance)",
            "CREATE INDEX IF NOT EXISTS idx_memory_concepts_memory ON memory_concepts (memory_id)",
            "CREATE INDEX IF NOT EXISTS idx_memory_concepts_concept ON memory_concepts (concept_id)",
            "CREATE INDEX IF NOT EXISTS idx_rag_manifest_doc ON rag_document_manifests (namespace, doc_id)",
            "CREATE INDEX IF NOT EXISTS idx_rag_manifest_active ON rag_document_manifests (namespace, doc_id, is_active)",
            "CREATE INDEX IF NOT EXISTS idx_rag_manifest_status ON rag_document_manifests (namespace, status)"
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
        
        conn.commit()
        print("[OK] SQLite 数据库表和索引创建完成")
    
    def add_memory(
        self,
        memory_id: str,
        user_id: str,
        content: str,
        memory_type: str,
        timestamp: int,
        importance: float,
        properties: Dict[str, Any] = None
    ) -> str:
        """添加记忆"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # 确保用户存在
        cursor.execute("INSERT OR IGNORE INTO users (id, name) VALUES (?, ?)", (user_id, user_id))
        
        # 插入记忆
        cursor.execute("""
            INSERT OR REPLACE INTO memories 
            (id, user_id, content, memory_type, timestamp, importance, properties, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (
            memory_id,
            user_id,
            content,
            memory_type,
            timestamp,
            importance,
            json.dumps(properties) if properties else None
        ))
        
        conn.commit()
        return memory_id
    
    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """获取单个记忆"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, user_id, content, memory_type, timestamp, importance, properties, created_at
            FROM memories
            WHERE id = ?
        """, (memory_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return {
            "memory_id": row["id"],
            "user_id": row["user_id"],
            "content": row["content"],
            "memory_type": row["memory_type"],
            "timestamp": row["timestamp"],
            "importance": row["importance"],
            "properties": json.loads(row["properties"]) if row["properties"] else {},
            "created_at": row["created_at"]
        }
    
    def search_memories(
        self,
        user_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        importance_threshold: Optional[float] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """搜索记忆"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # 构建查询条件
        where_conditions = []
        params = []
        
        if user_id:
            where_conditions.append("user_id = ?")
            params.append(user_id)
        
        if memory_type:
            where_conditions.append("memory_type = ?")
            params.append(memory_type)
        
        if start_time:
            where_conditions.append("timestamp >= ?")
            params.append(start_time)
        
        if end_time:
            where_conditions.append("timestamp <= ?")
            params.append(end_time)
        
        if importance_threshold:
            where_conditions.append("importance >= ?")
            params.append(importance_threshold)
        
        where_clause = ""
        if where_conditions:
            where_clause = "WHERE " + " AND ".join(where_conditions)
        
        cursor.execute(f"""
            SELECT id, user_id, content, memory_type, timestamp, importance, properties, created_at
            FROM memories
            {where_clause}
            ORDER BY importance DESC, timestamp DESC
            LIMIT ?
        """, params + [limit])
        
        memories = []
        for row in cursor.fetchall():
            memories.append({
                "memory_id": row["id"],
                "user_id": row["user_id"],
                "content": row["content"],
                "memory_type": row["memory_type"],
                "timestamp": row["timestamp"],
                "importance": row["importance"],
                "properties": json.loads(row["properties"]) if row["properties"] else {},
                "created_at": row["created_at"]
            })
        
        return memories
    
    def update_memory(
        self,
        memory_id: str,
        content: str = None,
        importance: float = None,
        properties: Dict[str, Any] = None
    ) -> bool:
        """更新记忆"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # 构建更新字段
        update_fields = []
        params = []
        
        if content is not None:
            update_fields.append("content = ?")
            params.append(content)
        
        if importance is not None:
            update_fields.append("importance = ?")
            params.append(importance)
        
        if properties is not None:
            update_fields.append("properties = ?")
            params.append(json.dumps(properties))
        
        if not update_fields:
            return False
        
        update_fields.append("updated_at = CURRENT_TIMESTAMP")
        params.append(memory_id)
        
        cursor.execute(f"""
            UPDATE memories
            SET {', '.join(update_fields)}
            WHERE id = ?
        """, params)
        
        conn.commit()
        return cursor.rowcount > 0
    
    def delete_memory(self, memory_id: str) -> bool:
        """删除记忆"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        deleted_count = cursor.rowcount
        
        conn.commit()
        return deleted_count > 0
    
    def get_database_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        stats = {}
        
        # 统计各表的记录数
        tables = ["users", "memories", "concepts", "memory_concepts", "concept_relationships", "rag_document_manifests"]
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
            stats[f"{table}_count"] = cursor.fetchone()["count"]
        
        # 统计记忆类型分布
        cursor.execute("""
            SELECT memory_type, COUNT(*) as count
            FROM memories
            GROUP BY memory_type
        """)
        memory_types = {}
        for row in cursor.fetchall():
            memory_types[row["memory_type"]] = row["count"]
        stats["memory_types"] = memory_types
        
        # 统计用户分布
        cursor.execute("""
            SELECT user_id, COUNT(*) as count
            FROM memories
            GROUP BY user_id
            ORDER BY count DESC
            LIMIT 10
        """)
        top_users = {}
        for row in cursor.fetchall():
            top_users[row["user_id"]] = row["count"]
        stats["top_users"] = top_users
        
        stats["store_type"] = "sqlite"
        stats["db_path"] = self.db_path
        
        return stats

    def _row_to_rag_manifest(self, row: sqlite3.Row) -> Optional[Dict[str, Any]]:
        """将 SQLite 行转换为 RAG manifest 字典。"""
        if not row:
            return None
        return {
            "namespace": row["namespace"],
            "doc_id": row["doc_id"],
            "doc_version_hash": row["doc_version_hash"],
            "source_path": row["source_path"],
            "content_hash": row["content_hash"],
            "chunk_count": int(row["chunk_count"] or 0),
            "chunk_size": int(row["chunk_size"] or 0),
            "chunk_overlap": int(row["chunk_overlap"] or 0),
            "parser_version": row["parser_version"],
            "embedding_model": row["embedding_model"],
            "embedding_dimension": int(row["embedding_dimension"] or 0),
            "qdrant_collection": row["qdrant_collection"],
            "status": row["status"],
            "is_active": bool(row["is_active"]),
            "chunks_added": int(row["chunks_added"] or 0),
            "chunks_deleted": int(row["chunks_deleted"] or 0),
            "last_error": row["last_error"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "imported_at": row["imported_at"],
            "metadata": json.loads(row["metadata_json"]) if row["metadata_json"] else {},
        }

    def get_active_rag_manifests(self, namespace: str, doc_id: str) -> List[Dict[str, Any]]:
        """获取某个 namespace/doc_id 下所有 active 的 RAG manifest。"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT *
            FROM rag_document_manifests
            WHERE namespace = ? AND doc_id = ? AND is_active = 1
            ORDER BY imported_at DESC, updated_at DESC
        """, (namespace, doc_id))
        return [self._row_to_rag_manifest(row) for row in cursor.fetchall()]

    def get_rag_manifest(
        self,
        namespace: str,
        doc_id: str,
        doc_version_hash: str
    ) -> Optional[Dict[str, Any]]:
        """按 namespace/doc_id/doc_version_hash 获取单个 RAG manifest。"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT *
            FROM rag_document_manifests
            WHERE namespace = ? AND doc_id = ? AND doc_version_hash = ?
        """, (namespace, doc_id, doc_version_hash))
        return self._row_to_rag_manifest(cursor.fetchone())

    def create_rag_manifest_pending(
        self,
        *,
        namespace: str,
        doc_id: str,
        doc_version_hash: str,
        source_path: str,
        content_hash: str,
        chunk_count: int,
        chunk_size: int,
        chunk_overlap: int,
        parser_version: str,
        embedding_model: str,
        embedding_dimension: int,
        qdrant_collection: str,
        metadata: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """创建或覆盖一个 pending 状态的 RAG manifest。"""
        import time

        now = int(time.time())
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO rag_document_manifests (
                namespace, doc_id, doc_version_hash, source_path, content_hash,
                chunk_count, chunk_size, chunk_overlap, parser_version,
                embedding_model, embedding_dimension, qdrant_collection,
                status, is_active, chunks_added, chunks_deleted, last_error,
                created_at, updated_at, imported_at, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            namespace,
            doc_id,
            doc_version_hash,
            source_path,
            content_hash,
            int(chunk_count),
            int(chunk_size),
            int(chunk_overlap),
            parser_version,
            embedding_model,
            int(embedding_dimension),
            qdrant_collection,
            "pending",
            0,
            0,
            0,
            None,
            now,
            now,
            None,
            json.dumps(metadata or {}, ensure_ascii=False),
        ))
        conn.commit()
        return self.get_rag_manifest(namespace, doc_id, doc_version_hash)

    def mark_rag_manifest_imported(
        self,
        namespace: str,
        doc_id: str,
        doc_version_hash: str,
        *,
        chunks_added: int = 0,
        chunks_deleted: int = 0,
        metadata: Dict[str, Any] = None,
    ) -> bool:
        """将 RAG manifest 标记为已导入并设为 active。"""
        import time

        now = int(time.time())
        conn = self._get_connection()
        cursor = conn.cursor()
        fields = [
            "status = ?",
            "is_active = 1",
            "chunks_added = ?",
            "chunks_deleted = ?",
            "last_error = NULL",
            "updated_at = ?",
            "imported_at = ?",
        ]
        params: List[Any] = ["imported", int(chunks_added), int(chunks_deleted), now, now]
        if metadata is not None:
            fields.append("metadata_json = ?")
            params.append(json.dumps(metadata, ensure_ascii=False))
        params.extend([namespace, doc_id, doc_version_hash])
        cursor.execute(f"""
            UPDATE rag_document_manifests
            SET {', '.join(fields)}
            WHERE namespace = ? AND doc_id = ? AND doc_version_hash = ?
        """, params)
        conn.commit()
        return cursor.rowcount > 0

    def mark_rag_manifest_failed(
        self,
        namespace: str,
        doc_id: str,
        doc_version_hash: str,
        *,
        error: str,
        chunks_added: int = 0,
        chunks_deleted: int = 0,
    ) -> bool:
        """将 RAG manifest 标记为失败；失败版本不参与 active 幂等判断。"""
        import time

        now = int(time.time())
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE rag_document_manifests
            SET status = ?, is_active = 0, chunks_added = ?, chunks_deleted = ?,
                last_error = ?, updated_at = ?
            WHERE namespace = ? AND doc_id = ? AND doc_version_hash = ?
        """, (
            "failed",
            int(chunks_added),
            int(chunks_deleted),
            error,
            now,
            namespace,
            doc_id,
            doc_version_hash,
        ))
        conn.commit()
        return cursor.rowcount > 0

    def deactivate_rag_manifests(
        self,
        namespace: str,
        doc_id: str,
        *,
        exclude_version: str = None,
        reason: str = "replaced",
    ) -> int:
        """停用某个 doc_id 下的 active manifests，可排除当前新版本。"""
        import time

        now = int(time.time())
        conn = self._get_connection()
        cursor = conn.cursor()
        if exclude_version:
            cursor.execute("""
                UPDATE rag_document_manifests
                SET status = ?, is_active = 0, updated_at = ?
                WHERE namespace = ? AND doc_id = ? AND is_active = 1
                  AND doc_version_hash != ?
            """, (reason, now, namespace, doc_id, exclude_version))
        else:
            cursor.execute("""
                UPDATE rag_document_manifests
                SET status = ?, is_active = 0, updated_at = ?
                WHERE namespace = ? AND doc_id = ? AND is_active = 1
            """, (reason, now, namespace, doc_id))
        conn.commit()
        return int(cursor.rowcount or 0)

    def clear_rag_manifests(self, namespace: str) -> int:
        """停用一个 namespace 下所有 active RAG manifests。"""
        import time

        now = int(time.time())
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE rag_document_manifests
            SET status = ?, is_active = 0, updated_at = ?
            WHERE namespace = ? AND is_active = 1
        """, ("cleared", now, namespace))
        conn.commit()
        return int(cursor.rowcount or 0)

    def get_rag_manifest_stats(self, namespace: str = None) -> Dict[str, Any]:
        """获取 RAG manifest 统计信息。"""
        conn = self._get_connection()
        cursor = conn.cursor()
        params: List[Any] = []
        where_clause = ""
        if namespace:
            where_clause = "WHERE namespace = ?"
            params.append(namespace)

        cursor.execute(f"""
            SELECT
                COUNT(*) AS total_manifests,
                SUM(CASE WHEN is_active = 1 THEN 1 ELSE 0 END) AS active_versions,
                COUNT(DISTINCT CASE WHEN is_active = 1 THEN doc_id ELSE NULL END) AS active_documents,
                SUM(CASE WHEN is_active = 1 THEN chunk_count ELSE 0 END) AS expected_chunk_count,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed_manifests
            FROM rag_document_manifests
            {where_clause}
        """, params)
        row = cursor.fetchone()
        stats = {
            "namespace": namespace,
            "total_manifests": int(row["total_manifests"] or 0),
            "active_versions": int(row["active_versions"] or 0),
            "active_documents": int(row["active_documents"] or 0),
            "expected_chunk_count": int(row["expected_chunk_count"] or 0),
            "failed_manifests": int(row["failed_manifests"] or 0),
            "db_path": self.db_path,
        }

        cursor.execute(f"""
            SELECT status, COUNT(*) AS count
            FROM rag_document_manifests
            {where_clause}
            GROUP BY status
        """, params)
        stats["status_counts"] = {
            row["status"]: int(row["count"] or 0)
            for row in cursor.fetchall()
        }
        return stats
    
    def add_document(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """添加文档"""
        import uuid
        import time
        
        doc_id = str(uuid.uuid4())
        user_id = metadata.get("user_id", "system") if metadata else "system"
        
        return self.add_memory(
            memory_id=doc_id,
            user_id=user_id,
            content=content,
            memory_type="document",
            timestamp=int(time.time()),
            importance=0.5,
            properties=metadata or {}
        )
    
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """获取文档"""
        return self.get_memory(document_id)

    def close(self):
        """关闭数据库连接"""
        if hasattr(self.local, 'connection'):
            self.local.connection.close()
            delattr(self.local, 'connection')
            print("[OK] SQLite 连接已关闭")

import os
import httpx
from typing import Any, List, Optional
from supabase import create_client
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from src.config import (
    SUPABASE_URL, SUPABASE_KEY, SUPABASE_TABLE, SUPABASE_QUERY_NAME,
    SUPABASE_HYBRID_QUERY_NAME, SUPABASE_PROXY,
)


class HybridSupabaseRetriever(BaseRetriever):
    """混合检索器：同时传递 query_embedding 和 query_text 到 Supabase RPC。

    LangChain 内置的 SupabaseVectorStore.as_retriever() 只传嵌入向量，
    本类绕过它直接调用 hybrid_match_papers SQL 函数，实现向量 + 文本混合检索。
    """

    client: Any
    """Supabase 客户端（httpx-backed）."""

    embedding: Any
    """Embedding 模型，需有 embed_query(text) -> List[float] 方法."""

    query_name: str = SUPABASE_HYBRID_QUERY_NAME
    """Supabase RPC 函数名."""

    k: int = 4
    """返回文档数量."""

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        # 1. 生成查询向量
        query_embedding: List[float] = self.embedding.embed_query(query)

        # 2. 调用 hybrid_match_papers（同时传入向量和文本）
        response = (
            self.client
            .rpc(self.query_name, {
                "query_embedding": query_embedding,
                "query_text": query,
                "match_count": self.k,
            })
            .execute()
        )

        # 3. 构造 Document 列表
        docs: List[Document] = []
        for row in response.data:
            doc = Document(
                page_content=row["out_content"],
                metadata={
                    **(row.get("out_meta") or {}),
                    "id": row["out_id"],
                },
            )
            docs.append(doc)
        return docs


def get_supabase_client():
    """创建 Supabase 客户端（支持代理）"""
    from supabase.lib.client_options import SyncClientOptions

    if SUPABASE_PROXY:
        transport = httpx.HTTPTransport(proxy=SUPABASE_PROXY)
        httpx_client = httpx.Client(transport=transport)
        options = SyncClientOptions(httpx_client=httpx_client)
        return create_client(SUPABASE_URL, SUPABASE_KEY, options=options)
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def create_supabase_vector_store(embeddings):
    """从现有 Supabase 表加载向量库（不插入新数据）"""
    client = get_supabase_client()
    return SupabaseVectorStore(
        client=client,
        embedding=embeddings,
        table_name=SUPABASE_TABLE,
        query_name=SUPABASE_QUERY_NAME,
    )


def add_documents_to_supabase(docs, embeddings):
    """将文档写入 Supabase 向量库"""
    from src.splitter import split_documents
    texts = split_documents(docs)
    print(f"Split into {len(texts)} chunks, uploading to Supabase...")

    client = get_supabase_client()
    store = SupabaseVectorStore(
        client=client,
        embedding=embeddings,
        table_name=SUPABASE_TABLE,
        query_name=SUPABASE_QUERY_NAME,
    )
    store.add_documents(texts)
    print(f"Uploaded {len(texts)} chunks to Supabase.")
    return store


def get_supabase_retriever(embeddings):
    """获取 Supabase 混合检索器（只读，用于问答）

    使用自定义 HybridSupabaseRetriever，将查询文本和向量同时发送到
    hybrid_match_papers SQL 函数，实现向量 + 关键词混合检索（权重 0.5:0.5）。
    """
    client = get_supabase_client()
    return HybridSupabaseRetriever(
        client=client,
        embedding=embeddings,
        query_name=SUPABASE_HYBRID_QUERY_NAME,
        k=4,
    )

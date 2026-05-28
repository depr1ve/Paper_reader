import os
import httpx
from supabase import create_client
from langchain_community.vectorstores import SupabaseVectorStore

from config import (
    SUPABASE_URL, SUPABASE_KEY, SUPABASE_TABLE, SUPABASE_QUERY_NAME,
    SUPABASE_PROXY,
)


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
    from splitter import split_documents
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
    """获取 Supabase 检索器（只读，用于问答）"""
    store = create_supabase_vector_store(embeddings)
    return store.as_retriever(search_kwargs={"k": 4})

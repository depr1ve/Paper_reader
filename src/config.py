import os
from dotenv import load_dotenv

load_dotenv()


def _get(key, default=""):
    """1.Streamlit secrets (secrets.toml / Cloud) 2.环境变量(.env/sidebar) 3.默认值"""
    try:
        import streamlit as st
        if st.secrets and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    val = os.getenv(key)
    if val:
        return val
    return default


# 输入：论文/文档存放目录
DATA_DIR = _get("DATA_DIR", "./data")

# 输出：向量数据库持久化目录（本地 Chroma 用）
VECTOR_STORE_DIR = _get("VECTOR_STORE_DIR", "./store")

# Chroma 集合名称
CHROMA_COLLECTION = _get("CHROMA_COLLECTION", "papers")

# DeepSeek API
DEEPSEEK_API_KEY = _get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = _get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = _get("DEEPSEEK_MODEL", "deepseek-chat")

# HuggingFace 镜像（国内用户设为 https://hf-mirror.com 加速下载）
HF_ENDPOINT = _get("HF_ENDPOINT", "")

# Embedding 模型（本地下载）
EMBEDDING_MODEL = _get("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

# Supabase 云向量库
SUPABASE_URL = _get("SUPABASE_URL", "")
SUPABASE_KEY = _get("SUPABASE_KEY", "")
SUPABASE_TABLE = _get("SUPABASE_TABLE", "papers")
SUPABASE_QUERY_NAME = _get("SUPABASE_QUERY_NAME", "match_papers")
SUPABASE_HYBRID_QUERY_NAME = _get("SUPABASE_HYBRID_QUERY_NAME", "hybrid_match_papers")
SUPABASE_PROXY = _get("SUPABASE_PROXY", "")

# 向量库后端: "chroma" (本地) 或 "supabase" (云端)
VECTOR_BACKEND = _get("VECTOR_BACKEND", "supabase")

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

# Embedding 模型（优先本地 models/ 目录，否则在线下载）
EMBEDDING_MODEL = _get("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

# ---- 本地模型目录 & 解析 ----
_MODELS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
)


def _resolve_local_model(model_id):
    """在 models/hub/ 下按 HuggingFace 缓存命名查找本地模型。

    格式: models--org--model-name（如 models--BAAI--bge-small-zh-v1.5）
    支持两种目录结构：
      - 扁平：config.json 直接在模型目录根
      - 标准缓存：config.json 在 snapshots/<hash>/ 子目录下
    找到返回实际包含 config.json 的目录路径，否则返回原始 model_id（走在线下载）。
    """
    if not os.path.isdir(_MODELS_DIR):
        return model_id
    cache_name = "models--" + model_id.replace("/", "--")
    for base in (os.path.join(_MODELS_DIR, "hub"), _MODELS_DIR):
        candidate = os.path.join(base, cache_name)
        if not os.path.isdir(candidate):
            continue
        # 情况 1: config.json 直接在根目录（如手动放置的模型）
        if os.path.isfile(os.path.join(candidate, "config.json")):
            return candidate
        # 情况 2: HuggingFace 标准缓存 snapshots/<hash>/
        snapshots_dir = os.path.join(candidate, "snapshots")
        if os.path.isdir(snapshots_dir):
            for name in os.listdir(snapshots_dir):
                snap_path = os.path.join(snapshots_dir, name)
                if os.path.isdir(snap_path) and os.path.isfile(
                    os.path.join(snap_path, "config.json")
                ):
                    return snap_path
    return model_id


# 将 Embedding 模型解析为本地路径（若存在）
EMBEDDING_MODEL = _resolve_local_model(EMBEDDING_MODEL)

# Supabase 云向量库
SUPABASE_URL = _get("SUPABASE_URL", "")
SUPABASE_KEY = _get("SUPABASE_KEY", "")
SUPABASE_TABLE = _get("SUPABASE_TABLE", "papers")
SUPABASE_QUERY_NAME = _get("SUPABASE_QUERY_NAME", "match_papers")
SUPABASE_HYBRID_QUERY_NAME = _get("SUPABASE_HYBRID_QUERY_NAME", "hybrid_match_papers")
SUPABASE_PROXY = _get("SUPABASE_PROXY", "")

# 向量库后端: "chroma" (本地) 或 "supabase" (云端)
VECTOR_BACKEND = _get("VECTOR_BACKEND", "supabase")

# ---- Reranker 重排序 ----
def _find_model_dir(base_dir):
    """递归扫描目录，返回第一个包含 config.json 的子目录路径（HuggingFace 模型标记）。

    若找不到，返回空字符串。
    """
    if not os.path.isdir(base_dir):
        return ""
    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]  # 跳过隐藏目录
        if "config.json" in files:
            return root
    return ""


_RERANKER_MODEL_ID = "BAAI/bge-reranker-base"
_RERANKER_PATH = _get("RERANKER_MODEL_PATH", "")
if not _RERANKER_PATH:
    _RERANKER_PATH = _resolve_local_model(_RERANKER_MODEL_ID)
    if _RERANKER_PATH == _RERANKER_MODEL_ID:          # 未找到标准命名，扫描兜底
        _RERANKER_PATH = _find_model_dir(_MODELS_DIR)
# 解析相对路径为绝对路径
if _RERANKER_PATH and not os.path.isabs(_RERANKER_PATH):
    _RERANKER_PATH = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", _RERANKER_PATH)
    )
RERANKER_MODEL_PATH = _RERANKER_PATH


def get_model(repo_id="DeepSeek", **kwargs):
    """创建 LLM 实例。DeepSeek 兼容 OpenAI 协议，使用 ChatOpenAI 调用。"""
    from langchain_openai import ChatOpenAI

    if repo_id == "DeepSeek":
        kwargs.setdefault("base_url", DEEPSEEK_BASE_URL)
        kwargs.setdefault("api_key", DEEPSEEK_API_KEY)
        kwargs.setdefault("model", DEEPSEEK_MODEL)
    return ChatOpenAI(temperature=0, **kwargs)

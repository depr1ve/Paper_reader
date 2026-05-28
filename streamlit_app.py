import os
import sys

import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from src.config import (
    EMBEDDING_MODEL, VECTOR_BACKEND, HF_ENDPOINT,
    DEEPSEEK_API_KEY as ENV_DEEPSEEK_KEY,
)

# ---- HuggingFace 镜像（国内加速，解决下载卡住问题） ----
if HF_ENDPOINT:
    os.environ["HF_ENDPOINT"] = HF_ENDPOINT

st.set_page_config(page_title="RAG Paper Reader", page_icon="📄")
st.title("📄 RAG Paper Reader - 论文问答")
st.caption("上传论文 PDF，基于 RAG 进行智能问答 | 支持 LaTeX 公式渲染")


# ---- 侧边栏 ----
with st.sidebar:
    st.subheader("🔑 API 配置")
    deepseek_key = st.text_input(
        "DeepSeek API Key",
        value=ENV_DEEPSEEK_KEY if ENV_DEEPSEEK_KEY else "",
        type="password",
        placeholder="sk-...",
    )
    if deepseek_key:
        os.environ["DEEPSEEK_API_KEY"] = deepseek_key

    st.divider()
    st.subheader("🗄️ 向量库后端")
    backend = st.radio(
        "选择后端",
        options=["chroma", "supabase"],
        format_func=lambda x: "💻 本地 Chroma" if x == "chroma" else "☁️ Supabase 云端",
        index=0 if VECTOR_BACKEND == "chroma" else 1,
        help="本地：向量存在磁盘。云端：向量存在 Supabase，永久保存，多地共享。",
    )

    if backend == "supabase":
        has_url = bool(os.getenv("SUPABASE_URL"))
        has_key = bool(os.getenv("SUPABASE_KEY"))
        if has_url and has_key:
            st.success("✅ Supabase 已配置（从 Secrets）")
        else:
            supabase_url = st.text_input(
                "Supabase URL",
                value=os.getenv("SUPABASE_URL", ""),
                placeholder="https://xxx.supabase.co",
            )
            supabase_key_input = st.text_input(
                "Supabase Key",
                value=os.getenv("SUPABASE_KEY", ""),
                type="password",
                placeholder="sb_secret_...",
                help="需 service_role key 才能写入向量",
            )
            if supabase_url:
                os.environ["SUPABASE_URL"] = supabase_url
            if supabase_key_input:
                os.environ["SUPABASE_KEY"] = supabase_key_input

    st.divider()
    st.subheader("📤 上传论文")
    uploaded_file = st.file_uploader(
        "支持 PDF / TXT 格式",
        type=["pdf", "txt"],
        accept_multiple_files=False,
    )

    if st.button("🗑️ 清空聊天记录"):
        st.session_state.messages = []
        st.rerun()


# ---- 加载 Embedding 模型 ----
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


# ---- 检索器 ----
@st.cache_resource
def get_local_retriever():
    from src.local_loader import load_documents
    from src.ensemble import ensemble_retriever_from_docs
    docs = load_documents()
    embeddings = get_embeddings()
    return ensemble_retriever_from_docs(docs, embeddings=embeddings)


@st.cache_resource
def get_supabase_retriever(_url, _key):
    from src.supabase_store import get_supabase_retriever as sup_ret
    embeddings = get_embeddings()
    return sup_ret(embeddings)


# ---- 处理上传 ----
if uploaded_file is not None:
    with st.spinner(f"正在处理 {uploaded_file.name} ..."):
        try:
            from src.local_loader import get_document_text
            docs = get_document_text(uploaded_file)
            if not docs:
                st.error("无法提取文档内容，请检查文件是否损坏。")
            else:
                embeddings = get_embeddings()

                if backend == "supabase":
                    from src.supabase_store import add_documents_to_supabase
                    n_chunks = add_documents_to_supabase(docs, embeddings)
                    st.success(f"✅ 已上传至 Supabase 云端: {uploaded_file.name}")
                    get_supabase_retriever.clear()
                else:
                    from src.splitter import split_documents
                    from src.vector_store import create_vector_db
                    from src.config import CHROMA_COLLECTION
                    texts = split_documents(docs)
                    create_vector_db(texts, embeddings, collection_name=CHROMA_COLLECTION)
                    st.success(f"✅ 已添加至本地向量库: {uploaded_file.name}（{len(texts)} chunks）")
                    get_local_retriever.clear()
        except Exception as e:
            st.error(f"上传失败: {e}")


# ---- 构建对话链（延迟加载，首次提问时才初始化模型） ----
@st.cache_resource
def build_chain(backend_choice, _supabase_url, _supabase_key):
    if backend_choice == "supabase":
        if not _supabase_url or not _supabase_key:
            return None
        retriever = get_supabase_retriever(_supabase_url, _supabase_key)
    else:
        retriever = get_local_retriever()

    from src.full_chain import create_full_chain
    return create_full_chain(
        retriever,
        chat_memory=StreamlitChatMessageHistory(key="langchain_messages"),
    )


# ---- 初始化 session state ----
if "chain_ready" not in st.session_state:
    st.session_state.chain_ready = False
    st.session_state.chain = None
    st.session_state.current_backend = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# 切换后端时重建 chain
if st.session_state.current_backend != backend:
    st.session_state.chain_ready = False
    st.session_state.chain = None
    st.session_state.current_backend = backend


# ---- 聊天界面 ----
def ensure_chain():
    """确保 chain 已初始化，未初始化时显示加载提示。返回 (chain, ok)"""
    if st.session_state.chain_ready:
        return st.session_state.chain, True

    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_KEY", "")

    if backend == "supabase" and (not url or not key):
        return None, False

    with st.spinner("⏳ 正在加载 Embedding 模型（首次运行需下载约 100MB，请耐心等待）..."):
        try:
            chain = build_chain(backend, url, key)
            st.session_state.chain = chain
            st.session_state.chain_ready = True
            return chain, True
        except Exception as e:
            st.error(f"模型加载失败: {e}")
            return None, False


# ---- 主流程 ----
if not deepseek_key:
    st.info("👈 请在侧边栏输入 DeepSeek API Key")
    st.stop()

# 预检查 supabase 配置
if backend == "supabase":
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_KEY", "")
    if not url or not key:
        st.error("👈 请在侧边栏填写 Supabase URL 和 Key")
        st.stop()

# 显示聊天记录
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown("请输入你的问题...")

# 接收用户输入
if prompt := st.chat_input("在此输入问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    chain, ok = ensure_chain()
    if not ok:
        st.error("请先在侧边栏完成配置")
    elif chain is None:
        st.error("对话链初始化失败")
    else:
        with st.chat_message("assistant"):
            with st.spinner("🤔 思考中..."):
                try:
                    from src.full_chain import ask_question
                    response = ask_question(chain, prompt)
                    st.markdown(response.content)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response.content}
                    )
                except Exception as e:
                    st.error(f"出错了: {e}")

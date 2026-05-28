# Paper Reader - RAG 论文问答系统

## 项目概述

基于 LangChain + Streamlit 的 RAG 论文阅读问答系统。上传 PDF/TXT 论文后，通过自然语言提问，系统自动检索相关段落并生成回答。

## 技术栈

| 层级 | 技术 |
|------|------|
| LLM | **DeepSeek** (`deepseek-chat`)，通过 OpenAI 兼容 API 调用 |
| Embedding | **BAAI/bge-small-zh-v1.5**（本地运行，维度 512，中文优化） |
| 向量库（本地） | Chroma，持久化到 `store/` |
| 向量库（云端） | **Supabase pgvector**（表 `papers`，函数 `match_papers`） |
| 检索策略 | BM25（关键词）+ 向量（语义）混合检索，权重 0.5:0.5 |
| 分块 | `RecursiveCharacterTextSplitter`，chunk_size=1000，overlap=200 |
| 对话记忆 | `RunnableWithMessageHistory`，自动将追问改写为独立问题 |
| UI | Streamlit，Markdown + LaTeX 公式渲染 |

## 目录结构

```
streamlit_app.py          # Streamlit 主页面
run_app.py                # PyCharm 本地入口
src/
├── config.py             # 配置（优先 env，其次 Streamlit secrets）
├── basic_chain.py        # LLM 模型初始化（DeepSeek）
├── local_loader.py       # 加载 data/ 下的 PDF/TXT
├── splitter.py           # 文档切分
├── vector_store.py       # Chroma 本地向量库
├── supabase_store.py     # Supabase 云端向量库（支持代理）
├── ensemble.py           # BM25 + 向量混合检索
├── rag_chain.py          # RAG 核心链
├── memory.py             # 多轮对话记忆
├── full_chain.py         # 完整问答链（含 system prompt）
└── filter.py             # 检索结果去重+重排序
```

## 配置方式

### 本地开发
- `.env` 文件存放 API Key
- 国内用户设置 `HF_ENDPOINT=https://hf-mirror.com` 加速 HuggingFace 模型下载
- 向量库默认 `chroma`，可在侧边栏切换 `supabase`
- Embedding 模型首次提问时延迟加载，不会阻塞页面启动

### Streamlit Cloud 部署
- 必须用 Cloud Secrets 配置（`secrets.toml` 不会推送到 GitHub）
- 必须用 `VECTOR_BACKEND=supabase`（Cloud 磁盘不持久化）
- Cloud 使用 Python 3.14，`requirements.txt` 不锁版本号以保持兼容
- 当 API Key / Supabase 已从 Secrets 配置时，侧边栏自动隐藏敏感信息

## 配置项

| 变量 | 说明 | 必填 |
|------|------|------|
| `DEEPSEEK_API_KEY` | DeepSeek API Key | 是 |
| `DEEPSEEK_BASE_URL` | DeepSeek API 地址 | 否（默认 api.deepseek.com） |
| `DEEPSEEK_MODEL` | DeepSeek 模型名 | 否（默认 deepseek-chat） |
| `EMBEDDING_MODEL` | HuggingFace Embedding 模型 | 否（默认 BAAI/bge-small-zh-v1.5） |
| `HF_ENDPOINT` | HuggingFace 镜像（国内加速） | 否（国内推荐 hf-mirror.com） |
| `VECTOR_BACKEND` | 向量后端：`chroma` 或 `supabase` | 否（Cloud 部署必须 supabase） |
| `SUPABASE_URL` | Supabase 项目 URL | 仅 supabase 模式 |
| `SUPABASE_KEY` | Supabase service_role key | 仅 supabase 模式 |
| `SUPABASE_PROXY` | Supabase 代理地址 | 否（国内访问需要） |
| `CHROMA_COLLECTION` | Chroma 集合名 | 否（默认 papers） |

## Supabase 表结构

```sql
create extension if not exists vector;

create table if not exists papers (
    id uuid primary key default gen_random_uuid(),
    content text,
    metadata jsonb,
    embedding vector(512)
);

create or replace function match_papers(
    query_embedding vector(512),
    match_count int default 4
)
returns table (
    id uuid,
    content text,
    metadata jsonb,
    similarity float
)
language plpgsql
as $$
begin
    return query
    select
        p.id,
        p.content,
        p.metadata,
        1 - (p.embedding <=> query_embedding) as similarity
    from papers p
    order by p.embedding <=> query_embedding
    limit match_count;
end;
$$;
```

## 关键依赖

```
streamlit                 # UI
langchain, langchain-community, langchain-core, langchain-openai, langchain-classic
chromadb                  # 本地向量库
supabase, httpx           # 云端向量库
sentence-transformers     # BAAI/bge-small-zh-v1.5 Embedding
pypdf, rank-bm25, rich, python-dotenv
```

依赖不锁版本号，以兼容 Streamlit Cloud 的 Python 3.14。

## 注意事项

- `.streamlit/` 和 `.env` 在 `.gitignore` 中，不会提交到 GitHub
- Embedding 模型首次运行从 HuggingFace 下载（约 100MB），国内需配置 `HF_ENDPOINT` 镜像
- Streamlit Cloud 部署前确保已建好 Supabase 表和 `match_papers` 函数
- 本地 Chroma 模式：论文 PDF 放在 `data/` 目录，启动后上传或自动加载
- Supabase 从国内访问可能需要代理，配置 `SUPABASE_PROXY`
- `__pycache__/` 目录已 gitignore，可随时删除

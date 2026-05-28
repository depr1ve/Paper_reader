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
- 向量库默认 `chroma`，可在侧边栏切换 `supabase`

### Streamlit Cloud 部署
- 用 `secrets.toml` 或 Cloud Secrets 配置
- 必须用 `VECTOR_BACKEND=supabase`（Cloud 磁盘不持久化）

## Supabase 表结构

```sql
papers (
    id uuid PRIMARY KEY,
    content text,
    metadata jsonb,
    embedding vector(512)
)
-- 检索函数: match_papers(query_embedding, match_count)
```

## 关键依赖

```
langchain, langchain-openai, langchain-community
chromadb (仅本地模式)
supabase, pgvector (仅云端模式)
sentence-transformers (BAAI/bge-small-zh-v1.5)
pypdf, streamlit, rank-bm25
```

## API Key

- DeepSeek: `DEEPSEEK_API_KEY`（平台: platform.deepseek.com）
- Supabase: `SUPABASE_URL` + `SUPABASE_KEY`（service_role key）

## 注意事项

- `.streamlit/` 和 `.env` 在 `.gitignore` 中，不会提交到 GitHub
- Supabase 从国内访问可能需要代理，配置 `SUPABASE_PROXY`
- Embedding 模型首次运行从 HuggingFace 下载（约 100MB），后续用缓存
- 论文 PDF 放在 `data/` 目录，启用 `chroma` 后端时自动加载

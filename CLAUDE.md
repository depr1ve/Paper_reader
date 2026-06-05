# Paper Reader - RAG 论文问答系统

## 项目概述

基于 LangChain + Streamlit 的 RAG 论文阅读问答系统。上传 PDF/TXT 论文后，通过自然语言提问，系统自动检索相关段落并生成回答。

## 技术栈

| 层级 | 技术 |
|------|------|
| LLM | **DeepSeek** (`deepseek-chat`)，通过 OpenAI 兼容 API 调用 |
| Embedding | **BAAI/bge-small-zh-v1.5**（本地运行，维度 512，中文优化） |
| Reranker | **BAAI/bge-reranker-base**（本地 CrossEncoder，召回 20 → 精排 5） |
| 向量库（本地） | Chroma，持久化到 `store/` |
| 向量库（云端） | **Supabase pgvector**（表 `papers`，函数 `match_papers`） |
| 检索策略 | 关键词+向量混合检索，**RRF 融合** `0.5/(rank+60)`（Chroma: BM25, Supabase: PGroonga BM25-like）+ **BGE CrossEncoder 重排序**（召回 20 → 精排 5） |
| 分块 | `RecursiveCharacterTextSplitter`，chunk_size=1000，overlap=200 |
| 元数据 | L1：论文标题、章节标题、章节类型、chunk 序号 |
| 对话记忆 | `RunnableWithMessageHistory`，自动将追问改写为独立问题 |
| UI | Streamlit，Markdown + LaTeX 公式渲染 |

## 目录结构

```
streamlit_app.py          # Streamlit 主页面
run_app.py                # PyCharm 本地入口
src/
├── config.py             # 配置 + LLM 工厂（get_model）
├── local_loader.py       # 加载 data/ 下的 PDF/TXT
├── splitter.py           # 文档切分 + 章节检测 + metadata 注入
├── vector_store.py       # Chroma 本地向量库
├── supabase_store.py     # Supabase 云端向量库（支持代理）
├── ensemble.py           # BM25 + 向量混合检索（本地 Chroma 用）
├── reranker.py           # BGE CrossEncoder 重排序（召回 20 → 精排 5）
└── full_chain.py         # 完整问答链：问题改写 + RAG + 多轮对话记忆
models/                   # 本地模型缓存（gitignore，自动发现）
SQL/
└── supabase.sql          # Supabase 建表 + 函数 SQL
```

## Metadata

切分阶段自动为每个 chunk 注入 4 个元数据字段：

| 字段 | 来源 | 说明 |
|------|------|------|
| `paper_title` | local_loader | 论文标题（优先 PDF 元数据，回退文件名） |
| `section_title` | splitter | 章节标题（正则匹配，扫描前 5 行） |
| `section_type` | splitter | 章节分类（abstract/introduction/methods/experiments/discussion/conclusion/appendix/body） |
| `chunk_index` | splitter | chunk 在全文中的序号 |

`format_docs()` 以章节标题标注上下文，同一章节的多个 chunk 合并避免重复标签：
```
## 《论文名》· 3 实验设计

chunk1 正文...

chunk2 正文...

## 《论文名》· 5 实验结果与分析

chunk3 正文...
```

## 配置方式

### 本地开发
- `.env` 文件存放 API Key
- 国内用户设置 `HF_ENDPOINT=https://hf-mirror.com` 加速 HuggingFace 模型下载
- 向量库默认 `chroma`，可在侧边栏切换 `supabase`
- Embedding + Reranker 模型启动时预加载（从 `models/` 自动发现，无网络）

### Streamlit Cloud 部署
- 必须用 Cloud Secrets 配置（`secrets.toml` 不会推送到 GitHub）
- 必须用 `VECTOR_BACKEND=supabase`（Cloud 磁盘不持久化）
- Cloud 使用 Python 3.14，`requirements.txt` 不锁版本号以保持兼容
- Supabase 需启用三个扩展：`vector`、`pgroonga`（全库 SQL 见 `SQL/supabase.sql`）

## 配置项

| 变量 | 说明 | 必填 |
|------|------|------|
| `DEEPSEEK_API_KEY` | DeepSeek API Key | 是 |
| `DEEPSEEK_BASE_URL` | DeepSeek API 地址 | 否（默认 api.deepseek.com） |
| `DEEPSEEK_MODEL` | DeepSeek 模型名 | 否（默认 deepseek-chat） |
| `EMBEDDING_MODEL` | HuggingFace Embedding 模型 | 否（默认 BAAI/bge-small-zh-v1.5，自动从 models/ 加载） |
| `HF_ENDPOINT` | HuggingFace 镜像（国内加速） | 否（国内推荐 hf-mirror.com） |
| `VECTOR_BACKEND` | 向量后端：`chroma` 或 `supabase` | 否（Cloud 部署必须 supabase） |
| `SUPABASE_URL` | Supabase 项目 URL | 仅 supabase 模式 |
| `SUPABASE_KEY` | Supabase service_role key | 仅 supabase 模式 |
| `SUPABASE_PROXY` | Supabase 代理地址 | 否（国内访问需要） |
| `CHROMA_COLLECTION` | Chroma 集合名 | 否（默认 papers） |
| `RERANKER_MODEL_PATH` | BGE Reranker 模型本地路径 | 否（自动从 models/ 加载） |

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
- `models/` 存放本地模型（Embedding + Reranker），启动时自动发现、预加载，无需联网下载
  - 目录结构：`models/hub/models--{org}--{model}/`（支持扁平或 snapshots 格式）
  - 手动放置模型到此目录即可，无需修改配置
- Streamlit Cloud 部署前确保已建好 Supabase 表和函数（完整 SQL 见 `SQL/supabase.sql`，在 Supabase SQL Editor 中粘贴执行即可）
  - 混合检索使用 **PGroonga + pgvector** 双路 RRF 融合，需启用 `pgroonga` 扩展
- 本地 Chroma 模式：论文 PDF 放在 `data/` 目录，启动后上传或自动加载



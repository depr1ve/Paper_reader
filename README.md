# 📄 RAG Paper Reader — 论文智能问答系统

基于 Retrieval-Augmented Generation (RAG) 的论文阅读助手。上传 PDF/TXT 论文后，通过自然语言提问，系统自动检索相关段落并结合 LLM 生成高质量回答。

**Live Demo**: [Streamlit Cloud](https://paperreader.streamlit.app/)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://paperreader.streamlit.app/)

---

## 架构概览

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  📤 论文上传  │ ──▶ │  📦 文档分块   │ ──▶ │  🧬 Embedding   │
│  (PDF/TXT)   │     │ chunk=1000    │     │  bge-small-zh   │
└─────────────┘     │ overlap=200    │     └───────┬─────────┘
                    └──────────────┘             │
                                          ┌──────▼─────────┐
                                          │  🗄️ 向量数据库   │
                                          │  Supabase/Chroma│
                                          └──────┬─────────┘
                                                 │
┌─────────────┐     ┌──────────────┐     ┌──────▼─────────┐
│  💬 用户提问  │ ──▶ │  🔄 问题改写   │ ──▶ │  🔍 混合检索   │
│              │     │  (对话记忆)    │     │  BM25 + 向量   │
└─────────────┘     └──────────────┘     └──────┬─────────┘
                                                 │
                                          ┌──────▼─────────┐
                                          │  🧠 LLM 生成    │
                                          │  DeepSeek-Chat │
                                          └──────┬─────────┘
                                                 │
                                          ┌──────▼─────────┐
                                          │  📝 回答输出    │
                                          │  Markdown+LaTeX│
                                          └────────────────┘
```

---

## 核心技术

### 1. Embedding 模型

| | 详情 |
|---|---|
| **模型** | `BAAI/bge-small-zh-v1.5` |
| **维度** | 512 |
| **特点** | 中文优化，本地运行无 API 成本，轻量（~100MB） |
| **运行方式** | `sentence-transformers` 加载，首次自动从 HuggingFace 下载并缓存 |
| **国内加速** | 设置 `HF_ENDPOINT=https://hf-mirror.com` 走镜像 |

选用原因：bge-small-zh 在中文语义理解 benchmark 上表现优异，512 维在精度和检索速度之间取得良好平衡。本地运行避免了 Embedding API 的调用延迟和费用。

### 2. 向量库双后端

| 后端 | 技术 | 适用场景 |
|---|---|---|
| **Supabase pgvector** | PostgreSQL + pgvector 扩展 + HNSW 索引 | 生产部署、多端共享、永久存储 |
| **Chroma** | 本地向量库，持久化到磁盘 | 本地开发、离线使用、零配置 |

两个后端通过侧边栏一键切换，检索接口完全统一。Supabase 模式下，论文向量存储在云端 PostgreSQL 中，通过 `match_papers` 存储函数执行余弦相似度检索，支持 HNSW 索引加速 Top-K 查询。

```sql
-- 基础向量检索函数（写入路径使用）
create or replace function match_papers(
    query_embedding vector(512),
    match_count int default 4
) returns table (
    id uuid, content text, metadata jsonb, similarity float
) language plpgsql as $$
begin
    return query
    select p.id, p.content, p.metadata,
           1 - (p.embedding <=> query_embedding) as similarity
    from papers p
    order by p.embedding <=> query_embedding
    limit match_count;
end;
$$;

-- 混合检索函数（读取路径使用，需先启用 pg_trgm 扩展）
create extension if not exists pg_trgm;

create or replace function hybrid_match_papers(
    query_embedding vector(512),
    query_text text,
    match_count int default 4
) returns table (
    id uuid, content text, metadata jsonb, similarity float
) language plpgsql as $$
begin
    return query
    select p.id, p.content, p.metadata,
           (0.5 * (1 - (p.embedding <=> query_embedding))
            + 0.5 * coalesce(similarity(p.content, query_text), 0)) as similarity
    from papers p
    where p.embedding is not null and p.content is not null
    order by similarity desc
    limit match_count;
end;
$$;
```

### 3. 混合检索策略

系统采用 **关键词（稀疏）+ 向量（稠密）** 混合检索，权重 0.5:0.5：

- **本地 Chroma**：基于 `rank-bm25` 实现 BM25 + 向量，`EnsembleRetriever` 融合
- **Supabase**：基于 `pg_trgm` 扩展的 `similarity()` 实现关键词 + `pgvector` 向量余弦相似度，`hybrid_match_papers` SQL 函数内融合
- **融合方式**：两组分数按 0.5:0.5 加权求和，互补各自盲区

### 4. 检索结果优化

- **去重**：`EmbeddingsRedundantFilter` 基于向量相似度剔除重复段落
- **重排序**：`LongContextReorder` 解决 "Lost in the Middle" 问题——将最相关结果置于上下文窗口的首尾位置，提升 LLM 引用准确率

### 5. 多轮对话记忆

- `RunnableWithMessageHistory` 自动管理对话历史
- 追问时自动将上下文压缩为独立问题（contextualization），避免检索歧义
- 基于 `StreamlitChatMessageHistory` 实现会话内持久化

### 6. LLM 模型

| | 详情 |
|---|---|
| **模型** | DeepSeek-Chat (`deepseek-chat`) |
| **调用方式** | OpenAI 兼容 API，通过 `langchain-openai` |
| **回答格式** | Markdown + LaTeX 公式渲染（`$...$` 行内，`$$...$$` 独立） |

### 7. 文档处理

- **解析**：`pypdf` 提取 PDF 文本，保留页码元数据
- **分块**：`RecursiveCharacterTextSplitter` 递归切分，chunk_size=1000，overlap=200，保证语义段落完整
- **延迟加载**：Embedding 模型在首次提问时才初始化，页面秒开，不阻塞启动

---

## 目录结构

```
streamlit_app.py          # Streamlit 主页面（双后端、延迟加载）
run_app.py                # PyCharm 本地入口
src/
├── config.py             # 多源配置（env > Streamlit Secrets > 默认值）
├── basic_chain.py        # DeepSeek 模型初始化
├── local_loader.py       # PDF/TXT 加载、解析
├── splitter.py           # 递归文档切分
├── vector_store.py       # Chroma 本地向量库（含 EmbeddingProxy 限速）
├── supabase_store.py     # Supabase 云端向量库（支持 HTTP 代理）
├── ensemble.py           # BM25 + 向量混合检索
├── rag_chain.py          # RAG 核心管道（检索→格式化→生成）
├── memory.py             # 多轮对话记忆链
├── full_chain.py         # 完整问答链（System Prompt + 检索 + 记忆）
└── filter.py             # 检索去重 + 重排序
```

---

## 快速开始

### 1. 环境配置

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

创建 `.env` 文件：

```env
DEEPSEEK_API_KEY=sk-xxxxxxxx
VECTOR_BACKEND=chroma        # 本地 Chroma，或 supabase 走云端
HF_ENDPOINT=https://hf-mirror.com  # 国内用户加速 HuggingFace
```

### 3. 启动

```bash
streamlit run streamlit_app.py
# 或 PyCharm 中直接运行 run_app.py
```

### 4. 使用 Supabase（可选）

1. 在 [Supabase](https://supabase.com) 创建项目
2. 在 SQL Editor 中执行建表语句（见上方 SQL）
3. `.env` 中补充：
```env
VECTOR_BACKEND=supabase
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=sb_secret_xxx
```

---

## Streamlit Cloud 部署

1. Fork 本仓库
2. 在 Streamlit Cloud 设置 **Secrets**（Settings → Secrets），填入所有必填配置项
3. Cloud 自动部署，模型首次加载需下载约 100MB

| 必填 Secrets | 说明 |
|---|---|
| `DEEPSEEK_API_KEY` | DeepSeek API Key |
| `VECTOR_BACKEND` | 设为 `supabase` |
| `SUPABASE_URL` | Supabase 项目 URL |
| `SUPABASE_KEY` | Supabase service_role Key |

---

## 技术选型与权衡

| 决策点 | 选择 | 原因 |
|---|---|---|
| LLM | DeepSeek vs OpenAI | 中文能力强，API 成本更低 |
| Embedding | 本地 bge-small-zh vs API | 零调用成本，离线可用，512 维轻量 |
| 检索 | 混合检索 vs 纯向量 | 关键词 + 语义互补，召回率更高 |
| 分块 | 1000/200 vs 更大窗口 | 平衡检索精度与上下文完整性 |
| 向量库双后端 | Chroma + Supabase | 本地开发零配置，生产部署高可用 |
| 重排序 | LongContextReorder | 解决 LLM 长上下文中段信息忽略问题 |
| 模型加载 | 延迟加载 vs 启动加载 | 页面秒开，首次加载有明确进度提示 |

---

## References

- Lewis, P., et al. (2020). [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401). NeurIPS.
- Robertson, S., & Zaragoza, H. (2009). [The Probabilistic Relevance Framework: BM25 and Beyond](https://dl.acm.org/doi/10.1561/1500000019). Foundations and Trends in IR.
- Liu, N. F., et al. (2024). [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172). TACL.
- BGE Embedding: [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5)
- DeepSeek API: [platform.deepseek.com](https://platform.deepseek.com)

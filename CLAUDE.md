# RAG Paper Agent 结构说明

## 项目概述

基于 LangChain + Streamlit 的 RAG（检索增强生成）论文阅读问答系统。上传论文/文档后，通过自然语言提问获取答案。

## 核心文件

| 文件 | 职责 |
|------|------|
| `streamlit_app.py` | Web UI 入口，聊天界面 |
| `full_chain.py` | 组装完整 RAG 链（检索 + 记忆 + 生成） |
| `rag_chain.py` | RAG 核心链：检索 → 格式化 → LLM 生成 |
| `ensemble.py` | 混合检索器：BM25 + 向量检索融合 |
| `vector_store.py` | Chroma 向量数据库 + OpenAI Embeddings |
| `splitter.py` | 文档分割（RecursiveCharacterTextSplitter） |
| `local_loader.py` | 加载本地 txt/pdf/csv 文件 |
| `remote_loader.py` | 加载网页、Wikipedia、在线 PDF |
| `memory.py` | 多轮对话记忆（问题改写 + 历史管理） |
| `filter.py` | 检索结果去重 + 重排序 |
| `basic_chain.py` | LLM 模型初始化（OpenAI / HuggingFace） |

## 数据流

```
文档加载 → 文本分割 → 向量存储 + BM25索引 → 混合检索 → LLM生成答案
                              ↑
                    用户提问 ──┘
```

## 关键设计

- **混合检索**：BM25（关键词）+ 向量检索（语义），权重 0.5:0.5，互补长短
- **多轮对话**：自动将上下文依赖问题改写为独立问题后再检索
- **重排序**：解决 "Lost in the Middle"，高相关文档放首尾
- **多模型**：支持 ChatGPT、Zephyr-7B、Mistral-7B

## 使用方式

1. 将论文/文档（txt/pdf）放入 `data/` 目录
2. 配置 OpenAI API Key
3. `streamlit run streamlit_app.py` 启动 Web 界面

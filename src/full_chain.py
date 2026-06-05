from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.config import get_model
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages.base import BaseMessage


def format_docs(docs):
    """将检索到的文档拼接为 LLM 上下文，以章节标签标注来源。"""
    parts = []
    prev_section = None
    for doc in docs:
        meta = doc.metadata
        section = meta.get('section_title') or '正文'
        paper = meta.get('paper_title', '')

        if section == prev_section:
            parts.append(doc.page_content)
        else:
            label = f"《{paper}》· {section}" if paper else section
            parts.append(f"## {label}\n\n{doc.page_content}")
            prev_section = section

    return "\n\n".join(parts)


def get_question(input):
    if not input:
        return None
    elif isinstance(input, str):
        return input
    elif isinstance(input, dict) and 'question' in input:
        return input['question']
    elif isinstance(input, BaseMessage):
        return input.content
    else:
        raise Exception("string or dict with 'question' key expected as RAG chain input.")


def make_rag_chain(model, retriever, rag_prompt=None):
    if not rag_prompt:
        raise ValueError("rag_prompt is required.")

    return (
        {
            "context": RunnableLambda(get_question) | retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | rag_prompt
        | model
    )


def create_full_chain(retriever, chat_memory=ChatMessageHistory()):
    model = get_model("DeepSeek")

    # 多轮对话：将追问改写为独立问题
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", "根据聊天记录，将用户的最新问题改写为独立问题，使其不依赖上下文也能理解。不要回答问题，只改写。"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    # RAG 问答
    rag_system_prompt = """你是一个严谨的论文阅读助手。根据以下上下文和聊天记录来回答问题。

## 规则
1. 严格基于上下文回答。上下文以「## 章节名」标注来源。
2. 回答时引用章节名称，如"根据第3.2节实验设计..."。如果某个信息跨越多个章节，也要说明。
3. 如果上下文中不包含某信息，直接说"论文中未提及"，不要猜测或使用你的预训练知识补全。
4. 数学公式使用 Markdown LaTeX 格式：行内 $...$，独立 $$...$$。不要使用 \\[...\\]。

---
上下文:
{context}
---
问题: """

    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", rag_system_prompt),
        ("human", "{question}"),
    ])

    rag_chain = make_rag_chain(model, retriever, rag_prompt=rag_prompt)
    full_chain = contextualize_prompt | model | rag_chain

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        return chat_memory

    return RunnableWithMessageHistory(
        full_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )


def ask_question(chain, query):
    return chain.invoke(
        {"question": query},
        config={"configurable": {"session_id": "foo"}}
    )

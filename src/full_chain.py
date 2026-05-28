import os

from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate

from src.basic_chain import get_model
from src.filter import ensemble_retriever_from_docs
from src.local_loader import load_documents
from src.memory import create_memory_chain
from src.rag_chain import make_rag_chain


def create_full_chain(retriever, chat_memory=ChatMessageHistory()):
    model = get_model("DeepSeek")
    system_prompt = """你是一个帮助阅读论文的AI助手。
根据以下上下文和用户的聊天记录来帮助用户回答问题。
如果你不知道答案，就说不知道。

如果回答中包含数学公式，请使用 Markdown LaTeX 格式：
行内公式使用 $...$，
独立公式使用 $$...$$。
不要使用 \[...\] 或裸露的 LaTeX 代码。

---
上下文: {context}
---
问题: """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )

    rag_chain = make_rag_chain(model, retriever, rag_prompt=prompt)
    chain = create_memory_chain(model, rag_chain, chat_memory)
    return chain


def ask_question(chain, query):
    response = chain.invoke(
        {"question": query},
        config={"configurable": {"session_id": "foo"}}
    )
    return response


def main():
    load_dotenv()

    from rich.console import Console
    from rich.markdown import Markdown
    console = Console()

    docs = load_documents()
    ensemble_retriever = ensemble_retriever_from_docs(docs)
    chain = create_full_chain(ensemble_retriever)

    queries = [
        "请帮我总结这篇论文的主要内容。"
    ]

    for query in queries:
        response = ask_question(chain, query)
        console.print(Markdown(response.content))


if __name__ == '__main__':
    # this is to quiet parallel tokenizers warning.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()

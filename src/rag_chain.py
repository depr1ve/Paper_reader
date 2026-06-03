import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages.base import BaseMessage

from src.config import get_model
from src.splitter import split_documents
from src.vector_store import create_vector_db


def find_similar(vs, query):
    docs = vs.similarity_search(query)
    return docs


def format_docs(docs):
    """将检索到的文档拼接为 LLM 上下文，以章节标签标注来源。"""
    parts = []
    prev_section = None
    for doc in docs:
        meta = doc.metadata
        section = meta.get('section_title') or '正文'
        paper = meta.get('paper_title', '')

        # 同一章节的多个 chunk 合并，避免重复标注
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
    elif isinstance(input,str):
        return input
    elif isinstance(input,dict) and 'question' in input:
        return input['question']
    elif isinstance(input,BaseMessage):
        return input.content
    else:
        raise Exception("string or dict with 'question' key expected as RAG chain input.")


def make_rag_chain(model, retriever, rag_prompt = None):
    if not rag_prompt:
        raise ValueError("rag_prompt is required. Please provide a ChatPromptTemplate.")

    # And we will use the LangChain RunnablePassthrough to add some custom processing into our chain.
    rag_chain = (
            {
                "context": RunnableLambda(get_question) | retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | rag_prompt
            | model
    )

    return rag_chain


def main():
    load_dotenv()
    from src.local_loader import load_documents

    model = get_model("DeepSeek")
    docs = load_documents()
    texts = split_documents(docs)
    vs = create_vector_db(texts)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professor who teaches philosophical concepts to beginners."),
        ("user", "{input}")
    ])
    # Besides similarly search, you can also use maximal marginal relevance (MMR) for selecting results.
    # retriever = vs.as_retriever(search_type="mmr")
    retriever = vs.as_retriever()

    output_parser = StrOutputParser()
    base_chain = prompt | model | output_parser
    rag_chain = make_rag_chain(model, retriever) | output_parser

    questions = [
        "What were the most important contributions of Bertrand Russell to philosophy?",
        "What was the first book Bertrand Russell published?",
        "What was most notable about \"An Essay on the Foundations of Geometry\"?",
    ]
    for q in questions:
        print("\n--- QUESTION: ", q)
        print("* BASE:\n", base_chain.invoke({"input": q}))
        print("* RAG:\n", rag_chain.invoke(q))


if __name__ == '__main__':
    # this is to quite parallel tokenizers warning.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()

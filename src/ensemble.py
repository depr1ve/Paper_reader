from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from src.splitter import split_documents
from src.vector_store import create_vector_db


def ensemble_retriever_from_docs(docs, embeddings=None, k=20):
    """构建混合检索器：BM25 + 向量，各召回 k 个候选。

    k 应设为较大的值（如 20），后续由 reranker 精排压缩。
    """
    from src.config import CHROMA_COLLECTION
    texts = split_documents(docs)
    vs = create_vector_db(texts, embeddings, collection_name=CHROMA_COLLECTION)
    vs_retriever = vs.as_retriever(search_kwargs={"k": k})

    bm25_retriever = BM25Retriever.from_texts([t.page_content for t in texts])
    bm25_retriever.k = k

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vs_retriever],
        weights=[0.5, 0.5])

    return ensemble_retriever




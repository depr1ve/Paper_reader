from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from src.splitter import split_documents
from src.vector_store import create_vector_db


def ensemble_retriever_from_docs(docs, embeddings=None):
    from src.config import CHROMA_COLLECTION
    texts = split_documents(docs)
    vs = create_vector_db(texts, embeddings, collection_name=CHROMA_COLLECTION)
    vs_retriever = vs.as_retriever()

    bm25_retriever = BM25Retriever.from_texts([t.page_content for t in texts])

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vs_retriever],
        weights=[0.5, 0.5])

    return ensemble_retriever




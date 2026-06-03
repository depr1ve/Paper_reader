import logging
import os
from typing import List

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from src.local_loader import load_documents
from src.splitter import split_documents
from dotenv import load_dotenv
from time import sleep

from src.config import EMBEDDING_MODEL

EMBED_DELAY = 0.02  # 20 milliseconds


# This is to get the Streamlit app to use less CPU while embedding documents into Chromadb.
class EmbeddingProxy:
    def __init__(self, embedding):
        self.embedding = embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        sleep(EMBED_DELAY)
        return self.embedding.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        sleep(EMBED_DELAY)
        return self.embedding.embed_query(text)


# This happens all at once, not ideal for large datasets.
def create_vector_db(texts, embeddings=None, collection_name="chroma"):
    if not texts:
        logging.warning("Empty texts passed in to create vector database")
    # Select embeddings
    if not embeddings:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    proxy_embeddings = EmbeddingProxy(embeddings)
    # Create a vectorstore from documents
    # this will be a chroma collection with a default name.
    from src.config import VECTOR_STORE_DIR
    db = Chroma(collection_name=collection_name,
                embedding_function=proxy_embeddings,
                persist_directory=os.path.join(VECTOR_STORE_DIR, collection_name))
    db.add_documents(texts)

    return db


def find_similar(vs, query):
    docs = vs.similarity_search(query)
    return docs


def main():
    load_dotenv()

    from src.local_loader import load_documents

    docs = load_documents()
    texts = split_documents(docs)
    vs = create_vector_db(texts)

    results = find_similar(vs, query="这篇论文的主要内容是什么？")
    MAX_CHARS = 300
    print("=== Results ===")
    for i, text in enumerate(results):
        content = text.page_content
        n = max(content.find(' ', MAX_CHARS), MAX_CHARS)
        content = text.page_content[:n]
        print(f"Result {i + 1}:\n {content}\n")


if __name__ == "__main__":
    main()

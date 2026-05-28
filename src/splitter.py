# Split documents into chunks
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False)

    # split_documents 直接传 Document 列表以保留 metadata
    texts = text_splitter.split_documents(docs)
    n_chunks = len(texts)
    print(f"Split into {n_chunks} chunks")
    return texts

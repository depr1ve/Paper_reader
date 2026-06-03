import os
from pathlib import Path

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader

from src.config import DATA_DIR


def list_files(data_dir=None, pattern="**/*.txt"):
    if data_dir is None:
        data_dir = DATA_DIR
    paths = Path(data_dir).glob(pattern)
    for path in paths:
        yield str(path)


def load_txt_files(data_dir=None):
    if data_dir is None:
        data_dir = DATA_DIR
    docs = []
    for path in list_files(data_dir, "**/*.txt"):
        print(f"Loading TXT: {path}")
        loader = TextLoader(path)
        docs.extend(loader.load())
    return docs


def load_pdf_files(data_dir=None):
    if data_dir is None:
        data_dir = DATA_DIR
    docs = []
    for path in list_files(data_dir, "**/*.pdf"):
        print(f"Loading PDF: {path}")
        reader = PdfReader(path)
        filename = os.path.basename(path)

        # 从 PDF metadata 读取标题；如果不可用则用文件名
        paper_title = filename
        if reader.metadata:
            pdf_title = reader.metadata.get('/Title', '')
            if pdf_title and pdf_title.strip() and len(pdf_title.strip()) > 3:
                # 过滤掉无意义的短标题（如模板填充的"分类号"）
                garbage_titles = {'分类号', '学号', '密级', '学校代码', 'UDC', '编号', 'Untitled'}
                if pdf_title.strip() not in garbage_titles:
                    paper_title = pdf_title.strip()

        for num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                doc = Document(
                    page_content=text,
                    metadata={
                        'source': filename,
                        'paper_title': paper_title,
                    }
                )
                docs.append(doc)
        print(f"  -> {len(docs)} pages extracted")
    return docs


def load_documents(data_dir=None):
    """加载 data/ 下所有 txt 和 pdf 文件"""
    docs = load_txt_files(data_dir)
    docs.extend(load_pdf_files(data_dir))
    print(f"Total documents loaded: {len(docs)}")
    return docs


def load_csv_files(data_dir=None):
    if data_dir is None:
        data_dir = DATA_DIR
    docs = []
    for path in list_files(data_dir, "**/*.csv"):
        loader = CSVLoader(file_path=str(path))
        docs.extend(loader.load())
    return docs


# Use with result of file_to_summarize = st.file_uploader("Choose a file") or a string.
# or a file like object.
def get_document_text(uploaded_file, title=None):
    docs = []
    fname = uploaded_file.name
    if not title:
        title = os.path.basename(fname)
    if fname.lower().endswith('pdf'):
        pdf_reader = PdfReader(uploaded_file)
        for num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                doc = Document(
                    page_content=page_text,
                    metadata={
                        'title': title,
                        'source': fname,
                        'paper_title': title,
                    }
                )
                docs.append(doc)

    else:
        # assume text
        doc_text = uploaded_file.read().decode()
        docs.append(Document(
            page_content=doc_text,
            metadata={'title': title, 'source': fname, 'paper_title': title}
        ))

    return docs


if __name__ == "__main__":
    docs = load_documents()
    for doc in docs[:3]:
        m = doc.metadata
        print(f"paper={m.get('paper_title','?')[:50]}, len={len(doc.page_content)}")

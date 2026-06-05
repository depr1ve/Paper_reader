import os
from pathlib import Path

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader

from src.config import DATA_DIR


def list_files(data_dir=None, pattern="**/*.txt"):
    if data_dir is None:
        data_dir = DATA_DIR
    for path in Path(data_dir).glob(pattern):
        yield str(path)


def load_txt_files(data_dir=None):
    if data_dir is None:
        data_dir = DATA_DIR
    docs = []
    for path in list_files(data_dir, "**/*.txt"):
        print(f"加载 TXT: {path}")
        loader = TextLoader(path)
        docs.extend(loader.load())
    return docs


def load_pdf_files(data_dir=None):
    if data_dir is None:
        data_dir = DATA_DIR
    docs = []
    for path in list_files(data_dir, "**/*.pdf"):
        print(f"加载 PDF: {path}")
        reader = PdfReader(path)
        filename = os.path.basename(path)

        # 优先使用 PDF 元数据标题，回退为文件名
        paper_title = filename
        if reader.metadata:
            pdf_title = reader.metadata.get('/Title', '')
            if pdf_title and pdf_title.strip() and len(pdf_title.strip()) > 3:
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
        print(f"  -> 提取 {len(docs)} 页")
    return docs


def load_documents(data_dir=None):
    """加载 data/ 下所有 txt 和 pdf 文件"""
    docs = load_txt_files(data_dir)
    docs.extend(load_pdf_files(data_dir))
    print(f"共加载 {len(docs)} 篇文档")
    return docs


def get_document_text(uploaded_file, title=None):
    """从 Streamlit 上传文件中提取文本，返回 Document 列表"""
    docs = []
    fname = uploaded_file.name
    if not title:
        title = os.path.basename(fname)

    if fname.lower().endswith('pdf'):
        pdf_reader = PdfReader(uploaded_file)
        for num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                docs.append(Document(
                    page_content=page_text,
                    metadata={'title': title, 'source': fname, 'paper_title': title},
                ))
    else:
        doc_text = uploaded_file.read().decode()
        docs.append(Document(
            page_content=doc_text,
            metadata={'title': title, 'source': fname, 'paper_title': title},
        ))
    return docs

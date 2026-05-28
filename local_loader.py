import os
from pathlib import Path

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader

from config import DATA_DIR


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
        title = os.path.basename(path)
        for num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                doc = Document(page_content=text, metadata={'source': title, 'page': num + 1})
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
            page = page.extract_text()
            doc = Document(page_content=page, metadata={'title': title, 'page': (num + 1)})
            docs.append(doc)

    else:
        # assume text
        doc_text = uploaded_file.read().decode()
        docs.append(doc_text)

    return docs


if __name__ == "__main__":
    example_pdf_path = "examples/healthy_meal_10_tips.pdf"
    docs = get_document_text(open(example_pdf_path, "rb"))
    for doc in docs:
        print(doc)
    docs = get_document_text(open("examples/us_army_recipes.txt", "rb"))
    for doc in docs:
        print(doc)
    txt_docs = load_txt_files("examples")
    for doc in txt_docs:
        print(doc)
    csv_docs = load_csv_files("examples")
    for doc in csv_docs:
        print(doc)

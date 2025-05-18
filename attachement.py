import tempfile
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader, UnstructuredFileLoader, Docx2txtLoader
)
from langchain_community.embeddings import FastEmbedEmbeddings
import os

def add_attachment(uploaded_file, llm):
    if uploaded_file.size > 5 * 1024 * 1024:
        raise ValueError("File too large. Please upload a file under 5MB.")

    file_extension = uploaded_file.name.split('.')[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    if file_extension == "pdf":
        loader = PyPDFLoader(tmp_path)
    elif file_extension in ["txt", "md"]:
        loader = TextLoader(tmp_path)
    elif file_extension == "csv":
        loader = CSVLoader(file_path=tmp_path)
    elif file_extension in ["docx", "doc"]:
        loader = Docx2txtLoader(tmp_path)
    else:
        loader = UnstructuredFileLoader(tmp_path)

    pages = loader.load_and_split()

    embeddings = FastEmbedEmbeddings()

    vectorstore = FAISS.from_documents(pages, embedding=embeddings)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )

    return qa_chain

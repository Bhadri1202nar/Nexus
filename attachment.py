import tempfile
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader, UnstructuredFileLoader, Docx2txtLoader
)
import  easyocr
from PIL import Image
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.docstore.document import Document
import  numpy as np
import io
import os
ocr_model=easyocr.Reader(['en'])
def extract_text_from_image(image_path:str) -> str :
    results = ocr_model.readtext(image_path, detail=0)  # detail=0 returns just the text
    return "\n".join(results)





def add_attachment(uploaded_files, llm):
    if not isinstance(uploaded_files,list):
        uploaded_files=[uploaded_files]
    all_pages=[]
    for uploaded_file in uploaded_files:
        if uploaded_file.size > 5 * 1024 * 1024:
            raise ValueError("File too large. Please upload a file under 5MB.")

    file_extension = uploaded_file.name.split('.')[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    if file_extension == "pdf":
        loader = PyPDFLoader(tmp_path)
        pages=loader.load_and_split()
    elif file_extension in ["txt", "md"]:
        loader = TextLoader(tmp_path)
        pages = loader.load_and_split()
    elif file_extension == "csv":
        loader = CSVLoader(file_path=tmp_path)
        pages = loader.load_and_split()
    elif file_extension in ["jpg", "jpeg" ,"png"]:
        extracted_text = extract_text_from_image(tmp_path)
        pages = [Document(page_content=extracted_text)]
    elif file_extension in ["docx", "doc"]:
        loader=Docx2txtLoader(tmp_path)
        pages=loader.load_and_split()

    else:
        loader = UnstructuredFileLoader(tmp_path)
        pages = loader.load_and_split()
    all_pages.extend(pages)

    #pages = loader.load_and_split()

    embeddings = FastEmbedEmbeddings()

    vectorstore = FAISS.from_documents(pages, embedding=embeddings)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )

    return qa_chain

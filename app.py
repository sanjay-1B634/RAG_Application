import os
os.environ['HF_HOME'] = "C:/RAG_Custom_HF_Cache"
os.makedirs("C:/RAG_Custom_HF_Cache", exist_ok=True)
from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import shutil
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
import uvicorn

os.makedirs("static", exist_ok=True)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def form_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask", response_class=HTMLResponse)
async def process_pdf(
    request: Request,
    file: UploadFile = Form(...),
    question: str = Form(...)
):
    
    file_location = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    loader = PyPDFLoader(file_location)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks = splitter.split_documents(documents)


    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = FAISS.from_documents(chunks, embedding_model)

    text_gen_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        tokenizer="google/flan-t5-large",
        max_length=256,
        do_sample=False,
        temperature=0.3,
        device=0
    )
    llm = HuggingFacePipeline(pipeline=text_gen_pipeline)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    from langchain.prompts import PromptTemplate
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Answer the following question using only the provided context. "
            "If the answer is not in the context, say 'I don't know.'\n\n"
            "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
    )
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    answer = qa_chain.run(question)

    return templates.TemplateResponse("answer.html", {"request": request, "answer": answer})
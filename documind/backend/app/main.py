# ./backend/app/main.py

import os
import boto3
import uuid
import logging
import tempfile
import pdfplumber
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException, Path
from pydantic import BaseModel, Path, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import update
from sqlalchemy.future import select
from typing import List
from . import models
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from pinecone import Pinecone, ServerlessSpec
from transformers import pipeline

# --- Configuration & Setup ---
#load_dotenv(dotenv_path="../.env")
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database Connection
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_async_engine(DATABASE_URL)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

print("DATABASE_URL = ", DATABASE_URL)

# S3 Client
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://minio:9000")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET = os.getenv("S3_BUCKET")

s3_client = boto3.client(
    's3',
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

# Pinecone Client
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD")
PINECONE_REGION = os.getenv("PINECONE_REGION")

pc = Pinecone(api_key=PINECONE_API_KEY)
PINECONE_INDEX_NAME = "documind-index" 

# Embedding Model & LLM
embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
hf_pipeline = pipeline(
    "text-generation", 
    model="microsoft/Phi-3-mini-4k-instruct", 
    trust_remote_code=True, # Required for Phi-3
    torch_dtype="auto"
)
llm = HuggingFacePipeline(pipeline=hf_pipeline)


app = FastAPI(title="DocuMind AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # or ["*"] for quick dev testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # or ["*"] for quick dev testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class QARequest(BaseModel):
    question: str

class QAResponse(BaseModel):
    answer: str
    source_chunks: list[str]

# --- Background Task (The Processing Pipeline) ---

async def process_document(doc_id: int, s3_key: str):
    logger.info(f"Starting processing for doc_id: {doc_id} with s3_key: {s3_key}")
    
    with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_file:
        try:
            # 1. Download & Extract
            s3_client.download_fileobj(S3_BUCKET, s3_key, temp_file)
            temp_file.seek(0)
            full_text = ""
            with pdfplumber.open(temp_file.name) as pdf:
                for i, page in enumerate(pdf.pages):
                    full_text += page.extract_text() + "\n"
                    logger.info(f"Extracted text from page {i+1}")
            
            logger.info(f"Total extracted text (first 500 chars): {full_text[:500]}")

            # Future steps will go here (chunking, embedding, etc.)

        # 3. Update the document status in Postgres to "READY"
        await set_document_status(doc_id, models.DocumentStatus.READY)
        logger.info(f"Updated status to READY for doc_id: {doc_id}")

        except Exception as e:
            logger.error(f"Failed to process document {doc_id}: {e}")
            async with AsyncSessionLocal() as session:
                async with session.begin():
                    await session.execute(update(models.Document).where(models.Document.id == doc_id).values(status=models.DocumentStatus.FAILED))

# --- API Endpoints ---

@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(models.Base.metadata.create_all)
    
    try:
        s3_client.head_bucket(Bucket=S3_BUCKET)
    except s3_client.exceptions.ClientError:
        logger.info(f"Bucket '{S3_BUCKET}' not found. Creating it.")
        s3_client.create_bucket(Bucket=S3_BUCKET)

@app.get("/")
def read_root():
    return {"message": "Welcome to DocuMind AI"}

@app.post("/documents", status_code=202)
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file name provided.")

    file_extension = os.path.splitext(file.filename)[1]
    s3_key = f"uploads/{uuid.uuid4()}{file_extension}"
    
    try:
        s3_client.upload_fileobj(file.file, S3_BUCKET, s3_key)
        logger.info(f"File '{file.filename}' uploaded to S3 as '{s3_key}'")
    except Exception as e:
        logger.error(f"Failed to upload to S3: {e}")
        raise HTTPException(status_code=500, detail="Could not upload file.")

    doc_id = None
    async with AsyncSessionLocal() as session:
        async with session.begin():
            doc = models.Document(title=file.filename, s3_key=s3_key, status=models.DocumentStatus.PROCESSING)
            session.add(doc)
        await session.refresh(doc)
        doc_id = doc.id

    if doc_id is None:
        raise HTTPException(status_code=500, detail="Could not create document record.")
    
    background_tasks.add_task(process_document, doc_id, s3_key)
    logger.info(f"Queued processing for doc_id: {doc_id}")
    
    return {"doc_id": doc_id, "s3_key": s3_key, "status": "processing"}

@app.post("/documents/{doc_id}/qa", response_model=QAResponse)
async def question_answering(
    doc_id: int = Path(..., title="The ID of the document to query."),
    request: QARequest = ...
):
    # Check if document is ready
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(models.Document).where(models.Document.id == doc_id))
        doc = result.scalar_one_or_none()
        if not doc or doc.status != models.DocumentStatus.READY:
            raise HTTPException(status_code=404, detail="Document not found or not ready for querying.")

    # Set up LangChain RAG
    vectorstore = PineconeVectorStore.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding=embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2, "filter": {"doc_id": doc_id}})
    
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    # Get answer
    result = await qa_chain.ainvoke({"query": request.question})
    
    return {
        "answer": result["result"],
        "source_chunks": [doc.page_content for doc in result["source_documents"]]
    }


@app.post("/documents/{doc_id}/qa", response_model=QAResponse)
async def question_answering(
    doc_id: int = Path(..., title="The ID of the document to query."),
    request: QARequest = ...
):
    # Check if document is ready
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(models.Document).where(models.Document.id == doc_id))
        doc = result.scalar_one_or_none()
        if not doc or doc.status != models.DocumentStatus.READY:
            raise HTTPException(status_code=404, detail="Document not found or not ready for querying.")

    # Set up LangChain RAG
    vectorstore = PineconeVectorStore.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding=embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2, "filter": {"doc_id": doc_id}})
    
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    # Get answer
    result = await qa_chain.ainvoke({"query": request.question})
    
    return {
        "answer": result["result"],
        "source_chunks": [doc.page_content for doc in result["source_documents"]]
    }


@app.get("/documents")
async def list_documents():
    """Return all documents in the DB."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(models.Document))
        docs = result.scalars().all()

        # Convert SQLAlchemy models to dict (fastapi can auto JSONify pydantic or dicts)
        return [
            {
                "id": doc.id,
                "filename": doc.title,
                "status": (doc.status.value if hasattr(doc.status, "value") else str(doc.status)).upper(),
                "s3_key": doc.s3_key,
            }
            for doc in docs
        ]
    
@app.get("/documents/{doc_id}")
async def get_document(doc_id: int):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(models.Document).where(models.Document.id == doc_id)
        )
        doc = result.scalar_one_or_none()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        return {
            "id": doc.id,
            "filename": doc.title,
            "status": (doc.status.value if hasattr(doc.status, "value") else str(doc.status)).upper(),
            "s3_key": doc.s3_key,
        }
    
@app.get("/files/{doc_id}")
async def get_file(doc_id: int):
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(models.Document).where(models.Document.id == doc_id))
        doc = result.scalar_one_or_none()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        # Download file stream from S3/MinIO
        s3_object = s3_client.get_object(Bucket=S3_BUCKET, Key=doc.s3_key)
        return StreamingResponse(
            s3_object["Body"], media_type="application/pdf"
        )
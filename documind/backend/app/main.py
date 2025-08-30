# ./backend/app/main.py

import os
import boto3
import uuid
import logging
import tempfile
import pdfplumber
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, update
from . import models

# --- Configuration & Setup ---
load_dotenv(dotenv_path="../.env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database Connection
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_async_engine(DATABASE_URL)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# S3 Client
S3_ENDPOINT = os.getenv("S3_ENDPOINT")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET = os.getenv("S3_BUCKET")

s3_client = boto3.client(
    's3',
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

app = FastAPI(title="DocuMind AI")

# --- Background Task (The Processing Pipeline) ---

async def process_document(doc_id: int, s3_key: str):
    logger.info(f"Starting processing for doc_id: {doc_id} with s3_key: {s3_key}")
    
    # Create a temporary file to download the document from S3
    with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_file:
        try:
            # 1. Download file from S3 to the temporary file
            s3_client.download_fileobj(S3_BUCKET, s3_key, temp_file)
            temp_file.seek(0) # Go back to the beginning of the file
            
            logger.info(f"Successfully downloaded {s3_key} to temporary file.")

            # 2. Perform text extraction with pdfplumber
            full_text = ""
            with pdfplumber.open(temp_file.name) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"
                        logger.info(f"Extracted text from page {i+1}")

            # For now, we'll just log the first 500 characters
            logger.info(f"Total extracted text (first 500 chars): {full_text[:500]}")

            # 3. In the next steps, we will chunk, embed, and store this text.

            # 4. Update document status to "ready"
            async with AsyncSessionLocal() as session:
                async with session.begin():
                    await session.execute(
                        update(models.Document)
                        .where(models.Document.id == doc_id)
                        .values(status=models.DocumentStatus.READY)
                    )
                # The context manager handles the commit here
            logger.info(f"Updated status to READY for doc_id: {doc_id}")

        except Exception as e:
            logger.error(f"Failed to process document {doc_id}: {e}")
            # Update status to "failed" in case of an error
            async with AsyncSessionLocal() as session:
                async with session.begin():
                    await session.execute(
                        update(models.Document)
                        .where(models.Document.id == doc_id)
                        .values(status=models.DocumentStatus.FAILED)
                    )
                # The context manager handles the commit here

# --- API Endpoints ---

@app.on_event("startup")
async def startup():
    try:
        s3_client.head_bucket(Bucket=S3_BUCKET)
    except s3_client.exceptions.ClientError:
        logger.info(f"Bucket '{S3_BUCKET}' not found. Creating it.")
        s3_client.create_bucket(Bucket=S3_BUCKET)
    
    async with engine.begin() as conn:
        await conn.run_sync(models.Base.metadata.create_all)

@app.get("/")
def read_root():
    return {"message": "Welcome to DocuMind AI"}

@app.post("/documents", status_code=202)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
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

    # Create the document record in the database
    doc_id = None
    async with AsyncSessionLocal() as session:
        async with session.begin():
            # Create the document with status 'processing' from the start
            doc = models.Document(
                title=file.filename, 
                s3_key=s3_key,
                status=models.DocumentStatus.PROCESSING
            )
            session.add(doc)
        
        # After the 'begin' block, the transaction is committed.
        # We now refresh the 'doc' object to get the ID assigned by the database.
        await session.refresh(doc)
        doc_id = doc.id

    if doc_id is None:
        raise HTTPException(status_code=500, detail="Could not create document record.")
    
    # Queue the background task for processing
    background_tasks.add_task(process_document, doc_id, s3_key)
    logger.info(f"Queued processing for doc_id: {doc_id}")
    
    return {"doc_id": doc_id, "s3_key": s3_key, "status": "processing"}

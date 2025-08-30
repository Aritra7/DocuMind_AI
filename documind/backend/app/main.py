# ./backend/app/main.py

import os
import boto3
import uuid
import logging
import tempfile
import pdfplumber # <-- ADD THIS IMPORT
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import update
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

async def set_document_status(doc_id: int, status: models.DocumentStatus):
    """Helper function to update document status in the database."""
    async with AsyncSessionLocal() as session:
        async with session.begin():
            stmt = (
                update(models.Document)
                .where(models.Document.id == doc_id)
                .values(status=status)
            )
            await session.execute(stmt)
            await session.commit()

async def process_document(doc_id: int, s3_key: str):
    """
    Main background task: downloads a file from S3, extracts text, and updates status.
    """
    logger.info(f"Starting processing for doc_id: {doc_id} with s3_key: {s3_key}")
    await set_document_status(doc_id, models.DocumentStatus.PROCESSING)

    try:
        # Create a temporary file to download the document
        with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_file:
            # 1. Download the file from S3
            s3_client.download_fileobj(S3_BUCKET, s3_key, temp_file)
            temp_file.seek(0) # Go to the beginning of the file
            logger.info(f"Successfully downloaded {s3_key} to temporary file.")

            # 2. Perform Text Extraction
            full_text = ""
            with pdfplumber.open(temp_file.name) as pdf:
                for i, page in enumerate(pdf.pages):
                    full_text += page.extract_text() + "\n"
                    logger.info(f"Extracted text from page {i+1}")
            
            logger.info(f"Total extracted text (first 500 chars): {full_text[:500]}")

            # Future steps will go here (chunking, embedding, etc.)

        # 3. Update the document status in Postgres to "ready"
        await set_document_status(doc_id, models.DocumentStatus.READY)
        logger.info(f"Updated status to READY for doc_id: {doc_id}")

    except Exception as e:
        logger.error(f"Failed to process document {doc_id}: {e}")
        await set_document_status(doc_id, models.DocumentStatus.FAILED)


# --- API Endpoints ---

@app.on_event("startup")
async def startup():
    # Create a bucket if it doesn't exist (for MinIO)
    try:
        s3_client.head_bucket(Bucket=S3_BUCKET)
    except s3_client.exceptions.ClientError:
        logger.info(f"Bucket '{S3_BUCKET}' not found. Creating it.")
        s3_client.create_bucket(Bucket=S3_BUCKET)
    
    # Create database tables
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
    
    # 1. Upload file to S3
    try:
        s3_client.upload_fileobj(file.file, S3_BUCKET, s3_key)
        logger.info(f"File '{file.filename}' uploaded to S3 as '{s3_key}'")
    except Exception as e:
        logger.error(f"Failed to upload to S3: {e}")
        raise HTTPException(status_code=500, detail="Could not upload file.")

    # 2. Create DB record
    doc_id = None
    async with AsyncSessionLocal() as session:
        async with session.begin():
            doc = models.Document(title=file.filename, s3_key=s3_key)
            session.add(doc)
            # The context manager will automatically commit here
        # After the block, the object is expired, but we can get the ID
        doc_id = doc.id
    
    if not doc_id:
        raise HTTPException(status_code=500, detail="Failed to create document record in database.")

    # 3. Queue background task for processing
    background_tasks.add_task(process_document, doc_id, s3_key)
    logger.info(f"Queued processing for doc_id: {doc_id}")
    
    return {"doc_id": doc_id, "s3_key": s3_key, "status": "processing"}

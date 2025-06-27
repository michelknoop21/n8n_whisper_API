from fastapi import FastAPI, HTTPException, UploadFile, File, Header, Request, BackgroundTasks, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
import torch
from transformers import pipeline
import os
import uuid
import asyncio
from datetime import datetime
import aiohttp
import tempfile
from pathlib import Path

# Create main FastAPI app
app = FastAPI(title="Insanely Fast Whisper API")

# Create API router
api_router = APIRouter(prefix="/api")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=os.getenv("CORS_METHODS", "*").split(","),
    allow_headers=os.getenv("CORS_HEADERS", "*").split(","),
    expose_headers=["Content-Type", "Authorization", "X-Requested-With"],
    max_age=3600
)

# Models
class TranscriptionRequest(BaseModel):
    url: Optional[str] = None
    task: str = "transcribe"
    language: Optional[str] = None
    batch_size: int = 64
    timestamp: str = "chunk"
    diarise_audio: bool = False
    webhook: Optional[Dict] = None
    is_async: bool = False
    managed_task_id: Optional[str] = None

class TranscriptionResponse(BaseModel):
    text: str
    task_id: str
    status: str
    created_at: datetime

# Initialize Whisper pipeline
whisper_pipeline = None

def get_whisper_pipeline():
    global whisper_pipeline
    if whisper_pipeline is None:
        whisper_pipeline = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-base",
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
    return whisper_pipeline

# Utility functions
async def download_audio(url: str) -> str:
    """Download audio from URL and return local file path."""
    temp_dir = Path(tempfile.gettempdir())
    temp_file = temp_dir / f"audio_{uuid.uuid4()}.wav"
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise HTTPException(status_code=400, detail="Failed to download audio file")
            
            with open(temp_file, "wb") as f:
                while True:
                    chunk = await response.content.read(1024)
                    if not chunk:
                        break
                    f.write(chunk)
    
    return str(temp_file)

# API Endpoints
@api_router.post("/transcribe")
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    request: Optional[TranscriptionRequest] = None,
    x_admin_api_key: str = Header(None)
):
    # Validate API key
    if x_admin_api_key != os.getenv("ADMIN_KEY"):
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Check if either file or URL is provided
    if file is None and (request is None or request.url is None):
        raise HTTPException(status_code=400, detail="Either file or URL must be provided")
    
    # Generate task ID
    task_id = str(uuid.uuid4())
    
    try:
        if file:
            # Handle file upload
            temp_file = Path(tempfile.gettempdir()) / f"audio_{task_id}_{file.filename}"
            with open(temp_file, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Get the transcription pipeline
            pipeline = get_whisper_pipeline()
            
            # Transcribe audio
            result = pipeline(
                str(temp_file),
                batch_size=64,
                return_timestamps=False
            )
            
            # Clean up
            temp_file.unlink()
            
            return TranscriptionResponse(
                text=result["text"],
                task_id=task_id,
                status="completed",
                created_at=datetime.utcnow(),
            )
            
        elif request and request.url:
            # Handle URL transcription (async)
            if request.is_async:
                background_tasks.add_task(process_async_transcription, request, task_id)
                return {"task_id": task_id, "status": "processing"}
            else:
                return await process_sync_transcription(request, task_id)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def process_sync_transcription(request: TranscriptionRequest, task_id: str):
    """Process transcription synchronously."""
    try:
        # Download audio
        audio_path = await download_audio(request.url)
        
        # Get the transcription pipeline
        pipeline = get_whisper_pipeline()
        
        # Transcribe audio
        result = pipeline(
            audio_path,
            language=request.language,
            batch_size=request.batch_size,
            return_timestamps=request.timestamp != "none"
        )
        
        # Clean up
        Path(audio_path).unlink()
        
        return TranscriptionResponse(
            text=result["text"],
            task_id=task_id,
            status="completed",
            created_at=datetime.utcnow(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def process_async_transcription(request: TranscriptionRequest, task_id: str):
    """Process transcription asynchronously."""
    try:
        # This would typically involve adding the task to a queue
        # and having a separate worker process it
        pass
    except Exception as e:
        # Log the error
        print(f"Error in async transcription: {e}")

@api_router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "timestamp": datetime.utcnow()}

@api_router.get("/tasks")
async def get_tasks():
    # Implement task management
    return []

@api_router.get("/status/{task_id}")
async def get_task_status(task_id: str):
    # Implement task status retrieval
    return {"status": "completed"}

@api_router.delete("/tasks/{task_id}")
async def cancel_task(task_id: str):
    # Implement task cancellation
    return {"status": "cancelled"}

# Include the API router
app.include_router(api_router)

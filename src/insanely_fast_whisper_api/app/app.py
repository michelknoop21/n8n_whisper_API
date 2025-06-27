from fastapi import FastAPI, HTTPException, UploadFile, File, Header, Request, BackgroundTasks, APIRouter, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List, Tuple
import torch
from transformers import pipeline
import os
import uuid
import asyncio
from datetime import datetime
import aiohttp
import tempfile
from pathlib import Path
import mimetypes
import logging
import subprocess
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supported audio formats
SUPPORTED_AUDIO_TYPES = [
    'audio/wav',
    'audio/mpeg',
    'audio/mp3',
    'audio/x-wav',
    'audio/x-m4a',
    'audio/mp4',
    'audio/aac',
    'audio/ogg',
    'audio/webm'
]

# Maximum file size (50MB)
MAX_FILE_SIZE = 50 * 1024 * 1024

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
        logger.info("Initializing Whisper pipeline...")
        device = 0 if torch.cuda.is_available() else -1
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        logger.info(f"Using device: {'CUDA' if device >= 0 else 'CPU'}")
        logger.info(f"Using torch_dtype: {torch_dtype}")
        
        # Use a larger model for better accuracy
        model_name = "openai/whisper-small"
        logger.info(f"Loading model: {model_name}")
        
        try:
            whisper_pipeline = pipeline(
                "automatic-speech-recognition",
                model=model_name,
                device=device,
                torch_dtype=torch_dtype,
                chunk_length_s=30,  # Process in 30-second chunks
                stride_length_s=5,   # Overlap chunks by 5 seconds
                return_timestamps=False
            )
            logger.info("Whisper pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Whisper pipeline: {str(e)}")
            raise
            
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

def convert_audio(input_path: str, output_path: str = None) -> str:
    """Convert audio file to WAV format using FFmpeg."""
    if output_path is None:
        output_path = str(Path(input_path).with_suffix('.wav'))
    
    try:
        # Convert to WAV with 16kHz sample rate and mono channel
        cmd = [
            'ffmpeg',
            '-i', input_path,    # Input file
            '-ar', '16000',     # Audio sample rate
            '-ac', '1',         # Mono audio
            '-y',               # Overwrite output file if it exists
            output_path
        ]
        
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logger.info(f"Audio converted successfully: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed: {e.stderr.decode()}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Kon het audiobestand niet converteren naar een bruikbaar formaat"
        )

async def validate_uploaded_file(file: UploadFile) -> dict:
    """Validate the uploaded file and return file info."""
    # Check file size
    file.file.seek(0, 2)  # Move to end of file
    file_size = file.file.tell()
    file.file.seek(0)  # Reset file pointer
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Bestand is te groot. Maximale grootte is {MAX_FILE_SIZE/1024/1024}MB"
        )
    
    # Get file extension
    file_extension = Path(file.filename).suffix.lower()
    
    # Check if FFmpeg is available
    ffmpeg_available = shutil.which('ffmpeg') is not None
    
    # If FFmpeg is not available, only allow WAV files
    if not ffmpeg_available and file_extension != '.wav':
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Alleen WAV-bestanden worden ondersteund. Installeer FFmpeg voor ondersteuning van meer formaten."
        )
    
    # Check file type
    content_type = file.content_type
    if not content_type:
        # Try to guess content type from filename
        content_type = mimetypes.guess_type(file.filename)[0]
    
    # If we couldn't determine content type, try to proceed anyway if FFmpeg is available
    if not content_type and not ffmpeg_available:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Kon het bestandstype niet bepalen"
        )
    
    # Check if the file needs conversion
    needs_conversion = file_extension not in ['.wav', '.mp3']
    
    return {
        'needs_conversion': needs_conversion,
        'extension': file_extension,
        'content_type': content_type
    }

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
            # Create temp directory if it doesn't exist
            temp_dir = Path(tempfile.gettempdir()) / "whisper_uploads"
            temp_dir.mkdir(exist_ok=True, parents=True)
            
            # Save uploaded file
            temp_file = temp_dir / f"audio_{task_id}_{file.filename}"
            output_file = temp_file.with_suffix('.wav')
            
            try:
                # Save uploaded file
                with open(temp_file, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                
                logger.info(f"File saved to {temp_file}")
                
                # Validate file and check if conversion is needed
                file_info = await validate_uploaded_file(file)
                
                # Convert audio if needed
                if file_info['needs_conversion']:
                    logger.info(f"Converting {temp_file} to WAV format...")
                    convert_audio(str(temp_file), str(output_file))
                    # Remove original file after conversion
                    temp_file.unlink()
                else:
                    output_file = temp_file
                
                # Get the transcription pipeline
                pipeline = get_whisper_pipeline()
                
                # Get language from form data or use auto-detect
                language = 'nl'  # Default to Dutch
                
                # Log the language being used
                logger.info(f"Starting transcription of {output_file} (language: {language or 'auto'})...")
                
                try:
                    # Prepare generation kwargs
                    generate_kwargs = {"language": language} if language else {}
                    
                    result = pipeline(
                        str(output_file),
                        batch_size=16,  # Reduced batch size for better memory management
                        return_timestamps=False,
                        generate_kwargs=generate_kwargs if generate_kwargs else None
                    )
                    
                    # Log the raw result for debugging
                    logger.info(f"Raw transcription result: {result}")
                    
                    # Ensure we have text in the result
                    if not result or 'text' not in result or not result['text'].strip():
                        logger.warning("No text was transcribed. The audio might be silent or contain no speech.")
                        result['text'] = "Geen spraak herkend in het audiobestand."
                    else:
                        logger.info("Transcription completed successfully")
                        
                except Exception as e:
                    logger.error(f"Error during transcription: {str(e)}")
                    raise
                
                return TranscriptionResponse(
                    text=result["text"],
                    task_id=task_id,
                    status="completed",
                    created_at=datetime.utcnow(),
                )
                
            except Exception as e:
                logger.error(f"Error during transcription: {str(e)}", exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Fout bij het verwerken van het audiobestand: {str(e)}"
                )
                
            finally:
                # Clean up
                try:
                    if temp_file.exists():
                        temp_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {temp_file}: {e}")
            
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

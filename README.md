# Insanely Fast Whisper API

An API to transcribe audio with OpenAI's Whisper Large v3! Powered by 🤗 Transformers, Optimum & flash-attn

## Features

- Transcribe audio files or URLs
- Support for both synchronous and asynchronous processing
- CORS enabled for web frontend integration
- Health check endpoint
- Admin key authentication

## Prerequisites

- Python 3.9+
- Docker and Docker Compose
- NVIDIA GPU with CUDA support (recommended)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jigsawstack/insanely-fast-whisper-api.git
   cd insanely-fast-whisper-api
   ```

2. Install dependencies (optional when using Docker):
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Copy `.env.example` to `.env` and adjust the values for your environment:

```env
ADMIN_KEY=your_secure_admin_key_here
WHISPER_MODEL=openai/whisper-small
MAX_FILE_SIZE=52428800
CORS_ORIGINS=*
CORS_METHODS=*
CORS_HEADERS=*
USE_REDIS_QUEUE=0
REDIS_URL=redis://redis:6379/0
```

## Running with Docker

1. Build and start the containers:
   ```bash
   docker-compose up --build
   ```

2. The API will be available at `http://localhost:5000`
3. When `USE_REDIS_QUEUE=1` a worker and Redis container will handle queued jobs.

## API Endpoints

### Transcribe Audio

```
POST /transcribe
```

**Headers:**
- `X-Admin-API-Key`: Your admin API key

**Form Data:**
- `file`: Audio file to transcribe (optional if URL is provided)

**JSON Body (optional):**
```json
{
  "url": "https://example.com/audio.mp3",
  "task": "transcribe",
  "language": "en",
  "batch_size": 64,
  "timestamp": "chunk",
  "diarise_audio": false,
  "is_async": false
}
```

### Health Check

```
GET /health
```

## Development

1. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

2. Run the development server:
   ```bash
   uvicorn app.app:app --reload
   ```

## License

MIT

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

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Create a `.env` file in the project root with the following variables:

```env
ADMIN_KEY=your_secure_admin_key_here
CORS_ORIGINS=*
CORS_METHODS=*
CORS_HEADERS=*
```

## Running with Docker

1. Build and start the container:
   ```bash
   docker-compose up --build
   ```

2. The API will be available at `http://localhost:5000`

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

## Generalisatie en Template

Deze applicatie biedt een REST API op basis van FastAPI waarmee audiobestanden worden getranscribeerd via OpenAI's Whisper-model.
Om dit project verder te generaliseren voor gebruik bij verschillende klanten kunnen onderstaande stappen worden gevolgd:

1. **Configuratie scheiden**: gebruik `.env` of container-omgevingsvariabelen voor waarden zoals het model, toegestane domeinen (CORS) en de admin-sleutel.
2. **Docker templatiseren**: maak een standaard `docker-compose.yml` die eenvoudig aanpasbaar is per klant en een `Dockerfile` die variabelen accepteert.
3. **Webhook integraties**: definieer duidelijke hooks (zoals `/webhook/transcribe`) die per klant ingesteld kunnen worden.
4. **Opschalen en queueing**: implementeer optioneel een wachtrij (bijv. via Redis) zodat grote aantallen bestanden asynchroon verwerkt kunnen worden.
5. **Documentatie**: lever een stappenplan voor installatie en configuratie zodat nieuwe omgevingen snel uitgerold kunnen worden.

Met deze aanpak kan dezelfde basiscode als sjabloon dienen terwijl specifieke instellingen per klant in configuratiebestanden worden vastgelegd.


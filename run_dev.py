import uvicorn
import os

def main():
    """Run the FastAPI development server with hot-reload."""
    os.environ["PYTHONPATH"] = "."
    
    uvicorn.run(
        "src.insanely_fast_whisper_api.app.app:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        reload_dirs=["src"],
        log_level="info"
    )

if __name__ == "__main__":
    main()

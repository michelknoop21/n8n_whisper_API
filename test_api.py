import requests
import os
from pathlib import Path
from typing import Optional
import argparse

class WhisperAPIClient:
    """Client for interacting with the Whisper API."""
    
    def __init__(self, base_url: str = "http://localhost:5000", api_key: Optional[str] = None):
        """Initialize the API client.
        
        Args:
            base_url: Base URL of the API
            api_key: Admin API key (can also be set via ADMIN_KEY environment variable)
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key or os.getenv("ADMIN_KEY")
        
        if not self.api_key:
            print("Warning: No API key provided. Set ADMIN_KEY environment variable or pass api_key parameter.")
    
    def _get_headers(self, content_type: str = "application/json") -> dict:
        """Get headers for API requests."""
        headers = {}
        if self.api_key:
            headers["X-Admin-API-Key"] = self.api_key
        if content_type:
            headers["Content-Type"] = content_type
        return headers
    
    def health_check(self) -> dict:
        """Check if the API is healthy."""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def transcribe_file(
        self,
        file_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
        timestamp: str = "chunk",
        batch_size: int = 64,
        is_async: bool = False
    ) -> dict:
        """Transcribe an audio file.
        
        Args:
            file_path: Path to the audio file
            language: Language code (e.g., 'en', 'fr', 'es')
            task: 'transcribe' or 'translate'
            timestamp: How to include timestamps ('chunk' or 'none')
            batch_size: Batch size for processing
            is_async: Whether to process asynchronously
            
        Returns:
            Transcription result
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        headers = self._get_headers(None)  # Let requests set Content-Type with boundary
        
        with open(file_path, 'rb') as f:
            files = {
                'file': (os.path.basename(file_path), f, 'audio/wav')
            }
            
            data = {
                'task': task,
                'language': language,
                'timestamp': timestamp,
                'batch_size': str(batch_size),
                'is_async': str(is_async).lower()
            }
            
            # Remove None values
            data = {k: v for k, v in data.items() if v is not None}
            
            response = requests.post(
                f"{self.base_url}/transcribe",
                headers=headers,
                files=files,
                data=data
            )
            
        response.raise_for_status()
        return response.json()
    
    def transcribe_url(
        self,
        url: str,
        language: Optional[str] = None,
        task: str = "transcribe",
        timestamp: str = "chunk",
        batch_size: int = 64,
        is_async: bool = False
    ) -> dict:
        """Transcribe audio from a URL.
        
        Args:
            url: URL of the audio file
            language: Language code (e.g., 'en', 'fr', 'es')
            task: 'transcribe' or 'translate'
            timestamp: How to include timestamps ('chunk' or 'none')
            batch_size: Batch size for processing
            is_async: Whether to process asynchronously
            
        Returns:
            Transcription result
        """
        headers = self._get_headers()
        
        data = {
            'url': url,
            'task': task,
            'language': language,
            'timestamp': timestamp,
            'batch_size': batch_size,
            'is_async': is_async
        }
        
        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}
        
        response = requests.post(
            f"{self.base_url}/transcribe",
            headers=headers,
            json=data
        )
        
        response.raise_for_status()
        return response.json()

def main():
    """Run the test script."""
    parser = argparse.ArgumentParser(description='Test Whisper API')
    parser.add_argument('--url', type=str, help='URL of the Whisper API', default='http://localhost:5000')
    parser.add_argument('--key', type=str, help='Admin API key', default=None)
    parser.add_argument('--file', type=str, help='Path to audio file for transcription', default=None)
    parser.add_argument('--audio-url', type=str, help='URL of audio file for transcription', default=None)
    parser.add_argument('--language', type=str, help='Language code (e.g., en, fr, es)', default=None)
    
    args = parser.parse_args()
    
    client = WhisperAPIClient(base_url=args.url, api_key=args.key)
    
    try:
        # Test health check
        print("\n=== Testing Health Check ===")
        health = client.health_check()
        print(f"Status: {health['status']}")
        print(f"Timestamp: {health['timestamp']}")
        
        # Test file transcription if file is provided
        if args.file:
            print("\n=== Testing File Transcription ===")
            print(f"File: {args.file}")
            result = client.transcribe_file(
                file_path=args.file,
                language=args.language
            )
            print(f"Task ID: {result['task_id']}")
            print(f"Status: {result['status']}")
            print(f"Transcription: {result['text']}")
        
        # Test URL transcription if URL is provided
        if args.audio_url:
            print("\n=== Testing URL Transcription ===")
            print(f"URL: {args.audio_url}")
            result = client.transcribe_url(
                url=args.audio_url,
                language=args.language
            )
            print(f"Task ID: {result['task_id']}")
            print(f"Status: {result['status']}")
            print(f"Transcription: {result['text']}")
        
        if not args.file and not args.audio_url:
            print("\nNo file or URL provided for transcription. Use --file or --audio-url to test transcription.")
    
    except requests.exceptions.HTTPError as e:
        print(f"\n=== Error ===")
        print(f"Status Code: {e.response.status_code}")
        try:
            print(f"Response: {e.response.json()}")
        except:
            print(f"Response: {e.response.text}")
    except Exception as e:
        print(f"\n=== Error ===")
        print(f"{type(e).__name__}: {str(e)}")

if __name__ == "__main__":
    main()

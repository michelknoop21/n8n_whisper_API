import os
import requests
from pathlib import Path

def download_test_audio():
    """Download a small test audio file for testing."""
    test_dir = Path("test_audio")
    test_dir.mkdir(exist_ok=True)
    
    # URL of a small test audio file (public domain)
    audio_url = "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"
    output_file = test_dir / "test_audio.mp3"
    
    print(f"Downloading test audio from {audio_url}...")
    
    response = requests.get(audio_url, stream=True)
    response.raise_for_status()
    
    with open(output_file, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"Test audio saved to: {output_file.absolute()}")
    return str(output_file.absolute())

if __name__ == "__main__":
    download_test_audio()

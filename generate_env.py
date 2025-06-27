import secrets

def generate_api_key():
    return secrets.token_urlsafe(32)  # Genereer een veilige willekeurige key

if __name__ == "__main__":
    new_key = generate_api_key()
    print(f"Nieuwe API key: {new_key}")
    
    with open(".env", "w") as f:
        f.write(f"ADMIN_KEY={new_key}\n")
        f.write("WHISPER_MODEL=openai/whisper-small\n")
        f.write("MAX_FILE_SIZE=52428800  # 50MB in bytes\n")
    
    print(".env bestand is aangemaakt met een nieuwe API key.")
    print("Zorg ervoor dat je .env toevoegt aan .gitignore als dat nog niet het geval is!")

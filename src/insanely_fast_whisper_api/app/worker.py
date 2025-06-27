import json
import logging
from pathlib import Path
import redis

from dotenv import load_dotenv
from pydantic import BaseSettings

from .app import get_whisper_pipeline

load_dotenv()


class Settings(BaseSettings):
    redis_url: str = "redis://redis:6379/0"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
redis_client = redis.from_url(settings.redis_url)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    pipeline = get_whisper_pipeline()
    logger.info("Worker started, waiting for tasks")
    while True:
        job = redis_client.blpop("transcription_tasks")
        if not job:
            continue
        _, data = job
        task = json.loads(data)
        task_id = task["task_id"]
        file_path = task["file_path"]
        language = task.get("language", "nl")
        logger.info(f"Processing task {task_id}")
        try:
            result = pipeline(
                file_path,
                batch_size=16,
                return_timestamps=False,
                generate_kwargs={"language": language},
            )
            redis_client.set(
                f"result:{task_id}",
                json.dumps({"text": result["text"], "status": "completed"}),
            )
        except Exception as e:
            logger.error(f"Error processing task {task_id}: {e}")
            redis_client.set(
                f"result:{task_id}", json.dumps({"status": "error", "detail": str(e)})
            )
        finally:
            Path(file_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()

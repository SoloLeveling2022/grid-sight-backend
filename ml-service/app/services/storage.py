import os
import base64
from abc import ABC, abstractmethod

class BaseStorageDriver(ABC):
    @abstractmethod
    def save_image(self, filename: str, data: bytes) -> str:
        """
        Saves the binary image data and returns the public accessibility URL.
        """
        pass

class LocalStorageDriver(BaseStorageDriver):
    def __init__(self, upload_dir: str = "static/evidence", base_url: str = None):
        self.upload_dir = upload_dir
        os.makedirs(self.upload_dir, exist_ok=True)
        # Use host URL from environment or default to localhost
        self.base_url = base_url or os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

    def save_image(self, filename: str, data: bytes) -> str:
        filepath = os.path.join(self.upload_dir, filename)
        with open(filepath, "wb") as f:
            f.write(data)
        # Return URL mapping to StaticFiles mount point
        return f"{self.base_url}/static/evidence/{filename}"

class S3StorageDriver(BaseStorageDriver):
    def __init__(self):
        self.bucket = os.getenv("AWS_BUCKET_NAME")
        self.region = os.getenv("AWS_REGION", "us-east-1")
        # Initialize boto3 dynamically to prevent import failure if not installed
        try:
            import boto3
            self.s3_client = boto3.client(
                "s3",
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=self.region
            )
        except Exception as e:
            print(f"[Storage S3] Failed to initialize S3 driver: {e}. Falling back to Local.")
            self.s3_client = None

    def save_image(self, filename: str, data: bytes) -> str:
        if not self.s3_client or not self.bucket:
            # Fallback to local storage if S3 config is missing or client failed to init
            fallback = LocalStorageDriver()
            return fallback.save_image(filename, data)
        
        try:
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=f"evidence/{filename}",
                Body=data,
                ContentType="image/jpeg"
            )
            return f"https://{self.bucket}.s3.{self.region}.amazonaws.com/evidence/{filename}"
        except Exception as e:
            print(f"[Storage S3] Upload failed: {e}. Falling back to Local.")
            fallback = LocalStorageDriver()
            return fallback.save_image(filename, data)

def get_storage_driver() -> BaseStorageDriver:
    provider = os.getenv("STORAGE_PROVIDER", "local").strip().lower()
    if provider == "s3":
        return S3StorageDriver()
    return LocalStorageDriver()

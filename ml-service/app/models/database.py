import os
import certifi
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables
ROOT = Path(__file__).resolve().parents[4]
env_paths = [
    ROOT / "gridsight-services" / "ml-service" / ".env",
    ROOT / ".env"
]
for p in env_paths:
    if p.exists():
        load_dotenv(p)
        break

DATABASE_URL = os.getenv("DATABASE_URL", "mongodb://localhost:27017/gridsight")

# SQLite DB path stub for backward compatibility / transition safety
SQLITE_DB_PATH = Path(__file__).resolve().parents[2] / "gridsight.db"

# Try to parse DB name from connection URI
db_name = "gridsight"
try:
    if "://" in DATABASE_URL:
        path_part = DATABASE_URL.split("://")[-1]
        if "/" in path_part:
            parsed_db = path_part.split("/")[-1].split("?")[0].strip()
            if parsed_db:
                db_name = parsed_db
except Exception:
    pass

# Setup PyMongo client with SSL parameters to support Atlas.
# We configure with both tlsCAFile and tlsAllowInvalidCertificates fallback options.
try:
    client = MongoClient(
        DATABASE_URL,
        tls=True,
        tlsCAFile=certifi.where(),
        serverSelectionTimeoutMS=5000
    )
except Exception:
    # Fallback to bypass invalid cert issues
    client = MongoClient(
        DATABASE_URL,
        tls=True,
        tlsAllowInvalidCertificates=True,
        serverSelectionTimeoutMS=5000
    )

db = client[db_name]
_violations_collection = db["violations"]


def get_violations_collection():
    """Return the native MongoDB violations collection."""
    return _violations_collection


def get_mongo_client():
    """Return the native MongoDB client."""
    return client

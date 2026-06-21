import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from app.models.database import get_violations_collection
from app.routes.violations import router as violations_router
from app.routes.ml import router as ml_router

# Load environment variables
load_dotenv()

from fastapi.staticfiles import StaticFiles

# Verify MongoDB connection on startup
try:
    print("[FastAPI Startup] Connecting to MongoDB Atlas ...")
    from app.models.database import get_mongo_client
    client = get_mongo_client()
    client.admin.command('ping')
    collection = get_violations_collection()
    count = collection.count_documents({})
    print(f"[FastAPI Startup] MongoDB OK — {count:,} violation records ready on cluster.")
except Exception as e:
    print(f"[FastAPI Startup] WARNING: MongoDB connection check failed: {e}")

app = FastAPI(
    title="GridSight Scalable Backend Service",
    description="Scalable APIs and machine learning inference pipeline for parking violation intelligence.",
    version="1.1.0"
)

# Mount Static Files for media storage
os.makedirs("static/evidence", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure CORS Middleware (allows frontend cross-origin requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://grid-sight-mu.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routes
app.include_router(violations_router)
app.include_router(ml_router)

@app.get("/health")
def health_check():
    try:
        from app.models.database import get_mongo_client
        client = get_mongo_client()
        client.admin.command('ping')
        collection = get_violations_collection()
        count = collection.count_documents({})
        db_status = f"MongoDB OK ({count:,} rows)"
    except Exception as e:
        db_status = f"MongoDB ERROR: {e}"
    return {
        "status": "healthy",
        "service": "GridSight Scalable API Server",
        "database": "MongoDB",
        "db_status": db_status,
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"[FastAPI Server] Starting server on {host}:{port}...")
    uvicorn.run("app.main:app", host=host, port=port, reload=True)

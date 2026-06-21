import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
uri = os.getenv("DATABASE_URL")
print(f"Connecting to MongoDB at: {uri}")
try:
    client = MongoClient(uri)
    dbs = client.list_database_names()
    print("Databases on cluster:", dbs)
    for db_name in dbs:
        if db_name in ['admin', 'local', 'config']:
            continue
        db = client[db_name]
        collections = db.list_collection_names()
        print(f"Database '{db_name}' collections:", collections)
        for c in collections:
            count = db[c].count_documents({})
            print(f"  Collection '{c}' has {count:,} documents.")
except Exception as e:
    print("Failed to connect to MongoDB:", e)

import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

def get_db():
    mongo_uri = os.getenv("MONGO_URI")
    db_name = os.getenv("MONGO_DB", "clientele")

    if not mongo_uri:
        raise ValueError("MONGO_URI is missing. Put it in your .env file.")

    client = MongoClient(mongo_uri)
    return client[db_name]

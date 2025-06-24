import chromadb
from crewai.utilities.paths import db_storage_path
import os

# Connect to your ChromaDB storage
storage_path = db_storage_path()
short_term_path = os.path.join(storage_path, "short_term_memory")

if os.path.exists(short_term_path):
    client = chromadb.PersistentClient(path=short_term_path)
    
    # List collections
    collections = client.list_collections()
    for collection in collections:
        print(f"Collection: {collection.name}")
        print(f"Documents: {collection.count()}")
        
        # Peek at some stored data
        results = collection.peek(limit=2)
        print(f"Sample documents: {results['documents'][:2]}")
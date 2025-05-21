# Database module for storing embeddings and metadata 
import sqlite3
import json
import numpy as np # Assuming embeddings are numpy arrays

DATABASE_FILE = "diskuss.db"

def initialize_database(db_path=DATABASE_FILE):
    """Initializes the SQLite database and creates the documents table."""
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE,
                embedding BLOB
            )
        """)
        conn.commit()
        print(f"Database initialized at {db_path}")
    except sqlite3.Error as e:
        print(f"Database error during initialization: {e}")
    finally:
        if conn:
            conn.close()

def add_document(db_path, file_path, embedding):
    """Adds a document's file path and embedding to the database."""
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Serialize numpy array to bytes for BLOB storage
        embedding_blob = embedding.tobytes() if isinstance(embedding, np.ndarray) else embedding

        cursor.execute("""
            INSERT OR REPLACE INTO documents (file_path, embedding)
            VALUES (?, ?)
        """, (file_path, embedding_blob))
        conn.commit()
        print(f"Added/Updated document: {file_path}")
    except sqlite3.Error as e:
        print(f"Database error adding document {file_path}: {e}")
    finally:
        if conn:
            conn.close()

def get_all_documents(db_path=DATABASE_FILE):
    """Retrieves all documents from the database."""
    conn = None
    documents = []
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT file_path, embedding FROM documents")
        rows = cursor.fetchall()
        for row in rows:
            file_path = row[0]
            # Deserialize the embedding BLOB back to numpy array (assuming float32)
            embedding = np.frombuffer(row[1], dtype=np.float32) if row[1] else None
            documents.append({'file_path': file_path, 'embedding': embedding})
    except sqlite3.Error as e:
        print(f"Database error retrieving documents: {e}")
    finally:
        if conn:
            conn.close()
    return documents

def search_documents(db_path, query_embedding, k=5):
    """Searches for documents most similar to the query embedding.
       NOTE: This is a placeholder. Actual vector similarity search requires
       a vector database or SQLite extension (e.g., sqlite-vss).
       This implementation fetches all and calculates similarity in Python (inefficient).
    """
    all_documents = get_all_documents(db_path)
    if not all_documents:
        return []

    # Calculate cosine similarity manually (inefficient for large datasets)
    similarities = []
    query_embedding_np = np.array(query_embedding, dtype=np.float32)

    for doc in all_documents:
        if doc['embedding'] is not None:
            doc_embedding_np = doc['embedding']
            # Cosine similarity formula: (A . B) / (||A|| ||B||)
            similarity = np.dot(query_embedding_np, doc_embedding_np) / (np.linalg.norm(query_embedding_np) * np.linalg.norm(doc_embedding_np))
            similarities.append((similarity, doc['file_path']))

    # Sort by similarity in descending order and get top k
    similarities.sort(key=lambda x: x[0], reverse=True)

    return [(file_path, score) for score, file_path in similarities[:k]]

def count_total_documents(db_path=DATABASE_FILE):
    """Counts the total number of documents in the database."""
    conn = None
    count = 0
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents")
        count = cursor.fetchone()[0]
    except sqlite3.Error as e:
        print(f"Database error counting documents: {e}")
    finally:
        if conn:
            conn.close()
    return count

if __name__ == '__main__':
    # Example Usage (for testing)
    initialize_database()

    # Create some dummy data (replace with actual embeddings from Ollama)
    dummy_embedding_1 = np.random.rand(768).astype(np.float32) # Assuming embedding dimension 768
    dummy_embedding_2 = np.random.rand(768).astype(np.float32)
    dummy_embedding_3 = np.random.rand(768).astype(np.float32)

    add_document(DATABASE_FILE, "/path/to/document1.docx", dummy_embedding_1)
    add_document(DATABASE_FILE, "/path/to/document2.pdf", dummy_embedding_2)
    add_document(DATABASE_FILE, "/path/to/document3.txt", dummy_embedding_3)

    # Simulate a query embedding
    query_embedding = np.random.rand(768).astype(np.float32)

    print("\nSearching for similar documents:")
    search_results = search_documents(DATABASE_FILE, query_embedding, k=2)
    for file_path, score in search_results:
        print(f"File: {file_path}, Similarity: {score:.4f}")

    # Clean up the test database (optional)
    # import os
    # if os.path.exists(DATABASE_FILE):
    #     os.remove(DATABASE_FILE)
    #     print(f"Cleaned up {DATABASE_FILE}") 
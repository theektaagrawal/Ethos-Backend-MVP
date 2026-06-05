import json
import os

DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "documents.json")

def load_documents():
    if not os.path.exists(DATA_FILE):
        return []
    try:
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []

def save_document(doc):
    docs = load_documents()
    # Replace if exists by filename
    docs = [d for d in docs if d.get("filename") != doc.get("filename")]
    docs.append(doc)
    
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    with open(DATA_FILE, "w") as f:
        json.dump(docs, f, indent=2)

def delete_document(doc_id):
    docs = load_documents()
    docs = [d for d in docs if d.get("id") != doc_id]
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    with open(DATA_FILE, "w") as f:
        json.dump(docs, f, indent=2)

def delete_document_by_filename(filename):
    docs = load_documents()
    docs = [d for d in docs if d.get("filename") != filename and d.get("name") != filename]
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    with open(DATA_FILE, "w") as f:
        json.dump(docs, f, indent=2)

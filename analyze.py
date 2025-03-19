from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
import json

sentiment_pipeline = pipeline("sentiment-analysis")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

class_location = "classes.json"

def load_class_file():
    try:
        with open(class_location, "r") as f:
            data = json.load(f)
        return data.get("classes", [])  # Return list of classes
    except FileNotFoundError:
        return []

#EMAIL_CLASSES = load_class_file()

def load_classes():
    with open(class_location, "r") as f:
        data = json.load(f)
    return data["classes"]

def save_classes(new_classes):
    with open(class_location, "w") as f:
        json.dump({"classes": new_classes}, f, indent=4)

def get_sentiment(text):
    response = sentiment_pipeline(text)
    return response

def compute_embeddings():
    embeddings = load_class_file()
    embeddings = model.encode(embeddings)
    return zip(load_class_file(), embeddings)

def classify_email(text):
    # Encode the input text
    text_embedding = model.encode([text])[0]
    
    # Get embeddings for all classes
    class_embeddings = compute_embeddings()
    
    # Calculate distances and return results
    results = []
    for class_name, class_embedding in class_embeddings:
        # Compute cosine similarity between text and class embedding
        similarity = np.dot(text_embedding, class_embedding) / (np.linalg.norm(text_embedding) * np.linalg.norm(class_embedding))
        results.append({
            "class": class_name,
            "similarity": float(similarity)  # Convert tensor to float for JSON serialization
        })
    
    # Sort by similarity score descending
    results.sort(key=lambda x: x["similarity"], reverse=True)
    
    return results
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import chromadb
import scanner_core
import uuid
from datetime import datetime
from pathlib import Path

app = Flask(__name__)
CORS(app)

# Setup database
db = chromadb.PersistentClient(path="./chroma_db")
try:
    collection = db.get_collection("documents")
except:
    collection = db.create_collection("documents")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    ext = Path(file.filename).suffix.lower()
    
    # Extract text based on file type
    if ext in ['.png', '.jpg', '.jpeg']:
        text = scanner_core.extract_text_from_image(file)
    elif ext == '.pdf':
        text = scanner_core.extract_text_from_pdf(file)
    elif ext == '.docx':
        text = scanner_core.extract_text_from_docx(file)
    else:
        text = file.read().decode('utf-8')
    
    # Analyze document
    analysis = scanner_core.analyze_document(text)
    
    # Create embedding
    embedding = scanner_core.create_embedding(text)
    
    # Save to database
    doc_id = str(uuid.uuid4())
    collection.add(
        embeddings=[embedding],
        documents=[text[:500]],
        metadatas=[{
            'filename': file.filename,
            'timestamp': datetime.now().isoformat()
        }],
        ids=[doc_id]
    )
    
    return jsonify({'success': True, 'data': analysis})

@app.route('/search', methods=['POST'])
def search():
    query = request.json.get('query', '')
    
    # Create query embedding
    query_vec = scanner_core.create_embedding(query)
    
    # Search database
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=5
    )
    
    # Format results
    formatted = []
    if results['ids'] and results['ids'][0]:
        for i in range(len(results['ids'][0])):
            distance = results['distances'][0][i]
            similarity = 1 / (1 + distance)  # Convert to 0-1 score
            formatted.append({
                'text': results['documents'][0][i],
                'score': round(similarity * 100, 1),
                'filename': results['metadatas'][0][i]['filename']
            })
    
    return jsonify({'results': formatted})

# This is what the notebook imports
def run_flask():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
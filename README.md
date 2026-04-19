scanNLP
Scan documents. Find anything. No setup headaches.

What it does
Upload a PDF, Word doc, or image. DocuMind reads the text, pulls out useful stuff like emails, dates, and company names, then lets you search by meaning - not just keywords.

Search "show me contracts from last month" and it actually understands.

How to run
1. Install packages
Open Jupyter notebook and run:

text
pip install flask chromadb sentence-transformers easyocr PyPDF2 python-docx pillow
2. Start the app
Run the notebook cells. The last one starts a server.

3. Open your browser
Go to http://localhost:5000

That's it.

What works
Format	Support
Images (PNG, JPG)	Yes
PDF	Yes
Word docs	Yes
Text files	Yes
What it finds
Email addresses

Phone numbers

Dates

Company names (looks for Inc, LLC, Corp, etc.)

Person names (looks for Mr., Ms., Name:, etc.)

Project files
text
DocuMind/
├── scanner_core.py    # Reads documents, finds entities
├── flask_app.py       # Web server
├── templates/
│   └── index.html     # The UI
└── chroma_db/         # Where documents are stored (auto-created)
Clear all documents
Stop the server, then delete the chroma_db folder.

Or run this before starting the server:

python
import chromadb
db = chromadb.PersistentClient(path="./chroma_db")
db.delete_collection("documents")
Need to change what it finds?
Open scanner_core.py. Look for the find_entities() function. Add or remove patterns there.

Built with
Flask - web server

ChromaDB - vector search

EasyOCR - reads images

sentence-transformers - understands meaning

Prototype status
This is a demo. Works fine for testing. Don't put it in production.

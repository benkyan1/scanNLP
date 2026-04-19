import re
import numpy as np
from PIL import Image
import docx
import PyPDF2
import pdfplumber
import easyocr
from sentence_transformers import SentenceTransformer
from gliner import GLiNER

# Initialize models
reader = easyocr.Reader(['en'], gpu=False, verbose=False)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize GLiNER - lightweight NER that can extract ANY entity types
# Use "gliner_small-v2.1" for fastest performance, "gliner_medium-v2.1" for better accuracy
gliner_model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")

class DocumentScanner:
    def extract_text_from_image(self, image_file):
        image = Image.open(image_file)
        image_np = np.array(image)
        results = reader.readtext(image_np, detail=0)
        return '\n'.join(results)
    
    def extract_text_from_pdf(self, pdf_file):
        text = ""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        except:
            pdf_file.seek(0)
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
        return text if text.strip() else "No text extracted"
    
    def extract_text_from_docx(self, docx_file):
        doc = docx.Document(docx_file)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    
    def extract_text_from_txt(self, txt_file):
        return txt_file.read().decode('utf-8')
    
    def extract_entities(self, text):
        """
        Extract entities using GLiNER.
        You can customize 'labels' to extract ANY entity types you want!
        """
        # Define what entities you want to find - YOU control this list
        # Add or remove any entity types based on your document needs
        labels = [
            "company name",
            "person name", 
            "signature",
            "date",
            "location",
            "email address",
            "phone number",
            "job title",
            "product name",
            "invoice number",
            "contract clause",
            "price",
            "organization",
            "address",
            "city",
            "country",
            "position",
            "skill",
            "qualification"
        ]
        
        # GLiNER predicts entities
        # threshold controls sensitivity (lower = more entities, higher = more precise)
        entities = gliner_model.predict_entities(text, labels, threshold=0.3)
        
        # Organize by entity type
        organized_entities = {}
        for entity in entities:
            label = entity["label"]
            text_found = entity["text"]
            
            if label not in organized_entities:
                organized_entities[label] = []
            if text_found not in organized_entities[label]:  # Remove duplicates
                organized_entities[label].append(text_found)
        
        return organized_entities
    
    def extract_pii_entities(self, text):
        """
        Specialized PII (Personally Identifiable Information) extraction.
        Uses GLiNER model fine-tuned for PII detection.
        """
        # Load PII-specialized model (first time downloads, then caches)
        try:
            pii_model = GLiNER.from_pretrained("urchade/gliner_multi_pii-v1")
            pii_labels = [
                "person", "organization", "phone number", "address", 
                "passport number", "email", "credit card number", 
                "social security number", "date of birth", "bank account number"
            ]
            entities = pii_model.predict_entities(text, pii_labels, threshold=0.3)
            
            organized = {}
            for entity in entities:
                label = entity["label"]
                text_found = entity["text"]
                if label not in organized:
                    organized[label] = []
                if text_found not in organized[label]:
                    organized[label].append(text_found)
            return organized
        except:
            return {}  # Fallback if model fails to load
    
    def extract_with_spacy_fallback(self, text):
        """
        Optional: Use spaCy as fallback for common entity types.
        Requires: pip install spacy && python -m spacy download en_core_web_sm
        """
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text[:100000])
            
            entities = {}
            for ent in doc.ents:
                if ent.label_ not in entities:
                    entities[ent.label_] = []
                if ent.text not in entities[ent.label_]:
                    entities[ent.label_].append(ent.text)
            return entities
        except:
            return {}
    
    def analyze_document(self, text):
        """Complete document analysis using GLiNER"""
        
        # Extract entities with GLiNER
        entities = self.extract_entities(text)
        
        # Additional regex extraction for common patterns (good backup)
        additional_info = {
            'emails': re.findall(r'\b[\w\.-]+@[\w\.-]+\.\w{2,}\b', text),
            'phones': re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text),
            'urls': re.findall(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', text),
            'dates': re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text)
        }
        
        # Remove duplicates
        for key in additional_info:
            additional_info[key] = list(dict.fromkeys(additional_info[key]))[:5]
        
        # Generate document summary
        summary = {
            'entities': entities,
            'additional_info': additional_info,
            'total_entities': sum(len(v) for v in entities.values()),
            'entity_types': list(entities.keys()),
            'document_preview': text[:500]
        }
        
        return summary
    
    def create_embedding(self, text):
        """Create vector embedding for search"""
        return embedding_model.encode(text[:1000]).tolist()

# Create global instance
scanner = DocumentScanner()
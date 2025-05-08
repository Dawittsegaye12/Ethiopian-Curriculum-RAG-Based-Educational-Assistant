import PyPDF2
import fitz
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sentence_transformers import SentenceTransformer
import chromadb

# Initialize spaCy and embedding model
nlp = spacy.load('en_core_web_sm')
model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.Client()
collection = client.create_collection('pdf_embeddings')

# Extract Raw Text

def extract_text_pypdf2(pdf_file_path):
    text = ''
    with open(pdf_file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text


def extract_text_pymupdf(pdf_file_path):
    doc = fitz.open(pdf_file_path)
    text = ''
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

# Clean Text

def clean_text(text):
    doc = nlp(text)
    cleaned_text = ' '.join([token.text.strip() for token in doc if not token.is_space and not token.is_punct])
    return cleaned_text.replace('\n', ' ').strip()

# Tokenize and Lemmatize Text

def tokenize_text(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]


def lemmatize_text(sentences):
    lemmatized_sentences = []
    for sent in sentences:
        doc = nlp(sent)
        lemmatized_sentences.append(' '.join([token.lemma_ for token in doc]))
    return lemmatized_sentences

# Remove Stopwords and Punctuation

def remove_stopwords(sentences):
    filtered_sentences = []
    for sent in sentences:
        doc = nlp(sent)
        filtered_tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
        filtered_sentences.append(' '.join(filtered_tokens))
    return filtered_sentences

# Filter Short Sentences

def filter_short_sentences(sentences, min_words=5):
    return [sent for sent in sentences if len(sent.split()) >= min_words]

# Chunk Sentences

def chunk_sentences(sentences, chunk_size=300):
    chunks = []
    current_chunk = ''
    for sent in sentences:
        if len(current_chunk.split()) + len(sent.split()) <= chunk_size:
            current_chunk += ' ' + sent
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sent
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Generate Embeddings

def generate_embeddings(chunks):
    return model.encode(chunks)

# Store in Vector Database

def store_embeddings(chunks, embeddings):
    for idx, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            embeddings=[embeddings[idx]],
            metadatas=[{'chunk_index': idx}]
        )

# Main Execution Flow

def main(pdf_file_path):
    # Extract raw text
    text = extract_text_pypdf2(pdf_file_path)
    # Clean text
    clean_text_data = clean_text(text)
    # Tokenize and lemmatize
    sentences = tokenize_text(clean_text_data)
    lemmatized_sentences = lemmatize_text(sentences)
    # Remove stopwords and filter short sentences
    filtered_sentences = remove_stopwords(lemmatized_sentences)
    meaningful_sentences = filter_short_sentences(filtered_sentences)
    # Chunk sentences
    chunks = chunk_sentences(meaningful_sentences)
    # Generate and store embeddings
    embeddings = generate_embeddings(chunks)
    store_embeddings(chunks, embeddings)

# Run the pipeline
if __name__ == '__main__':
    main('sample.pdf')


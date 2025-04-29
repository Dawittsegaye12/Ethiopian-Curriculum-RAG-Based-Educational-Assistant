# Ethiopian Curriculum PDF Processing and Embedding

This project processes Ethiopian curriculum textbooks in PDF format, cleans and preprocesses the text, generates embeddings using Sentence Transformers, and stores them in a vector database (ChromaDB) for efficient retrieval.

## Project Structure

- **`grade 12-chemistry_fetena_net_ef88.pdf`**: Example PDF file for Grade 12 Chemistry.
- **`grade 12-history_fetena_net_0ee1.pdf`**: Example PDF file for Grade 12 History.
- **`process_pdf_and_store_embeddings.ipynb`**: Jupyter Notebook for interactively processing PDFs, generating embeddings, and storing them in ChromaDB.
- **`process_pdf_and_store_embeddings.py`**: Python script for automating the same process as the notebook.
- **`requirements.txt`**: List of dependencies required for the project.

## Features

1. **PDF Text Extraction**:
   - Extracts text from PDF files using `PyPDF2`.

2. **Text Preprocessing**:
   - Cleans raw text (removes headers, footers, and unnecessary characters).
   - Tokenizes and lemmatizes text using `spaCy`.
   - Removes stopwords and punctuation.

3. **Chunking**:
   - Splits preprocessed text into manageable chunks for embedding.

4. **Embedding Generation**:
   - Generates vector embeddings for text chunks using `Sentence Transformers`.

5. **Vector Database Storage**:
   - Stores embeddings and metadata in a ChromaDB collection for efficient retrieval.

## Setup Instructions

1. **Install Dependencies**:
   Run the following command to install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Notebook**:
   Open `process_pdf_and_store_embeddings.ipynb` in Jupyter Notebook and execute the cells step by step.

3. **Run the Script**:
   Alternatively, run the Python script for automated processing:
   ```bash
   python process_pdf_and_store_embeddings.py
   ```

## Requirements

- Python 3.8 or higher
- Libraries:
  - `sentence-transformers`
  - `chromadb`
  - `PyPDF2`
  - `spacy`

## Notes

- Ensure that the PDF files are placed in the same directory as the script or notebook.
- The processed embeddings are stored in the `vector_db` directory for retrieval.

## Future Enhancements

- Add support for additional preprocessing steps.
- Integrate a user-friendly interface for querying the vector database.
- Expand support for more Ethiopian curriculum subjects.
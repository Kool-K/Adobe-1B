import json
import os
import pdfplumber
from sentence_transformers import SentenceTransformer, util

def chunk_pdf_by_structure(pdf_path, outline_data):
    """
    Extracts text chunks from a PDF based on the structural outline (headings).
    """
    print(f"\n📄 Reading and chunking PDF: {pdf_path}")
    try:
        with pdfplumber.open(pdf_path) as pdf:
            full_text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    except Exception as e:
        print(f"❌ Error reading PDF file: {e}")
        return []

    chunks = []
    headings = outline_data.get('outline', [])
    headings.sort(key=lambda x: x.get('page', 0))

    for i, heading in enumerate(headings):
        cleaned_heading_text = heading['text'].split('.')[0].strip()
        heading_text = cleaned_heading_text

        start_index = full_text.find(heading_text)
        if start_index == -1:
            continue

        end_index = len(full_text)
        if i + 1 < len(headings):
            next_heading_text = headings[i+1]['text'].split('.')[0].strip()
            next_heading_index = full_text.find(next_heading_text, start_index + len(heading_text))
            if next_heading_index != -1:
                end_index = next_heading_index

        chunk_content = full_text[start_index : end_index].strip()
        chunks.append({
            "heading": heading_text,
            "content": chunk_content,
            "page": heading.get('page')
        })
    
    print(f"✅ Created {len(chunks)} chunks.")
    return chunks

# --- Main execution ---
if __name__ == "__main__":
    # --- 1. CONFIGURATION ---
    PDF_FILE_PATH = os.path.join('test_files', 'sample.pdf')
    OUTLINE_FILE_PATH = os.path.join('test_files', 'sample_outline.json')
    USER_QUERY = "What is the syntax for creating a link in Markdown?"
    
    # --- THIS IS THE MODIFIED PART ---
    # Instead of a name, we now point to the local folder where the model is saved.
    MODEL_PATH = './local_minilm_model' 
    
    # --- 2. LOAD & PARSE OUTLINE ---
    print(f"Loading outline from: {OUTLINE_FILE_PATH}")
    try:
        with open(OUTLINE_FILE_PATH, 'r', encoding='utf-8') as f:
            all_labeled_data = json.load(f)

        title_text = "No Title Found"
        outline_headings = []
        for item in all_labeled_data:
            if item.get('label_type') == 'TITLE':
                title_text = item.get('text')
            elif item.get('label_type') in ['H1', 'H2', 'H3', 'H4']:
                outline_headings.append({
                    "text": item.get('text'), "page": item.get('page')
                })
        outline_data = {"title": title_text, "outline": outline_headings}
        print("✅ Outline data loaded and parsed successfully.")

        # --- 3. CREATE CHUNKS ---
        document_chunks = chunk_pdf_by_structure(PDF_FILE_PATH, outline_data)
        
        if document_chunks:
            # --- 4. LOAD MODEL & GET RANKINGS ---
            # --- AND WE USE THE NEW VARIABLE HERE ---
            print(f"\n🧠 Loading model from local path: '{MODEL_PATH}'...")
            model = SentenceTransformer(MODEL_PATH)
            
            chunk_contents = [chunk['content'] for chunk in document_chunks]
            
            print("🔍 Converting query and chunks to vectors...")
            query_embedding = model.encode(USER_QUERY, convert_to_tensor=True)
            chunk_embeddings = model.encode(chunk_contents, convert_to_tensor=True)

            print("📊 Calculating relevance scores...")
            cosine_scores = util.pytorch_cos_sim(query_embedding, chunk_embeddings)
            
            for i, score in enumerate(cosine_scores[0]):
                document_chunks[i]['relevance_score'] = score.item()

            ranked_chunks = sorted(document_chunks, key=lambda x: x['relevance_score'], reverse=True)

            # --- 5. DISPLAY RESULTS ---
            print("\n--- Top 5 Most Relevant Sections ---")
            for chunk in ranked_chunks[:5]:
                print(f"  - Score: {chunk['relevance_score']:.4f} | Page: {chunk['page']} | Heading: {chunk['heading']}")

    except FileNotFoundError:
        print(f"❌ Error: Make sure input files exist.")
    except Exception as e:
        print(f"An error occurred: {e}")
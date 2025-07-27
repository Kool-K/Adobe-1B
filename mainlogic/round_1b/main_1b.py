import json
import os
import pdfplumber
from sentence_transformers import SentenceTransformer, util
import sys
import datetime

def chunk_pdf_by_structure(pdf_path, outline_data):
    """Extracts text chunks from a PDF based on the structural outline (headings)."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            full_text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    except Exception as e:
        print(f"❌ Error reading PDF file {pdf_path}: {e}")
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
            "section_title": heading_text,
            "refined_text": chunk_content,
            "page_number": heading.get('page'),
            "document_filename": os.path.basename(pdf_path)
        })
    
    return chunks

# --- Main execution ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main_1b.py <path_to_input_json>")
        sys.exit(1)
    
    master_input_path = sys.argv[1]
    
    try:
        with open(master_input_path, 'r', encoding='utf-8') as f:
            master_input_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Master input file not found at '{master_input_path}'")
        sys.exit(1)

    # --- 1. EXTRACT INFORMATION FROM MASTER JSON ---
    persona = master_input_data.get('persona', {})
    job_to_be_done = master_input_data.get('job_to_be_done', {})
    documents_to_process = master_input_data.get('documents', [])
    user_query = f"As a {persona.get('role', 'User')}, I need to {job_to_be_done.get('task', '')}."
    print(f"🔍 Running query: \"{user_query}\"")

    # --- 2. SET UP FILE PATHS BASED ON YOUR STRUCTURE ---
    PDF_DIR = "mainlogic/round_1b/test_files/pdfs"
    OUTLINE_DIR = "mainlogic/round_1b/test_files/outlines"
    MODEL_PATH = './mainlogic/local_minilm_model'
    
    # --- 3. GATHER AND CHUNK ALL DOCUMENTS ---
    all_document_chunks = []
    for doc_info in documents_to_process:
        pdf_filename = doc_info.get('filename')
        if not pdf_filename:
            continue
            
        pdf_path = os.path.join(PDF_DIR, pdf_filename)
        outline_filename = pdf_filename.replace('.pdf', '_outline.json') # Your naming convention
        outline_path = os.path.join(OUTLINE_DIR, outline_filename)

        if not os.path.exists(pdf_path) or not os.path.exists(outline_path):
            print(f"⚠️ Warning: Missing PDF or outline for {pdf_filename}. Skipping.")
            continue
        
        with open(outline_path, 'r', encoding='utf-8') as f:
            raw_json = json.load(f)
            # This logic handles your hand-labeled JSON format
            title_text = next((item.get('text') for item in raw_json if item.get('label_type') == 'TITLE'), "N/A")
            outline_headings = [{"text": item.get('text'), "page": item.get('page')} for item in raw_json if item.get('label_type') in ['H1', 'H2', 'H3', 'H4']]
            outline_data = {"title": title_text, "outline": outline_headings}

        all_document_chunks.extend(chunk_pdf_by_structure(pdf_path, outline_data))
    
    if not all_document_chunks:
        print("❌ No chunks were created. Exiting.")
        sys.exit(1)

    # --- 4. LOAD MODEL, EMBED, AND RANK ---
    print(f"\n🧠 Loading model from local path: '{MODEL_PATH}'...")
    model = SentenceTransformer(MODEL_PATH)
    
    chunk_contents = [chunk['refined_text'] for chunk in all_document_chunks]
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    chunk_embeddings = model.encode(chunk_contents, convert_to_tensor=True)

    cosine_scores = util.pytorch_cos_sim(query_embedding, chunk_embeddings)
    
    for i, score in enumerate(cosine_scores[0]):
        all_document_chunks[i]['relevance_score'] = score.item()

    ranked_chunks = sorted(all_document_chunks, key=lambda x: x['relevance_score'], reverse=True)

    # --- 5. PRODUCE FINAL JSON OUTPUT IN THE REQUIRED FORMAT ---
    timestamp = datetime.datetime.now().isoformat()
    
    final_output = {
        "metadata": {
            "input_documents": documents_to_process,
            "persona": persona,
            "job_to_be_done": job_to_be_done,
            "processing_timestamp": timestamp
        },
        "extracted_sections": [],
        "subsection_analysis": []
    }

    # Populate the lists based on the ranked chunks
    for rank, chunk in enumerate(ranked_chunks):
        final_output["extracted_sections"].append({
            "document": chunk["document_filename"],
            "section_title": chunk["section_title"],
            "importance_rank": rank + 1,
            "page_number": chunk["page_number"]
        })
        final_output["subsection_analysis"].append({
            "document": chunk["document_filename"],
            "refined_text": chunk["refined_text"],
            "page_number": chunk["page_number"]
        })
        
    output_filename = "1b_final_output.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
        
    print(f"\n✅ Successfully saved final results to '{output_filename}'")
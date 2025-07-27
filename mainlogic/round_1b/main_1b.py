import json
import os
import pdfplumber
from sentence_transformers import SentenceTransformer, util
import sys

def chunk_pdf_by_structure(pdf_path, outline_data):
    """
    Extracts text chunks from a PDF based on the structural outline (headings).
    """
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
            "heading": heading_text,
            "content": chunk_content,
            "page": heading.get('page'),
            "source_pdf": os.path.basename(pdf_path) # Track the source file
        })
    
    print(f"✅ Created {len(chunks)} chunks from {os.path.basename(pdf_path)}.")
    return chunks

# --- Main execution ---
if __name__ == "__main__":
    # --- 1. PARSE INPUTS from command line ---
    if len(sys.argv) < 5:
        print("Usage: python main_1b.py <path_to_pdfs> <path_to_outlines> \"<Persona>\" \"<Job-to-be-Done>\"")
        print("Example: python main_1b.py ./pdfs ./outlines \"Investment Analyst\" \"Analyze revenue trends and market positioning\"")
        sys.exit(1)

    pdf_dir = sys.argv[1]
    outline_dir = sys.argv[2]
    persona = sys.argv[3]
    job_to_be_done = sys.argv[4]

    # Combine persona and job for a richer query
    user_query = f"As a {persona}, I need to {job_to_be_done}."
    print(f"🔍 Running query: \"{user_query}\"")

    # --- 2. CONFIGURATION ---F
    MODEL_PATH = './mainlogic/local_minilm_model'    
    # --- 3. GATHER AND CHUNK ALL DOCUMENTS ---
    all_document_chunks = []
    print(f"\nLooking for PDFs in '{pdf_dir}' and outlines in '{outline_dir}'...")

    for filename in os.listdir(pdf_dir):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_dir, filename)
            # Assumes a naming convention like 'doc1.pdf' and 'doc1_outline.json'
            outline_filename = filename.replace('.pdf', '_outline.json') 
            outline_path = os.path.join(outline_dir, outline_filename)

            if not os.path.exists(outline_path):
                print(f"⚠️ Warning: No matching outline found for {filename}. Skipping.")
                continue
            
            with open(outline_path, 'r', encoding='utf-8') as f:
                # This needs to be robust to the two types of JSONs we've discussed
                raw_json = json.load(f)
                if isinstance(raw_json, list): # It's the labeled data format
                    title_text = next((item.get('text') for item in raw_json if item.get('label_type') == 'TITLE'), "N/A")
                    outline_headings = [{"text": item.get('text'), "page": item.get('page')} for item in raw_json if item.get('label_type') in ['H1', 'H2', 'H3', 'H4']]
                    outline_data = {"title": title_text, "outline": outline_headings}
                else: # It's the simple {title, outline} format
                    outline_data = raw_json

            all_document_chunks.extend(chunk_pdf_by_structure(pdf_path, outline_data))
    
    if not all_document_chunks:
        print("❌ No chunks were created from any of the documents. Exiting.")
        sys.exit(1)

    # --- 4. LOAD MODEL & GET RANKINGS ---
    print(f"\n🧠 Loading model from local path: '{MODEL_PATH}'...")
    model = SentenceTransformer(MODEL_PATH)
    
    chunk_contents = [chunk['content'] for chunk in all_document_chunks]
    
    print("VECTORIZING: Converting query and all chunks to vectors...")
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    chunk_embeddings = model.encode(chunk_contents, convert_to_tensor=True)

    print("📊 Calculating relevance scores...")
    cosine_scores = util.pytorch_cos_sim(query_embedding, chunk_embeddings)
    
    for i, score in enumerate(cosine_scores[0]):
        all_document_chunks[i]['relevance_score'] = score.item()

    ranked_chunks = sorted(all_document_chunks, key=lambda x: x['relevance_score'], reverse=True)

    # --- 5. PRODUCE FINAL JSON OUTPUT (Example) ---
    # This part needs to be formatted to the *exact* required output spec from the hackathon
    final_output = {
        "metadata": {
            "persona": persona,
            "job_to_be_done": job_to_be_done,
        },
        "results": [
            {
                "source_pdf": chunk['source_pdf'],
                "page": chunk['page'],
                "section_title": chunk['heading'],
                "importance_rank": rank + 1,
                "score": chunk['relevance_score'],
                "refined_text": chunk['content'][:500] + "..." # Truncating for preview
            } for rank, chunk in enumerate(ranked_chunks[:10]) # Get top 10 results
        ]
    }

    output_filename = "1b_final_output.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4)
        
    print(f"\n✅ Successfully saved final results to '{output_filename}'")
import os, json, datetime, sys
from pathlib import Path
import pdfplumber
from sentence_transformers import SentenceTransformer, util

def extract_structured_chunks(pdf_path, outline_data):
    with pdfplumber.open(pdf_path) as pdf:
        page_texts = [page.extract_text() or "" for page in pdf.pages]

    outline = sorted(outline_data.get('outline', []), key=lambda x: (x['page'], x['text']))

    chunks = []
    for i, heading in enumerate(outline):
        start_page = heading['page'] - 1
        section_title = heading['text'].strip()

        end_page = len(page_texts) - 1
        if i + 1 < len(outline):
            end_page = outline[i + 1]['page'] - 1

        content = "\n".join(page_texts[start_page:end_page + 1]).strip()
        if section_title in content:
            start_idx = content.find(section_title) + len(section_title)
            content = content[start_idx:].strip()

        if len(content) > 20:
            chunks.append({
                "section_title": section_title,
                "refined_text": content,
                "page_number": heading['page'],
                "document_filename": os.path.basename(pdf_path)
            })
    return chunks

def main():
    if len(sys.argv) < 2:
        print("Usage: python main_1b.py input.json")
        return

    input_path = sys.argv[1]
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    persona = data["persona"]
    job = data["job_to_be_done"]
    user_query = f"As a {persona['role']}, I need to {job['task']}."

    PDF_DIR = "mainlogic/round_1b/test_files/pdfs"
    OUTLINE_DIR = "mainlogic/round_1b/test_files/outlines"
    MODEL_PATH = "./mainlogic/local_minilm_model"

    model = SentenceTransformer(MODEL_PATH)

    all_chunks = []
    for doc in data["documents"]:
        pdf_name = doc["filename"]
        outline_name = pdf_name.replace(".pdf", "_outline.json")

        pdf_path = os.path.join(PDF_DIR, pdf_name)
        outline_path = os.path.join(OUTLINE_DIR, outline_name)

        if not os.path.exists(pdf_path) or not os.path.exists(outline_path):
            print(f"⚠️ Skipping missing: {pdf_name}")
            continue

        with open(outline_path, 'r', encoding='utf-8') as f:
            outline_raw = json.load(f)
            title = next((x["text"] for x in outline_raw if x["label_type"] == "TITLE"), "")
            headings = [x for x in outline_raw if x["label_type"] in ["H1", "H2", "H3", "H4"]]
            outline = {"title": title, "outline": headings}

        chunks = extract_structured_chunks(pdf_path, outline)
        all_chunks.extend(chunks)

    if not all_chunks:
        print("❌ No valid chunks found.")
        return

    print("🧠 Ranking chunks...")
    query_embed = model.encode(user_query, convert_to_tensor=True)
    chunk_texts = [chunk['refined_text'] for chunk in all_chunks]
    chunk_embeds = model.encode(chunk_texts, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embed, chunk_embeds)

    for i, score in enumerate(scores[0]):
        all_chunks[i]["relevance_score"] = score.item()

    ranked = sorted(all_chunks, key=lambda x: x["relevance_score"], reverse=True)

    timestamp = datetime.datetime.now().isoformat()
    output = {
        "metadata": {
            "input_documents": [d["filename"] for d in data["documents"]],
            "persona": persona["role"],
            "job_to_be_done": job["task"],
            "processing_timestamp": timestamp
        },
        "extracted_sections": [],
        "subsection_analysis": []
    }

    for rank, chunk in enumerate(ranked[:5]):
        output["extracted_sections"].append({
            "document": chunk["document_filename"],
            "section_title": chunk["section_title"],
            "importance_rank": rank + 1,
            "page_number": chunk["page_number"]
        })
        output["subsection_analysis"].append({
            "document": chunk["document_filename"],
            "refined_text": chunk["refined_text"],
            "page_number": chunk["page_number"]
        })

    with open("1b_final_output.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print("✅ Final output saved to 1b_final_output.json")

if __name__ == "__main__":
    main()

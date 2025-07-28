# Adobe India Hackathon 2025: Round 1B - Persona-Driven Document Intelligence

This project is a solution for Round 1B of the "Connecting the Dots" Hackathon. It acts as an intelligent document analyst, extracting and prioritizing the most relevant sections from a collection of documents based on a specific persona and their job-to-be-done.

## 🚀 Features

* **Multi-Document Analysis:** Processes a collection of 3-10 PDF documents simultaneously to find the most relevant information.
* **Structural Chunking:** Leverages the structural outline (headings) from Round 1A to create semantically coherent text chunks for analysis.
* **Semantic Search:** Uses a pre-trained sentence-transformer model (`all-MiniLM-L6-v2`) to understand the meaning of the user's query and the document chunks.
* **Relevance Ranking:** Calculates the cosine similarity between the user query and all text chunks to rank the results by importance.
* **Offline Execution:** The entire process, including loading the AI model, runs completely offline to comply with the hackathon constraints.
* **Flexible Input:** Accepts a master JSON file as input, making it easy for judges to test with different document collections and queries.

## 📁 File Structure

For the script to run correctly, your project should be organized with the following folder structure:
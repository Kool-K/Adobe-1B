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

## ⚙️ Setup Instructions

To set up and run this project, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone <your_repository_url>
    cd <repository_name>
    ```

2.  **Create and Activate Virtual Environment:**
    ```bash
    # Create the environment
    python -m venv venv

    # Activate on Windows
    .\venv\Scripts\activate

    # Activate on Mac/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    All required Python libraries are listed in `requirements.txt`. Install them with:
    ```bash
    pip install -r requirements.txt
    ```

## ▶️ How to Run

The script is executed from the command line and takes the path to a master input JSON file as its only argument.

1.  **Prepare Inputs:**
    * Place all the required PDF documents into the `source_pdfs/` folder.
    * Place their corresponding JSON outlines (from Round 1A) into the `labeled_json_original/` folder. Ensure the outline file names match the PDF names (e.g., `mydoc.pdf` and `mydoc_labels.json`).
    * Create a master `input.json` file in the root directory that lists the documents to be processed, the persona, and the job-to-be-done.

2.  **Execute the Script:**
    Run the script from the root project directory, passing the path to your master input file:
    ```bash
    python mainlogic/round_1b/main_1b.py input.json
    ```

## 📝 Input and Output Format

### Input (`input.json`)

The script expects a single master JSON file as input, which defines the entire task.

* **`documents`**: A list of objects, where each object specifies the `filename` of a PDF to be processed.
* **`persona`**: An object containing the `role` of the user.
* **`job_to-be_done`**: An object containing the `task` the user wants to accomplish.

*(You can include the sample `input.json` we created here as an example).*

### Output (`1b_final_output.json`)

The script will generate a single output file named `1b_final_output.json` in the root directory. This file contains the ranked list of extracted sections and subsections that are most relevant to the input query, formatted according to the official hackathon specifications.
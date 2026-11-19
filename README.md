# SimilarityX - Text Similarity & Duplicate Detection System

SimilarityX is a Streamlit application that compares two sentences or questions
using traditional TF-IDF vectorization and semantic sentence embeddings. It
classifies the relationship as **Duplicate**, **Similar**, or **Different**.

## Features

- Side-by-side text comparison
- Character counters and input validation
- TF-IDF cosine similarity
- Semantic cosine similarity using `all-MiniLM-L6-v2`
- Configurable duplicate and similarity thresholds
- Session comparison history
- Plotly similarity visualization
- Text comparison, methodology, and project pages
- Responsive custom CSS

## Technology

- Python 3.12
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Sentence Transformers
- Plotly
- Pytest

## Architecture

```text
SimilarityX/
|-- app.py                 # Streamlit user interface
|-- requirements.txt        # Deployment dependencies
|-- README.md
|-- .gitignore
|-- assets/style.css        # Custom application styling
|-- core/
|   |-- preprocessing.py    # Text normalization
|   |-- tfidf_model.py      # TF-IDF cosine similarity
|   `-- semantic_model.py   # Sentence Transformer embeddings
|-- utils/similarity.py     # Unified comparison and classification
`-- tests/
    |-- test_similarity.py
    `-- test_app.py
```

## Classification Model

The initial threshold defaults are:

- Duplicate: semantic score >= 85%
- Similar: semantic score >= 60% and < 85%
- Different: semantic score < 60%

These thresholds are demonstration defaults and are not scientifically calibrated
probabilities.

## Local Installation

### Windows

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the URL printed by Streamlit, normally `http://localhost:8501`.

## Usage

1. Enter a sentence or question in both text panels.
2. Optionally use an example.
3. Select **Compare text**.
4. Review semantic similarity, TF-IDF similarity, classification, timestamp,
   explanation, and chart.
5. Open **Comparison History** to review current-session comparisons.

## Model Behavior

SimilarityX uses `sentence-transformers/all-MiniLM-L6-v2`. The model is downloaded
by Sentence Transformers on first use and cached by Streamlit and the Hugging Face
runtime. No model weights are stored in this repository. Initial first-use
downloads require network access.

## Testing

```bash
python -m pytest -q
```

The suite covers preprocessing, TF-IDF, semantic integration, classification,
invalid input, repeated comparisons, and Streamlit application smoke behavior.

## Screenshots

Add screenshots to this section after deployment or local review:

- Application text comparison page
- Comparison result and chart
- Comparison history
- Methodology page

## GitHub Upload

1. Initialize Git in the project directory if not already initialized.
2. Review `git status` and inspect untracked files.
3. Add and commit the source files.
4. Create an empty GitHub repository.
5. Add the GitHub repository as `origin`.
6. Push the main branch.

Example:

```bash
git init
git add .
git status
git commit -m "Initial SimilarityX application"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Do not run the push command until you have confirmed the repository name and
authentication details.

## Streamlit Community Cloud

1. Push the project to GitHub.
2. Open Streamlit Community Cloud.
3. Choose **Create app**.
4. Select the GitHub repository and main branch.
5. Set the entry point to `app.py`.
6. Deploy the app.

Streamlit Community Cloud should install packages from `requirements.txt`.
The app does not require secrets for normal operation.

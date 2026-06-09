# AI Candidate Ranking System for India Runs Hackathon

This repository contains the full source code for the **AI Candidate Ranking System**, developed for **Track 1: The Data & AI Challenge** of the India Runs Hackathon (Prize: 10 Lakhs).

The system uses a **2-Stage Local Retrieval and Ranking Pipeline** designed to handle 50,000+ candidates under strict performance limitations (<6 seconds per job, <4GB RAM) using **zero network / external API calls** (running completely locally).

---

## 2-Stage Local Architecture

```
           +-----------------------------------------+
           |       Candidates Pool (50k Resumes)     |
           +-----------------------------------------+
                                |
                                v
       +---------------------------------------------------+
       | Stage 1: Dense Retrieval (Sentence-BERT + FAISS)  |
       |          Retrieves top 200 candidates per job     |
       +---------------------------------------------------+
                                |
                                v
       +---------------------------------------------------+
       | Stage 2: XGBoost Ranker (15-feature model)        |
       |          Ranks and selects top 100 candidates     |
       +---------------------------------------------------+
                                |
                                v
           +-----------------------------------------+
           |       submission.csv (Final Rankings)   |
           +-----------------------------------------+
```

### Fallback Chain
- **Stage 1 (FAISS)** fails -> Falls back to standard **TF-IDF Dense/Sparse Retrieval**.
- **Stage 2 (XGBoost)** fails -> Falls back to **Weighted Heuristic Scoring** (Platform Activity, Experience Match, Skill Match, Education Match).

---

## One-Command Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Getting Started (Hackathon Participant Bundle)

### 1) Read the official docs (recommended order)
Use the files from:
`[PUB] India_runs_data_and_ai_challenge/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/`
- `job_description.txt`
- `submission_spec.txt`
- `redrob_signals_doc.txt`
- `candidate_schema.json`

### 2) Prepare the candidate pool
The official bundle provides `candidates.jsonl.gz` (100,000 candidates).

Option A (unpack locally):
```bash
gunzip -k candidates.jsonl.gz
# creates candidates.jsonl (100000 lines)
```

Option B (load gz directly in Python):
```python
import gzip, json
with gzip.open("candidates.jsonl.gz","rt",encoding="utf-8") as f:
    candidates = [json.loads(line) for line in f if line.strip()]
print(len(candidates))
```

### 3) Build your ranker (produce `top 100` + reasoning)
Run this repo’s pipeline to generate a submission CSV.

> Output CSV must contain **exactly** these columns (header order matters):  
`candidate_id,rank,score,reasoning`

Example (JSONL/GZ/CSV supported for candidates):
```bash
python rank.py --candidates ./candidates.jsonl.gz --out ./submission.csv
```

If your dataset uses `candidates.csv` + `jobs.csv` in a directory:
```bash
python run.py --data_path ./data --output ./submission.csv
```

---

## Data Formats

### Expected Input Format

The pipeline expects two files in the input directory (`./data`):
1. **`jobs.csv`**: Contains job descriptions.
   - `job_id`: Unique identifier for the job.
   - `title`: Job title.
   - `description`: Detailed job description including required skills, education, and experience.
   
2. **`candidates.csv`** (or `candidates.jsonl` / `candidates.jsonl.gz`): Contains candidate resumes.
   - `candidate_id`: Unique identifier for the candidate.
   - `resume_text`: Full text content of the candidate's resume.
   - `education`: (Optional) College/degree details.
   - `experience`: (Optional) Raw experience years or details.
   - `location`: (Optional) City or state of residence.
   - `platform_activity`: (Optional) Activity score (0.0 to 1.0) on the hosting site.

### Expected Output Format

The output file **`<registered_participant_id>.csv`** (e.g. `team_xxx.csv` or `submission.csv`) must contain the final sorted ranks in exactly 100 data rows:
- `candidate_id`: The identifier of the ranked candidate (matching `CAND_XXXXXXX` format).
- `rank`: Contiguous rank number from 1 to 100 (where 1 is the best-fit candidate).
- `score`: Monotonically non-increasing similarity score (float).
- `reasoning`: A text reason summarizing the candidate suitability.

The CSV must contain exactly 100 data rows following a 1-row header.

---

## Performance and Constraints

- **Execution Speed**: Less than 6 seconds per job description.
- **Resource Constraints**: Consumes less than 4GB RAM, suitable for local development/deployment.
- **Scale**: Handles databases with 50,000+ candidates efficiently via Stage 1 filtering.
- **Local First**: Sentence-BERT embeddings, FAISS indices, and XGBoost models are run completely offline with no network dependencies.

---

## Troubleshooting

- **FAISS Installation Errors**:
  - On Windows, if `faiss-cpu` fails to build, make sure you have the Microsoft Visual C++ Redistributable installed. Alternatively, run `pip install faiss-cpu` in a Conda environment.
  
- **CUDA/GPU Out of Memory**:
  - The pipeline runs on CPU by default to maintain the `<4GB RAM` constraint. Force CPU mode by setting `CUDA_VISIBLE_DEVICES=""` if your GPU runs out of VRAM.

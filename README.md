# 🤖 AI Candidate Ranking System — India Runs Hackathon

> **Track 1: The Data & AI Challenge** · Prize: ₹10 Lakhs  
> **Team:** Smarpit — AI Ranker · **Contact:** smarpitmalik@gmail.com

A production-grade **2-Stage Local Retrieval & Ranking Pipeline** that identifies the top 100 candidates from a 50,000+ candidate pool — with **zero network calls**, under **6 seconds per job**, and within **4 GB RAM**.

---

## 📐 Architecture

```
            ┌─────────────────────────────────────────────┐
            │       Candidates Pool  (50k+ Resumes)       │
            └─────────────────────┬───────────────────────┘
                                  │
                                  ▼
        ┌─────────────────────────────────────────────────┐
        │  Stage 1 — Dense Retrieval                      │
        │  Sentence-BERT (all-MiniLM-L6-v2) + FAISS      │
        │  Retrieves top 200 candidates per job           │
        └─────────────────────────┬───────────────────────┘
                                  │
                                  ▼
        ┌─────────────────────────────────────────────────┐
        │  Stage 2 — XGBoost Pairwise Ranker              │
        │  15-feature model trained on synthetic labels    │
        │  Ranks and selects top 100 candidates           │
        └─────────────────────────┬───────────────────────┘
                                  │
                                  ▼
            ┌─────────────────────────────────────────────┐
            │  submission.csv / submission.xlsx            │
            │  (candidate_id, rank, score, reasoning)     │
            └─────────────────────────────────────────────┘
```

### Fallback Chain

| Stage | Primary | Fallback |
|-------|---------|----------|
| Stage 1 (Retrieval) | Sentence-BERT + FAISS dense retrieval | TF-IDF sparse retrieval |
| Stage 2 (Ranking) | XGBoost pairwise ranker | Weighted heuristic scoring (Platform Activity, Experience, Skills, Education) |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline

```bash
# Primary entry point (JSONL / JSONL.GZ / CSV supported)
python rank.py --candidates ./candidates.jsonl.gz --out ./output/submission.csv

# Alternative entry point (data directory mode)
python run.py --data_path ./data --output ./output/submission.csv
```

### 3. CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--candidates` | *required* | Path to candidates file (`.jsonl`, `.jsonl.gz`, `.json`, `.csv`) |
| `--out` | *required* | Output path for the ranked CSV |
| `--jobs` | `data/jobs.csv` | Path to jobs CSV |
| `--no_llm` | `false` | Disable Stage 3 LLM re-ranking |
| `--build_index` | `false` | Force rebuild of FAISS index |
| `--train_ranker` | `false` | Force retrain of XGBoost model |

---

## 📁 Project Structure

```
hackathonpresent/
├── rank.py                    # CLI entry point (primary)
├── run.py                     # CLI entry point (data-directory mode)
├── requirements.txt           # Python dependencies
├── submission_metadata.yaml   # Hackathon submission metadata
├── context.txt                # Job description context
├── pytest.ini                 # Test configuration
│
├── src/                       # Core pipeline modules
│   ├── config.py              # Paths, hyperparameters, skill/education/location maps
│   ├── pipeline.py            # End-to-end orchestrator (Stage 1 → Stage 2)
│   ├── embeddings.py          # Sentence-BERT encoding + FAISS index build/load/query
│   ├── features.py            # 15-feature extraction engine
│   ├── ranker.py              # XGBoost training, loading, and prediction
│   ├── synthetic_labels.py    # Synthetic training data generation
│   ├── preprocessing.py       # Text cleaning, skill aliasing, education/location normalization
│   ├── data_loader.py         # Multi-format candidate data loading
│   ├── job_description.py     # JD parsing utilities
│   └── reasoning_generator.py # Human-readable reasoning text per candidate
│
├── tests/                     # Comprehensive test suite
│   ├── conftest.py            # Shared fixtures
│   ├── test_config.py         # Config sanity checks
│   ├── test_features.py       # Feature extraction unit tests
│   ├── test_preprocessing.py  # Text processing unit tests
│   ├── test_ranker.py         # XGBoost ranker tests
│   ├── test_synthetic_labels.py # Synthetic data generation tests
│   ├── test_pipeline.py       # Pipeline integration tests
│   ├── test_hardening.py      # Edge-case & robustness tests
│   └── test_e2e.py            # End-to-end tests
│
├── data/                      # Input data (candidates, jobs, synthetic training)
├── models/                    # Persisted artefacts (FAISS index, XGBoost model)
├── output/                    # Pipeline outputs
│   ├── submission.csv         # Final ranked CSV
│   └── submission.xlsx        # Spreadsheet version
└── cache/                     # Intermediate caches
```

---

## 🧠 The 15-Feature Model

The XGBoost ranker is trained on **15 engineered features** that capture multi-dimensional candidate–job fit:

| # | Feature | Description |
|---|---------|-------------|
| 1 | **Skill Match Count** | Number of JD-required skills found in resume (alias-aware) |
| 2 | **Skill Match Ratio** | Fraction of JD skills matched |
| 3 | **Candidate Experience** | Parsed years of experience (with resume-text fallback) |
| 4 | **Experience Match** | Step-function score comparing candidate vs required experience |
| 5 | **Education Match** | Education level comparison with college-tier and activity bonuses |
| 6 | **Location Match** | Geo proximity (exact city → metro region → national → international) |
| 7 | **Semantic Similarity** | FAISS cosine similarity from Sentence-BERT embeddings |
| 8 | **Platform Activity** | Normalized engagement score (0–1) |
| 9 | **Resume Completeness** | Fraction of key profile fields filled |
| 10 | **Career Progression** | Title seniority × experience years |
| 11 | **Resume Length** | Word count of resume text |
| 12 | **Keyword Density** | Non-stopword JD term overlap ratio |
| 13 | **Section Completeness** | Presence of standard resume sections |
| 14 | **Project Diversity** | Count of distinct project domains mentioned |
| 15 | **Title Match** | Lexical overlap between JD title and candidate's current title |

### Synthetic Training Data

Training labels are generated synthetically using a weighted composite score:

| Weight | Signal |
|--------|--------|
| 0.35 | Skill Match |
| 0.30 | Semantic Similarity |
| 0.20 | Experience Match |
| 0.15 | Platform Activity |

---

## 🇮🇳 India-Specific Handling

- **Hinglish Education Degrees**: Recognizes Hindi-script degree names (बीटेक, एमटेक, पीएचडी) alongside English equivalents
- **Indian City Normalization**: Maps 30+ Indian city aliases (Bengaluru↔Bangalore, Bombay↔Mumbai, Gurugram↔Gurgaon, etc.)
- **NCR Metro Region**: Delhi, Gurgaon, and Noida treated as same metro for location scoring
- **College Tiers**: IIT/NIT/BITS (Tier 1), VIT/SRM/Manipal (Tier 2) with score bonuses
- **Tier-2/3 Activity Bonus**: High-activity candidates from non-Tier-1 colleges get a +0.05 boost
- **Consulting vs Product Company Detection**: Identifies company backgrounds (TCS, Infosys vs Swiggy, Razorpay)
- **Honeypot Filtering**: Detects and filters honeypot candidates to ensure zero honeypot rate

---

## 📊 Output Format

### CSV (`submission.csv`)

| Column | Type | Description |
|--------|------|-------------|
| `candidate_id` | string | `CAND_XXXXXXX` format identifier |
| `rank` | integer | 1–100 (1 = best fit) |
| `score` | float | Monotonically non-increasing similarity score |
| `reasoning` | string | Human-readable suitability explanation |

### Excel (`submission.xlsx`)

Same data as above, exported as an Excel spreadsheet for easy viewing and sharing.

---

## 🧪 Testing

Run the full test suite:

```bash
pytest tests/ -v
```

| Test File | Coverage |
|-----------|----------|
| `test_config.py` | Config constants, weight invariants, path existence |
| `test_features.py` | All 15 feature extraction functions |
| `test_preprocessing.py` | Text cleaning, skill aliasing, education/location normalization |
| `test_ranker.py` | XGBoost train/predict/save/load cycle |
| `test_synthetic_labels.py` | Synthetic label generation and thresholds |
| `test_pipeline.py` | Pipeline integration (index build → rank → validate) |
| `test_hardening.py` | Edge cases, missing data, malformed inputs |
| `test_e2e.py` | End-to-end ranking pipeline |

---

## ⚡ Performance & Constraints

| Metric | Target | Achieved |
|--------|--------|----------|
| Execution speed | < 6 sec / job | ✅ |
| RAM usage | < 4 GB | ✅ |
| Candidate scale | 50,000+ | ✅ |
| Network calls during ranking | 0 | ✅ |
| GPU required | No | ✅ (CPU-only) |

---

## 🔧 Dependencies

```
pandas==3.0.3
numpy==2.4.6
scikit-learn==1.9.0
xgboost==3.2.0
sentence-transformers==5.5.1
faiss-cpu==1.14.2
PyPDF2==3.0.1
pdfplumber==0.11.9
python-dotenv==1.2.2
pytest==9.0.3
```

---

## 🛠 Troubleshooting

### FAISS Installation Errors
On Windows, if `faiss-cpu` fails to build, ensure the **Microsoft Visual C++ Redistributable** is installed. Alternatively, install inside a Conda environment:
```bash
conda install -c conda-forge faiss-cpu
```

### CUDA / GPU Out of Memory
The pipeline runs on **CPU by default**. Force CPU mode if GPU VRAM is insufficient:
```bash
CUDA_VISIBLE_DEVICES="" python rank.py --candidates ./candidates.jsonl.gz --out ./submission.csv
```

### openpyxl for Excel Export
To generate `.xlsx` output, install openpyxl:
```bash
pip install openpyxl
```

---

## 📜 License

Built for the India Runs Hackathon by **Smarpit**. All code is original work.

## 🔗 Links

- **GitHub**: [github.com/smarpitm/Hack2skill](https://github.com/smarpitm/Hack2skill)
- **Sandbox**: [huggingface.co/spaces/smarpitm/redrob-ranker](https://huggingface.co/spaces/smarpitm/redrob-ranker)

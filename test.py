import pandas as pd
import numpy as np
from src.embeddings import build_index_from_dataframe, retrieve_candidates
from src.features import extract_all_features
from src.synthetic_labels import generate_synthetic_labels
from src.ranker import train_ranker, predict_rankings

# Create dummy data
candidates = pd.DataFrame({
    "candidate_id": [f"CAND_{i:03d}" for i in range(100)],
    "resume_text": [f"Python developer with {i%10} years experience" for i in range(100)],
    "skills": ["python, django" if i % 2 == 0 else "java, spring" for i in range(100)],
    "experience_years": [i % 10 for i in range(100)],
    "education": ["B.Tech" for _ in range(100)],
    "location": ["Bangalore" for _ in range(100)]
})

jobs = pd.DataFrame({
    "job_id": ["JOB_001"],
    "job_description": ["Senior Python developer needed"],
    "required_skills": ["python, django"],
    "experience_required": [5],
    "education_required": ["B.Tech"],
    "location": ["Bangalore"]
})

# Test 1: Build FAISS index
index, ids, embedder = build_index_from_dataframe(candidates, save_dir="./models")
print(f"Index built with {index.ntotal} candidates")

# Test 2: Retrieve for 1 job
scores, indices, retrieved_ids = retrieve_candidates(
    "Python developer", index, ids, embedder, top_k=10
)
print(f"Retrieved {len(retrieved_ids)} candidates")

# Test 3: Feature extraction
features = extract_all_features(jobs.iloc[0], candidates.iloc[0], faiss_score=0.9)
print(f"Features shape: {features.shape}")
print(f"Features: {features}")

# Test 4: Synthetic labels
synthetic = generate_synthetic_labels(jobs, candidates, n_samples=50)
print(f"Synthetic data shape: {synthetic.shape}")

# Test 5: Train ranker
ranker = train_ranker(synthetic, "./models/test_ranker.json")
print("Ranker trained successfully")

# Test 6: Predict
feature_cols = [col for col in synthetic.columns if col not in {"job_id", "candidate_id", "label"}]
preds = predict_rankings(synthetic[feature_cols].values, ranker)
print(f"Predictions shape: {preds.shape}")


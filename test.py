import pandas as pd
from src.llm_reranker import GroqReranker, create_candidate_summary, validate_ranking

# Test 1: Create summaries
candidates = pd.DataFrame({
    "candidate_id": ["CAND_001", "CAND_002", "CAND_003"],
    "experience_years": [5, 3, 7],
    "skills": ["Python, Django", "Java, Spring", "Python, AWS"],
    "resume_text": ["Built web apps", "Backend services", "Cloud infrastructure"]
})
for _, row in candidates.iterrows():
    print(create_candidate_summary(row))

# Test 2: Initialize reranker (set a dummy key to test init)
import os
os.environ["GROQ_API_KEY"] = "dummy_key_for_testing"
reranker = GroqReranker()
print("Reranker initialized")

# Test 3: Test cache
key = reranker._get_cache_key("test prompt")
print(f"Cache key: {key}")

# Test 4: Test validation
print(validate_ranking(["CAND_001", "CAND_002"], ["CAND_001", "CAND_002"]))  # True
print(validate_ranking(["CAND_001", "CAND_001"], ["CAND_001", "CAND_002"]))  # False (duplicate)



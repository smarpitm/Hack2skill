"""
tests/test_e2e.py — End-to-end integration tests for the ranking pipeline.
"""

import os
import json
import pytest
import pandas as pd
from pathlib import Path

from src.pipeline import CandidateRankingPipeline
from src.data_loader import is_honeypot_candidate


def test_end_to_end_pipeline(tmp_path):
    # 1. Create a dummy candidates list
    candidates = []
    
    # Add some normal candidates
    for i in range(1, 25):
        # We need a mix of skills and locations
        skills = [{"name": "Python", "proficiency": "expert", "endorsements": 5, "duration_months": 24}]
        if i % 3 == 0:
            skills.append({"name": "FAISS", "proficiency": "intermediate", "endorsements": 2, "duration_months": 12})
        if i % 2 == 0:
            skills.append({"name": "XGBoost", "proficiency": "advanced", "endorsements": 3, "duration_months": 18})
            
        location = "Pune" if i % 2 == 0 else "Bangalore"
        
        candidates.append({
            "candidate_id": f"CAND_{i:07d}",
            "profile": {
                "anonymized_name": f"Candidate {i}",
                "headline": "AI Engineer",
                "summary": "Building retrieval systems and ML solutions.",
                "location": location,
                "country": "India",
                "years_of_experience": 3.0 + (i % 8),
                "current_title": "AI Engineer",
                "current_company": "Zomato" if i % 2 == 0 else "TCS"
            },
            "skills": skills,
            "education": [
                {"degree": "B.Tech", "institution": "IIT Bombay", "field_of_study": "CS", "start_year": 2015, "end_year": 2019, "tier": "tier_1"}
            ],
            "career_history": [
                {"company": "Zomato" if i % 2 == 0 else "TCS", "title": "AI Engineer", "start_date": "2019-06-01", "end_date": None, "duration_months": 48, "is_current": True, "industry": "IT", "company_size": "501-1000", "description": "Working on embeddings and ranking systems."}
            ],
            "redrob_signals": {
                "profile_completeness_score": 85.0,
                "signup_date": "2019-01-01",
                "last_active_date": "2026-06-01",
                "open_to_work_flag": True,
                "profile_views_received_30d": 10,
                "applications_submitted_30d": 2,
                "recruiter_response_rate": 0.8,
                "avg_response_time_hours": 2.0,
                "skill_assessment_scores": {"Python": 90.0},
                "connection_count": 50,
                "endorsements_received": 10,
                "notice_period_days": 15,
                "expected_salary_range_inr_lpa": {"min": 12.0, "max": 20.0},
                "preferred_work_mode": "hybrid",
                "willing_to_relocate": True,
                "github_activity_score": 80.0,
                "search_appearance_30d": 30,
                "saved_by_recruiters_30d": 5,
                "interview_completion_rate": 0.95,
                "offer_acceptance_rate": 0.9,
                "verified_email": True,
                "verified_phone": True,
                "linkedin_connected": True
            }
        })
        
    # Add a couple of honeypot candidates (expert skills with 0 duration)
    for i in range(25, 27):
        candidates.append({
            "candidate_id": f"CAND_{i:07d}",
            "profile": {
                "anonymized_name": f"Honeypot {i}",
                "headline": "Staff AI Engineer",
                "summary": "Expert in all things.",
                "location": "Pune",
                "country": "India",
                "years_of_experience": 8.0,
                "current_title": "Senior Engineer",
                "current_company": "Krutrim"
            },
            "skills": [
                {"name": "Python", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
                {"name": "Java", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
                {"name": "Go", "proficiency": "expert", "endorsements": 0, "duration_months": 0}
            ],
            "education": [
                {"degree": "B.Tech", "institution": "IIT Delhi", "field_of_study": "CS", "start_year": 2015, "end_year": 2019, "tier": "tier_1"}
            ],
            "career_history": [
                {"company": "Krutrim", "title": "AI Engineer", "start_date": "2020-01-01", "end_date": None, "duration_months": 48, "is_current": True, "industry": "IT", "company_size": "51-200", "description": "Expert engineering."}
            ],
            "redrob_signals": {
                "profile_completeness_score": 90.0,
                "signup_date": "2020-01-01",
                "last_active_date": "2026-06-01",
                "open_to_work_flag": True,
                "profile_views_received_30d": 10,
                "applications_submitted_30d": 2,
                "recruiter_response_rate": 0.8,
                "avg_response_time_hours": 2.0,
                "skill_assessment_scores": {"Python": 90.0},
                "connection_count": 50,
                "endorsements_received": 10,
                "notice_period_days": 15,
                "expected_salary_range_inr_lpa": {"min": 12.0, "max": 20.0},
                "preferred_work_mode": "hybrid",
                "willing_to_relocate": True,
                "github_activity_score": 80.0,
                "search_appearance_30d": 30,
                "saved_by_recruiters_30d": 5,
                "interview_completion_rate": 0.95,
                "offer_acceptance_rate": 0.9,
                "verified_email": True,
                "verified_phone": True,
                "linkedin_connected": True
            }
        })
        
    candidates_file = tmp_path / "candidates.jsonl"
    with open(candidates_file, "w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")
            
    # 2. Create jobs CSV
    jobs_df = pd.DataFrame([
        {
            "job_id": "JOB_001",
            "description": "Looking for a Senior AI Engineer with Python, FAISS, and XGBoost expertise. Pune/Noida locations preferred.",
            "required_skills": "Python, FAISS, XGBoost",
            "experience_required": 5.0,
            "education_required": "B.Tech",
            "location": "Pune",
            "job_title": "Senior AI Engineer",
        }
    ])
    jobs_file = tmp_path / "jobs.csv"
    jobs_df.to_csv(jobs_file, index=False)
    
    # 3. Initialize and run the pipeline
    pipeline = CandidateRankingPipeline()
    
    # Force rebuild index from our new dummy candidates file
    pipeline.build_index(candidates_path=str(candidates_file), force_rebuild=True)
    
    # Force train ranker using the new dummy candidates file
    pipeline.train_ranker(candidates_path=str(candidates_file), force_retrain=True)
    
    # Run ranking for all jobs
    out_file = tmp_path / "submission.csv"
    submission_df = pipeline.process_all_jobs(
        jobs_path=str(jobs_file),
        candidates_path=str(candidates_file),
        output_path=str(out_file),
        use_llm=False,
        top_k=10  # use smaller top_k for quick e2e validation
    )
    
    # 4. Verify outputs
    assert out_file.exists()
    assert len(submission_df) == 10
    assert list(submission_df.columns) == ["candidate_id", "rank", "score", "reasoning"]
    
    # Check monotonicity
    scores = submission_df["score"].tolist()
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1]
        
    # Check that honeypots were completely excluded
    honeypot_ids = {f"CAND_{i:07d}" for i in range(25, 27)}
    ranked_ids = set(submission_df["candidate_id"].tolist())
    assert len(ranked_ids.intersection(honeypot_ids)) == 0, "Honeypot candidate was ranked!"
    
    # Validate using pipeline's validator
    validation = pipeline.validate_submission(submission_df, strict=False)
    assert validation["valid"] is True, f"Validation failed: {validation['errors']}"

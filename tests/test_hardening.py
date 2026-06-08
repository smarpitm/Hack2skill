"""
tests/test_hardening.py — Tests for the hardened and review-polished features.
"""

import pytest
import numpy as np
import pandas as pd
from src.embeddings import NumpyFaissFallbackIndex
from src.preprocessing import (
    normalize_education,
    extract_experience_years,
    parse_resume_sections,
)
from src.features import compute_education_match


class TestNumpyFaissFallbackIndex:
    """Tests for the pure numpy FAISS fallback index implementation."""

    def test_search_correctness(self):
        # Create 5 normalized mock embeddings of dimension 4
        embeddings = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.8, 0.6, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.5, 0.5, 0.5, 0.5],
        ], dtype=np.float32)
        
        # Normalize just to be sure
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        index = NumpyFaissFallbackIndex(embeddings)
        assert index.ntotal == 5

        # Query close to index 0 and 2
        query = np.array([[0.9, 0.1, 0.0, 0.0]], dtype=np.float32)
        
        scores, indices = index.search(query, top_k=3)
        
        assert scores.shape == (1, 3)
        assert indices.shape == (1, 3)
        # Ranks should be 0, 2, then 4 or 1
        assert indices[0][0] == 0
        assert indices[0][1] == 2
        assert scores[0][0] > scores[0][1]


class TestHinglishEducationParsing:
    """Tests for Hinglish degree normalization in normalize_education."""

    def test_hinglish_normalization(self):
        # B.Tech level (3)
        level, name, tier = normalize_education("बीटेक in CS")
        assert level == 3
        assert name == "बीटेक"

        level, name, tier = normalize_education("बीई in Electronics")
        assert level == 3
        assert name == "बीई"

        # M.Tech / MCA level (4)
        level, name, tier = normalize_education("एमटेक from NIT")
        assert level == 4
        assert name == "एमटेक"

        level, name, tier = normalize_education("एमसीए from college")
        assert level == 4
        assert name == "एमसीए"

        # PhD level (5)
        level, name, _ = normalize_education("पीएचडी from IIT")
        assert level == 5
        assert name == "पीएचडी"


class TestExperienceExtractionEdgeCases:
    """Tests for additional experience year formats."""

    def test_experience_formats(self):
        assert extract_experience_years("3 years+") == 3.0
        assert extract_experience_years("over 5 years of experience") == 5.0
        assert extract_experience_years("more than 7 years") == 7.0
        assert extract_experience_years("at least 1.5 years experience") == 1.5


class TestPlatformActivityEducationBonus:
    """Tests for the Tier-2/3 + high activity bonus."""

    def test_activity_bonus(self):
        # Tier 1 (IIT) -> gets standard tier 1 bonus, no extra activity bonus needed
        score_t1 = compute_education_match(3, 3, resume_college_tier=1, platform_activity=0.9)
        assert score_t1 == 1.0  # meets requirement, capped at 1.0

        # Tier 2, low activity -> no activity bonus (only tier 2 base bonus +0.02)
        score_t2_low = compute_education_match(3, 3, resume_college_tier=2, platform_activity=0.3)
        assert score_t2_low == pytest.approx(1.0) # wait, meets requirement (base_score = 1.0, capped at 1.0)

        # Let's test where requirement is 4, candidate is level 3 (base_score = 0.7)
        # Tier 2, low activity -> 0.7 + 0.02 = 0.72
        score_t2_low_diff = compute_education_match(4, 3, resume_college_tier=2, platform_activity=0.5)
        assert score_t2_low_diff == pytest.approx(0.72)

        # Tier 2, high activity (>=0.8) -> 0.7 + 0.02 + 0.05 = 0.77
        score_t2_high_diff = compute_education_match(4, 3, resume_college_tier=2, platform_activity=0.85)
        assert score_t2_high_diff == pytest.approx(0.77)

        # Tier 3 (others), high activity (>=0.8) -> 0.7 + 0.0 + 0.05 = 0.75
        score_t3_high_diff = compute_education_match(4, 3, resume_college_tier=3, platform_activity=0.90)
        assert score_t3_high_diff == pytest.approx(0.75)


class TestRobustSectionDetection:
    """Tests for section detection with plural variations."""

    def test_plural_keywords(self):
        text = "Academics: scored high. Key Skills: Python. Personal Projects: Built a chatbot."
        sections = parse_resume_sections(text)
        assert sections["education"] is True
        assert sections["skills"] is True
        assert sections["projects"] is True


class TestCandidateDataLoading:
    """Tests for loading candidates from various formats (CSV, JSON, JSONL, JSONL.GZ)."""

    @pytest.fixture
    def mock_candidate_dict(self):
        return {
            "candidate_id": "CAND_0000001",
            "profile": {
                "anonymized_name": "Test Candidate",
                "headline": "Software Engineer",
                "summary": "Experienced engineer.",
                "location": "Pune",
                "country": "India",
                "years_of_experience": 5.0,
                "current_title": "Software Engineer",
                "current_company": "TCS"
            },
            "skills": [
                {"name": "Python", "proficiency": "expert", "endorsements": 10, "duration_months": 36},
                {"name": "SQL", "proficiency": "intermediate", "endorsements": 5, "duration_months": 24}
            ],
            "education": [
                {"degree": "B.Tech", "institution": "IIT Bombay", "field_of_study": "CS", "start_year": 2015, "end_year": 2019, "tier": "tier_1"}
            ],
            "career_history": [
                {"company": "TCS", "title": "Software Engineer", "start_date": "2019-06-01", "end_date": None, "duration_months": 48, "is_current": True, "industry": "IT", "company_size": "10001+", "description": "Developed web applications."}
            ],
            "redrob_signals": {
                "profile_completeness_score": 90.0,
                "signup_date": "2019-01-01",
                "last_active_date": "2026-06-01",
                "open_to_work_flag": True,
                "profile_views_received_30d": 12,
                "applications_submitted_30d": 3,
                "recruiter_response_rate": 0.8,
                "avg_response_time_hours": 4.0,
                "skill_assessment_scores": {"Python": 85.0},
                "connection_count": 120,
                "endorsements_received": 15,
                "notice_period_days": 30,
                "expected_salary_range_inr_lpa": {"min": 15.0, "max": 25.0},
                "preferred_work_mode": "hybrid",
                "willing_to_relocate": True,
                "github_activity_score": 75.0,
                "search_appearance_30d": 45,
                "saved_by_recruiters_30d": 5,
                "interview_completion_rate": 0.9,
                "offer_acceptance_rate": 0.8,
                "verified_email": True,
                "verified_phone": True,
                "linkedin_connected": True
            }
        }

    def test_json_loading(self, tmp_path, mock_candidate_dict):
        import json
        from src.pipeline import CandidateRankingPipeline

        json_file = tmp_path / "candidates.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump([mock_candidate_dict], f)

        pipeline = CandidateRankingPipeline()
        df = pipeline.load_candidates_dataframe(str(json_file))
        assert len(df) == 1
        assert df.iloc[0]["candidate_id"] == "CAND_0000001"
        assert "Python" in df.iloc[0]["skills"]
        assert df.iloc[0]["experience_years"] == 5.0
        assert df.iloc[0]["is_honeypot"] == 0

    def test_jsonl_gz_loading(self, tmp_path, mock_candidate_dict):
        import json
        import gzip
        from src.pipeline import CandidateRankingPipeline

        gz_file = tmp_path / "candidates.jsonl.gz"
        with gzip.open(gz_file, "wt", encoding="utf-8") as f:
            f.write(json.dumps(mock_candidate_dict) + "\n")

        pipeline = CandidateRankingPipeline()
        df = pipeline.load_candidates_dataframe(str(gz_file))
        assert len(df) == 1
        assert df.iloc[0]["candidate_id"] == "CAND_0000001"
        assert df.iloc[0]["experience_years"] == 5.0
        assert df.iloc[0]["is_honeypot"] == 0


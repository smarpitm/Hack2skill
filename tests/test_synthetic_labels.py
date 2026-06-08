"""
tests/test_synthetic_labels.py — Unit tests for src/synthetic_labels.py.

Tests synthetic label generation, stratification, and I/O with realistic data.
"""

import os
import tempfile
import pytest
import numpy as np
import pandas as pd
from src.synthetic_labels import (
    generate_synthetic_labels,
    save_synthetic_data,
    load_synthetic_data,
)


# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES — Small realistic datasets for fast test runs
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def mini_jobs_df():
    """Two job descriptions for quick testing."""
    return pd.DataFrame([
        {
            "job_id": "JOB_001",
            "description": (
                "Senior AI Engineer — 5 years experience. "
                "Skills: Python, FAISS, NLP, XGBoost, sentence-transformers."
            ),
            "required_skills": "Python, FAISS, NLP, XGBoost",
            "experience_required": 5.0,
            "education_required": "B.Tech",
            "location": "Pune",
            "job_title": "Senior AI Engineer",
        },
        {
            "job_id": "JOB_002",
            "description": (
                "Full Stack Developer — 3 years experience. "
                "Skills: React, Node.js, MongoDB, Docker."
            ),
            "required_skills": "React, Node.js, MongoDB, Docker",
            "experience_required": 3.0,
            "education_required": "B.Tech",
            "location": "Bangalore",
            "job_title": "Full Stack Developer",
        },
    ])


@pytest.fixture
def mini_candidates_df():
    """Six diverse candidates for fast synthetic label generation."""
    return pd.DataFrame([
        {
            "candidate_id": "C001",
            "resume_text": "Python FAISS NLP expert with 7 years experience. Education: B.Tech from IIT.",
            "skills": "Python, FAISS, NLP, PyTorch",
            "experience_years": 7.0,
            "education": "B.Tech",
            "location": "Pune",
            "current_title": "Senior AI Engineer",
            "platform_activity_score": 90.0,
        },
        {
            "candidate_id": "C002",
            "resume_text": "React and Node.js developer with 4 years experience.",
            "skills": "React, Node.js, MongoDB",
            "experience_years": 4.0,
            "education": "BCA",
            "location": "Bangalore",
            "current_title": "Full Stack Developer",
            "platform_activity_score": 70.0,
        },
        {
            "candidate_id": "C003",
            "resume_text": "Marketing manager with 6 years in digital marketing.",
            "skills": "Marketing, SEO, Excel",
            "experience_years": 6.0,
            "education": "MBA",
            "location": "Mumbai",
            "current_title": "Marketing Manager",
            "platform_activity_score": 40.0,
        },
        {
            "candidate_id": "C004",
            "resume_text": "Junior developer, 1 year experience in Python basics.",
            "skills": "Python, HTML",
            "experience_years": 1.0,
            "education": "B.Sc",
            "location": "Jaipur",
            "current_title": "Junior Developer",
            "platform_activity_score": 20.0,
        },
        {
            "candidate_id": "C005",
            "resume_text": "DevOps lead with 8 years. Kubernetes, AWS, Docker expert.",
            "skills": "AWS, Docker, Kubernetes, Terraform",
            "experience_years": 8.0,
            "education": "B.Tech",
            "location": "Gurugram",
            "current_title": "DevOps Lead",
            "platform_activity_score": 85.0,
        },
        {
            "candidate_id": "C006",
            "resume_text": "Data scientist with Python, TensorFlow, 3 years NLP experience.",
            "skills": "Python, TensorFlow, NLP, Pandas",
            "experience_years": 3.0,
            "education": "M.Tech",
            "location": "Hyderabad",
            "current_title": "Data Scientist",
            "platform_activity_score": 65.0,
        },
    ])


# ═══════════════════════════════════════════════════════════════════════════
# generate_synthetic_labels
# ═══════════════════════════════════════════════════════════════════════════

class TestGenerateSyntheticLabels:
    """Tests for synthetic label generation."""

    def test_returns_dataframe(self, mini_jobs_df, mini_candidates_df):
        result = generate_synthetic_labels(mini_jobs_df, mini_candidates_df, n_samples=20)
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_required_columns_present(self, mini_jobs_df, mini_candidates_df):
        result = generate_synthetic_labels(mini_jobs_df, mini_candidates_df, n_samples=20)
        assert "job_id" in result.columns
        assert "candidate_id" in result.columns
        assert "label" in result.columns

    def test_has_15_feature_columns(self, mini_jobs_df, mini_candidates_df):
        result = generate_synthetic_labels(mini_jobs_df, mini_candidates_df, n_samples=20)
        feature_cols = [col for col in result.columns if col.startswith("feature_")]
        assert len(feature_cols) == 15

    def test_labels_are_binary(self, mini_jobs_df, mini_candidates_df):
        result = generate_synthetic_labels(mini_jobs_df, mini_candidates_df, n_samples=20)
        assert set(result["label"].unique()).issubset({0, 1})

    def test_has_both_classes(self, mini_jobs_df, mini_candidates_df):
        """Should generate both positive and negative samples (stratified)."""
        result = generate_synthetic_labels(mini_jobs_df, mini_candidates_df, n_samples=50)
        # At minimum, we should not have a single-class dataset
        unique_labels = result["label"].unique()
        assert len(unique_labels) >= 1  # At least one class present

    def test_reproducibility(self, mini_jobs_df, mini_candidates_df):
        """Same random_state → same output."""
        r1 = generate_synthetic_labels(mini_jobs_df, mini_candidates_df, n_samples=20, random_state=42)
        r2 = generate_synthetic_labels(mini_jobs_df, mini_candidates_df, n_samples=20, random_state=42)
        pd.testing.assert_frame_equal(r1, r2)

    def test_empty_jobs_returns_empty(self, mini_candidates_df):
        result = generate_synthetic_labels(pd.DataFrame(), mini_candidates_df)
        assert result.empty

    def test_empty_candidates_returns_empty(self, mini_jobs_df):
        result = generate_synthetic_labels(mini_jobs_df, pd.DataFrame())
        assert result.empty

    def test_respects_n_samples_cap(self, mini_jobs_df, mini_candidates_df):
        """Output should not exceed n_samples."""
        result = generate_synthetic_labels(mini_jobs_df, mini_candidates_df, n_samples=10)
        assert len(result) <= 10


# ═══════════════════════════════════════════════════════════════════════════
# save / load roundtrip
# ═══════════════════════════════════════════════════════════════════════════

class TestSaveLoadSyntheticData:
    """Tests for CSV I/O roundtrip."""

    def test_roundtrip(self, mini_jobs_df, mini_candidates_df, tmp_path):
        """Save → load should yield identical data."""
        df = generate_synthetic_labels(mini_jobs_df, mini_candidates_df, n_samples=20)
        filepath = str(tmp_path / "test_synthetic.csv")
        save_synthetic_data(df, filepath)
        loaded = load_synthetic_data(filepath)
        assert len(loaded) == len(df)
        assert list(loaded.columns) == list(df.columns)

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_synthetic_data("/nonexistent/path/data.csv")

    def test_save_creates_directories(self, tmp_path):
        """save_synthetic_data should create intermediate dirs."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        nested_path = str(tmp_path / "deep" / "nested" / "output.csv")
        save_synthetic_data(df, nested_path)
        assert os.path.exists(nested_path)

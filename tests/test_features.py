"""
tests/test_features.py — Comprehensive unit tests for src/features.py.

Tests all 15 feature computations individually plus the full
extract_all_features() pipeline, using real-world candidate/job data.
"""

import pytest
import numpy as np
import pandas as pd
from src.features import (
    compute_skill_match,
    compute_experience_match,
    compute_education_match,
    compute_location_match,
    compute_semantic_similarity,
    compute_platform_activity,
    compute_career_progression,
    compute_resume_completeness,
    compute_keyword_density,
    compute_project_diversity,
    compute_current_title_match,
    extract_all_features,
)
from src import config


# ═══════════════════════════════════════════════════════════════════════════
# compute_skill_match
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeSkillMatch:
    """Tests for skill overlap computation."""

    def test_perfect_match(self):
        jd = ["Python", "Docker", "AWS"]
        resume = ["Python", "Docker", "AWS"]
        count, ratio = compute_skill_match(jd, resume)
        assert count == 3
        assert ratio == pytest.approx(1.0)

    def test_partial_match(self):
        jd = ["Python", "Docker", "AWS", "Kubernetes"]
        resume = ["Python", "Docker"]
        count, ratio = compute_skill_match(jd, resume)
        assert count == 2
        assert ratio == pytest.approx(0.5)

    def test_no_match(self):
        jd = ["Python", "Docker"]
        resume = ["Java", "Spring Boot"]
        count, ratio = compute_skill_match(jd, resume)
        assert count == 0
        assert ratio == pytest.approx(0.0)

    def test_empty_jd_skills(self):
        count, ratio = compute_skill_match([], ["Python"])
        assert count == 0
        assert ratio == 0.0

    def test_empty_resume_skills(self):
        count, ratio = compute_skill_match(["Python"], [])
        assert count == 0
        assert ratio == pytest.approx(0.0)

    def test_alias_matching(self):
        """Node.js in JD should match Node.js in resume via aliases."""
        jd = ["Node.js"]
        resume = ["Node.js"]
        count, ratio = compute_skill_match(jd, resume)
        assert count == 1

    def test_real_ai_role_match(self):
        """Test with realistic AI role skill lists."""
        jd = ["Python", "FAISS", "NLP", "XGBoost", "PyTorch", "Docker"]
        resume = ["Python", "PyTorch", "FAISS", "NLP", "Docker", "AWS"]
        count, ratio = compute_skill_match(jd, resume)
        assert count >= 4  # At least Python, FAISS, NLP, Docker match
        assert ratio >= 0.6


# ═══════════════════════════════════════════════════════════════════════════
# compute_experience_match
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeExperienceMatch:
    """Tests for experience matching score."""

    def test_meets_requirement(self):
        assert compute_experience_match(5.0, 7.0) == 1.0

    def test_exact_match(self):
        assert compute_experience_match(5.0, 5.0) == 1.0

    def test_70_percent_match(self):
        """3.5 years with 5 required = 70% threshold."""
        result = compute_experience_match(5.0, 3.5)
        assert result == 0.7

    def test_50_percent_match(self):
        result = compute_experience_match(10.0, 5.0)
        assert result == 0.4

    def test_below_50_percent(self):
        result = compute_experience_match(10.0, 2.0)
        assert result == 0.1

    def test_zero_requirement(self):
        """No experience requirement → default score."""
        result = compute_experience_match(0.0, 5.0)
        assert result == config.DEFAULT_EXPERIENCE_MATCH

    def test_zero_candidate_exp(self):
        result = compute_experience_match(5.0, 0.0)
        assert result == 0.1


# ═══════════════════════════════════════════════════════════════════════════
# compute_education_match
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeEducationMatch:
    """Tests for education level matching with tier bonuses."""

    def test_meets_requirement(self):
        assert compute_education_match(3, 4) == 1.0  # M.Tech >= B.Tech

    def test_exact_match(self):
        assert compute_education_match(3, 3) == 1.0

    def test_one_level_below(self):
        assert compute_education_match(4, 3) == 0.7

    def test_two_levels_below(self):
        assert compute_education_match(5, 3) == 0.3

    def test_tier1_bonus(self):
        """IIT tier bonus of +0.05."""
        result = compute_education_match(3, 3, resume_college_tier=1)
        assert result == pytest.approx(1.0)  # capped at 1.0

    def test_tier2_bonus(self):
        result = compute_education_match(4, 3, resume_college_tier=2)
        assert result == pytest.approx(0.72)

    def test_capped_at_1(self):
        """Score should never exceed 1.0."""
        result = compute_education_match(3, 5, resume_college_tier=1)
        assert result <= 1.0

    def test_zero_requirement(self):
        result = compute_education_match(0, 3)
        assert result == config.DEFAULT_EDUCATION_MATCH


# ═══════════════════════════════════════════════════════════════════════════
# compute_location_match
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeLocationMatch:
    """Tests for location matching with Indian geography context."""

    def test_exact_match(self):
        assert compute_location_match("Pune", "Pune") == 1.0

    def test_alias_match(self):
        """'Bengaluru' and 'Bangalore' resolve to same."""
        assert compute_location_match("Bengaluru", "Bangalore") == 1.0

    def test_ncr_metro(self):
        """Delhi-Gurgaon-Noida are same metro."""
        assert compute_location_match("Delhi", "Gurgaon") == 0.8

    def test_different_indian_cities(self):
        """Both Indian but different cities."""
        result = compute_location_match("Pune", "Chennai")
        assert result == 0.3

    def test_unknown_locations(self):
        result = compute_location_match("Mars", "Jupiter")
        assert result == config.DEFAULT_LOCATION_MATCH

    def test_outside_india(self):
        """USA location → 0.0 match."""
        result = compute_location_match("Pune", "Austin, USA")
        assert result == 0.0

    def test_empty_locations(self):
        result = compute_location_match("", "")
        assert result == config.DEFAULT_LOCATION_MATCH


# ═══════════════════════════════════════════════════════════════════════════
# compute_semantic_similarity
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeSemanticSimilarity:
    """Tests for FAISS score passthrough."""

    def test_returns_float(self):
        assert compute_semantic_similarity(0.85) == 0.85

    def test_zero_score(self):
        assert compute_semantic_similarity(0.0) == 0.0

    def test_negative_score(self):
        assert compute_semantic_similarity(-0.1) == -0.1


# ═══════════════════════════════════════════════════════════════════════════
# compute_platform_activity
# ═══════════════════════════════════════════════════════════════════════════

class TestComputePlatformActivity:
    """Tests for platform activity normalization."""

    def test_percentage_normalization(self):
        """Score > 1.0 and <= 100 → normalized to 0–1 range."""
        row = {"platform_activity_score": 92.3}
        result = compute_platform_activity(row)
        assert 0.0 <= result <= 1.0
        assert result == pytest.approx(0.923)

    def test_already_normalized(self):
        row = {"platform_activity_score": 0.75}
        assert compute_platform_activity(row) == 0.75

    def test_zero_activity(self):
        row = {"platform_activity_score": 0.0}
        assert compute_platform_activity(row) == 0.0

    def test_missing_key(self):
        """Missing activity keys → default."""
        row = {"name": "test"}
        assert compute_platform_activity(row) == config.DEFAULT_PLATFORM_ACTIVITY

    def test_nan_value(self):
        row = {"platform_activity_score": float("nan")}
        assert compute_platform_activity(row) == config.DEFAULT_PLATFORM_ACTIVITY

    def test_above_100(self):
        """Values above 100 capped at 1.0."""
        row = {"platform_activity_score": 150.0}
        assert compute_platform_activity(row) == 1.0

    def test_fallback_column_names(self):
        """Test alternative column names: activity_score, engagement_score."""
        row = {"activity_score": 80.0}
        result = compute_platform_activity(row)
        assert result == pytest.approx(0.80)


# ═══════════════════════════════════════════════════════════════════════════
# compute_career_progression
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeCareerProgression:
    """Tests for career progression scoring."""

    def test_senior_title_high_exp(self):
        assert compute_career_progression("Senior AI Engineer", 7.0) == 1.0

    def test_lead_title_mid_exp(self):
        assert compute_career_progression("Tech Lead", 4.0) == 0.8

    def test_no_seniority_high_exp(self):
        assert compute_career_progression("Software Engineer", 6.0) == 0.7

    def test_junior_low_exp(self):
        assert compute_career_progression("Junior Developer", 1.0) == 0.3

    def test_unknown_title(self):
        assert compute_career_progression("unknown", 5.0) == 0.3

    def test_empty_title(self):
        assert compute_career_progression("", 10.0) == 0.3

    def test_principal_title(self):
        assert compute_career_progression("Principal Engineer", 10.0) == 1.0

    def test_cto_title(self):
        assert compute_career_progression("CTO", 15.0) == 1.0


# ═══════════════════════════════════════════════════════════════════════════
# compute_resume_completeness
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeResumeCompleteness:
    """Tests for resume field completeness scoring."""

    def test_fully_complete(self):
        row = {
            "resume_text": "Full resume here.",
            "skills": ["Python", "Java"],
            "experience_years": 5.0,
            "education": "B.Tech",
            "current_title": "Engineer",
        }
        assert compute_resume_completeness(row) == pytest.approx(1.0)

    def test_partially_complete(self):
        row = {
            "resume_text": "Some text.",
            "skills": ["Python"],
            "experience_years": 3.0,
            "education": "",
            "current_title": "unknown",
        }
        result = compute_resume_completeness(row)
        assert 0.0 < result < 1.0

    def test_empty_resume(self):
        row = {
            "resume_text": "",
            "skills": [],
            "experience_years": 0.0,
            "education": "",
            "current_title": "",
        }
        result = compute_resume_completeness(row)
        assert result == pytest.approx(0.0)

    def test_experience_fallback_key(self):
        """'experience' used if 'experience_years' missing."""
        row = {
            "resume_text": "text",
            "skills": ["Python"],
            "experience": 3.0,
            "education": "B.Tech",
            "current_title": "Dev",
        }
        result = compute_resume_completeness(row)
        assert result >= 0.6


# ═══════════════════════════════════════════════════════════════════════════
# compute_keyword_density
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeKeywordDensity:
    """Tests for keyword density matching."""

    def test_high_overlap(self):
        jd = "Python machine learning FAISS embeddings NLP"
        resume = "Expert in Python and FAISS for NLP and machine learning with embeddings"
        result = compute_keyword_density(jd, resume)
        assert result > 0.5

    def test_no_overlap(self):
        jd = "accounting finance excel spreadsheets"
        resume = "Python machine learning deep learning"
        result = compute_keyword_density(jd, resume)
        assert result < 0.3

    def test_empty_jd(self):
        assert compute_keyword_density("", "Python developer") == 0.0

    def test_empty_resume(self):
        assert compute_keyword_density("Python developer", "") == 0.0

    def test_both_empty(self):
        assert compute_keyword_density("", "") == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# compute_project_diversity
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeProjectDiversity:
    """Tests for project domain diversity scoring."""

    def test_diverse_projects(self):
        text = (
            "Built a web application for e-commerce. "
            "Developed a machine learning model for NLP. "
            "Created a mobile app for Android. "
            "Implemented a cloud deployment pipeline on AWS."
        )
        result = compute_project_diversity(text)
        assert result > 0.0

    def test_no_projects(self):
        text = "I studied computer science at university."
        result = compute_project_diversity(text)
        assert result == 0.0

    def test_empty_text(self):
        assert compute_project_diversity("") == 0.0

    def test_nan_input(self):
        assert compute_project_diversity(float("nan")) == 0.0

    def test_capped_at_1(self):
        """Score should not exceed 1.0 even with many domains."""
        text = (
            "Built web, frontend, backend, fullstack, mobile, iOS, Android projects. "
            "Developed cloud, devops, machine learning, data science, NLP, computer vision, "
            "blockchain, database, cybersecurity, embedded, IoT solutions."
        )
        result = compute_project_diversity(text)
        assert result <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# compute_current_title_match
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeCurrentTitleMatch:
    """Tests for title overlap scoring."""

    def test_exact_match(self):
        result = compute_current_title_match("Senior AI Engineer", "Senior AI Engineer")
        assert result == 1.0

    def test_partial_match(self):
        result = compute_current_title_match("Senior AI Engineer", "AI Engineer")
        assert result >= 0.6

    def test_no_match(self):
        result = compute_current_title_match("Senior AI Engineer", "Marketing Manager")
        assert result == 0.2

    def test_unknown_titles(self):
        assert compute_current_title_match("unknown", "unknown") == 0.2

    def test_empty_titles(self):
        assert compute_current_title_match("", "") == 0.2


# ═══════════════════════════════════════════════════════════════════════════
# extract_all_features — Integration
# ═══════════════════════════════════════════════════════════════════════════

class TestExtractAllFeatures:
    """Integration tests for the full 15-feature extraction pipeline."""

    def test_returns_15_features(self, sample_job_dict, strong_candidate_row):
        feats = extract_all_features(sample_job_dict, strong_candidate_row, faiss_score=0.85)
        assert isinstance(feats, np.ndarray)
        assert feats.shape == (15,)
        assert feats.dtype == np.float32

    def test_strong_candidate_scores_high(self, sample_job_dict, strong_candidate_row):
        """CAND_001 (Sr. AI Engineer, IIT, Pune) should score well."""
        feats = extract_all_features(sample_job_dict, strong_candidate_row, faiss_score=0.90)
        skill_ratio = feats[1]
        exp_match = feats[3]
        semantic_sim = feats[6]
        assert skill_ratio > 0.0, "Strong candidate should have >0 skill match"
        assert exp_match >= 0.7, "7 yrs exp vs 5 req → should be high"
        assert semantic_sim == pytest.approx(0.90)

    def test_weak_candidate_scores_low(self, sample_job_dict, weak_candidate_row):
        """CAND_006 (Marketing Manager) should score poorly for AI role."""
        feats = extract_all_features(sample_job_dict, weak_candidate_row, faiss_score=0.15)
        skill_ratio = feats[1]
        assert skill_ratio < 0.3, "Marketing manager should have low skill match for AI role"

    def test_empty_candidate_does_not_crash(self, sample_job_dict, empty_candidate_row):
        """CAND_008 (all empty fields) must not raise."""
        feats = extract_all_features(sample_job_dict, empty_candidate_row, faiss_score=0.0)
        assert feats.shape == (15,)
        assert not np.any(np.isnan(feats)), "No NaN values should be produced"

    def test_all_features_in_valid_range(self, sample_job_dict, strong_candidate_row):
        """All features should be finite numbers."""
        feats = extract_all_features(sample_job_dict, strong_candidate_row, faiss_score=0.80)
        for i, val in enumerate(feats):
            assert np.isfinite(val), f"Feature {i} is not finite: {val}"

    def test_feature_ordering_consistency(self, sample_job_dict, strong_candidate_row):
        """Features should be in the documented order:
        [skill_count, skill_ratio, candidate_exp, exp_match, edu_match,
         loc_match, sem_sim, plat_act, res_complete, career_prog,
         res_len, kw_density, sec_complete, proj_div, title_match]
        """
        feats = extract_all_features(sample_job_dict, strong_candidate_row, faiss_score=0.85)
        # Feature 2 = candidate_exp → should be 7.0 for CAND_001
        assert feats[2] == pytest.approx(7.0, abs=0.5)
        # Feature 6 = semantic_similarity → should be the faiss_score
        assert feats[6] == pytest.approx(0.85)

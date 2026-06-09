"""
tests/test_config.py — Unit tests for src/config.py constants and invariants.

Validates that all configuration values are sane, consistent,
and compatible with downstream pipeline expectations.
"""

import pytest
from src import config


class TestWeights:
    """Synthetic label weights must be valid probability distribution."""

    def test_weights_sum_to_one(self):
        total = (
            config.WEIGHT_SKILL_MATCH
            + config.WEIGHT_SEMANTIC
            + config.WEIGHT_EXPERIENCE
            + config.WEIGHT_ACTIVITY
        )
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_all_weights_positive(self):
        assert config.WEIGHT_SKILL_MATCH > 0
        assert config.WEIGHT_SEMANTIC > 0
        assert config.WEIGHT_EXPERIENCE > 0
        assert config.WEIGHT_ACTIVITY > 0

    def test_threshold_in_valid_range(self):
        assert 0.0 < config.SYNTHETIC_LABEL_THRESHOLD < 1.0


class TestTopK:
    """Top-K values must maintain stage pipeline ordering: RETRIEVAL > RANKER > LLM."""

    def test_retrieval_k_largest(self):
        assert config.RETRIEVAL_K > config.RANKER_K

    def test_ranker_k_greater_than_llm_k(self):
        assert config.RANKER_K >= config.LLM_K

    def test_all_positive(self):
        assert config.RETRIEVAL_K > 0
        assert config.RANKER_K > 0
        assert config.LLM_K > 0


class TestSkillDictionary:
    """Skill dictionary must be non-empty and contain key skills."""

    def test_not_empty(self):
        assert len(config.SKILL_DICTIONARY) > 50

    def test_core_skills_present(self):
        """Key AI/ML skills for this hackathon must be in the dictionary."""
        expected = ["Python", "Docker", "AWS", "React", "SQL", "Git"]
        for skill in expected:
            assert skill in config.SKILL_DICTIONARY, f"Missing core skill: {skill}"


class TestEducationEquivalence:
    """Indian education equivalence map must cover all common degrees."""

    def test_btech_equals_be(self):
        assert config.EDUCATION_EQUIVALENCE_MAP["b.tech"] == config.EDUCATION_EQUIVALENCE_MAP["b.e"]

    def test_mca_equals_mtech(self):
        """MCA should be same level as M.Tech (Indian context rule)."""
        assert config.EDUCATION_EQUIVALENCE_MAP["mca"] == config.EDUCATION_EQUIVALENCE_MAP["m.tech"]

    def test_phd_highest(self):
        assert config.EDUCATION_EQUIVALENCE_MAP["phd"] == 5
        assert max(config.EDUCATION_EQUIVALENCE_MAP.values()) == 5

    def test_levels_are_1_to_5(self):
        for key, level in config.EDUCATION_EQUIVALENCE_MAP.items():
            assert 1 <= level <= 5, f"Invalid level {level} for '{key}'"


class TestLocationNormalization:
    """Location normalization map sanity checks."""

    def test_bengaluru_and_bangalore(self):
        assert config.LOCATION_NORMALIZATION_MAP["bengaluru"] == "bangalore"
        assert config.LOCATION_NORMALIZATION_MAP["bangalore"] == "bangalore"

    def test_bombay_to_mumbai(self):
        assert config.LOCATION_NORMALIZATION_MAP["bombay"] == "mumbai"

    def test_ncr_consistency(self):
        """Delhi and New Delhi must normalize to same value."""
        assert config.LOCATION_NORMALIZATION_MAP["delhi"] == config.LOCATION_NORMALIZATION_MAP["new delhi"]


class TestCollegeTiers:
    """Tier keyword lists must be non-empty."""

    def test_tier1_has_keywords(self):
        assert len(config.TIER_1_KEYWORDS) > 0
        assert "iit" in config.TIER_1_KEYWORDS

    def test_tier2_has_keywords(self):
        assert len(config.TIER_2_KEYWORDS) > 0

    def test_no_overlap_between_tiers(self):
        """No keyword should appear in both tier lists."""
        overlap = set(config.TIER_1_KEYWORDS) & set(config.TIER_2_KEYWORDS)
        assert len(overlap) == 0, f"Overlapping tier keywords: {overlap}"




class TestDefaults:
    """Default fallback values must be sensible."""

    def test_defaults_in_range(self):
        for val in [
            config.DEFAULT_PLATFORM_ACTIVITY,
            config.DEFAULT_EXPERIENCE_MATCH,
            config.DEFAULT_EDUCATION_MATCH,
            config.DEFAULT_LOCATION_MATCH,
        ]:
            assert 0.0 <= val <= 1.0


class TestPaths:
    """Path configuration should point to valid Path objects."""

    def test_paths_are_path_objects(self):
        from pathlib import Path
        for path in [config.DATA_DIR, config.MODELS_DIR, config.CACHE_DIR, config.SUBMISSIONS_DIR]:
            assert isinstance(path, Path)

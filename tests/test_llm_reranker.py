"""
tests/test_llm_reranker.py — Unit tests for src/llm_reranker.py.

Tests candidate summary creation, ranking validation, prompt building,
response parsing, and caching logic *without* making real API calls.
"""

import os
import json
import pytest
import hashlib
import tempfile
import pandas as pd
from unittest.mock import patch, MagicMock
from src.llm_reranker import (
    create_candidate_summary,
    validate_ranking,
    GroqReranker,
)


# ═══════════════════════════════════════════════════════════════════════════
# create_candidate_summary
# ═══════════════════════════════════════════════════════════════════════════

class TestCreateCandidateSummary:
    """Tests for candidate summary compression."""

    def test_basic_format(self):
        row = pd.Series({
            "candidate_id": "CAND_001",
            "experience_years": 7.0,
            "skills": "Python, PyTorch, FAISS",
            "resume_text": "Built ML pipelines for production systems.",
        })
        result = create_candidate_summary(row)
        assert "CAND_001" in result
        assert "7y" in result
        assert "Python" in result

    def test_max_length_respected(self):
        row = pd.Series({
            "candidate_id": "CAND_001",
            "experience_years": 5.0,
            "skills": "Python, Java, Docker, Kubernetes, AWS, GCP, React, Angular",
            "resume_text": "A" * 1000,
        })
        result = create_candidate_summary(row, max_length=200)
        assert len(result) <= 210  # Small tolerance for "..." suffix

    def test_skills_as_list(self):
        row = pd.Series({
            "candidate_id": "CAND_002",
            "experience_years": 3.0,
            "skills": ["React", "Node.js"],
            "resume_text": "Frontend dev.",
        })
        result = create_candidate_summary(row)
        assert "React" in result

    def test_missing_fields_handled(self):
        """Should not crash with missing fields."""
        row = pd.Series({"candidate_id": "CAND_003"})
        result = create_candidate_summary(row)
        assert "CAND_003" in result

    def test_decimal_experience(self):
        row = pd.Series({
            "candidate_id": "C",
            "experience_years": 2.5,
            "skills": "",
            "resume_text": "",
        })
        result = create_candidate_summary(row)
        assert "2.5y" in result

    def test_integer_experience(self):
        row = pd.Series({
            "candidate_id": "C",
            "experience_years": 5.0,
            "skills": "",
            "resume_text": "",
        })
        result = create_candidate_summary(row)
        assert "5y" in result


# ═══════════════════════════════════════════════════════════════════════════
# validate_ranking
# ═══════════════════════════════════════════════════════════════════════════

class TestValidateRanking:
    """Tests for ranking validation logic."""

    def test_valid_ranking(self):
        assert validate_ranking(
            ["CAND_001", "CAND_002", "CAND_003"],
            ["CAND_001", "CAND_002", "CAND_003"],
        ) is True

    def test_valid_reordered(self):
        assert validate_ranking(
            ["CAND_003", "CAND_001", "CAND_002"],
            ["CAND_001", "CAND_002", "CAND_003"],
        ) is True

    def test_duplicate_ids(self):
        assert validate_ranking(
            ["CAND_001", "CAND_001"],
            ["CAND_001", "CAND_002"],
        ) is False

    def test_length_mismatch(self):
        assert validate_ranking(
            ["CAND_001"],
            ["CAND_001", "CAND_002"],
        ) is False

    def test_unknown_id(self):
        assert validate_ranking(
            ["CAND_001", "CAND_999"],
            ["CAND_001", "CAND_002"],
        ) is False

    def test_not_a_list(self):
        assert validate_ranking("CAND_001", ["CAND_001"]) is False

    def test_non_string_items(self):
        assert validate_ranking([1, 2], ["1", "2"]) is False

    def test_empty_lists(self):
        assert validate_ranking([], []) is True


# ═══════════════════════════════════════════════════════════════════════════
# GroqReranker — Unit Tests (no real API calls)
# ═══════════════════════════════════════════════════════════════════════════

class TestGroqReranker:
    """Tests for GroqReranker internals without API calls."""

    @pytest.fixture
    def reranker(self, tmp_path):
        """Create a GroqReranker with a dummy API key and temp cache."""
        with patch.dict(os.environ, {"GROQ_API_KEY": "test_key_for_unit_tests"}):
            cache_path = str(tmp_path / "test_cache.json")
            rr = GroqReranker(api_key="test_key_for_unit_tests", cache_path=cache_path)
            return rr

    def test_init_with_api_key(self, reranker):
        assert reranker.client is not None
        assert reranker.model_name is not None

    def test_init_without_api_key_raises(self, tmp_path):
        """Missing GROQ_API_KEY should raise ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GROQ_API_KEY", None)
            with pytest.raises(ValueError, match="GROQ API key"):
                GroqReranker(cache_path=str(tmp_path / "cache.json"))

    def test_cache_key_is_md5(self, reranker):
        prompt = "test prompt for hashing"
        key = reranker._get_cache_key(prompt)
        expected = hashlib.md5(prompt.encode("utf-8")).hexdigest()
        assert key == expected

    def test_cache_key_deterministic(self, reranker):
        k1 = reranker._get_cache_key("same prompt")
        k2 = reranker._get_cache_key("same prompt")
        assert k1 == k2

    def test_cache_key_different_for_different_prompts(self, reranker):
        k1 = reranker._get_cache_key("prompt A")
        k2 = reranker._get_cache_key("prompt B")
        assert k1 != k2

    def test_cache_save_load_roundtrip(self, reranker):
        """Cache should persist to disk and reload correctly."""
        reranker.cache["test_key"] = {"response": "test_response", "timestamp": 12345}
        reranker._save_cache()

        # Reload
        reranker.cache = {}
        reranker._load_cache()
        assert "test_key" in reranker.cache
        assert reranker.cache["test_key"]["response"] == "test_response"

    def test_build_prompt_format(self, reranker):
        prompt = reranker._build_prompt(
            "Senior AI Engineer role",
            ["ID: CAND_001, Skills: Python", "ID: CAND_002, Skills: Java"],
        )
        assert "CAND_001" in prompt
        assert "CAND_002" in prompt
        assert "Senior AI Engineer" in prompt
        assert "JSON" in prompt

    def test_parse_valid_json_array(self, reranker):
        response = '["CAND_001", "CAND_002", "CAND_003"]'
        expected = ["CAND_001", "CAND_002", "CAND_003"]
        result = reranker._parse_response(response, expected)
        assert result == expected

    def test_parse_json_with_markdown(self, reranker):
        """Handle ```json ... ``` wrapping."""
        response = '```json\n["CAND_001", "CAND_002"]\n```'
        expected = ["CAND_001", "CAND_002"]
        result = reranker._parse_response(response, expected)
        assert result == expected

    def test_parse_json_object_with_list(self, reranker):
        """Handle when model wraps list in a JSON object."""
        response = '{"ranked_candidates": ["CAND_001", "CAND_002"]}'
        expected = ["CAND_001", "CAND_002"]
        result = reranker._parse_response(response, expected)
        assert result == expected

    def test_parse_invalid_json_returns_none(self, reranker):
        result = reranker._parse_response("not valid json at all", ["CAND_001"])
        assert result is None

    def test_parse_wrong_ids_returns_none(self, reranker):
        """Valid JSON but wrong IDs should fail validation."""
        response = '["CAND_999", "CAND_998"]'
        expected = ["CAND_001", "CAND_002"]
        result = reranker._parse_response(response, expected)
        assert result is None

    def test_rerank_empty_df_returns_empty(self, reranker):
        result = reranker.rerank("Some job", pd.DataFrame())
        assert result == []

    def test_rerank_missing_columns_returns_none(self, reranker):
        """Missing required columns → None."""
        df = pd.DataFrame({"name": ["Alice"]})
        result = reranker.rerank("Some job", df)
        assert result is None

    def test_rerank_with_cache_hit(self, reranker):
        """If cache has valid entry, API should not be called."""
        candidates_df = pd.DataFrame({
            "candidate_id": ["CAND_001", "CAND_002"],
            "resume_text": ["AI engineer", "Web dev"],
            "skills": ["Python, FAISS", "React, Node"],
            "experience_years": [5.0, 3.0],
        })
        jd = "Senior AI Engineer"

        # Pre-populate cache with valid response
        prompt = reranker._build_prompt(jd, [
            create_candidate_summary(candidates_df.iloc[0]),
            create_candidate_summary(candidates_df.iloc[1]),
        ])
        cache_key = reranker._get_cache_key(prompt)
        reranker.cache[cache_key] = {
            "response": '["CAND_001", "CAND_002"]',
            "timestamp": 12345,
        }

        # Mock API to ensure it's NOT called
        with patch.object(reranker, "_call_groq_api") as mock_api:
            result = reranker.rerank(jd, candidates_df)
            mock_api.assert_not_called()

        assert result == ["CAND_001", "CAND_002"]

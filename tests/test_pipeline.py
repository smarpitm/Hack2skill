"""
tests/test_pipeline.py — Integration tests for src/pipeline.py.

Tests the CandidateRankingPipeline orchestrator end-to-end,
including validation logic, without requiring a GROQ API key.
"""

import pytest
import numpy as np
import pandas as pd
from src.pipeline import CandidateRankingPipeline


# ═══════════════════════════════════════════════════════════════════════════
# validate_submission — Pure logic, no model/index needed
# ═══════════════════════════════════════════════════════════════════════════

class TestValidateSubmission:
    """Tests for submission validation rules."""

    @pytest.fixture
    def pipeline(self):
        return CandidateRankingPipeline()

    def test_valid_submission(self, pipeline):
        df = pd.DataFrame({
            "candidate_id": ["CAND_001", "CAND_002", "CAND_003"],
            "rank": [1, 2, 3],
            "score": [1.0, 0.67, 0.33],
            "reasoning": ["Good fit", "Decent fit", "Weak fit"],
        })
        result = pipeline.validate_submission(df)
        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_missing_column(self, pipeline):
        df = pd.DataFrame({
            "candidate_id": ["CAND_001"],
            "rank": [1],
            # Missing 'score' and 'reasoning'
        })
        result = pipeline.validate_submission(df)
        assert result["valid"] is False
        assert any("score" in e for e in result["errors"])

    def test_null_candidate_ids(self, pipeline):
        df = pd.DataFrame({
            "candidate_id": [None, "CAND_002"],
            "rank": [1, 2],
            "score": [1.0, 0.5],
            "reasoning": ["a", "b"],
        })
        result = pipeline.validate_submission(df)
        assert result["valid"] is False
        assert any("missing" in e.lower() for e in result["errors"])

    def test_duplicate_candidate_ids(self, pipeline):
        df = pd.DataFrame({
            "candidate_id": ["CAND_001", "CAND_001"],
            "rank": [1, 2],
            "score": [1.0, 0.5],
            "reasoning": ["a", "b"],
        })
        result = pipeline.validate_submission(df)
        assert result["valid"] is False
        assert any("duplicate" in e.lower() for e in result["errors"])

    def test_non_contiguous_ranks(self, pipeline):
        df = pd.DataFrame({
            "candidate_id": ["CAND_001", "CAND_002"],
            "rank": [1, 3],  # Missing rank 2
            "score": [1.0, 0.5],
            "reasoning": ["a", "b"],
        })
        result = pipeline.validate_submission(df)
        assert result["valid"] is False
        assert any("contiguous" in e.lower() for e in result["errors"])

    def test_non_monotonic_score(self, pipeline):
        df = pd.DataFrame({
            "candidate_id": ["CAND_001", "CAND_002"],
            "rank": [1, 2],
            "score": [0.5, 0.9],  # Score increases with rank → invalid
            "reasoning": ["a", "b"],
        })
        result = pipeline.validate_submission(df)
        assert result["valid"] is False
        assert any("non-increasing" in e.lower() for e in result["errors"])

    def test_single_candidate_valid(self, pipeline):
        df = pd.DataFrame({
            "candidate_id": ["CAND_001"],
            "rank": [1],
            "score": [1.0],
            "reasoning": ["Perfect fit"],
        })
        result = pipeline.validate_submission(df)
        assert result["valid"] is True

    def test_large_valid_submission(self, pipeline):
        """50-candidate submission should validate."""
        n = 50
        df = pd.DataFrame({
            "candidate_id": [f"CAND_{i:04d}" for i in range(n)],
            "rank": list(range(1, n + 1)),
            "score": [round(1.0 - i / n, 4) for i in range(n)],
            "reasoning": [f"Reason {i}" for i in range(n)],
        })
        result = pipeline.validate_submission(df)
        assert result["valid"] is True

    def test_float_ranks_that_are_integers(self, pipeline):
        """Ranks like 1.0, 2.0 should still be valid."""
        df = pd.DataFrame({
            "candidate_id": ["CAND_001", "CAND_002"],
            "rank": [1.0, 2.0],
            "score": [1.0, 0.5],
            "reasoning": ["a", "b"],
        })
        result = pipeline.validate_submission(df)
        assert result["valid"] is True


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline State Management
# ═══════════════════════════════════════════════════════════════════════════

class TestPipelineState:
    """Tests for pipeline initialization and state flags."""

    def test_initial_state(self):
        p = CandidateRankingPipeline()
        assert p.index_built is False
        assert p.ranker_trained is False
        assert p.faiss_index is None
        assert p.embedder is None
        assert p.candidate_ids is None

    def test_custom_config(self):
        """Pipeline should accept custom config module."""
        import types
        mock_cfg = types.SimpleNamespace(
            EMBEDDING_MODEL="all-MiniLM-L6-v2",
            MODELS_DIR="./test_models",
            RETRIEVAL_K=50,
            RANKER_K=20,
            LLM_K=10,
        )
        p = CandidateRankingPipeline(config_module=mock_cfg)
        assert p._cfg.RETRIEVAL_K == 50

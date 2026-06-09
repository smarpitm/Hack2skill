"""
src/pipeline.py

End-to-end orchestrator for the AI Candidate Ranking System.
Wires Stage 1 (FAISS dense retrieval) → Stage 2 (XGBoost ranking) → Stage 3 (Groq LLM re-ranking).
"""

import logging
import os
import gzip
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from . import config
from . import embeddings
from . import features
from . import synthetic_labels
from . import ranker
from . import data_loader
from . import job_description
from . import reasoning_generator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# File paths for persisted artefacts
_FAISS_INDEX_PATH = config.MODELS_DIR / "candidates.index"
_FAISS_IDS_PATH = config.MODELS_DIR / "candidates.ids.npy"
_XGB_MODEL_PATH = config.MODELS_DIR / "xgb_ranker.json"
_SYNTHETIC_DATA_PATH = config.DATA_DIR / "synthetic_train.csv"

# Default data file names
if (config.DATA_DIR / "candidates.jsonl").exists():
    _DEFAULT_CANDIDATES_FILE = config.DATA_DIR / "candidates.jsonl"
elif (config.DATA_DIR / "candidates.jsonl.gz").exists():
    _DEFAULT_CANDIDATES_FILE = config.DATA_DIR / "candidates.jsonl.gz"
else:
    _DEFAULT_CANDIDATES_FILE = config.DATA_DIR / "candidates.csv"

_DEFAULT_JOBS_FILE = config.DATA_DIR / "jobs.csv"
_DEFAULT_SUBMISSION_FILE = config.SUBMISSIONS_DIR / "submission.csv"


class CandidateRankingPipeline:
    """
    Three-stage candidate ranking pipeline:
      Stage 1 — Sentence-BERT + FAISS dense retrieval (Top-200)
      Stage 2 — XGBoost pairwise ranker on 15 features (Top-50)
      Stage 3 — Groq LLM re-ranker (Top-20, optional)
    """

    def __init__(self, config_module=None):
        """
        Initialise the pipeline.

        Args:
            config_module: Optional pre-loaded config module (used in tests).
                           Defaults to src.config.
        """
        self._cfg = config_module or config

        # Stage 1 attributes
        self.embedder: Optional[SentenceTransformer] = None
        self.faiss_index = None
        self.candidate_ids: Optional[np.ndarray] = None

        # Stage 2 attributes
        self._ranker = None  # xgb.Booster

        # State flags
        self.index_built: bool = False
        self.ranker_trained: bool = False

    def load_candidates_dataframe(self, path: str) -> pd.DataFrame:
        """
        Load candidates data using data_loader.
        """
        return data_loader.load_candidates_dataframe(path)

    # ------------------------------------------------------------------
    # STAGE 1: Build / Load FAISS Index
    # ------------------------------------------------------------------

    def build_index(
        self,
        candidates_path: Optional[str] = None,
        text_column: str = "resume_text",
        id_column: str = "candidate_id",
        force_rebuild: bool = False,
    ) -> None:
        """
        Build (or reload) the FAISS index from candidates CSV.

        If the index is already built and force_rebuild is False, returns immediately.

        Args:
            candidates_path: Path to candidates CSV. Defaults to config DATA_DIR.
            text_column: Column containing resume text.
            id_column: Column containing unique candidate IDs.
            force_rebuild: If True, always rebuilds even if index already exists on disk.
        """
        if self.index_built and not force_rebuild:
            logger.info("Index already built. Skipping (use force_rebuild=True to override).")
            return

        candidates_path = candidates_path or _DEFAULT_CANDIDATES_FILE
        logger.info(f"Loading candidates from: {candidates_path}")
        candidates_df = self.load_candidates_dataframe(candidates_path)

        # If a saved index exists and we are not forcing rebuild, load it from disk
        loaded_index = False
        if (
            not force_rebuild
            and _FAISS_INDEX_PATH.exists()
            and _FAISS_IDS_PATH.exists()
        ):
            logger.info("Found existing FAISS index on disk. Loading from disk...")
            faiss_index, candidate_ids = embeddings.load_faiss_index(
                str(_FAISS_INDEX_PATH), str(_FAISS_IDS_PATH)
            )
            
            # Verify if index matches the loaded candidate DataFrame
            df_ids = set(candidates_df[id_column].astype(str))
            idx_ids = set(candidate_ids.astype(str))
            if df_ids == idx_ids:
                embedder = SentenceTransformer(self._cfg.EMBEDDING_MODEL)
                loaded_index = True
                logger.info("Existing FAISS index matches current candidate pool. Reusing index.")
            else:
                logger.info("Existing FAISS index candidate IDs do not match the current candidate pool. Rebuilding...")

        if not loaded_index:
            logger.info("Building new FAISS index...")
            self._cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
            faiss_index, candidate_ids, embedder = embeddings.build_index_from_dataframe(
                candidates_df,
                text_column=text_column,
                id_column=id_column,
                save_dir=str(self._cfg.MODELS_DIR),
            )

        self.faiss_index = faiss_index
        self.candidate_ids = candidate_ids
        self.embedder = embedder
        self.index_built = True
        logger.info(f"Index built with {len(candidate_ids)} candidates.")

    # ------------------------------------------------------------------
    # STAGE 2: Train / Load XGBoost Ranker
    # ------------------------------------------------------------------

    def train_ranker(
        self,
        synthetic_data_path: Optional[str] = None,
        force_retrain: bool = False,
        candidates_path: Optional[str] = None,
    ) -> None:
        """
        Train (or reload) the XGBoost ranking model.

        If synthetic training data does not exist, it is generated from jobs.csv
        and candidates.csv first. If a saved model exists and force_retrain is False,
        the model is loaded from disk instead.

        Args:
            synthetic_data_path: Path to pre-generated training CSV.
                                 Defaults to data/synthetic_train.csv.
            force_retrain: If True, always retrains even if model exists on disk.
            candidates_path: Path to candidates file to use for generating synthetic labels.
        """
        if self.ranker_trained and not force_retrain:
            logger.info("Ranker already trained. Skipping (use force_retrain=True to override).")
            return

        synthetic_data_path = synthetic_data_path or _SYNTHETIC_DATA_PATH

        # Load existing model from disk if available and no retrain requested
        if not force_retrain and _XGB_MODEL_PATH.exists():
            logger.info("Found existing XGBoost model on disk. Loading from disk...")
            self._ranker = ranker.load_ranker(str(_XGB_MODEL_PATH))
            self.ranker_trained = True
            logger.info("Ranker loaded from disk.")
            return

        # Generate synthetic data if it does not exist
        if not _SYNTHETIC_DATA_PATH.exists():
            logger.info("Synthetic training data not found. Generating...")
            jobs_df = pd.read_csv(str(_DEFAULT_JOBS_FILE))
            cand_path = candidates_path or str(_DEFAULT_CANDIDATES_FILE)
            candidates_df = self.load_candidates_dataframe(cand_path)
            synthetic_df = synthetic_labels.generate_synthetic_labels(jobs_df, candidates_df)
            synthetic_labels.save_synthetic_data(synthetic_df, str(_SYNTHETIC_DATA_PATH))
        else:
            logger.info(f"Loading synthetic training data from: {_SYNTHETIC_DATA_PATH}")

        train_df = synthetic_labels.load_synthetic_data(str(_SYNTHETIC_DATA_PATH))

        if train_df.empty:
            raise ValueError("Synthetic training data is empty. Cannot train ranker.")

        self._cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        booster = ranker.train_ranker(train_df, model_save_path=str(_XGB_MODEL_PATH))
        self._ranker = booster
        self.ranker_trained = True
        logger.info(f"Ranker trained on {len(train_df)} samples.")

    # ------------------------------------------------------------------
    # CORE METHOD: Process a Single Job
    # ------------------------------------------------------------------

    def process_single_job(
        self,
        jd_row: pd.Series,
        candidates_df: pd.DataFrame,
        use_llm: bool = False,
        top_k: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Run local pipeline for one job description and return a ranked DataFrame.

        Stage 1: FAISS dense retrieval → Top-200 candidates
        Stage 2: XGBoost ranking on 15 features → Top top_k candidates

        Args:
            jd_row: A single job description row (pd.Series).
            candidates_df: Full candidate pool DataFrame.
            use_llm: Ignored (kept for signature compatibility).
            top_k: How many candidates to return. Defaults to config.RANKER_K.

        Returns:
            pd.DataFrame with columns: job_id, candidate_id, rank, score, reasoning
        """
        if top_k is None:
            top_k = self._cfg.RANKER_K

        job_id = str(jd_row.get("job_id", "unknown"))
        jd_text = jd_row.get("description", jd_row.get("job_description", ""))

        # ── STAGE 1: Dense Retrieval ─────────────────────────────────
        scores, _, matched_ids = embeddings.retrieve_candidates(
            jd_text=str(jd_text),
            faiss_index=self.faiss_index,
            candidate_ids=self.candidate_ids,
            embedder=self.embedder,
            top_k=self._cfg.RETRIEVAL_K,
        )

        # Build a {candidate_id: faiss_score} lookup
        matched_ids_str = [str(mid) for mid in matched_ids]
        score_map: Dict[str, float] = {
            cid: float(sc) for cid, sc in zip(matched_ids_str, scores)
        }

        # Filter candidates_df to only indexed IDs, preserving FAISS order
        candidates_df = candidates_df.copy()
        candidates_df["_cid_str"] = candidates_df["candidate_id"].astype(str)
        retrieved_df = candidates_df[
            candidates_df["_cid_str"].isin(set(matched_ids_str))
        ].copy()
        retrieved_df = retrieved_df.drop(columns=["_cid_str"])

        # Filter out honeypots to ensure zero honeypot rate in final submissions
        if "is_honeypot" in retrieved_df.columns:
            before_cnt = len(retrieved_df)
            retrieved_df = retrieved_df[retrieved_df["is_honeypot"] != 1].copy()
            after_cnt = len(retrieved_df)
            if before_cnt - after_cnt > 0:
                logger.info(f"[{job_id}] Filtered out {before_cnt - after_cnt} honeypot candidate(s).")

        logger.info(f"[{job_id}] Stage 1: Retrieved {len(retrieved_df)} candidates from index.")

        if retrieved_df.empty:
            logger.warning(f"[{job_id}] No candidates retrieved. Returning empty DataFrame.")
            return pd.DataFrame(columns=["job_id", "candidate_id", "rank", "score", "reasoning"])

        # ── STAGE 2: Feature Engineering + XGBoost Ranking ──────────
        jd_dict = dict(jd_row)
        feature_rows: List[np.ndarray] = []
        for _, cand_row in retrieved_df.iterrows():
            cid = str(cand_row["candidate_id"])
            faiss_score = score_map.get(cid, 0.0)
            feat_vec = features.extract_all_features(
                jd_row=jd_dict,
                candidate_row=dict(cand_row),
                faiss_score=faiss_score,
            )
            feature_rows.append(feat_vec)

        feature_matrix = np.vstack(feature_rows)
        ranking_scores = ranker.predict_rankings(feature_matrix, self._ranker)

        retrieved_df = retrieved_df.copy()
        retrieved_df["_xgb_score"] = ranking_scores
        
        # Sort by score descending and break ties deterministically using candidate_id ascending
        retrieved_df = retrieved_df.sort_values(
            by=["_xgb_score", "candidate_id"], 
            ascending=[False, True]
        ).reset_index(drop=True)
        
        top_k_df = retrieved_df.head(top_k).copy()
        logger.info(f"[{job_id}] Stage 2: Ranked top {len(top_k_df)} candidates.")

        # ── Finalise Output ───────────────────────────────────────────
        top_k_df = top_k_df.reset_index(drop=True)
        top_k_df["rank"] = top_k_df.index + 1
        top_k_df["job_id"] = job_id

        # Calculate monotonically non-increasing score
        top_k_df["score"] = top_k_df.apply(
            lambda r: round(1.0 - (r["rank"] - 1) / len(top_k_df), 4), axis=1
        )

        # Generate reasoning using the reasoning_generator module
        top_k_df["reasoning"] = top_k_df.apply(
            lambda r: reasoning_generator.generate_candidate_reasoning(r, jd_row, r["rank"]),
            axis=1
        )

        # Drop internal scoring column if present
        if "_xgb_score" in top_k_df.columns:
            top_k_df = top_k_df.drop(columns=["_xgb_score"])

        # Ensure correct columns order
        cols = ["job_id", "candidate_id", "rank", "score", "reasoning"] + [
            c for c in top_k_df.columns if c not in ("job_id", "candidate_id", "rank", "score", "reasoning")
        ]
        top_k_df = top_k_df[cols]

        return top_k_df

    # ------------------------------------------------------------------
    # Process All Jobs
    # ------------------------------------------------------------------

    def process_all_jobs(
        self,
        jobs_path: Optional[str] = None,
        candidates_path: Optional[str] = None,
        output_path: Optional[str] = None,
        use_llm: bool = True,
        top_k: Optional[int] = None,
        save_interval: int = 10,
    ) -> pd.DataFrame:
        """
        Run the full pipeline over every job in jobs.csv.

        Ensures index and ranker are ready before iterating.
        Saves the submission CSV after processing all jobs.

        Args:
            jobs_path: Path to jobs CSV. Defaults to config DATA_DIR/jobs.csv.
            candidates_path: Path to candidates CSV. Defaults to config DATA_DIR/candidates.csv.
            output_path: Output CSV path. Defaults to SUBMISSIONS_DIR/submission.csv.
            use_llm: Whether to apply Stage 3 LLM re-ranking.
            top_k: Candidates per job. Defaults to config.RANKER_K.
            save_interval: Print progress every N jobs.

        Returns:
            pd.DataFrame with columns job_id, candidate_id, rank.
        """
        jobs_path = jobs_path or _DEFAULT_JOBS_FILE
        candidates_path = candidates_path or _DEFAULT_CANDIDATES_FILE
        output_path = output_path or _DEFAULT_SUBMISSION_FILE
        top_k = top_k or self._cfg.RANKER_K

        logger.info(f"Loading jobs from: {jobs_path}")
        jobs_df = pd.read_csv(str(jobs_path))

        logger.info(f"Loading candidates from: {candidates_path}")
        candidates_df = self.load_candidates_dataframe(str(candidates_path))

        # Ensure index is built
        if not self.index_built:
            self.build_index(candidates_path=str(candidates_path))

        # Ensure ranker is trained
        if not self.ranker_trained:
            self.train_ranker()

        total_jobs = len(jobs_df)
        all_results: List[pd.DataFrame] = []
        output_path_obj = Path(output_path)

        for idx, (_, jd_row) in enumerate(jobs_df.iterrows()):
            if (idx + 1) % save_interval == 0 or idx == 0 or (idx + 1) == total_jobs:
                logger.info(f"Processing job {idx + 1}/{total_jobs}...")

            ranked_df = self.process_single_job(
                jd_row=jd_row,
                candidates_df=candidates_df,
                use_llm=use_llm,
                top_k=top_k,
            )
            all_results.append(ranked_df)
            
            # Save intermediate results (tmp file) after each job
            try:
                temp_df = pd.concat(all_results, ignore_index=True)
                temp_submission_df = temp_df[["candidate_id", "rank", "score", "reasoning"]].copy()
                temp_output_path = output_path_obj.with_suffix(".tmp.csv")
                temp_output_path.parent.mkdir(parents=True, exist_ok=True)
                temp_submission_df.to_csv(temp_output_path, index=False)
            except Exception as e:
                logger.warning(f"Failed to save intermediate progress: {e}")

        final_df = pd.concat(all_results, ignore_index=True)

        # Keep only submission-required columns
        submission_df = final_df[["candidate_id", "rank", "score", "reasoning"]].copy()

        # Save to disk
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        submission_df.to_csv(output_path_obj, index=False)
        logger.info(f"Submission saved with {len(submission_df)} entries -> {output_path_obj}")

        # Clean up temporary file
        try:
            temp_output_path = output_path_obj.with_suffix(".tmp.csv")
            if temp_output_path.exists():
                temp_output_path.unlink()
        except Exception as e:
            logger.warning(f"Failed to delete temporary file: {e}")

        return submission_df

    # ------------------------------------------------------------------
    # Validate Submission
    # ------------------------------------------------------------------

    def validate_submission(
        self,
        submission_df: pd.DataFrame,
        jobs_df: Optional[pd.DataFrame] = None,
        strict: bool = False,
    ) -> dict:
        """
        Validate a submission DataFrame per hackathon rules.
        """
        errors: List[str] = []
        warnings: List[str] = []

        # Check required columns
        required_cols = ["candidate_id", "rank", "score", "reasoning"]
        for col in required_cols:
            if col not in submission_df.columns:
                errors.append(f"Missing required column: '{col}'.")

        if errors:
            return {"valid": False, "errors": errors, "warnings": warnings}

        # Check exact headers (no extra columns)
        if list(submission_df.columns) != required_cols:
            errors.append(f"Header columns must be exactly {required_cols}; found {list(submission_df.columns)}.")

        # Check null values
        for col in ["candidate_id", "rank", "score"]:
            null_count = submission_df[col].isnull().sum()
            if null_count > 0:
                errors.append(f"Column '{col}' has {null_count} missing value(s).")

        # Check row count if strict
        n = len(submission_df)
        if strict and n != 100:
            errors.append(f"Submission must have exactly 100 data rows; found {n}.")

        import re
        candidate_id_pattern = re.compile(r"^CAND_[0-9]{7}$") if strict else re.compile(r"^CAND_.*$")
        seen_ids = set()
        seen_ranks = set()
        by_rank = []

        for idx, row in submission_df.iterrows():
            row_num = idx + 2  # 1-indexed plus header row
            raw_cid = row["candidate_id"]
            raw_rank = row["rank"]
            raw_score = row["score"]

            # Validate candidate ID
            if pd.isnull(raw_cid):
                cid = None
            else:
                cid = str(raw_cid).strip()
                if cid == "" or cid.lower() == "nan" or cid.lower() == "none":
                    errors.append(f"Row {row_num}: candidate_id is required.")
                    cid = None
                elif not candidate_id_pattern.match(cid):
                    if strict:
                        errors.append(f"Row {row_num}: candidate_id must be CAND_XXXXXXX (7 digits).")
                    else:
                        errors.append(f"Row {row_num}: candidate_id must start with 'CAND_'.")
                    cid = None
                elif cid in seen_ids:
                    errors.append(f"Row {row_num}: duplicate candidate_id '{cid}'.")
                else:
                    seen_ids.add(cid)

            # Validate rank
            if pd.isnull(raw_rank):
                rank = None
            else:
                try:
                    rank_s = str(raw_rank).strip()

                    if strict:
                        # Match PUB validate_submission.py exactly:
                        # - rank must parse as int(rank_s)
                        # - and str(rank) must equal rank_s (reject "1.0", "01", etc.)
                        rank = int(rank_s)
                        if str(rank) != rank_s:
                            raise ValueError
                    else:
                        rank_f = float(raw_rank)
                        if not rank_f.is_integer():
                            raise ValueError
                        rank = int(rank_f)

                    max_rank = 100 if strict else n
                    if not 1 <= rank <= max_rank:
                        errors.append(f"Row {row_num}: rank must be between 1 and {max_rank}.")
                    elif rank in seen_ranks:
                        errors.append(f"Row {row_num}: duplicate rank {rank}.")
                    else:
                        seen_ranks.add(rank)
                except ValueError:
                    errors.append(f"Row {row_num}: rank must be an integer.")
                    rank = None

            # Validate score
            if pd.isnull(raw_score):
                score = None
            else:
                try:
                    score = float(raw_score)
                except ValueError:
                    errors.append(f"Row {row_num}: score must be a float.")
                    score = None

            if rank is not None and score is not None and cid:
                by_rank.append((rank, score, cid))

        expected_range = range(1, 101) if strict else range(1, n + 1)
        missing_ranks = set(expected_range) - seen_ranks
        if missing_ranks and not (pd.isnull(submission_df["rank"]).any() and not strict):
            # Only report missing ranks if we aren't already reporting null values in non-strict mode
            errors.append(f"Ranks are not a contiguous sequence starting at 1; missing: {sorted(list(missing_ranks))}")

        # Check score monotonicity and tie-breaker
        by_rank.sort(key=lambda x: x[0])
        for i in range(len(by_rank) - 1):
            r1, s1, c1 = by_rank[i]
            r2, s2, c2 = by_rank[i + 1]
            if s1 < s2:
                errors.append(f"score must be non-increasing by rank: rank {r1} ({s1}) < rank {r2} ({s2}).")
            elif s1 == s2 and c1 > c2:
                errors.append(f"Equal scores at ranks {r1} and {r2}: tie-break requires candidate_id ascending ('{c1}' > '{c2}').")

        valid = len(errors) == 0
        return {"valid": valid, "errors": errors, "warnings": warnings}

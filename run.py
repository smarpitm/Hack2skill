"""
run.py
CLI entry point for the AI Candidate Ranking System.
Usage: python run.py [--options]
"""

import argparse
import sys
from pathlib import Path
from src.pipeline import CandidateRankingPipeline
from src import config


def main():
    parser = argparse.ArgumentParser(
        description="AI Candidate Ranking System — Hack2skill Hackathon"
    )
    parser.add_argument(
        "--data_path", default=str(config.DATA_DIR),
        help="Path to data directory (must contain jobs.csv and candidates.csv)"
    )
    parser.add_argument(
        "--output", default=str(config.SUBMISSIONS_DIR / "submission.csv"),
        help="Output path for the ranked submission CSV"
    )
    parser.add_argument(
        "--top_k", type=int, default=100,
        help="Number of candidates to rank per job (default: 100)"
    )
    parser.add_argument(
        "--use_llm", action="store_true", default=True,
        help="Enable GROQ LLM re-ranking Stage 3 (default: enabled)"
    )
    parser.add_argument(
        "--no_llm", action="store_true",
        help="Disable GROQ LLM re-ranking (overrides --use_llm)"
    )
    parser.add_argument(
        "--build_index", action="store_true",
        help="Force rebuild of FAISS index even if one already exists"
    )
    parser.add_argument(
        "--train_ranker", action="store_true",
        help="Force retrain of XGBoost ranker even if a model already exists"
    )
    args = parser.parse_args()

    use_llm = False if args.no_llm else args.use_llm

    # Ensure required directories exist
    config.MODELS_DIR.mkdir(exist_ok=True, parents=True)
    config.SUBMISSIONS_DIR.mkdir(exist_ok=True, parents=True)
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data_path)
    candidates_csv = str(data_path / "candidates.csv")
    jobs_csv = str(data_path / "jobs.csv")
    faiss_index_file = config.MODELS_DIR / "candidates.index"
    xgb_model_file = config.MODELS_DIR / "xgb_ranker.json"

    print("\n=== AI Candidate Ranking System ===")
    print(f"  Data path   : {data_path}")
    print(f"  Output      : {args.output}")
    print(f"  Top-K       : {args.top_k}")
    print(f"  LLM stage   : {'enabled' if use_llm else 'disabled'}")
    print("===================================\n")

    pipeline = CandidateRankingPipeline()

    # Build/load FAISS index
    if args.build_index or not faiss_index_file.exists():
        print("Building FAISS index...")
        pipeline.build_index(candidates_path=candidates_csv, force_rebuild=args.build_index)
    else:
        print("Loading existing FAISS index from disk...")
        pipeline.build_index(candidates_path=candidates_csv, force_rebuild=False)

    # Train/load XGBoost ranker
    if args.train_ranker or not xgb_model_file.exists():
        print("Training XGBoost ranker...")
        pipeline.train_ranker(force_retrain=args.train_ranker)
    else:
        print("Loading existing XGBoost ranker from disk...")
        pipeline.train_ranker(force_retrain=False)

    print("\nProcessing jobs...\n")
    submission = pipeline.process_all_jobs(
        jobs_path=jobs_csv,
        candidates_path=candidates_csv,
        output_path=args.output,
        use_llm=use_llm,
        top_k=args.top_k,
    )

    print("\nValidating submission...")
    validation = pipeline.validate_submission(submission, strict=True)

    if not validation["valid"]:
        print("FAILED: Submission validation failed:")
        for error in validation["errors"]:
            print(f"  ERROR  : {error}")
        sys.exit(1)

    print(f"SUCCESS: Submission validated successfully - {len(submission)} rows saved to {args.output}")


if __name__ == "__main__":
    main()

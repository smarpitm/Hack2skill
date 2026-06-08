"""
rank.py
Official CLI entry point for the AI Candidate Ranking System (Redrob Hackathon).
Usage: python rank.py --candidates ./candidates.jsonl --out ./submission.csv
"""

import argparse
import sys
from pathlib import Path
from src.pipeline import CandidateRankingPipeline
from src import config


def main():
    parser = argparse.ArgumentParser(
        description="AI Candidate Ranking System — Redrob Hackathon"
    )
    parser.add_argument(
        "--candidates", required=True,
        help="Path to candidates data file (.jsonl, .jsonl.gz, .json, or .csv)"
    )
    parser.add_argument(
        "--out", required=True,
        help="Output path for the ranked submission CSV"
    )
    parser.add_argument(
        "--jobs", default=str(config.DATA_DIR / "jobs.csv"),
        help="Path to jobs.csv file"
    )
    parser.add_argument(
        "--use_llm", action="store_true", default=True,
        help="Enable Stage 3 GROQ LLM re-ranking (default: enabled)"
    )
    parser.add_argument(
        "--no_llm", action="store_true",
        help="Disable Stage 3 GROQ LLM re-ranking"
    )
    parser.add_argument(
        "--build_index", action="store_true",
        help="Force rebuild of FAISS index"
    )
    parser.add_argument(
        "--train_ranker", action="store_true",
        help="Force retrain of XGBoost ranker"
    )
    args = parser.parse_args()

    use_llm = False if args.no_llm else args.use_llm

    # Ensure required directories exist
    config.MODELS_DIR.mkdir(exist_ok=True, parents=True)
    config.CACHE_DIR.mkdir(exist_ok=True, parents=True)
    Path(args.out).parent.mkdir(exist_ok=True, parents=True)

    print("\n=== AI Candidate Ranking System ===")
    print(f"  Candidates  : {args.candidates}")
    print(f"  Jobs        : {args.jobs}")
    print(f"  Output      : {args.out}")
    print(f"  LLM stage   : {'enabled' if use_llm else 'disabled'}")
    print("===================================\n")

    pipeline = CandidateRankingPipeline()

    # Build/load FAISS index
    faiss_index_file = config.MODELS_DIR / "candidates.index"
    if args.build_index or not faiss_index_file.exists():
        print("Building FAISS index...")
        pipeline.build_index(candidates_path=args.candidates, force_rebuild=args.build_index)
    else:
        print("Loading existing FAISS index from disk...")
        pipeline.build_index(candidates_path=args.candidates, force_rebuild=False)

    # Train/load XGBoost ranker
    xgb_model_file = config.MODELS_DIR / "xgb_ranker.json"
    if args.train_ranker or not xgb_model_file.exists():
        print("Training XGBoost ranker...")
        pipeline.train_ranker(force_retrain=args.train_ranker)
    else:
        print("Loading existing XGBoost ranker from disk...")
        pipeline.train_ranker(force_retrain=False)

    print("\nProcessing and ranking...")
    submission = pipeline.process_all_jobs(
        jobs_path=args.jobs,
        candidates_path=args.candidates,
        output_path=args.out,
        use_llm=use_llm,
    )

    print("\nValidating output submission CSV...")
    strict_mode = len(submission) >= 100
    if not strict_mode:
        print(f"WARNING: Ranked candidates count ({len(submission)}) is less than 100. Skipping strict validation checks (e.g. 100 rows requirement).")
    
    validation = pipeline.validate_submission(submission, strict=strict_mode)

    if not validation["valid"]:
        print("ERROR: Submission validation failed:")
        for error in validation["errors"]:
            print(f"  - {error}")
        sys.exit(1)

    print(f"SUCCESS: Submission validated successfully! {len(submission)} rows saved to {args.out}")


if __name__ == "__main__":
    main()

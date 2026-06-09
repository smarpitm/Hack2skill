"""
src/data_loader.py

Data loading and parsing utilities for candidate resume data.
Supports CSV, JSON, JSONL, and JSONL.GZ formats.
"""

import gzip
import json
import logging
from pathlib import Path
import pandas as pd

from . import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_honeypot_candidate(cand_dict: dict) -> bool:
    """
    Helper function to detect honeypot candidate on the fly.
    Checks for impossible profile configurations.
    """
    skills = cand_dict.get("skills", [])
    expert_zero_duration_count = 0
    for s in skills:
        prof = str(s.get("proficiency", "")).lower()
        dur = s.get("duration_months", 0)
        if prof in ("expert", "advanced") and dur == 0:
            expert_zero_duration_count += 1
            
    history = cand_dict.get("career_history", [])
    recent_startups = {"krutrim", "sarvam ai"}
    foundation_anomaly = False
    for job in history:
        comp = str(job.get("company", "")).lower()
        start = job.get("start_date")
        duration = job.get("duration_months", 0)
        
        if comp in recent_startups:
            if start:
                try:
                    start_year = int(start.split("-")[0])
                    if start_year < 2023 or duration > 36:
                        foundation_anomaly = True
                except Exception:
                    continue
    return expert_zero_duration_count >= 3 or foundation_anomaly


def load_candidates_dataframe(path: str) -> pd.DataFrame:
    """
    Load candidates data from CSV, JSON, JSONL, or JSONL.GZ file.
    Parses and flattens JSON fields to match the expected DataFrame schema.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Candidate data file not found at: {path}")

    if path_obj.suffix.lower() == ".csv":
        logger.info(f"Reading candidates CSV: {path_obj}")
        df = pd.read_csv(str(path_obj))
        # Ensure is_honeypot is present, default to 0 if not present
        if "is_honeypot" not in df.columns:
            df["is_honeypot"] = 0
        return df

    logger.info(f"Parsing candidates JSON/JSONL: {path_obj}")
    records = []
    candidates_list = None

    # Read JSON file directly if it's a JSON array
    if path_obj.suffix.lower() == ".json":
        with open(str(path_obj), "r", encoding="utf-8") as f:
            candidates_list = json.load(f)
            if isinstance(candidates_list, dict):
                candidates_list = [candidates_list]

    if candidates_list is None:
        # Read JSONL or JSONL.GZ line-by-line
        is_gz = path_obj.suffix.lower() == ".gz" or path_obj.name.lower().endswith(".jsonl.gz")
        open_func = gzip.open if is_gz else open
        mode = "rt" if is_gz else "r"
        candidates_list = []
        with open_func(str(path_obj), mode, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                candidates_list.append(json.loads(line))

    for cand in candidates_list:
        profile = cand.get("profile", {})
        candidate_id = cand.get("candidate_id")
        headline = profile.get("headline", "")
        summary = profile.get("summary", "")
        location = profile.get("location", "")
        country = profile.get("country", "")
        years_exp = profile.get("years_of_experience", 0.0)
        current_title = profile.get("current_title", "")
        current_company = profile.get("current_company", "")
        
        skill_list = [s.get("name", "") for s in cand.get("skills", [])]
        skills_str = ", ".join(skill_list)
        
        edu_list = cand.get("education", [])
        edu_parts = []
        for edu in edu_list:
            edu_parts.append(f"{edu.get('degree', '')} from {edu.get('institution', '')} (tier: {edu.get('tier', 'unknown')})")
        edu_str = "; ".join(edu_parts) if edu_parts else "None"
        
        # Recreate resume_text matching clean_dataset.py exactly
        resume_parts = [
            f"Name: {profile.get('anonymized_name', '')}",
            f"Headline: {headline}",
            f"Summary: {summary}",
            f"Current Position: {current_title} at {current_company}",
            f"Location: {location}, {country}",
            f"Education: {edu_str}"
        ]
        
        history_parts = []
        for job in cand.get("career_history", []):
            history_parts.append(
                f"Role: {job.get('title', '')} at {job.get('company', '')} "
                f"({job.get('duration_months', 0)} months). Description: {job.get('description', '')}"
            )
        if history_parts:
            resume_parts.append("Work History: " + "; ".join(history_parts))
            
        resume_text = " | ".join(resume_parts)
        
        signals = cand.get("redrob_signals", {})
        platform_activity = signals.get("profile_completeness_score", 0.0)
        
        is_hp = 1 if is_honeypot_candidate(cand) else 0
        
        records.append({
            "candidate_id": candidate_id,
            "resume_text": resume_text,
            "skills": skills_str,
            "experience_years": years_exp,
            "education": edu_str,
            "location": location,
            "current_title": current_title,
            "platform_activity_score": platform_activity,
            "is_honeypot": is_hp
        })
        
    logger.info(f"Successfully loaded {len(records)} candidates from JSON/JSONL.")
    return pd.DataFrame(records)

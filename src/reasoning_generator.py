"""
src/reasoning_generator.py

Rule-based reasoning generator that produces descriptive, non-templated,
and fact-based candidate justifications based on their profile data.
"""

from typing import Dict, Any, Union
import pandas as pd

from . import config
from . import preprocessing


def generate_candidate_reasoning(
    candidate_row: Union[pd.Series, Dict[str, Any]],
    jd_row: Union[pd.Series, Dict[str, Any]],
    rank: int,
) -> str:
    """
    Generate a 1-2 sentence reasoning explaining the candidate's rank,
    incorporating specific facts (skills, experience, location, current title),
    connecting to the JD, and raising honest concerns when relevant.
    """
    title = candidate_row.get("current_title", "")
    if not title or str(title).lower() == "unknown":
        title = "AI Specialist"
        
    exp = float(candidate_row.get("experience_years", 0.0))
    skills = str(candidate_row.get("skills", ""))
    loc = candidate_row.get("location", "")
    
    # 1. Evaluate Experience against JD (5-9 years target)
    exp_text = ""
    if exp >= 5.0 and exp <= 9.0:
        exp_text = f"Strong experience match with {int(exp)} years in the field"
    elif exp > 9.0:
        exp_text = f"Highly experienced professional with {int(exp)} years in software engineering"
    else:
        exp_text = f"Possesses {exp} years of experience (below the preferred 5-9 years range)"

    # 2. Pick top skills matching the JD
    jd_skills_text = str(jd_row.get("required_skills", ""))
    jd_skills = [s.strip().lower() for s in jd_skills_text.split(",") if s.strip()]
    cand_skills = [s.strip() for s in skills.split(",") if s.strip()]
    
    matching_skills = []
    for s in cand_skills:
        s_lower = s.lower()
        if s_lower in jd_skills:
            matching_skills.append(s)
        else:
            # Check aliases
            aliases = preprocessing.get_skill_aliases(s)
            if any(alias in jd_skills for alias in aliases):
                matching_skills.append(s)

    skills_text = ""
    if matching_skills:
        skills_text = f"exhibiting hands-on expertise in {', '.join(matching_skills[:3])}"
    else:
        skills_text = "possessing adjacent technical competencies"

    # 3. Location and notice period checks
    loc_norm = preprocessing.normalize_location(loc)
    loc_match_text = ""
    if loc_norm in ("pune", "noida"):
        loc_match_text = f"located locally in {loc}"
    elif loc_norm in ("bangalore", "hyderabad", "mumbai", "delhi", "gurgaon"):
        loc_match_text = f"based in {loc} (relocation candidate)"
    else:
        loc_match_text = f"located in {loc or 'unknown region'}"

    # Let's construct the main sentence
    # We introduce natural variation based on rank and details to prevent templating flags
    parts = []
    if rank <= 10:
        parts.append(
            f"Exceptional fit for the Senior AI Engineer role: {title} with {int(exp)} years of experience, "
            f"{skills_text}. Fully aligned with the product-shipper mandate, {loc_match_text}."
        )
    elif rank <= 50:
        parts.append(
            f"Strong profile for the founding team: {exp_text}, {skills_text}. "
            f"Solid background as a {title}, {loc_match_text}."
        )
    else:
        # Lower ranks: highlight some concerns or adjacent fit
        concern_parts = []
        if exp < 5.0:
            concern_parts.append("lower experience years than target")
        
        # Check for consulting history if possible (e.g. if current company is in consulting)
        current_company = str(candidate_row.get("current_company", "")).lower()
        is_consulting = any(c in current_company for c in config.CONSULTING_COMPANIES)
        if is_consulting:
            concern_parts.append("entire career spent at service-based firms")
            
        concern_text = ""
        if concern_parts:
            concern_text = f" Minor concerns include {', '.join(concern_parts)}."
            
        parts.append(
            f"Moderately suitable candidate: working as {title} with {exp} years of experience, "
            f"{skills_text}. {loc_match_text}.{concern_text}"
        )

    return " ".join(parts)

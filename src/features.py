"""
src/features.py

Feature engineering module to extract candidate-job similarity features.
Calculates exactly 15 quantitative features used to train the XGBoost ranker.
"""

import re
import logging
from typing import List, Tuple, Dict, Any, Union
import numpy as np
import pandas as pd

from . import config
from . import preprocessing

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_skill_match(jd_skills: List[str], resume_skills: List[str]) -> Tuple[int, float]:
    """
    Compute count and ratio of skill intersection, taking skill aliases/variations into account.

    Args:
        jd_skills (List[str]): List of required skills.
        resume_skills (List[str]): List of candidate skills.

    Returns:
        Tuple[int, float]: (skill_match_count, skill_match_ratio)
    """
    if not jd_skills:
        return 0, 0.0
        
    # Generate alias sets for each resume skill
    resume_aliases_list = []
    for s in resume_skills:
        resume_aliases_list.append(set(preprocessing.get_skill_aliases(s)))
        
    # Check each JD skill
    match_count = 0
    for jd_s in jd_skills:
        jd_aliases = set(preprocessing.get_skill_aliases(jd_s))
        # Check if this JD skill matches any candidate skill (overlapping aliases)
        for res_aliases in resume_aliases_list:
            if jd_aliases.intersection(res_aliases):
                match_count += 1
                break
                
    ratio = match_count / len(jd_skills)
    return match_count, ratio


def compute_experience_match(jd_exp_required: float, resume_exp: float) -> float:
    """
    Compare candidate's experience years against job description requirement using step functions.

    Args:
        jd_exp_required (float): Required experience years.
        resume_exp (float): Candidate's experience years.

    Returns:
        float: Normalized match score.
    """
    if jd_exp_required <= 0:
        return config.DEFAULT_EXPERIENCE_MATCH
        
    if resume_exp >= jd_exp_required:
        return 1.0
    elif resume_exp >= jd_exp_required * 0.7:
        return 0.7
    elif resume_exp >= jd_exp_required * 0.5:
        return 0.4
    else:
        return 0.1


def compute_education_match(
    jd_edu_level: int, 
    resume_edu_level: int, 
    resume_college_tier: int = 0
) -> float:
    """
    Compare education levels and add college tier bonuses.

    Args:
        jd_edu_level (int): Required education level (1-5).
        resume_edu_level (int): Candidate's education level (1-5).
        resume_college_tier (int): Candidate's college tier (0=unknown, 1=tier 1, 2=tier 2, 3=others).

    Returns:
        float: Standardized education score capped at 1.0.
    """
    if jd_edu_level <= 0:
        return config.DEFAULT_EDUCATION_MATCH
        
    if resume_edu_level >= jd_edu_level:
        base_score = 1.0
    elif resume_edu_level >= jd_edu_level - 1:
        base_score = 0.7
    else:
        base_score = 0.3
        
    # College tier bonus
    bonus = 0.0
    if resume_college_tier == 1:
        bonus = 0.05
    elif resume_college_tier == 2:
        bonus = 0.02
        
    return min(base_score + bonus, 1.0)


def compute_location_match(jd_location: str, resume_location: str) -> float:
    """
    Compute location score checking for exact match, metro region, national, or international mismatches.

    Args:
        jd_location (str): Raw job location.
        resume_location (str): Raw candidate location.

    Returns:
        float: Location match score (0.0 to 1.0).
    """
    # Normalize locations
    jd_loc_norm = preprocessing.normalize_location(jd_location)
    res_loc_norm = preprocessing.normalize_location(resume_location)
    
    if jd_loc_norm == "unknown" or res_loc_norm == "unknown":
        return config.DEFAULT_LOCATION_MATCH
        
    if jd_loc_norm == res_loc_norm:
        return 1.0
        
    # Same NCR metro region (Delhi, Gurgaon, Noida)
    ncr_region = {"delhi", "gurgaon", "noida"}
    if jd_loc_norm in ncr_region and res_loc_norm in ncr_region:
        return 0.8
        
    # Check if both are in India
    # If normalized location is in our location map, we know it's a valid Indian city
    valid_cities = set(config.LOCATION_NORMALIZATION_MAP.values())
    if jd_loc_norm in valid_cities and res_loc_norm in valid_cities:
        return 0.3
        
    # Check if either location contains outside India indicators
    outside_keywords = ["usa", "uk", "united states", "london", "singapore", "dubai", "canada", "germany", "australia"]
    is_jd_outside = any(kw in str(jd_location).lower() for kw in outside_keywords)
    is_res_outside = any(kw in str(resume_location).lower() for kw in outside_keywords)
    
    if is_jd_outside or is_res_outside:
        return 0.0
        
    return 0.3


def compute_semantic_similarity(faiss_score: float) -> float:
    """
    Return FAISS dense similarity score directly.

    Args:
        faiss_score (float): FAISS cosine similarity.

    Returns:
        float: Cosine similarity score.
    """
    return float(faiss_score)


def compute_platform_activity(candidate_row: Union[pd.Series, Dict[str, Any]]) -> float:
    """
    Extract and normalize candidate platform activity score.

    Args:
        candidate_row: The candidate record.

    Returns:
        float: Activity score between 0.0 and 1.0.
    """
    # Look up potential column variants
    val = candidate_row.get("platform_activity_score")
    if val is None:
        val = candidate_row.get("activity_score")
    if val is None:
        val = candidate_row.get("engagement_score")
    if val is None:
        val = candidate_row.get("platform_activity")
        
    if val is None or (isinstance(val, float) and val != val):
        return config.DEFAULT_PLATFORM_ACTIVITY
        
    try:
        score = float(val)
        # Handle cases where score is given in percentage or scales other than 0-1
        if score > 1.0:
            if score <= 100.0:
                score = score / 100.0
            else:
                score = 1.0
        return max(0.0, min(score, 1.0))
    except (ValueError, TypeError):
        return config.DEFAULT_PLATFORM_ACTIVITY


def compute_career_progression(current_title: str, years_exp: float) -> float:
    """
    Evaluate career progression based on leadership keywords in job title and experience years.

    Args:
        current_title (str): Cleaned current job title.
        years_exp (float): Years of experience.

    Returns:
        float: Career progression score.
    """
    if not current_title or current_title.lower() == "unknown":
        return 0.3
        
    title_lower = current_title.lower()
    senior_indicators = [
        "senior", "lead", "manager", "principal", "architect", 
        "director", "head", "staff", "vp", "cto", "ceo"
    ]
    
    has_indicator = any(kw in title_lower for kw in senior_indicators)
    
    if has_indicator and years_exp >= 5:
        return 1.0
    elif has_indicator and years_exp >= 3:
        return 0.8
    elif years_exp >= 5:
        return 0.7
    elif years_exp >= 3:
        return 0.5
    else:
        return 0.3


def compute_resume_completeness(candidate_row: Union[pd.Series, Dict[str, Any]]) -> float:
    """
    Measure completeness of candidate resume profile.

    Args:
        candidate_row: Candidate record.

    Returns:
        float: Percentage of filled fields.
    """
    key_fields = ["resume_text", "skills", "experience_years", "education", "current_title"]
    filled_count = 0
    total_fields = len(key_fields)
    
    # We also check for fallback keys
    for field in key_fields:
        val = candidate_row.get(field)
        if field == "experience_years" and val is None:
            val = candidate_row.get("experience")
            
        if val is not None and not (isinstance(val, float) and val != val):
            if isinstance(val, (list, set, tuple)):
                if len(val) > 0:
                    filled_count += 1
            elif str(val).strip() != "" and str(val).lower() != "unknown":
                filled_count += 1
                
    return filled_count / total_fields if total_fields else 0.0


def compute_keyword_density(jd_text: str, resume_text: str) -> float:
    """
    Measure keyword density matching percentage of unique non-stopword JD terms present in resume.

    Args:
        jd_text (str): Job description text.
        resume_text (str): Candidate resume text.

    Returns:
        float: Match density ratio.
    """
    if not jd_text or not resume_text:
        return 0.0
        
    stopwords = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
        "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", 
        "by", "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", 
        "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "him", "his", 
        "how", "i", "if", "in", "into", "is", "it", "its", "me", "more", "most", "my", "myself", "no", 
        "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", 
        "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such", "than", 
        "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", 
        "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", 
        "what", "when", "where", "which", "while", "who", "whom", "why", "with", "would", "you", "your", 
        "yours", "yourself", "yourselves"
    }
    
    jd_clean = preprocessing.clean_text(jd_text)
    res_clean = preprocessing.clean_text(resume_text)
    
    jd_words = [w for w in jd_clean.split() if w not in stopwords and len(w) > 2]
    res_word_set = set(res_clean.split())
    
    if not jd_words:
        return 0.0
        
    unique_jd_words = set(jd_words)
    matches = sum(1 for w in unique_jd_words if w in res_word_set)
    
    return matches / len(unique_jd_words)


def compute_project_diversity(resume_text: str) -> float:
    """
    Measure project domain diversity based on indicators in matching resume sentences.

    Args:
        resume_text (str): Candidate resume text.

    Returns:
        float: Project diversity score.
    """
    if not resume_text or (isinstance(resume_text, float) and resume_text != resume_text):
        return 0.0
        
    resume_lower = str(resume_text).lower()
    
    # Split text into clauses or sentences
    sentences = re.split(r'[\n.!?•;]', resume_lower)
    
    project_indicators = ["project", "built", "developed", "created", "implemented"]
    domains = [
        "web", "frontend", "backend", "fullstack", "mobile", "ios", "android", 
        "cloud", "devops", "machine learning", "data science", "nlp", 
        "computer vision", "blockchain", "database", "cybersecurity", "embedded", "iot"
    ]
    
    found_domains = set()
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        # Check if sentence mentions a project indicator
        if any(ind in sentence for ind in project_indicators):
            # Check which domains appear in this sentence
            for domain in domains:
                if domain in sentence:
                    found_domains.add(domain)
                    
    count = len(found_domains)
    return min(count / 5.0, 1.0)


def compute_current_title_match(jd_title: str, resume_title: str) -> float:
    """
    Lexical match between job description title and candidate's current job title.

    Args:
        jd_title (str): Target job title.
        resume_title (str): Candidate's current title.

    Returns:
        float: Match score (0.2, 0.6, or 1.0).
    """
    jd_clean = preprocessing.clean_text(jd_title)
    res_clean = preprocessing.clean_text(resume_title)
    
    if not jd_clean or jd_clean == "unknown" or not res_clean or res_clean == "unknown":
        return 0.2
        
    jd_words = set(w for w in jd_clean.split() if len(w) > 2)
    res_words = set(w for w in res_clean.split() if len(w) > 2)
    
    if not jd_words or not res_words:
        return 0.2
        
    intersection = jd_words.intersection(res_words)
    if not intersection:
        return 0.2
        
    # Check ratio of match
    ratio = len(intersection) / len(jd_words)
    if ratio >= 0.5:
        return 1.0
    else:
        return 0.6


def extract_all_features(
    jd_row: Union[pd.Series, Dict[str, Any]], 
    candidate_row: Union[pd.Series, Dict[str, Any]], 
    faiss_score: float = 0.0
) -> np.ndarray:
    """
    Extract and compile all 15 features in the designated order.

    Args:
        jd_row: The job description record.
        candidate_row: The candidate profile record.
        faiss_score (float): The similarity score from dense retrieval.

    Returns:
        np.ndarray: A 1D array of shape (15,).
    """
    # --- 1. EXTRACT DATA SAFELY ---
    # Job Info
    jd_text = jd_row.get("description", jd_row.get("job_description", ""))
    jd_skills_raw = jd_row.get("skills", jd_row.get("required_skills", []))
    jd_exp_req = jd_row.get("experience_years")
    if jd_exp_req is None:
        jd_exp_req = jd_row.get("experience")
    if jd_exp_req is None:
        jd_exp_req = jd_row.get("experience_required", 0.0)
    jd_edu_raw = jd_row.get("education", jd_row.get("education_required", ""))
    jd_location = jd_row.get("location", "")
    jd_title = jd_row.get("title", jd_row.get("job_title", ""))
    
    # Candidate Info
    resume_text = candidate_row.get("resume_text", "")
    candidate_skills_raw = candidate_row.get("skills", [])
    candidate_exp = candidate_row.get("experience_years", 0.0)
    if candidate_exp is None:
        candidate_exp = candidate_row.get("experience", 0.0)
    candidate_edu_raw = candidate_row.get("education", "")
    candidate_location = candidate_row.get("location", "")
    candidate_title = candidate_row.get("current_title", "")
    if candidate_title is None or candidate_title == "":
        candidate_title = candidate_row.get("title", "")
        
    # Parse exp and titles if raw strings are present instead of pre-parsed values
    if isinstance(jd_exp_req, str):
        jd_exp_req = preprocessing.extract_experience_years(jd_exp_req)
    if isinstance(candidate_exp, str) or (isinstance(candidate_exp, float) and candidate_exp == 0.0 and resume_text):
        # Fallback to extract from resume text if not pre-extracted
        extracted_exp = preprocessing.extract_experience_years(resume_text)
        if extracted_exp > 0:
            candidate_exp = extracted_exp
            
    if not candidate_title or candidate_title == "unknown":
        extracted_title = preprocessing.extract_current_title(resume_text)
        if extracted_title != "unknown":
            candidate_title = extracted_title

    # Clean and parse skills if passed as raw text instead of list
    if isinstance(jd_skills_raw, str):
        jd_skills = preprocessing.extract_skills(jd_skills_raw)
    elif isinstance(jd_skills_raw, list):
        jd_skills = [str(s) for s in jd_skills_raw]
    else:
        jd_skills = []
        
    if isinstance(candidate_skills_raw, str):
        candidate_skills = preprocessing.extract_skills(candidate_skills_raw)
    elif isinstance(candidate_skills_raw, list):
        candidate_skills = [str(s) for s in candidate_skills_raw]
    else:
        # Fallback to extract from resume text if list is empty
        candidate_skills = preprocessing.extract_skills(resume_text)

    # Education parsing
    if isinstance(jd_edu_raw, (int, float)):
        jd_edu_level = int(jd_edu_raw)
    else:
        jd_edu_level, _, _ = preprocessing.normalize_education(jd_edu_raw)
        
    if isinstance(candidate_edu_raw, (int, float)):
        res_edu_level = int(candidate_edu_raw)
        res_tier = 0
    else:
        res_edu_level, _, res_tier = preprocessing.normalize_education(candidate_edu_raw)
        if res_edu_level == 0 and resume_text:
            # Fallback to extract from resume text
            res_edu_level, _, res_tier = preprocessing.normalize_education(resume_text)

    # Set defaults if values are missing
    try:
        jd_exp_req = float(jd_exp_req)
    except (ValueError, TypeError):
        jd_exp_req = 0.0
        
    try:
        candidate_exp = float(candidate_exp)
    except (ValueError, TypeError):
        candidate_exp = 0.0

    # --- 2. COMPUTE FEATURES ---
    skill_count, skill_ratio = compute_skill_match(jd_skills, candidate_skills)
    exp_match = compute_experience_match(jd_exp_req, candidate_exp)
    edu_match = compute_education_match(jd_edu_level, res_edu_level, res_tier)
    loc_match = compute_location_match(jd_location, candidate_location)
    sem_sim = compute_semantic_similarity(faiss_score)
    plat_act = compute_platform_activity(candidate_row)
    career_prog = compute_career_progression(candidate_title, candidate_exp)
    
    # Update candidate_row to contain parsed details for completeness check
    candidate_row_parsed = dict(candidate_row)
    candidate_row_parsed["resume_text"] = resume_text
    candidate_row_parsed["skills"] = candidate_skills
    candidate_row_parsed["experience_years"] = candidate_exp
    candidate_row_parsed["education"] = candidate_edu_raw
    candidate_row_parsed["current_title"] = candidate_title
    res_complete = compute_resume_completeness(candidate_row_parsed)
    
    res_len = len(str(resume_text).split()) if resume_text else 0
    kw_density = compute_keyword_density(jd_text, resume_text)
    
    # Section completeness
    sections_dict = preprocessing.parse_resume_sections(resume_text)
    if sections_dict:
        sec_complete = float(np.mean(list(sections_dict.values())))
    else:
        sec_complete = 0.0
        
    proj_div = compute_project_diversity(resume_text)
    title_match = compute_current_title_match(jd_title, candidate_title)

    # Compile 15 features in designated order
    features = [
        float(skill_count),
        float(skill_ratio),
        float(candidate_exp),
        float(exp_match),
        float(edu_match),
        float(loc_match),
        float(sem_sim),
        float(plat_act),
        float(res_complete),
        float(career_prog),
        float(res_len),
        float(kw_density),
        float(sec_complete),
        float(proj_div),
        float(title_match)
    ]
    
    return np.array(features, dtype=np.float32)

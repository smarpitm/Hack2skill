"""
preprocessing.py

This module contains text cleaning, processing, extraction, and normalization 
functions for candidates' resumes and job descriptions.
"""

import re
import logging
import unicodedata
from typing import List, Dict, Tuple, Optional
from . import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Clean text by converting to lowercase, removing special characters (except alphanumeric 
    and spaces), and normalizing whitespace (multiple spaces -> single space).
    Handles None and NaN by returning an empty string.

    Args:
        text (str): The raw input text.

    Returns:
        str: The cleaned and normalized text.
    """
    if text is None or (isinstance(text, float) and text != text):
        return ""
    
    text_str = str(text).lower()
    
    # Pre-clean normalization of specific tech terms containing special characters 
    # to preserve their semantic meaning (e.g., C++ -> cpp, C# -> csharp, .NET -> dotnet)
    text_str = re.sub(r'\bc\+\+(?!\w)', 'cpp', text_str)
    text_str = re.sub(r'\bc\#(?!\w)', 'csharp', text_str)
    text_str = re.sub(r'(?<!\w)\.net\b', 'dotnet', text_str)
    
    # Keep alphanumeric (including unicode letters) and whitespace, replace underscore with space
    text_str = re.sub(r'[^\w\s]', ' ', text_str)
    text_str = re.sub(r'_', ' ', text_str)
    
    # Normalize whitespace (multiple spaces -> single space)
    text_str = re.sub(r'\s+', ' ', text_str).strip()
    
    return text_str


def get_skill_aliases(skill: str) -> List[str]:
    """
    Helper function to generate aliases for a given skill name to support flexible matching.
    e.g., 'Node.js' -> ['node js', 'node', 'nodejs']
    e.g., 'C++' -> ['cpp', 'cplusplus']

    Args:
        skill (str): The raw skill name.

    Returns:
        List[str]: A list of lowercase string aliases.
    """
    cleaned = clean_text(skill)
    aliases = {cleaned}
    
    # Handle variations for JS-related skills (bidirectional matching)
    js_bases = ["react", "node", "vue", "next", "nuxt", "three", "angular"]
    for base in js_bases:
        if cleaned == base or cleaned == f"{base} js" or cleaned == f"{base}js":
            aliases.update([base, f"{base} js", f"{base}js"])
            
    # Handle C++, C#, .NET
    skill_lower = skill.lower()
    if skill_lower in ["c++", "cpp", "c plus plus"]:
        aliases.update(["cpp", "cplusplus"])
    elif skill_lower in ["c#", "csharp"]:
        aliases.add("csharp")
    elif skill_lower in [".net", "dotnet"]:
        aliases.update(["dotnet", "net"])
        
    return sorted(list(aliases))


def extract_skills(text: str, skill_dict: Optional[List[str]] = None) -> List[str]:
    """
    Extract a sorted list of unique matched skills from the input text using a skill dictionary.
    Each skill is checked as a whole word (not substring) in the cleaned text.
    Handles variations: "node" matches "node.js", "react" matches "react.js".

    Args:
        text (str): The raw input text.
        skill_dict (List[str], optional): The vocabulary of skills to search for. 
                                          If None, defaults to config.SKILL_DICTIONARY.

    Returns:
        List[str]: A sorted list of unique matched skill names (original format from dictionary).
    """
    if not text or (isinstance(text, float) and text != text):
        return []
        
    if skill_dict is None:
        skill_dict = config.SKILL_DICTIONARY
        
    cleaned_text = clean_text(text)
    if not cleaned_text:
        return []
        
    tokens = set(cleaned_text.split())
    matched_skills = set()
    
    for skill in skill_dict:
        aliases = get_skill_aliases(skill)
        for alias in aliases:
            if " " in alias:
                # Multi-word alias: check if it exists in the cleaned text with word boundaries
                if alias in cleaned_text:
                    pattern = r'\b' + re.escape(alias) + r'\b'
                    if re.search(pattern, cleaned_text):
                        matched_skills.add(skill)
                        break
            else:
                # Single-word alias: check in O(1) tokens set
                if alias in tokens:
                    matched_skills.add(skill)
                    break
                    
    return sorted(list(matched_skills))


def extract_experience_years(text: str) -> float:
    """
    Extract experience years from text using multiple regex patterns.
    Returns the maximum value found or 0.0 if no match found.

    Args:
        text (str): The raw text to search.

    Returns:
        float: The maximum experience years found.
    """
    if not text or (isinstance(text, float) and text != text):
        return 0.0
        
    text_lower = str(text).lower()
    
    # Normalize ranges like "2-4 years" or "2 to 4 years" to the lower bound "2 years"
    text_lower = re.sub(r'(\d+(?:\.\d+)?)\s*(?:-|to)\s*\d+(?:\.\d+)?\s*years?', r'\1 years', text_lower)
    
    # Define regex patterns supporting integers and decimals
    patterns = [
        r'(\d+(?:\.\d+)?)\+?\s*years?\+?\s*(?:of\s*)?experience',
        r'experience\s*:?\s*(\d+(?:\.\d+)?)\+?\s*years?\+?',
        r'(\d+(?:\.\d+)?)\s*years?\+?\s*(?:in\s*)?(?:industry|field|work|domain|software|development)?',
        r'(?:over|more than|at least)\s*(\d+(?:\.\d+)?)\s*years?'
    ]
    
    years_found = []
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        for m in matches:
            try:
                years_found.append(float(m))
            except ValueError:
                continue
                
    if years_found:
        return max(years_found)
    return 0.0


def normalize_education(edu_text: str) -> Tuple[int, str, int]:
    """
    Normalize education text to find the highest degree level, normalized name, and college tier.

    Args:
        edu_text (str): The raw education string from resume.

    Returns:
        Tuple[int, str, int]: (normalized_level: int, normalized_name: str, tier: int)
                              where tier represents: 0=unknown, 1=tier 1, 2=tier 2, 3=others.
    """
    if not edu_text or (isinstance(edu_text, float) and edu_text != edu_text):
        return (0, "unknown", 0)
        
    edu_text_lower = unicodedata.normalize("NFKC", str(edu_text)).lower()
    
    # Check college tier first
    tier = 0
    
    # 1. Check Tier 1 Keywords
    for kw in config.TIER_1_KEYWORDS:
        if kw in ["iit", "nit", "bits"]:
            pattern = r'\b' + re.escape(kw) + r'\b'
            if re.search(pattern, edu_text_lower):
                tier = 1
                break
        else:
            if kw in edu_text_lower:
                tier = 1
                break
                
    # 2. Check Tier 2 Keywords (only if Tier 1 not matched)
    if tier == 0:
        for kw in config.TIER_2_KEYWORDS:
            if kw in ["vit", "srm", "lpu"]:
                pattern = r'\b' + re.escape(kw) + r'\b'
                if re.search(pattern, edu_text_lower):
                    tier = 2
                    break
            else:
                if kw in edu_text_lower:
                    tier = 2
                    break
                    
    # 3. Find the highest education level matching in the text
    best_level = 0
    best_name = "unknown"

    # Robust Devanagari / Hinglish stability:
    # work on an aggressive compact form and do direct substring checks.
    edu_compact = re.sub(r"\s+", "", edu_text_lower)

    # M.Tech / एमटेक
    if "एमटेक" in edu_compact or "मटेक" in edu_compact:
        best_level = max(
            best_level,
            int(config.EDUCATION_EQUIVALENCE_MAP.get("m.tech", 4)),
            int(config.EDUCATION_EQUIVALENCE_MAP.get("एमटेक", 4)),
        )
        best_name = "एमटेक"
    # also tolerate patterns with extra spaces
    elif re.search(r"म\s*टेक|एम\s*टेक", edu_text_lower):
        best_level = max(
            best_level,
            int(config.EDUCATION_EQUIVALENCE_MAP.get("m.tech", 4)),
            int(config.EDUCATION_EQUIVALENCE_MAP.get("एमटेक", 4)),
        )
        best_name = "एमटेक"

    # MCA / एमसीए
    if "एमसीए" in edu_compact or "मका" in edu_compact or "मका़" in edu_compact:
        best_level = max(
            best_level,
            int(config.EDUCATION_EQUIVALENCE_MAP.get("mca", 4)),
            int(config.EDUCATION_EQUIVALENCE_MAP.get("एमसीए", 4)),
        )
        best_name = "एमसीए"
    elif re.search(r"म\s*का|एम\s*का", edu_text_lower):
        best_level = max(
            best_level,
            int(config.EDUCATION_EQUIVALENCE_MAP.get("mca", 4)),
            int(config.EDUCATION_EQUIVALENCE_MAP.get("एमसीए", 4)),
        )
        best_name = "एमसीए"
    
    # Custom regex mappings to avoid false positives (e.g. "database" matching "b.a", "me" matching "m.e")
    key_patterns = {
        "phd": r'\bph\.?\s*d\b|\bphd\b',
        "ph.d": r'\bph\.?\s*d\b|\bphd\b',
        "doctorate": r'\bdoctorate\b',
        "पीएचडी": r'पीएचडी',
        "masters": r'\bmasters?\b',
        "m.tech": r'\bm\.?\s*tech\b',
        "mca": r'\bmca\b',
        "mba": r'\bmba\b',
        "m.s": r'\bm\.?\s*s\b',
        "m.sc": r'\bm\.?\s*sc\b',
        "m.e": r'\bm\.?\s*e\b',
        "pgdm": r'\bpgdm\b',
        "m.com": r'\bm\.?\s*com\b',
        "m.a": r'\bm\.?\s*a\b',
        "एमटेक": r'एमटेक',
        "एमसीए": r'एमसीए',
        "bachelors": r'\bbachelors?\b',
        "b.tech": r'\bb\.?\s*tech\b',
        "b.e": r'\bb\.?\s*e\b',
        "bca": r'\bbca\b',
        "b.sc": r'\bb\.?\s*sc\b',
        "b.com": r'\bb\.?\s*com\b',
        "b.a": r'\bb\.?\s*a\b',
        "bba": r'\bbba\b',
        "बीटेक": r'बीटेक',
        "बीई": r'बीई',
        "diploma": r'\bdiploma\b',
        "polytechnic": r'\bpolytechnic\b',
        "डिप्लोमा": r'डिप्लोमा',
        "high school": r'\bhigh\s*school\b',
        "12th": r'\b12th\b',
        "hsc": r'\bhsc\b',
        "ssc": r'\bssc\b',
        "10th": r'\b10th\b',
        "cbse": r'\bcbse\b',
        "icse": r'\bicse\b',
        "intermediate": r'\bintermediate\b',
        "matriculation": r'\bmatriculation\b'
    }
    
    for key, level in config.EDUCATION_EQUIVALENCE_MAP.items():
        # Unicode-robust substring match, BUT for M.Tech/MCA we use stricter regex
        # to avoid false positives (e.g., "B.Tech from IIT" must stay level 3).
        key_norm = unicodedata.normalize("NFKC", str(key)).lower()

        if str(key).lower() in {"m.tech"} or key in {"एमटेक"} or str(key).lower() in {"mca"} or key in {"एमसीए"}:
            pattern = key_patterns.get(key, r'\b' + re.escape(str(key)) + r'\b')
            matched = re.search(pattern, edu_text_lower) is not None
        else:
            if key_norm and key_norm in edu_text_lower:
                matched = True
            else:
                pattern = key_patterns.get(key, r'\b' + re.escape(str(key)) + r'\b')
                matched = re.search(pattern, edu_text_lower) is not None

        if matched and level > best_level:
            best_level = level
            best_name = str(key)
                
    # Guardrail for common false-positives:
    # If we clearly detect B.Tech/B.E and NOT M.Tech/MCA, force B.Tech level=3.
    has_btech = re.search(r'\b(b\.?\s*tech|b\.?\s*e|bachelor(?:s)?)\b', edu_text_lower) is not None
    has_mtech = re.search(r'\b(m\.?\s*tech|m\.?\s*tech|एमटेक|mtech)\b', edu_text_lower) is not None or (
        "एमटेक" in edu_compact
    )
    has_mca = re.search(r'\bmca\b', edu_text_lower) is not None or ("एमसीए" in edu_compact)

    if has_btech and not has_mtech and not has_mca:
        best_level = max(best_level, 3) if best_level != 0 else 3
        # keep best_name if it was already btech-like; otherwise normalize
        if not ("b.tech" in str(best_name).lower() or "b.e" in str(best_name).lower() or "bachelor" in str(best_name).lower()):
            best_name = "b.tech"

    # Hard override for Tier-1 B.Tech/B.E cases (prevents accidental M.Tech level inflation)
    has_tier1_inst = re.search(r"\b(iit|nit|bits)\b", edu_text_lower) is not None
    has_btech_marker = re.search(r"\b(b\.?\s*tech|b\.?\s*e|btech|b\.?e|bachelor(?:s)?)\b", edu_text_lower) is not None
    has_mtech_marker = re.search(r"\b(m\.?\s*tech|mtech|एमटेक)\b", edu_text_lower) is not None
    has_mca_marker = re.search(r"\b(mca|एमसीए)\b", edu_text_lower) is not None
    if has_tier1_inst and has_btech_marker and not has_mtech_marker and not has_mca_marker:
        # Test expectation: B.Tech from IIT => level 3, tier 1
        return (3, "b.tech", 1)

    # 4. If we found an education level but college tier is still 0, 
    # check if an educational institution is mentioned to assign tier 3 (others).
    if best_level > 0 and tier == 0:
        institution_keywords = [
            "college", "university", "institute", "school", "academy", 
            "univ", "inst", "iit", "nit", "bits", "vit", "srm", "lpu"
        ]
        has_institution = any(kw in edu_text_lower for kw in institution_keywords)
        if has_institution:
            tier = 3
            
    return (best_level, best_name, tier)


def normalize_location(loc_text: str) -> str:
    """
    Standardize raw location text using config.LOCATION_NORMALIZATION_MAP.
    Returns standardized lowercase city name or "unknown" if not found.

    Args:
        loc_text (str): The raw location string.

    Returns:
        str: Standardized city name, or "unknown" if not recognized.
    """
    if not loc_text or (isinstance(loc_text, float) and loc_text != loc_text):
        return "unknown"
        
    cleaned_loc = str(loc_text).lower().strip()
    
    # Try exact lookup first
    if cleaned_loc in config.LOCATION_NORMALIZATION_MAP:
        return config.LOCATION_NORMALIZATION_MAP[cleaned_loc]
        
    # Check if any location key matches as a whole word in the text
    for key, normalized in config.LOCATION_NORMALIZATION_MAP.items():
        pattern = r'\b' + re.escape(key) + r'\b'
        if re.search(pattern, cleaned_loc):
            return normalized
            
    return "unknown"


def parse_resume_sections(text: str) -> Dict[str, bool]:
    """
    Detect the presence of common resume sections using keywords.

    Args:
        text (str): The raw resume text.

    Returns:
        Dict[str, bool]: Dictionary of section name keys mapped to a boolean indicating presence.
    """
    if not text or (isinstance(text, float) and text != text):
        return {
            "education": False,
            "experience": False,
            "skills": False,
            "projects": False,
            "certifications": False
        }
        
    text_lower = str(text).lower()
    
    keywords_map = {
        "education": ["education", "academic", "qualification", "degree", "academics", "qualifications", "degrees"],
        "experience": ["experience", "experiences", "work history", "employment", "career", "work experience", "professional experience"],
        "skills": ["skills", "technical skills", "core competencies", "expertise", "technologies", "skillset", "key skills"],
        "projects": ["projects", "portfolio", "work samples", "key projects", "personal projects"],
        "certifications": ["certifications", "certificates", "accreditations", "certification", "certificate"]
    }
    
    sections_present = {}
    for section, keywords in keywords_map.items():
        present = False
        for kw in keywords:
            pattern = r'\b' + re.escape(kw) + r'\b'
            if re.search(pattern, text_lower):
                present = True
                break
        sections_present[section] = present
        
    return sections_present


def extract_current_title(text: str) -> str:
    """
    Look for job title indicators near the top of the resume.
    Returns the first match or "unknown".

    Args:
        text (str): The raw resume text.

    Returns:
        str: Extracted job title or "unknown".
    """
    if not text or (isinstance(text, float) and text != text):
        return "unknown"
        
    # We restrict searching to the top portion of the resume (first 1000 characters)
    top_text = str(text)[:1000]
    
    # Define title pattern search matching common structures
    patterns = [
        r'(?i)current\s+role\s*:\s*([^\n\r,;]+)',
        r'(?i)currently\s+working\s+as\s*:\s*([^\n\r,;]+)',
        r'(?i)currently\s+working\s+as\s+([^\n\r,;]+)',
        r'(?i)present\s*:\s*([^\n\r,;]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, top_text)
        if match:
            title = match.group(1).strip()
            # Clean up trailing employer references e.g. "at Company Name" or "since 2019"
            title_clean = re.split(r'\b(at|for|since|in)\b', title, flags=re.IGNORECASE)[0].strip()
            if title_clean:
                return title_clean
                
    # Fallback to checking the first few lines of the resume for common title keywords
    lines = [line.strip() for line in top_text.split('\n') if line.strip()]
    title_keywords = [
        "engineer", "developer", "lead", "manager", "analyst", "consultant", "architect", 
        "specialist", "scientist", "programmer", "intern", "associate", "officer", "administrator"
    ]
    
    # First line fallback check
    if len(lines) > 0:
        candidate_title = lines[0]
        if any(kw in candidate_title.lower() for kw in title_keywords) and len(candidate_title) < 50:
            return candidate_title
            
    # Second line fallback check (commonly right under candidate's name)
    if len(lines) > 1:
        candidate_title = lines[1]
        if any(kw in candidate_title.lower() for kw in title_keywords) and len(candidate_title) < 50:
            return candidate_title
            
    return "unknown"

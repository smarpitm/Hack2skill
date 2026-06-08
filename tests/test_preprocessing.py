"""
tests/test_preprocessing.py — Comprehensive unit tests for src/preprocessing.py.

Tests every public function with real-world resume/JD text,
Indian-context edge cases, and adversarial inputs.
"""

import pytest
import math
from src.preprocessing import (
    clean_text,
    get_skill_aliases,
    extract_skills,
    extract_experience_years,
    normalize_education,
    normalize_location,
    parse_resume_sections,
    extract_current_title,
)


# ═══════════════════════════════════════════════════════════════════════════
# clean_text
# ═══════════════════════════════════════════════════════════════════════════

class TestCleanText:
    """Tests for text cleaning and normalization."""

    def test_basic_cleaning(self):
        """Standard lowercase + whitespace normalization."""
        assert clean_text("  Hello   World  ") == "hello world"

    def test_special_characters_removed(self):
        """Non-alphanum characters stripped."""
        result = clean_text("Python@3.9! & Django#4")
        assert "@" not in result
        assert "!" not in result
        assert "#" not in result

    def test_cpp_preserved(self):
        """C++ should be converted to 'cpp' before special-char strip."""
        assert "cpp" in clean_text("Expert in C++ and C# programming")

    def test_csharp_preserved(self):
        """C# should be converted to 'csharp'."""
        assert "csharp" in clean_text("C# developer with 5 years experience")

    def test_dotnet_preserved(self):
        """.NET should be converted to 'dotnet'."""
        assert "dotnet" in clean_text(".NET framework specialist")

    def test_none_input(self):
        """None returns empty string."""
        assert clean_text(None) == ""

    def test_nan_input(self):
        """float NaN returns empty string."""
        assert clean_text(float("nan")) == ""

    def test_numeric_input(self):
        """Numeric input coerced to string."""
        assert clean_text(12345) == "12345"

    def test_real_resume_snippet(self):
        """Real resume text should clean without crashing."""
        text = (
            "Name: Arjun Mehta | Headline: Senior AI Engineer | 7+ yrs experience | "
            "Built production FAISS pipelines, fine-tuned LLMs with LoRA and QLoRA."
        )
        result = clean_text(text)
        assert "arjun" in result
        assert "faiss" in result
        assert "|" not in result

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_whitespace_only(self):
        assert clean_text("   \t\n  ") == ""


# ═══════════════════════════════════════════════════════════════════════════
# get_skill_aliases
# ═══════════════════════════════════════════════════════════════════════════

class TestGetSkillAliases:
    """Tests for skill alias generation."""

    def test_nodejs_aliases(self):
        aliases = get_skill_aliases("Node.js")
        assert "node" in aliases
        assert "nodejs" in aliases or "node js" in aliases

    def test_react_aliases(self):
        aliases = get_skill_aliases("React")
        assert "react" in aliases

    def test_cpp_aliases(self):
        aliases = get_skill_aliases("C++")
        assert "cpp" in aliases

    def test_csharp_aliases(self):
        aliases = get_skill_aliases("C#")
        assert "csharp" in aliases

    def test_dotnet_aliases(self):
        aliases = get_skill_aliases(".NET")
        assert "dotnet" in aliases

    def test_simple_skill(self):
        aliases = get_skill_aliases("Python")
        assert "python" in aliases

    def test_always_returns_sorted_list(self):
        aliases = get_skill_aliases("Vue.js")
        assert isinstance(aliases, list)
        assert aliases == sorted(aliases)


# ═══════════════════════════════════════════════════════════════════════════
# extract_skills
# ═══════════════════════════════════════════════════════════════════════════

class TestExtractSkills:
    """Tests for skill extraction from resume/JD text."""

    def test_real_ai_resume(self):
        """Extract skills from a realistic AI engineer resume."""
        text = (
            "Expert in Python, PyTorch, and FAISS. Experience with NLP, "
            "Machine Learning, Docker, and AWS. Built sentence-transformer pipelines."
        )
        skills = extract_skills(text)
        assert "Python" in skills
        assert "Docker" in skills
        assert "AWS" in skills

    def test_frontend_resume(self):
        """Frontend developer skills detected."""
        text = "Proficient in React, Angular, TypeScript, HTML, CSS, and Webpack."
        skills = extract_skills(text)
        assert "React" in skills
        assert "TypeScript" in skills
        assert "HTML" in skills

    def test_no_skills_in_text(self):
        """Non-technical text returns empty list."""
        text = "I enjoy hiking and cooking in my spare time."
        skills = extract_skills(text)
        assert isinstance(skills, list)
        assert len(skills) == 0

    def test_empty_text(self):
        assert extract_skills("") == []

    def test_none_text(self):
        assert extract_skills(None) == []

    def test_nan_text(self):
        assert extract_skills(float("nan")) == []

    def test_returns_sorted_unique(self):
        """No duplicates and sorted order."""
        text = "Python Python Python Java Java"
        skills = extract_skills(text)
        assert skills == sorted(set(skills))

    def test_custom_skill_dict(self):
        """Custom skill dictionary works."""
        text = "I know Quantum Computing and Blockchain"
        skills = extract_skills(text, skill_dict=["Blockchain", "Quantum Computing"])
        assert "Blockchain" in skills

    def test_case_insensitive_matching(self):
        """Skills matched case-insensitively."""
        text = "PYTHON and DOCKER and KUBERNETES"
        skills = extract_skills(text)
        assert "Python" in skills
        assert "Docker" in skills


# ═══════════════════════════════════════════════════════════════════════════
# extract_experience_years
# ═══════════════════════════════════════════════════════════════════════════

class TestExtractExperienceYears:
    """Tests for experience year extraction from free text."""

    def test_standard_format(self):
        assert extract_experience_years("7 years of experience") == 7.0

    def test_plus_format(self):
        assert extract_experience_years("5+ years experience in ML") == 5.0

    def test_decimal_years(self):
        result = extract_experience_years("3.5 years of experience")
        assert result == 3.5

    def test_range_format(self):
        """'2-4 years' should extract the lower bound (2)."""
        result = extract_experience_years("2-4 years experience required")
        assert result == 2.0

    def test_experience_colon_format(self):
        result = extract_experience_years("Experience: 6 years")
        assert result == 6.0

    def test_industry_format(self):
        result = extract_experience_years("10 years in industry")
        assert result == 10.0

    def test_no_experience_mentioned(self):
        assert extract_experience_years("Just graduated from college") == 0.0

    def test_empty_string(self):
        assert extract_experience_years("") == 0.0

    def test_none_input(self):
        assert extract_experience_years(None) == 0.0

    def test_nan_input(self):
        assert extract_experience_years(float("nan")) == 0.0

    def test_multiple_patterns_returns_max(self):
        text = "3 years of experience. Previously, 5 years experience in data science."
        result = extract_experience_years(text)
        assert result == 5.0

    def test_real_jd_text(self):
        """Extract from the actual hackathon JD text."""
        jd = "Senior AI Engineer — 5-9 years experience. Required skills: Python, FAISS."
        result = extract_experience_years(jd)
        assert result == 5.0


# ═══════════════════════════════════════════════════════════════════════════
# normalize_education
# ═══════════════════════════════════════════════════════════════════════════

class TestNormalizeEducation:
    """Tests for education parsing with Indian context."""

    def test_btech_iit(self):
        """B.Tech from IIT should be level 3, tier 1."""
        level, name, tier = normalize_education("B.Tech from IIT Bombay")
        assert level == 3
        assert tier == 1

    def test_mtech_nit(self):
        """M.Tech from NIT → level 4, tier 1."""
        level, name, tier = normalize_education("M.Tech from NIT Trichy")
        assert level == 4
        assert tier == 1

    def test_phd(self):
        level, name, tier = normalize_education("PhD in Computer Science from IIT Delhi")
        assert level == 5
        assert tier == 1

    def test_bca_tier2(self):
        level, name, tier = normalize_education("BCA from VIT Vellore")
        assert level == 3
        assert tier == 2

    def test_mba_symbiosis(self):
        level, name, tier = normalize_education("MBA from Symbiosis Pune")
        assert level == 4
        assert tier == 2

    def test_diploma(self):
        level, name, tier = normalize_education("Diploma in Computer Science from Government Polytechnic")
        assert level >= 2

    def test_btech_equals_be(self):
        """B.Tech and B.E. should be same level (Indian equivalence)."""
        level_bt, _, _ = normalize_education("B.Tech in Computer Science")
        level_be, _, _ = normalize_education("B.E. in Computer Science from college")
        assert level_bt == level_be

    def test_mca_equals_mtech_cs(self):
        """MCA should be same level as M.Tech."""
        level_mca, _, _ = normalize_education("MCA from university")
        level_mt, _, _ = normalize_education("M.Tech from college")
        assert level_mca == level_mt

    def test_empty_string(self):
        level, name, tier = normalize_education("")
        assert level == 0
        assert name == "unknown"
        assert tier == 0

    def test_none_input(self):
        level, name, tier = normalize_education(None)
        assert level == 0

    def test_nan_input(self):
        level, name, tier = normalize_education(float("nan"))
        assert level == 0

    def test_tier3_other_college(self):
        level, name, tier = normalize_education("B.Tech from Pune University")
        assert level == 3
        assert tier == 3  # Known institution but not tier 1 or 2


# ═══════════════════════════════════════════════════════════════════════════
# normalize_location
# ═══════════════════════════════════════════════════════════════════════════

class TestNormalizeLocation:
    """Tests for Indian city normalization."""

    def test_bengaluru_to_bangalore(self):
        assert normalize_location("Bengaluru") == "bangalore"

    def test_bombay_to_mumbai(self):
        assert normalize_location("Bombay") == "mumbai"

    def test_gurugram_to_gurgaon(self):
        assert normalize_location("Gurugram") == "gurgaon"

    def test_ncr_to_delhi(self):
        assert normalize_location("NCR") == "delhi"

    def test_new_delhi(self):
        assert normalize_location("New Delhi") == "delhi"

    def test_madras_to_chennai(self):
        assert normalize_location("Madras") == "chennai"

    def test_calcutta_to_kolkata(self):
        assert normalize_location("Calcutta") == "kolkata"

    def test_cochin_to_kochi(self):
        assert normalize_location("Cochin") == "kochi"

    def test_trivandrum(self):
        assert normalize_location("Trivandrum") == "thiruvananthapuram"

    def test_unknown_location(self):
        assert normalize_location("Random City") == "unknown"

    def test_empty_string(self):
        assert normalize_location("") == "unknown"

    def test_none_input(self):
        assert normalize_location(None) == "unknown"

    def test_location_with_extra_text(self):
        """'Pune, India' should still resolve to pune."""
        result = normalize_location("Pune, Maharashtra, India")
        assert result == "pune"


# ═══════════════════════════════════════════════════════════════════════════
# parse_resume_sections
# ═══════════════════════════════════════════════════════════════════════════

class TestParseResumeSections:
    """Tests for section detection in resumes."""

    def test_all_sections_present(self):
        text = (
            "Education: B.Tech from IIT. "
            "Experience: 5 years at Google. "
            "Skills: Python, Java. "
            "Projects: Built ML pipeline. "
            "Certifications: AWS Certified."
        )
        sections = parse_resume_sections(text)
        assert sections["education"] is True
        assert sections["experience"] is True
        assert sections["skills"] is True
        assert sections["projects"] is True
        assert sections["certifications"] is True

    def test_partial_sections(self):
        text = "Education: B.Tech. Skills: Python."
        sections = parse_resume_sections(text)
        assert sections["education"] is True
        assert sections["skills"] is True
        assert sections["experience"] is False

    def test_empty_text(self):
        sections = parse_resume_sections("")
        assert all(v is False for v in sections.values())

    def test_none_input(self):
        sections = parse_resume_sections(None)
        assert all(v is False for v in sections.values())

    def test_returns_all_five_keys(self):
        sections = parse_resume_sections("Some text")
        expected_keys = {"education", "experience", "skills", "projects", "certifications"}
        assert set(sections.keys()) == expected_keys


# ═══════════════════════════════════════════════════════════════════════════
# extract_current_title
# ═══════════════════════════════════════════════════════════════════════════

class TestExtractCurrentTitle:
    """Tests for job title extraction from resume text."""

    def test_current_role_format(self):
        text = "Current Role: Senior Machine Learning Engineer\nEducation: B.Tech"
        result = extract_current_title(text)
        assert "Machine Learning Engineer" in result or "Senior" in result

    def test_title_in_first_line(self):
        text = "Software Engineer\nExperience: 5 years\nSkills: Python"
        result = extract_current_title(text)
        assert result != "unknown"

    def test_title_in_second_line(self):
        text = "John Doe\nData Analyst\nPune, India"
        result = extract_current_title(text)
        assert "Analyst" in result

    def test_no_title_found(self):
        text = "Random text without any job title indicators."
        result = extract_current_title(text)
        assert result == "unknown"

    def test_empty_text(self):
        assert extract_current_title("") == "unknown"

    def test_none_text(self):
        assert extract_current_title(None) == "unknown"

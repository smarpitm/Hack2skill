"""
conftest.py — Shared fixtures for the AI Candidate Ranking System test suite.

Provides realistic, representative test data modelled on the actual
candidates.csv / jobs.csv schema so that every test module starts from a
consistent, well-understood baseline.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure project root is importable (handles running `pytest` from repo root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  REALISTIC CANDIDATE FIXTURES                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@pytest.fixture
def sample_candidates_df():
    """
    10 realistic candidate profiles covering:
      - Strong match (senior AI/ML, IIT, Pune/Noida)
      - Partial match (some skill overlap, mid experience)
      - Weak match (unrelated domain, no relevant skills)
      - Edge cases (missing fields, zero experience, honeypot)
    """
    return pd.DataFrame([
        {
            "candidate_id": "CAND_001",
            "resume_text": (
                "Name: Arjun Mehta | Headline: Senior AI Engineer | 7+ yrs experience | "
                "Summary: Professional with 7 years of experience in Machine Learning, NLP, "
                "and retrieval systems. Built production FAISS pipelines, fine-tuned LLMs with "
                "LoRA and QLoRA. Expert in Python, PyTorch, sentence-transformers. "
                "Education: B.Tech from IIT Bombay. "
                "Work History: Role: Senior AI Engineer at Flipkart (36 months). "
                "Developed recommendation engine using embeddings and FAISS. "
                "Built NLP pipeline for resume parsing. "
                "Projects: built web search engine, developed backend microservices, "
                "created machine learning model for fraud detection."
            ),
            "skills": "Python, PyTorch, FAISS, NLP, Machine Learning, Deep Learning, sentence-transformers, Docker, AWS",
            "experience_years": 7.0,
            "education": "B.Tech from IIT Bombay (tier: tier_1)",
            "location": "Pune, India",
            "current_title": "Senior AI Engineer",
            "platform_activity_score": 92.3,
            "is_honeypot": 0,
        },
        {
            "candidate_id": "CAND_002",
            "resume_text": (
                "Name: Priya Sharma | Headline: ML Engineer | 5+ yrs experience | "
                "Summary: Machine Learning Engineer with expertise in XGBoost, Scikit-learn, "
                "and data pipelines. Experience with Elasticsearch and vector databases. "
                "Education: M.Tech from NIT Trichy. "
                "Work History: Role: ML Engineer at Amazon (24 months). "
                "Built learning-to-rank model for product search. "
                "Projects: developed cloud deployment pipeline, built data science dashboard."
            ),
            "skills": "Python, XGBoost, Scikit-learn, Elasticsearch, Pandas, SQL, Docker, Kubernetes",
            "experience_years": 5.0,
            "education": "M.Tech from NIT Trichy (tier: tier_1)",
            "location": "Noida, India",
            "current_title": "ML Engineer",
            "platform_activity_score": 78.5,
            "is_honeypot": 0,
        },
        {
            "candidate_id": "CAND_003",
            "resume_text": (
                "Name: Rahul Gupta | Headline: Data Scientist | 3+ yrs experience | "
                "Summary: Data Scientist specializing in NLP and computer vision. "
                "Familiar with embeddings and retrieval. "
                "Education: B.Tech from VIT Vellore. "
                "Work History: Role: Data Scientist at TCS (18 months). "
                "Built text classification model and sentiment analysis pipeline."
            ),
            "skills": "Python, TensorFlow, NLP, Pandas, NumPy, SQL",
            "experience_years": 3.0,
            "education": "B.Tech from VIT Vellore (tier: tier_2)",
            "location": "Bengaluru, India",
            "current_title": "Data Scientist",
            "platform_activity_score": 65.0,
            "is_honeypot": 0,
        },
        {
            "candidate_id": "CAND_004",
            "resume_text": (
                "Name: Sneha Patel | Headline: Full Stack Developer | 4+ yrs experience | "
                "Summary: Full stack developer with React, Node.js, and Django expertise. "
                "Education: BCA from Pune University. "
                "Work History: Role: Full Stack Developer at Infosys (30 months). "
                "Built e-commerce platform and REST APIs."
            ),
            "skills": "JavaScript, React, Node.js, Django, Python, MongoDB, HTML, CSS",
            "experience_years": 4.0,
            "education": "BCA from Pune University (tier: tier_3)",
            "location": "Pune, India",
            "current_title": "Full Stack Developer",
            "platform_activity_score": 55.0,
            "is_honeypot": 0,
        },
        {
            "candidate_id": "CAND_005",
            "resume_text": (
                "Name: Ankit Verma | Headline: DevOps Lead | 8+ yrs experience | "
                "Summary: DevOps engineer with strong cloud and CI/CD expertise. "
                "Education: B.Tech from BITS Pilani. "
                "Work History: Role: DevOps Lead at Razorpay (48 months). "
                "Managed Kubernetes clusters, Terraform infrastructure."
            ),
            "skills": "AWS, Docker, Kubernetes, Terraform, Jenkins, Python, Linux, CI/CD, Ansible",
            "experience_years": 8.0,
            "education": "B.Tech from BITS Pilani (tier: tier_1)",
            "location": "Gurugram, India",
            "current_title": "DevOps Lead",
            "platform_activity_score": 88.0,
            "is_honeypot": 0,
        },
        {
            "candidate_id": "CAND_006",
            "resume_text": (
                "Name: Kavita Reddy | Headline: Marketing Manager | 6+ yrs experience | "
                "Summary: Marketing professional specializing in digital campaigns "
                "and brand strategy. No technical background. "
                "Education: MBA from Symbiosis. "
                "Work History: Role: Marketing Manager at Swiggy (36 months)."
            ),
            "skills": "Marketing, SEO, Excel, PowerBI, Photoshop",
            "experience_years": 6.0,
            "education": "MBA from Symbiosis (tier: tier_2)",
            "location": "Hyderabad, India",
            "current_title": "Marketing Manager",
            "platform_activity_score": 40.0,
            "is_honeypot": 0,
        },
        {
            "candidate_id": "CAND_007",
            "resume_text": (
                "Name: Vikram Singh | Headline: Junior Developer | 1+ yrs experience | "
                "Summary: Fresh graduate with basic Python knowledge. "
                "Completed a few online courses in ML. "
                "Education: B.Sc from local college."
            ),
            "skills": "Python, HTML, CSS",
            "experience_years": 1.0,
            "education": "B.Sc from local college (tier: tier_4)",
            "location": "Jaipur, India",
            "current_title": "Junior Developer",
            "platform_activity_score": 20.0,
            "is_honeypot": 0,
        },
        {
            "candidate_id": "CAND_008",
            "resume_text": "",  # Edge: empty resume
            "skills": "",
            "experience_years": 0.0,
            "education": "",
            "location": "",
            "current_title": "",
            "platform_activity_score": 0.0,
            "is_honeypot": 0,
        },
        {
            "candidate_id": "CAND_009",
            "resume_text": (
                "Name: Deepak Joshi | Headline: Principal Engineer | 10+ yrs experience | "
                "Summary: 10 years experience building distributed systems. "
                "Expert in Python, Go, AWS, Kafka. Led teams of 15+ engineers. "
                "Education: PhD from IIT Delhi. "
                "Current Role: Principal Engineer at Google. "
                "Projects: built web application, developed cloud infrastructure, "
                "implemented machine learning pipeline for NLP."
            ),
            "skills": "Python, Go, AWS, Kafka, Docker, Kubernetes, Redis, PostgreSQL, Machine Learning",
            "experience_years": 10.0,
            "education": "PhD from IIT Delhi (tier: tier_1)",
            "location": "Bengaluru, India",
            "current_title": "Principal Engineer",
            "platform_activity_score": 95.0,
            "is_honeypot": 0,
        },
        {
            "candidate_id": "CAND_010",
            "resume_text": (
                "Name: Honeypot Candidate | Summary: Fake candidate for testing. "
                "Skills: Everything. Experience: 100 years."
            ),
            "skills": "Python, Java, C++, Everything",
            "experience_years": 100.0,
            "education": "PhD from MIT",
            "location": "USA",
            "current_title": "CTO",
            "platform_activity_score": 100.0,
            "is_honeypot": 1,  # <-- Honeypot
        },
    ])


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  REALISTIC JOB DESCRIPTION FIXTURES                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@pytest.fixture
def sample_job_row():
    """
    Real job description matching the hackathon's Senior AI Engineer role.
    Column names match the actual jobs.csv schema.
    """
    return pd.Series({
        "job_id": "REDROB_AI_SR_AI_ENG",
        "job_description": (
            "Senior AI Engineer — Founding Team. Pune/Noida, India (Hybrid). "
            "5-9 years experience. Required skills: embeddings-based retrieval systems, "
            "sentence-transformers, vector databases, FAISS, Python, LLM fine-tuning, "
            "LoRA, QLoRA, PEFT, learning-to-rank models, XGBoost, NLP, IR."
        ),
        "required_skills": (
            "embeddings-based retrieval systems, sentence-transformers, FAISS, Python, "
            "vector databases, XGBoost, NLP, LLM fine-tuning, LoRA, learning-to-rank models"
        ),
        "experience_required": 5.0,
        "education_required": "B.Tech",
        "location": "Pune/Noida, India",
        "job_title": "Senior AI Engineer",
    })


@pytest.fixture
def sample_jobs_df(sample_job_row):
    """Single-job DataFrame for pipeline tests."""
    return pd.DataFrame([dict(sample_job_row)])


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  HELPER FIXTURES                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@pytest.fixture
def strong_candidate_row(sample_candidates_df):
    """CAND_001 — strong match for the AI Engineer role."""
    return dict(sample_candidates_df[sample_candidates_df["candidate_id"] == "CAND_001"].iloc[0])


@pytest.fixture
def weak_candidate_row(sample_candidates_df):
    """CAND_006 — marketing manager, weak match for AI role."""
    return dict(sample_candidates_df[sample_candidates_df["candidate_id"] == "CAND_006"].iloc[0])


@pytest.fixture
def empty_candidate_row(sample_candidates_df):
    """CAND_008 — all fields empty (edge case)."""
    return dict(sample_candidates_df[sample_candidates_df["candidate_id"] == "CAND_008"].iloc[0])


@pytest.fixture
def sample_job_dict(sample_job_row):
    """Job row as a plain dict for feature extraction tests."""
    return dict(sample_job_row)

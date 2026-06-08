# create_dummy_data.py
import pandas as pd

candidates = pd.DataFrame({
    "candidate_id": ["CAND_001", "CAND_002", "CAND_003"],
    "resume_text": [
        "Python developer with 5 years experience in Django and AWS",
        "Java backend engineer with 3 years in Spring Boot",
        "Full stack developer with React, Node.js, and MongoDB"
    ],
    "skills": ["Python, Django, AWS", "Java, Spring Boot", "React, Node.js, MongoDB"],
    "experience_years": [5, 3, 4],
    "education": ["B.Tech", "B.E.", "MCA"],
    "location": ["Bangalore", "Delhi", "Bangalore"]
})
candidates.to_csv("data/candidates.csv", index=False)

jobs = pd.DataFrame({
    "job_id": ["JOB_001"],
    "job_description": ["Senior Python developer with Django and AWS experience"],
    "required_skills": ["Python, Django, AWS"],
    "experience_required": [5],
    "education_required": ["B.Tech"],
    "location": ["Bangalore"]
})
jobs.to_csv("data/jobs.csv", index=False)

print("Dummy data created")
"""
src/job_description.py

Job description class representing required fields and metrics.
Supports parsing job description details from CSV.
"""

import logging
from pathlib import Path
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JobDescription:
    """
    Represents a single job description with parsed required fields.
    """

    def __init__(
        self,
        job_id: str,
        title: str,
        description: str,
        required_skills: str = "",
        experience_required: float = 0.0,
        education_required: str = "",
        location: str = "",
    ):
        self.job_id = job_id
        self.title = title
        self.description = description
        self.required_skills = required_skills
        self.experience_required = experience_required
        self.education_required = education_required
        self.location = location

    @classmethod
    def load_from_csv(cls, path: str) -> list["JobDescription"]:
        """
        Load list of job descriptions from a CSV file.
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Jobs CSV not found at: {path}")

        logger.info(f"Loading job descriptions from: {path_obj}")
        df = pd.read_csv(str(path_obj))
        
        jobs = []
        for _, row in df.iterrows():
            desc_col = "description" if "description" in df.columns else ("job_description" if "job_description" in df.columns else None)
            if desc_col is None:
                desc_cols = [col for col in df.columns if "desc" in col.lower()]
                desc_col = desc_cols[0] if desc_cols else df.columns[0]
                
            required_skills = row.get("required_skills", row.get("skills", ""))
            
            exp_col = "experience_years" if "experience_years" in df.columns else ("experience_required" if "experience_required" in df.columns else ("experience" if "experience" in df.columns else None))
            exp_val = 0.0
            if exp_col and pd.notna(row.get(exp_col)):
                try:
                    exp_val = float(row.get(exp_col))
                except ValueError:
                    pass
                    
            edu_col = "education_required" if "education_required" in df.columns else ("education" if "education" in df.columns else None)
            edu_val = ""
            if edu_col and pd.notna(row.get(edu_col)):
                edu_val = str(row.get(edu_col))
                
            jobs.append(
                cls(
                    job_id=str(row.get("job_id", "unknown")),
                    title=str(row.get("job_title", row.get("title", ""))),
                    description=str(row.get(desc_col, "")),
                    required_skills=str(required_skills),
                    experience_required=exp_val,
                    education_required=edu_val,
                    location=str(row.get("location", "")),
                )
            )
        return jobs

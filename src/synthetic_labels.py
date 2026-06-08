"""
src/synthetic_labels.py

Synthetic label generation module. Generates binary labels (0/1) for candidate-job 
pairs to train the ranking model when no ground truth is available.
"""

import os
import logging
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config
from . import features
from . import preprocessing

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_labels(
    jobs_df: pd.DataFrame, 
    candidates_df: pd.DataFrame, 
    n_samples: int = 5000, 
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic candidate-job pair labels (0/1) using weighted scores.
    Ensures balanced stratification of positive and negative labels.

    Args:
        jobs_df (pd.DataFrame): DataFrame of job descriptions.
        candidates_df (pd.DataFrame): DataFrame of candidate resumes.
        n_samples (int): Total number of training samples to generate.
        random_state (int): Seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame containing job_id, candidate_id, 15 features, and label.
    """
    np.random.seed(random_state)
    
    if jobs_df.empty or candidates_df.empty:
        logger.warning("Empty jobs or candidates DataFrame. Returning empty training DataFrame.")
        return pd.DataFrame()
        
    logger.info("Initializing TF-IDF vectorizer for semantic similarity proxy...")
    # Clean text columns for TF-IDF
    desc_col = "description" if "description" in jobs_df.columns else ("job_description" if "job_description" in jobs_df.columns else None)
    if desc_col is None:
        desc_cols = [col for col in jobs_df.columns if "desc" in col.lower()]
        desc_col = desc_cols[0] if desc_cols else jobs_df.columns[0]
        
    jd_texts = jobs_df[desc_col].fillna("").apply(preprocessing.clean_text).tolist()
    resume_texts = candidates_df["resume_text"].fillna("").apply(preprocessing.clean_text).tolist()
    
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    try:
        # Fit TF-IDF on candidate resumes and transform
        candidate_tfidf = vectorizer.fit_transform(resume_texts)
        job_tfidf = vectorizer.transform(jd_texts)
        # Compute pairwise cosine similarity matrix
        sim_matrix = cosine_similarity(job_tfidf, candidate_tfidf)
    except Exception as e:
        logger.error(f"Failed to fit/transform TF-IDF: {str(e)}")
        # Fallback to random matrix if TF-IDF fails
        sim_matrix = np.random.uniform(0.1, 0.4, (len(jobs_df), len(candidates_df)))
        
    num_jobs = len(jobs_df)
    samples_per_job = int(np.ceil(n_samples / num_jobs))
    
    # We want roughly equal positive and negative samples per job
    target_pos_per_job = int(np.ceil(samples_per_job / 2))
    target_neg_per_job = int(np.ceil(samples_per_job / 2))
    
    records = []
    
    # Pre-parse candidate education and location details to avoid re-parsing in loop
    logger.info("Extracting candidate metadata for feature generation...")
    parsed_candidates = []
    for idx, row in candidates_df.iterrows():
        c_dict = dict(row)
        # Handle cases where columns are missing
        c_dict["candidate_id"] = row.get("candidate_id", f"c_{idx}")
        c_dict["resume_text"] = row.get("resume_text", "")
        c_dict["skills"] = row.get("skills", [])
        c_dict["experience_years"] = row.get("experience_years", 0.0)
        if c_dict["experience_years"] is None or c_dict["experience_years"] == 0.0:
            c_dict["experience_years"] = row.get("experience", 0.0)
        c_dict["education"] = row.get("education", "")
        c_dict["location"] = row.get("location", "")
        c_dict["current_title"] = row.get("current_title", row.get("title", ""))
        parsed_candidates.append(c_dict)

    logger.info(f"Generating synthetic labels across {num_jobs} jobs...")
    for j_idx, (_, job_row) in enumerate(jobs_df.iterrows()):
        job_id = job_row.get("job_id", f"j_{j_idx}")
        jd_dict = dict(job_row)
        
        # Get similarities for this job
        job_sims = sim_matrix[j_idx]
        
        # Retrieve candidates with highest and lowest similarities to balance positive/negative classes
        sorted_indices = np.argsort(job_sims)[::-1]
        
        # Split candidate pool for this job (handling small candidate pools safely)
        num_candidates = len(sorted_indices)
        num_top = min(num_candidates, max(200, target_pos_per_job * 3))
        if num_candidates <= 200:
            num_top = max(1, num_candidates // 2)
            
        top_candidates_idx = sorted_indices[:num_top]
        remaining_indices = sorted_indices[num_top:]
        
        target_neg_size = min(len(remaining_indices), target_neg_per_job * 4)
        if target_neg_size > 0:
            random_candidates_idx = np.random.choice(
                remaining_indices,
                size=target_neg_size,
                replace=False
            )
        else:
            random_candidates_idx = np.array([], dtype=int)
        
        job_positives = []
        job_negatives = []
        
        # Process potential positives and negatives
        candidate_indices_to_test = np.concatenate([top_candidates_idx, random_candidates_idx])
        
        for c_idx in candidate_indices_to_test:
            cand = parsed_candidates[c_idx]
            sim_score = float(job_sims[c_idx])
            
            # Extract all 15 features
            feats = features.extract_all_features(jd_dict, cand, faiss_score=sim_score)
            
            # Compute composite synthetic score:
            # score = skill_match_ratio * W_SKILL + semantic_similarity * W_SEMANTIC + experience_match * W_EXP + platform_activity * W_ACT
            comp_score = (
                feats[1] * config.WEIGHT_SKILL_MATCH +
                feats[6] * config.WEIGHT_SEMANTIC +
                feats[3] * config.WEIGHT_EXPERIENCE +
                feats[7] * config.WEIGHT_ACTIVITY
            )
            
            label = 1 if comp_score >= config.SYNTHETIC_LABEL_THRESHOLD else 0
            
            record = {
                "job_id": job_id,
                "candidate_id": cand["candidate_id"],
                "label": label
            }
            # Append features
            for f_idx, val in enumerate(feats):
                record[f"feature_{f_idx+1}"] = val
                
            if label == 1:
                job_positives.append(record)
            else:
                job_negatives.append(record)
                
        # Stratify: select balanced samples for this job
        num_pos = len(job_positives)
        num_neg = len(job_negatives)
        
        # Determine exact sample sizes to match targets or local constraints
        pos_to_take = min(num_pos, target_pos_per_job)
        neg_to_take = min(num_neg, target_neg_per_job)
        
        # Balance out if one side is lacking
        if pos_to_take < target_pos_per_job and num_neg > neg_to_take:
            neg_to_take = min(num_neg, samples_per_job - pos_to_take)
        elif neg_to_take < target_neg_per_job and num_pos > pos_to_take:
            pos_to_take = min(num_pos, samples_per_job - neg_to_take)
            
        # Sample randomly from collected sets
        if job_positives:
            sampled_pos = [job_positives[i] for i in np.random.choice(num_pos, size=pos_to_take, replace=False)]
            records.extend(sampled_pos)
        if job_negatives:
            sampled_neg = [job_negatives[i] for i in np.random.choice(num_neg, size=neg_to_take, replace=False)]
            records.extend(sampled_neg)

    df_out = pd.DataFrame(records)
    
    if df_out.empty:
        return df_out
        
    # Shuffle the final dataset
    df_out = df_out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    
    # Truncate to desired size if needed
    if len(df_out) > n_samples:
        df_out = df_out.iloc[:n_samples].reset_index(drop=True)
        
    logger.info(f"Generated {len(df_out)} training samples (Positives: {sum(df_out['label'] == 1)}, Negatives: {sum(df_out['label'] == 0)})")
    
    return df_out


def save_synthetic_data(df: pd.DataFrame, path: str) -> None:
    """
    Save synthetic training dataset to disk.

    Args:
        df (pd.DataFrame): Training DataFrame.
        path (str): File path to save CSV.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"Synthetic training data successfully saved to: {path}")
    except Exception as e:
        logger.error(f"Failed to save training data to {path}: {str(e)}")


def load_synthetic_data(path: str) -> pd.DataFrame:
    """
    Load synthetic training dataset from disk.

    Args:
        path (str): File path of the CSV.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Synthetic training data file not found at: {path}")
        
    df = pd.read_csv(path)
    logger.info(f"Synthetic training data successfully loaded from: {path}")
    return df

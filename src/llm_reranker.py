"""
src/llm_reranker.py

GROQ API integration for precision re-ranking of top candidates.
Provides the GroqReranker class and validation helper functions.
"""

import os
import json
import time
import logging
import hashlib
import pathlib
import importlib.util
from typing import List, Dict, Optional, Any, Union
import pandas as pd
from groq import Groq

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Handle config import with fallback for direct execution
try:
    from . import config
except ImportError:
    _config_path = pathlib.Path(__file__).resolve().parent / "config.py"
    spec = importlib.util.spec_from_file_location("config", _config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)


def create_candidate_summary(candidate_row: Union[pd.Series, Dict[str, Any]], max_length: int = 200) -> str:
    """
    Compress candidate profile into a single string of max_length characters.
    Format: "ID: {candidate_id}, Exp: {experience_years}y, Skills: {skills}, Background: {resume_text[:150]}..."
    """
    candidate_id = candidate_row.get("candidate_id", "unknown")
    
    # Extract experience years
    exp = candidate_row.get("experience_years")
    if exp is None:
        exp = candidate_row.get("experience", 0.0)
    try:
        exp_val = float(exp)
        exp_str = str(int(exp_val)) if exp_val.is_integer() else f"{exp_val:.1f}"
    except (ValueError, TypeError):
        exp_str = str(exp)
        
    # Extract skills
    skills_val = candidate_row.get("skills", "")
    if isinstance(skills_val, list):
        skills_str = ", ".join(map(str, skills_val))
    else:
        skills_str = str(skills_val)
    skills_str = " ".join(skills_str.split())
    
    # Extract resume text
    resume_text = str(candidate_row.get("resume_text", ""))
    resume_text = " ".join(resume_text.split())
    
    # Format template: "ID: {candidate_id}, Exp: {exp_str}y, Skills: {skills_str}, Background: "
    prefix = f"ID: {candidate_id}, Exp: {exp_str}y, Skills: {skills_str}, Background: "
    
    # Calculate remaining length for background text
    needed_chars_for_suffix = 3
    allowed_resume_len = max_length - len(prefix) - needed_chars_for_suffix
    
    if allowed_resume_len > 0:
        truncated_resume = resume_text[:allowed_resume_len]
        summary = f"{prefix}{truncated_resume}..."
    else:
        summary = prefix[:max_length]
        
    return summary


def validate_ranking(ranked_ids: list, expected_ids: list) -> bool:
    """
    Validate that ranked_ids is a valid permutation of expected_ids and conforms to requirements:
    - Must be a list
    - All elements must be strings
    - No duplicates
    - Elements must exactly match expected_ids
    """
    if not isinstance(ranked_ids, list):
        logger.warning("Validation failed: ranked_ids is not a list.")
        return False
        
    if len(ranked_ids) != len(expected_ids):
        logger.warning(f"Validation failed: length mismatch. Expected {len(expected_ids)}, got {len(ranked_ids)}.")
        return False
        
    expected_set = set(str(eid) for eid in expected_ids)
    seen = set()
    for item in ranked_ids:
        if not isinstance(item, str):
            logger.warning(f"Validation failed: item {item} is not a string.")
            return False
        if item not in expected_set:
            logger.warning(f"Validation failed: item {item} not in expected set of candidate IDs.")
            return False
        if item in seen:
            logger.warning(f"Validation failed: duplicate item {item} found.")
            return False
        seen.add(item)
        
    return True


class GroqReranker:
    """
    GROQ API wrapper for candidate re-ranking with disk-based persistent caching and retry logic.
    """
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model_name: Optional[str] = None, 
        cache_path: Optional[str] = None
    ):
        # Resolve API Key
        if not api_key:
            api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ API key must be provided via parameter or GROQ_API_KEY environment variable."
            )
            
        # Resolve Model Name
        self.model_name = model_name or config.GROQ_MODEL
        
        # Resolve Cache Path
        self.cache_path = cache_path or os.path.join(config.CACHE_DIR, "groq_cache.json")
        
        # Initialize Groq client
        self.client = Groq(api_key=api_key)
        
        # Load Cache
        self.cache = {}
        self._load_cache()

    def _get_cache_key(self, prompt: str) -> str:
        """
        Return MD5 hash of prompt string.
        """
        return hashlib.md5(prompt.encode("utf-8")).hexdigest()

    def _load_cache(self) -> None:
        """
        Load cache from disk JSON file if exists, else initialize empty dictionary.
        """
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
                logger.info(f"Successfully loaded {len(self.cache)} entries from cache at {self.cache_path}.")
            except Exception as e:
                logger.error(f"Failed to load cache from {self.cache_path}: {e}. Initializing empty cache.")
                self.cache = {}
        else:
            self.cache = {}

    def _save_cache(self) -> None:
        """
        Save self.cache to disk as pretty-printed JSON. Creates directories if needed.
        """
        try:
            dir_name = os.path.dirname(self.cache_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=4, ensure_ascii=False)
            logger.info(f"Successfully saved cache with {len(self.cache)} entries to {self.cache_path}.")
        except Exception as e:
            logger.error(f"Failed to save cache to {self.cache_path}: {e}.")

    def _build_prompt(self, job_description: str, candidate_summaries: list) -> str:
        """
        Construct the strict system prompt for ranking.
        """
        numbered_list = []
        for idx, summary in enumerate(candidate_summaries):
            numbered_list.append(f"{idx + 1}. {summary}")
        numbered_list_of_summaries = "\n".join(numbered_list)
        
        prompt = f"""You are an expert technical recruiter with 15 years of experience.
Rank these candidates for the given job from BEST fit to WORST fit.
Consider: technical depth, experience relevance, leadership potential, career trajectory, and cultural fit.

Job Description:
{job_description}

Candidates:
{numbered_list_of_summaries}

Return ONLY a valid JSON array of candidate IDs in ranked order (best first).
Example: ["CAND_001", "CAND_045", "CAND_089"]"""
        return prompt

    def _parse_response(self, response_text: str, expected_ids: list) -> Optional[list]:
        """
        Parse JSON array of candidate IDs from response text and validate it against expected IDs.
        """
        import re
        
        # Clean potential markdown wrappers
        cleaned_text = response_text.strip()
        if cleaned_text.startswith("```"):
            cleaned_text = re.sub(r"^```(?:json)?\s*", "", cleaned_text)
            cleaned_text = re.sub(r"\s*```$", "", cleaned_text)
        cleaned_text = cleaned_text.strip()
        
        try:
            data = json.loads(cleaned_text)
        except Exception as e:
            logger.error(f"Failed to parse JSON response: {e}. Raw response: {response_text[:200]}")
            return None
            
        ranked_ids = None
        if isinstance(data, list):
            ranked_ids = data
        elif isinstance(data, dict):
            # Attempt to extract list from common JSON keys if the model wrapped it in an object
            for val in data.values():
                if isinstance(val, list):
                    ranked_ids = val
                    break
                    
        if not isinstance(ranked_ids, list):
            logger.error("Parsed response is not a list and does not contain a list.")
            return None
            
        # Convert expected IDs and parsed IDs to string for uniform comparison
        ranked_ids = [str(rid) for rid in ranked_ids]
        expected_ids_str = [str(eid) for eid in expected_ids]
        
        if validate_ranking(ranked_ids, expected_ids_str):
            return ranked_ids
            
        logger.error("Parsed ranked IDs failed validation checks.")
        return None

    def _call_groq_api(self, prompt: str) -> Optional[str]:
        """
        Make the actual chat completion API call to Groq. Handles rate limiting and retries.
        """
        from groq import RateLimitError, APIError, APIConnectionError, APITimeoutError
        
        for attempt in range(2):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=config.GROQ_TEMPERATURE,
                    max_tokens=config.GROQ_MAX_TOKENS,
                    response_format={"type": "json_object"}
                )
                time.sleep(config.GROQ_DELAY_SECONDS)
                return response.choices[0].message.content
            except RateLimitError as e:
                logger.warning(f"Rate limit hit on attempt {attempt + 1}: {e}")
                time.sleep(config.GROQ_DELAY_SECONDS)
                if attempt == 0:
                    logger.info("Rate limit retry: sleeping 60 seconds before retrying...")
                    time.sleep(60)
                else:
                    logger.error("Rate limit hit again on retry. Giving up API call.")
                    return None
            except (APIError, APIConnectionError, APITimeoutError) as e:
                logger.error(f"GROQ API Error on attempt {attempt + 1}: {e}")
                time.sleep(config.GROQ_DELAY_SECONDS)
                return None
            except Exception as e:
                logger.error(f"Unexpected API call error on attempt {attempt + 1}: {e}")
                time.sleep(config.GROQ_DELAY_SECONDS)
                return None
        return None

    def rerank(self, job_description: str, candidates_df: pd.DataFrame) -> Optional[list]:
        """
        Rerank a dataframe of candidates for a job description using GROQ LLM.
        """
        if candidates_df.empty:
            logger.warning("Empty candidates DataFrame passed to rerank. Returning empty list.")
            return []
            
        # Extract candidate IDs and verify presence of required columns
        required_cols = {"candidate_id", "resume_text", "skills"}
        for col in required_cols:
            if col not in candidates_df.columns:
                logger.error(f"Missing required column '{col}' in candidates_df.")
                return None
                
        # Build summaries and gather expected IDs
        summaries = []
        expected_ids = []
        for _, row in candidates_df.iterrows():
            summaries.append(create_candidate_summary(row, max_length=200))
            expected_ids.append(str(row["candidate_id"]))
            
        # Build Prompt
        prompt = self._build_prompt(job_description, summaries)
        cache_key = self._get_cache_key(prompt)
        
        # Check Cache
        if cache_key in self.cache:
            logger.info("Cache hit! Returning cached ranking.")
            cached_val = self.cache[cache_key]["response"]
            parsed = self._parse_response(cached_val, expected_ids)
            if parsed is not None:
                return parsed
            logger.warning("Cached value failed validation. Making fresh API call.")
            
        # Cache Miss: Make API call
        logger.info(f"Cache miss. Calling GROQ API using model {self.model_name}...")
        response_text = self._call_groq_api(prompt)
        if response_text is None:
            logger.error("API call failed. Rerank returning None.")
            return None
            
        # Parse and Validate Response
        ranked_ids = self._parse_response(response_text, expected_ids)
        if ranked_ids is not None:
            # Save to Cache
            self.cache[cache_key] = {
                "response": response_text,
                "timestamp": time.time()
            }
            self._save_cache()
            logger.info("API call successful, cache updated, and ranking returned.")
            return ranked_ids
            
        logger.error("API response parsing/validation failed.")
        return None

    def batch_rerank(self, jobs_df: pd.DataFrame, ranked_candidates_dict: Dict[str, pd.DataFrame]) -> Dict[str, Optional[list]]:
        """
        Sequentially rerank top candidates across multiple jobs.
        """
        results = {}
        for idx, (_, job_row) in enumerate(jobs_df.iterrows()):
            # Resolve job ID safely
            job_id = job_row.get("job_id")
            if not job_id:
                logger.warning(f"Skipping job row {idx} due to missing 'job_id'.")
                continue
            job_id = str(job_id)
            
            if job_id not in ranked_candidates_dict:
                logger.warning(f"No candidate dataframe found in ranked_candidates_dict for job_id '{job_id}'. Skipping.")
                results[job_id] = None
                continue
                
            candidates_df = ranked_candidates_dict[job_id]
            job_desc = job_row.get("description", job_row.get("job_description", ""))
            
            logger.info(f"[{idx + 1}/{len(jobs_df)}] Batch reranking for Job ID: {job_id}...")
            ranked_ids = self.rerank(job_desc, candidates_df)
            results[job_id] = ranked_ids
            
        return results

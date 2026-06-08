"""
src/embeddings.py

Dense retrieval module using Sentence-BERT for text embeddings and FAISS for 
efficient similarity searching.
"""

import os
import logging
from typing import List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Try to import FAISS; handle CPU/GPU import failures gracefully
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False

from . import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def encode_texts(
    texts: List[str], 
    model_name: Optional[str] = None, 
    batch_size: int = 32, 
    show_progress: bool = True
) -> np.ndarray:
    """
    Encode a list of texts into dense vectors using a SentenceTransformer model.

    Args:
        texts (List[str]): List of strings to encode.
        model_name (str, optional): The name of the transformer model to use.
                                    If None, defaults to config.EMBEDDING_MODEL.
        batch_size (int): The batch size to use for encoding.
        show_progress (bool): Whether to show progress bar during encoding.

    Returns:
        np.ndarray: A 2D numpy array of shape (n_texts, embedding_dimension).
                    If input list is empty, returns an empty array of shape (0, 384).
    """
    if not texts:
        return np.empty((0, 384), dtype=np.float32)
        
    if model_name is None:
        model_name = config.EMBEDDING_MODEL
        
    try:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(
            texts, 
            batch_size=batch_size, 
            show_progress_bar=show_progress, 
            convert_to_numpy=True
        )
        return np.array(embeddings, dtype=np.float32)
    except Exception as e:
        logger.error(f"Error during SentenceTransformer encoding: {str(e)}")
        raise e


def build_faiss_index(
    candidate_texts: List[str], 
    candidate_ids: List[Any], 
    save_path: Optional[str] = None
) -> Any:
    """
    Build a FAISS index of candidate resumes for similarity search using Inner Product.
    Also saves candidate_ids mapping alongside index if save_path is provided.

    Args:
        candidate_texts (List[str]): List of cleaned candidate resumes.
        candidate_ids (List[Any]): List of candidate IDs corresponding to the texts.
        save_path (str, optional): Absolute path to save the FAISS index to.

    Returns:
        faiss.Index: The built FAISS IndexFlatIP index.
    """
    if not FAISS_AVAILABLE:
        logger.error("FAISS is not installed. Cannot build FAISS index.")
        raise ImportError("FAISS is required to build dense retrieval index. Please install faiss-cpu.")
        
    if len(candidate_texts) != len(candidate_ids):
        raise ValueError("Candidate texts and candidate IDs must have the same length.")
        
    if not candidate_texts:
        raise ValueError("Cannot build index for empty candidate lists.")

    logger.info(f"Encoding {len(candidate_texts)} candidate resumes...")
    embeddings = encode_texts(candidate_texts)
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    dimension = embeddings.shape[1]
    
    # Inner Product on normalized vectors is equivalent to Cosine Similarity
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    if save_path:
        try:
            # Write index to disk
            faiss.write_index(index, save_path)
            logger.info(f"FAISS index successfully saved to: {save_path}")
            
            # Save candidate IDs in the same folder with .ids.npy extension
            ids_path = save_path.rsplit('.', 1)[0] + ".ids.npy"
            np.save(ids_path, np.array(candidate_ids, dtype=object))
            logger.info(f"Candidate IDs mapped and saved to: {ids_path}")
        except Exception as e:
            logger.error(f"Failed to save FAISS index/IDs to disk: {str(e)}")
            
    return index


def load_faiss_index(index_path: str, ids_path: Optional[str] = None) -> Tuple[Any, np.ndarray]:
    """
    Load a pre-built FAISS index and candidate IDs mapping from disk.

    Args:
        index_path (str): Path to the saved FAISS index.
        ids_path (str, optional): Path to the candidate IDs numpy file.
                                  If None, defaults to the index filename with .ids.npy extension.

    Returns:
        Tuple[faiss.Index, np.ndarray]: Loaded FAISS index and the numpy array of candidate IDs.
    """
    if not FAISS_AVAILABLE:
        logger.error("FAISS is not installed. Cannot load FAISS index.")
        raise ImportError("FAISS is required to load dense retrieval index. Please install faiss-cpu.")
        
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index file not found at: {index_path}")
        
    index = faiss.read_index(index_path)
    logger.info(f"FAISS index successfully loaded from: {index_path}")
    
    if ids_path is None:
        ids_path = index_path.rsplit('.', 1)[0] + ".ids.npy"
        
    if not os.path.exists(ids_path):
        raise FileNotFoundError(f"Candidate IDs mapping file not found at: {ids_path}")
        
    candidate_ids = np.load(ids_path, allow_pickle=True)
    logger.info(f"Candidate IDs successfully loaded from: {ids_path}")
    
    return index, candidate_ids


def retrieve_candidates(
    jd_text: str, 
    faiss_index: Any, 
    candidate_ids: np.ndarray, 
    embedder: SentenceTransformer, 
    top_k: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Retrieve top K matching candidates for a given job description using FAISS index.

    Args:
        jd_text (str): The raw job description text.
        faiss_index (faiss.Index): The built FAISS index.
        candidate_ids (np.ndarray): Array mapping index offsets to actual candidate IDs.
        embedder (SentenceTransformer): The model to encode the job description.
        top_k (int, optional): The number of candidates to retrieve. 
                               If None, defaults to config.RETRIEVAL_K.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (scores, candidate_indices, matched_candidate_ids)
            - scores: 1D numpy array of cosine similarities (0 to 1).
            - candidate_indices: 1D numpy array of FAISS internal offsets.
            - matched_candidate_ids: 1D numpy array of the actual candidate IDs.
    """
    if not FAISS_AVAILABLE:
        logger.error("FAISS is not installed. Cannot perform retrieval.")
        raise ImportError("FAISS is required to retrieve candidates. Please install faiss-cpu.")
        
    if top_k is None:
        top_k = config.RETRIEVAL_K
        
    # Cap top_k at number of candidates in index
    top_k = min(top_k, faiss_index.ntotal)
    
    if top_k <= 0:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64), np.empty((0,), dtype=object)

    # Encode query
    jd_emb = embedder.encode(jd_text, convert_to_numpy=True).reshape(1, -1).astype(np.float32)
    
    # Normalize for cosine similarity
    faiss.normalize_L2(jd_emb)
    
    # Search the index
    scores, indices = faiss_index.search(jd_emb, top_k)
    
    # Flatten outputs for 1D convenience
    flat_scores = scores[0]
    flat_indices = indices[0]
    
    # Map back to original candidate IDs
    matched_ids = candidate_ids[flat_indices]
    
    return flat_scores, flat_indices, matched_ids


def build_index_from_dataframe(
    candidates_df: pd.DataFrame, 
    text_column: str = "resume_text", 
    id_column: str = "candidate_id", 
    save_dir: Optional[str] = None
) -> Tuple[Any, np.ndarray, SentenceTransformer]:
    """
    Convenience function to build a FAISS index from a pandas DataFrame.

    Args:
        candidates_df (pd.DataFrame): DataFrame containing candidate data.
        text_column (str): Name of the column with raw/cleaned resume text.
        id_column (str): Name of the column containing unique candidate IDs.
        save_dir (str, optional): Directory to save the built index and IDs to.

    Returns:
        Tuple[faiss.Index, np.ndarray, SentenceTransformer]:
            The built index, candidate IDs array, and the loader model.
    """
    if id_column not in candidates_df.columns:
        raise KeyError(f"ID column '{id_column}' not found in candidate DataFrame.")
    if text_column not in candidates_df.columns:
        raise KeyError(f"Text column '{text_column}' not found in candidate DataFrame.")
        
    texts = candidates_df[text_column].fillna("").astype(str).tolist()
    ids = candidates_df[id_column].tolist()
    
    save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "candidates.index")
        
    index = build_faiss_index(texts, ids, save_path=save_path)
    
    # Re-load or just return mapped IDs
    candidate_ids = np.array(ids, dtype=object)
    
    # Load model to return alongside index
    embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    
    return index, candidate_ids, embedder

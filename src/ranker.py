"""
src/ranker.py

XGBoost pairwise ranking model module. Implements preparation, training, 
prediction, and model analysis functions for candidate ranking.
"""

import os
import logging
import importlib.util
import pathlib
from typing import List, Dict, Optional, Any
import numpy as np
import pandas as pd
try:
    import xgboost as xgb
except ImportError as e:
    raise ImportError("XGBoost is required. Please install it with 'pip install xgboost'.") from e

try:
    from . import config
except ImportError:
    # Fallback: load config directly from file when executed as a script
    _config_path = pathlib.Path(__file__).resolve().parent / "config.py"
    spec = importlib.util.spec_from_file_location("config", _config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_dmatrix(
    train_df: pd.DataFrame, 
    feature_cols: List[str], 
    label_col: str = "label", 
    group_col: str = "job_id"
) -> xgb.DMatrix:
    """
    Prepare an XGBoost DMatrix for pairwise ranking by sorting and grouping by job_id.

    Args:
        train_df (pd.DataFrame): Training DataFrame.
        feature_cols (List[str]): List of column names representing features.
        label_col (str): Column name containing labels (0/1).
        group_col (str): Column name containing query/job IDs.

    Returns:
        xgb.DMatrix: DMatrix with group information configured.
    """
    if train_df.empty:
        raise ValueError("Cannot prepare DMatrix from an empty DataFrame.")
        
    for col in feature_cols + [label_col, group_col]:
        if col not in train_df.columns:
            raise KeyError(f"Required column '{col}' not found in training DataFrame.")
            
    # Sort DataFrame by group_col to ensure contiguous groups (required by XGBoost)
    sorted_df = train_df.sort_values(by=group_col).reset_index(drop=True)
    
    # Calculate group sizes (number of candidates per job)
    group_sizes = sorted_df.groupby(group_col).size().values
    
    # Validate that group sizes sum to total samples
    total_group_samples = int(np.sum(group_sizes))
    if total_group_samples != len(sorted_df):
        raise ValueError(
            f"Validation Failed: Sum of group sizes ({total_group_samples}) "
            f"does not match total sorted DataFrame samples ({len(sorted_df)})."
        )
        
    X = sorted_df[feature_cols].values
    y = sorted_df[label_col].values
    
    # Create DMatrix and set group sizes
    dmatrix = xgb.DMatrix(X, label=y, feature_names=feature_cols)
    dmatrix.set_group(group_sizes)
    
    return dmatrix


def train_ranker(
    train_df: pd.DataFrame, 
    model_save_path: str, 
    feature_cols: Optional[List[str]] = None, 
    params: Optional[Dict[str, Any]] = None
) -> xgb.Booster:
    """
    Train an XGBoost pairwise ranking model on synthetic candidate-job pair data.

    Args:
        train_df (pd.DataFrame): The training dataset.
        model_save_path (str): File path to save the trained Booster.
        feature_cols (List[str], optional): Features to train on. 
                                            If None, uses all columns except job_id, candidate_id, label.
        params (Dict[str, Any], optional): XGBoost parameters. If None, uses defaults.

    Returns:
        xgb.Booster: The trained Booster model.
    """
    if train_df.empty:
        raise ValueError("Cannot train ranking model on empty training data.")
        
    if feature_cols is None:
        # Automatically find feature columns (excluding metadata and labels)
        exclude_cols = {"job_id", "candidate_id", "label"}
        feature_cols = [col for col in train_df.columns if col not in exclude_cols]
        
    if params is None:
        params = {
            "objective": "rank:pairwise",
            "eval_metric": "ndcg",
            "learning_rate": 0.1,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_estimators": 100,
            "random_state": 42
        }
        
    logger.info(f"Preparing DMatrix for training with {len(feature_cols)} features...")
    dtrain = prepare_dmatrix(train_df, feature_cols, label_col="label", group_col="job_id")
    
    # Extract boosting rounds parameter and clean dict for xgb.train()
    local_params = dict(params)
    num_boost_round = local_params.pop("n_estimators", 100)
    
    logger.info(f"Training XGBoost ranker (rounds={num_boost_round})...")
    booster = xgb.train(
        params=local_params,
        dtrain=dtrain,
        num_boost_round=num_boost_round
    )
    
    # Save the model
    try:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        booster.save_model(model_save_path)
        logger.info(f"XGBoost ranking model successfully saved to: {model_save_path}")
    except Exception as e:
        logger.error(f"Failed to save model to disk at {model_save_path}: {str(e)}")
        
    return booster


def load_ranker(model_path: str) -> xgb.Booster:
    """
    Load a pre-trained XGBoost Booster model from disk.

    Args:
        model_path (str): Path to the saved model file.

    Returns:
        xgb.Booster: The loaded model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"XGBoost model file not found at: {model_path}")
        
    booster = xgb.Booster()
    booster.load_model(model_path)
    logger.info(f"XGBoost model successfully loaded from: {model_path}")
    return booster


def predict_rankings(features_matrix: np.ndarray, ranker: xgb.Booster) -> np.ndarray:
    """
    Predict pairwise ranking scores for a matrix of candidate features.

    Args:
        features_matrix (np.ndarray): 2D array of shape (n_candidates, n_features).
        ranker (xgb.Booster): Trained XGBoost ranking model.

    Returns:
        np.ndarray: 1D array of scores (higher scores indicate better rank position).
    """
    if features_matrix.ndim != 2:
        raise ValueError(f"Features matrix must be 2D, got shape {features_matrix.shape}")
        
    # Extract feature names from ranker booster to avoid feature mismatch errors
    feature_names = getattr(ranker, "feature_names", None)
    
    if isinstance(features_matrix, pd.DataFrame):
        dpredict = xgb.DMatrix(features_matrix)
    else:
        dpredict = xgb.DMatrix(features_matrix, feature_names=feature_names)
        
    preds = ranker.predict(dpredict)
    return preds


def get_feature_importance(ranker: xgb.Booster, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Extract and sort feature importance scores from the XGBoost ranking model.

    Args:
        ranker (xgb.Booster): Trained model.
        feature_names (List[str], optional): Map feature keys (e.g. f0, f1) to descriptive names.

    Returns:
        Dict[str, float]: Sorted feature importance scores.
    """
    # Get scores (gain, cover, weight, etc.). Default is weight (frequency)
    scores = ranker.get_score(importance_type="gain")
    
    mapped_scores = {}
    for key, val in scores.items():
        # XGBoost names feature columns as f0, f1, etc. if no headers provided
        if key.startswith("f") and key[1:].isdigit() and feature_names:
            idx = int(key[1:])
            if idx < len(feature_names):
                mapped_scores[feature_names[idx]] = float(val)
                continue
        mapped_scores[key] = float(val)
        
    # If some features didn't get split, assign 0.0
    if feature_names:
        for name in feature_names:
            if name not in mapped_scores:
                mapped_scores[name] = 0.0
                
    # Sort by importance value descending
    sorted_scores = dict(sorted(mapped_scores.items(), key=lambda item: item[1], reverse=True))
    return sorted_scores

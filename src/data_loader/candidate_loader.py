#!/usr/bin/env python3
"""
Candidate data loader for InsightHire AI
Loads and validates candidate data from JSONL format
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CandidateLoader:
    """Load and validate candidate data from JSONL files."""
    
    def __init__(self, data_path: str = "Data/raw/candidates.jsonl"):
        """
        Initialize the candidate loader.
        
        Args:
            data_path: Path to the candidates JSONL file
        """
        self.data_path = Path(data_path)
        self.candidates = []
        
    def load_candidates(self, limit: Optional[int] = None) -> List[Dict[Any, Any]]:
        """
        Load candidates from JSONL file.
        
        Args:
            limit: Maximum number of candidates to load (None for all)
            
        Returns:
            List of candidate dictionaries
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Candidates file not found: {self.data_path}")
            
        candidates = []
        
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if limit and i >= limit:
                        break
                        
                    line = line.strip()
                    if line:
                        try:
                            candidate = json.loads(line)
                            candidates.append(candidate)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipping invalid JSON on line {i+1}: {e}")
                            continue
                            
        except Exception as e:
            logger.error(f"Error loading candidates: {e}")
            raise
            
        self.candidates = candidates
        logger.info(f"Loaded {len(candidates)} candidates")
        return candidates
    
    def get_candidate_count(self) -> int:
        """
        Get total number of candidates without loading all data.
        
        Returns:
            Number of candidates in the file
        """
        if not self.data_path.exists():
            return 0
            
        count = 0
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        count += 1
        except Exception as e:
            logger.error(f"Error counting candidates: {e}")
            return 0
            
        return count
    
    def validate_candidate_structure(self, candidate: Dict[Any, Any]) -> bool:
        """
        Basic validation of candidate structure.
        
        Args:
            candidate: Candidate dictionary to validate
            
        Returns:
            True if valid structure, False otherwise
        """
        required_fields = ['candidate_id', 'profile', 'career_history', 'education', 'skills', 'redrob_signals']
        
        for field in required_fields:
            if field not in candidate:
                logger.warning(f"Missing required field: {field}")
                return False
                
        # Validate candidate_id format
        cid = candidate.get('candidate_id', '')
        if not cid.startswith('CAND_') or len(cid) != 12:
            logger.warning(f"Invalid candidate_id format: {cid}")
            return False
            
        return True
    
    def load_sample_candidates(self, sample_path: str = "Data/raw/sample_candidates.json") -> List[Dict[Any, Any]]:
        """
        Load sample candidates for testing.
        
        Args:
            sample_path: Path to sample candidates file
            
        Returns:
            List of sample candidate dictionaries
        """
        sample_file = Path(sample_path)
        
        if not sample_file.exists():
            logger.warning(f"Sample file not found: {sample_path}")
            return []
            
        try:
            with open(sample_file, 'r', encoding='utf-8') as f:
                samples = json.load(f)
                logger.info(f"Loaded {len(samples)} sample candidates")
                return samples
        except Exception as e:
            logger.error(f"Error loading sample candidates: {e}")
            return []


def main():
    """Test the candidate loader."""
    loader = CandidateLoader()
    
    # First try to count candidates
    total_count = loader.get_candidate_count()
    print(f"Total candidates in dataset: {total_count}")
    
    # Load a small sample first
    print("Loading first 10 candidates for testing...")
    candidates = loader.load_candidates(limit=10)
    
    if candidates:
        print(f"Successfully loaded {len(candidates)} candidates")
        
        # Validate first candidate
        if loader.validate_candidate_structure(candidates[0]):
            print(" Candidate structure validation passed")
        else:
            print(" Candidate structure validation failed")
            
        # Show first candidate ID
        print(f"First candidate: {candidates[0].get('candidate_id', 'Unknown')}")
    else:
        print(" No candidates loaded")


if __name__ == "__main__":
    main()
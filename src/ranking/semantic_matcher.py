#!/usr/bin/env python3
"""
Semantic Matcher - Embedding-based candidate similarity matching
Uses AI embeddings to find candidates semantically similar to job descriptions
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.feature_engineering.embeddings_features import EmbeddingsFeaturesExtractor
from src.job_requirements.job_schema import JobRequirements


class SemanticMatcher:
    """
    Semantic matching engine using text embeddings.
    Finds candidates most similar to job descriptions at a semantic level.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.embeddings_extractor = None
        self.job_embedding = None
        self.job_text = None
        
        # Initialize embeddings extractor
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize the embeddings extractor"""
        
        # Create a dummy job requirement for initialization
        dummy_job = JobRequirements()
        self.embeddings_extractor = EmbeddingsFeaturesExtractor(dummy_job, self.model_name)
        
        if not self.embeddings_extractor.model:
            print("Warning: Embeddings model not available. Semantic matching will use fallback scores.")
    
    def create_job_embedding(self, jd_profile: Dict[str, Any], 
                           job_requirements: JobRequirements) -> np.ndarray:
        """
        Create embedding for job description
        
        Args:
            jd_profile: Job description profile
            job_requirements: Job requirements object
            
        Returns:
            Job embedding vector
        """
        
        # Create comprehensive job description text
        job_text_parts = []
        
        # Basic job info
        job_info = jd_profile.get('job_info', {})
        role_title = job_info.get('role_title', '')
        role_type = job_info.get('role_type', '')
        role_level = job_info.get('role_level', '')
        
        if role_title:
            job_text_parts.append(f"Role: {role_title}")
        if role_type:
            job_text_parts.append(f"Type: {role_type}")
        if role_level:
            job_text_parts.append(f"Level: {role_level}")
        
        # Required skills (high importance)
        required_skills = jd_profile.get('required_skills', [])
        if required_skills:
            job_text_parts.append(f"Must have skills: {', '.join(required_skills)}")
        
        # Preferred skills
        preferred_skills = jd_profile.get('preferred_skills', [])
        if preferred_skills:
            job_text_parts.append(f"Preferred skills: {', '.join(preferred_skills)}")
        
        # Experience requirements
        exp_range = jd_profile.get('experience_range', [0, 20])
        if exp_range[0] > 0 or exp_range[1] < 20:
            job_text_parts.append(f"Experience: {exp_range[0]} to {exp_range[1]} years")
        
        # Positive signals (what we want to see)
        positive_signals = jd_profile.get('positive_signals', [])
        if positive_signals:
            job_text_parts.append(f"Positive indicators: {', '.join(positive_signals)}")
        
        # Industries
        industries = jd_profile.get('preferred_industries', [])
        if industries:
            job_text_parts.append(f"Industries: {', '.join(industries)}")
        
        # Company types
        company_types = jd_profile.get('company_types', [])
        if company_types:
            job_text_parts.append(f"Company types: {', '.join(company_types)}")
        
        self.job_text = " ".join(job_text_parts)
        
        # Generate embedding
        if self.embeddings_extractor and self.embeddings_extractor.model:
            self.job_embedding = self.embeddings_extractor._generate_embedding(self.job_text)
            print(f"Created job embedding (dim: {len(self.job_embedding) if self.job_embedding is not None else 0})")
            print(f"Job text: {self.job_text[:200]}...")
        else:
            self.job_embedding = None
            print("Warning: Cannot create job embedding - model not available")
        
        return self.job_embedding
    
    def calculate_semantic_similarities(self, candidates_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate semantic similarity between job and all candidates
        
        Args:
            candidates_df: DataFrame with candidate features including embeddings
            
        Returns:
            DataFrame with added semantic similarity scores
        """
        
        if self.job_embedding is None:
            print("Warning: No job embedding available. Using fallback similarity scores.")
            candidates_df['semantic_similarity'] = 0.5  # Neutral score
            return candidates_df
        
        similarities = []
        
        for idx, candidate in candidates_df.iterrows():
            candidate_embedding = candidate.get('candidate_embedding')
            
            if candidate_embedding is not None and len(candidate_embedding) > 0:
                # Convert to numpy array
                candidate_emb = np.array(candidate_embedding)
                
                # Calculate cosine similarity
                similarity = self._calculate_cosine_similarity(self.job_embedding, candidate_emb)
                similarities.append(similarity)
            else:
                # No embedding available for this candidate
                similarities.append(0.3)  # Low default similarity
        
        candidates_df['semantic_similarity'] = similarities
        
        # Convert to 0-100 scale for consistency with other scores
        candidates_df['semantic_similarity_score'] = candidates_df['semantic_similarity'] * 100
        
        print(f"Calculated semantic similarities for {len(candidates_df)} candidates")
        print(f"Average similarity: {candidates_df['semantic_similarity'].mean():.3f}")
        print(f"Max similarity: {candidates_df['semantic_similarity'].max():.3f}")
        print(f"Min similarity: {candidates_df['semantic_similarity'].min():.3f}")
        
        return candidates_df
    
    def _calculate_cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        
        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            
            # Ensure similarity is between 0 and 1
            similarity = max(0.0, min(1.0, similarity))
            
            return similarity
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.3  # Fallback similarity
    
    def get_top_semantic_matches(self, candidates_df: pd.DataFrame, 
                                top_k: int = 20) -> pd.DataFrame:
        """
        Get top candidates by semantic similarity
        
        Args:
            candidates_df: DataFrame with semantic similarities
            top_k: Number of top candidates to return
            
        Returns:
            DataFrame with top semantically similar candidates
        """
        
        if 'semantic_similarity' not in candidates_df.columns:
            print("Error: Semantic similarities not calculated. Run calculate_semantic_similarities first.")
            return candidates_df.head(top_k)
        
        # Sort by semantic similarity (descending)
        top_matches = candidates_df.nlargest(top_k, 'semantic_similarity')
        
        return top_matches
    
    def analyze_semantic_matches(self, top_matches: pd.DataFrame, 
                                original_candidates: List[Dict[str, Any]] = None):
        """
        Analyze and display top semantic matches for manual verification
        
        Args:
            top_matches: DataFrame with top semantic matches
            original_candidates: Original candidate data for detailed analysis
        """
        
        print("\n" + "="*60)
        print("SEMANTIC MATCHING ANALYSIS - TOP CANDIDATES")
        print("="*60)
        
        print(f"Job Description Text:\n{self.job_text}\n")
        print("-" * 60)
        
        for i, (idx, candidate) in enumerate(top_matches.iterrows(), 1):
            candidate_id = candidate['candidate_id']
            similarity = candidate['semantic_similarity']
            similarity_score = candidate.get('semantic_similarity_score', similarity * 100)
            
            print(f"\n{i:2d}. {candidate_id}")
            print(f"    Semantic Similarity: {similarity:.3f} ({similarity_score:.1f}/100)")
            
            # Show other relevant scores for context
            if 'technical_fit_score' in candidate:
                print(f"    Technical Fit:       {candidate['technical_fit_score']:.1f}")
            if 'career_score' in candidate:
                print(f"    Career Score:        {candidate['career_score']:.1f}")
            if 'overall_candidate_score' in candidate:
                print(f"    Overall Score:       {candidate['overall_candidate_score']:.1f}")
        
        # Summary statistics
        avg_similarity = top_matches['semantic_similarity'].mean()
        print(f"\n📊 SUMMARY:")
        print(f"    Average Similarity: {avg_similarity:.3f}")
        print(f"    Range: {top_matches['semantic_similarity'].min():.3f} - {top_matches['semantic_similarity'].max():.3f}")
        
        print(f"\n✅ Checkpoint 2: Review the top 20 candidates above")
        print(f"Ask yourself: Are they actually matching the job requirements?")
        print(f"If not, consider improving the job description text or candidate text extraction.")
    
    def save_semantic_results(self, candidates_df: pd.DataFrame, 
                            output_path: str = 'Data/processed/semantic_similarities.parquet'):
        """Save semantic similarity results"""
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Select relevant columns for saving
        columns_to_save = ['candidate_id', 'semantic_similarity', 'semantic_similarity_score']
        existing_columns = [col for col in columns_to_save if col in candidates_df.columns]
        
        semantic_results = candidates_df[existing_columns].copy()
        semantic_results.to_parquet(output_file, index=False)
        
        print(f"Semantic similarity results saved to: {output_file}")
        return output_file
    
    def create_enhanced_job_text(self, jd_profile: Dict[str, Any], 
                               job_requirements: JobRequirements,
                               include_examples: bool = True) -> str:
        """
        Create enhanced job description text with examples and context
        
        Args:
            jd_profile: Job description profile
            job_requirements: Job requirements
            include_examples: Whether to include example phrases
            
        Returns:
            Enhanced job description text
        """
        
        text_parts = []
        
        # Role description with context
        job_info = jd_profile.get('job_info', {})
        role_title = job_info.get('role_title', '')
        role_type = job_info.get('role_type', '')
        role_level = job_info.get('role_level', '')
        
        if role_title and role_level and role_type:
            text_parts.append(f"Seeking a {role_level} {role_title} for {role_type} role")
        
        # Detailed skill descriptions
        required_skills = jd_profile.get('required_skills', [])
        if required_skills:
            skills_text = f"Must have expertise in: {', '.join(required_skills)}"
            if include_examples:
                skills_text += ". Candidates should have hands-on experience and proven proficiency in these technologies."
            text_parts.append(skills_text)
        
        preferred_skills = jd_profile.get('preferred_skills', [])
        if preferred_skills:
            pref_text = f"Preferred additional skills: {', '.join(preferred_skills)}"
            if include_examples:
                pref_text += ". Experience with these technologies is a plus and will be considered favorably."
            text_parts.append(pref_text)
        
        # Experience with context
        exp_range = jd_profile.get('experience_range', [0, 20])
        if exp_range[0] > 0:
            exp_text = f"Requires {exp_range[0]} to {exp_range[1]} years of relevant experience"
            if include_examples:
                exp_text += " in similar roles or technologies"
            text_parts.append(exp_text)
        
        # Company and industry context
        industries = jd_profile.get('preferred_industries', [])
        if industries:
            text_parts.append(f"Experience in {', '.join(industries)} industry preferred")
        
        # Behavioral and cultural fit
        positive_signals = jd_profile.get('positive_signals', [])
        if positive_signals:
            text_parts.append(f"Looking for candidates who demonstrate: {', '.join(positive_signals)}")
        
        # What we don't want
        negative_signals = jd_profile.get('negative_signals', [])
        if negative_signals and include_examples:
            text_parts.append(f"Not suitable for candidates with primarily: {', '.join(negative_signals)}")
        
        return ". ".join(text_parts)


if __name__ == "__main__":
    # Test Semantic Matcher
    print("Testing Semantic Matcher...")
    
    # This would normally be run after we have candidate features with embeddings
    print("Note: This test requires candidate features with embeddings to be generated first.")
    print("Run the feature extraction pipeline with embeddings enabled to test semantic matching.")
    
    # Example usage:
    print("\nExample usage:")
    print("1. Create JD profile using jd_parser.py")
    print("2. Load candidate features with embeddings")  
    print("3. matcher = SemanticMatcher()")
    print("4. matcher.create_job_embedding(jd_profile, job_requirements)")
    print("5. candidates_df = matcher.calculate_semantic_similarities(candidates_df)")
    print("6. top_matches = matcher.get_top_semantic_matches(candidates_df, top_k=20)")
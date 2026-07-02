#!/usr/bin/env python3
"""
Embeddings Features Extractor - Creates semantic embeddings for candidate matching
Generates text embeddings using sentence-transformers for semantic similarity matching
"""

from typing import Dict, List, Any, Optional, Tuple
import sys
from pathlib import Path
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.job_requirements.job_schema import JobRequirements


class EmbeddingsFeaturesExtractor:
    """
    Creates semantic embeddings for candidate profiles and job descriptions.
    Uses sentence-transformers to generate vector representations for similarity matching.
    """
    
    def __init__(self, job_requirements: JobRequirements, model_name: str = 'all-MiniLM-L6-v2'):
        self.job_requirements = job_requirements
        self.model_name = model_name
        self.model = None
        self.embedding_dim = 384  # Default for all-MiniLM-L6-v2
        
        # Initialize the model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the sentence transformer model"""
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("Warning: sentence-transformers not available. Install with: pip install sentence-transformers")
            return
        
        try:
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"Initialized embedding model: {self.model_name} (dim: {self.embedding_dim})")
        except Exception as e:
            print(f"Error initializing embedding model: {e}")
            self.model = None
    
    def extract_features(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract embedding features for a candidate
        
        Args:
            candidate: Candidate data dictionary
            
        Returns:
            Dictionary with embedding features and similarity scores
        """
        
        if not self.model:
            return self._get_fallback_features()
        
        # Extract and combine candidate text
        candidate_text = self._extract_candidate_text(candidate)
        
        # Generate candidate embedding
        candidate_embedding = self._generate_embedding(candidate_text)
        
        # Create job description text from requirements
        job_text = self._create_job_description_text()
        
        # Generate job embedding
        job_embedding = self._generate_embedding(job_text)
        
        # Calculate similarity scores
        overall_similarity = self._calculate_cosine_similarity(candidate_embedding, job_embedding)
        
        # Calculate section-specific similarities
        profile_similarity = self._calculate_profile_similarity(candidate)
        skills_similarity = self._calculate_skills_similarity(candidate)
        experience_similarity = self._calculate_experience_similarity(candidate)
        
        return {
            'candidate_embedding': candidate_embedding.tolist() if candidate_embedding is not None else None,
            'overall_similarity_score': round(overall_similarity * 100, 2),
            'profile_similarity_score': round(profile_similarity * 100, 2),
            'skills_similarity_score': round(skills_similarity * 100, 2),
            'experience_similarity_score': round(experience_similarity * 100, 2),
            'embedding_dimension': self.embedding_dim,
            'text_length': len(candidate_text)
        }
    
    def _get_fallback_features(self) -> Dict[str, Any]:
        """Return fallback features when embedding model is not available"""
        
        return {
            'candidate_embedding': None,
            'overall_similarity_score': 50.0,  # Neutral score
            'profile_similarity_score': 50.0,
            'skills_similarity_score': 50.0,
            'experience_similarity_score': 50.0,
            'embedding_dimension': 0,
            'text_length': 0
        }
    
    def _extract_candidate_text(self, candidate: Dict[str, Any]) -> str:
        """
        Extract and combine all relevant text from candidate profile
        """
        
        text_parts = []
        
        # Profile information
        profile = candidate.get('profile', {})
        
        # Add headline and summary
        headline = profile.get('headline', '').strip()
        if headline:
            text_parts.append(f"Headline: {headline}")
        
        summary = profile.get('summary', '').strip()
        if summary:
            text_parts.append(f"Summary: {summary}")
        
        # Add current role information
        current_title = profile.get('current_title', '').strip()
        if current_title:
            text_parts.append(f"Current Role: {current_title}")
        
        current_industry = profile.get('current_industry', '').strip()
        if current_industry:
            text_parts.append(f"Industry: {current_industry}")
        
        # Add skills
        skills = candidate.get('skills', [])
        if skills:
            skill_names = [skill.get('name', '') for skill in skills[:15]]  # Top 15 skills
            skills_text = ', '.join([skill for skill in skill_names if skill])
            if skills_text:
                text_parts.append(f"Skills: {skills_text}")
        
        # Add career history (recent jobs)
        career_history = candidate.get('career_history', [])
        for i, job in enumerate(career_history[:3]):  # Most recent 3 jobs
            job_title = job.get('title', '')
            job_company = job.get('company', '')
            job_description = job.get('description', '')
            
            if job_title:
                text_parts.append(f"Job {i+1}: {job_title}")
                if job_company:
                    text_parts.append(f"Company: {job_company}")
                if job_description:
                    # Truncate long descriptions
                    desc = job_description[:200] + "..." if len(job_description) > 200 else job_description
                    text_parts.append(f"Description: {desc}")
        
        return " ".join(text_parts)
    
    def _create_job_description_text(self) -> str:
        """
        Create job description text from job requirements
        """
        
        text_parts = []
        
        # Role information
        if self.job_requirements.role_title:
            text_parts.append(f"Role: {self.job_requirements.role_title}")
        
        if self.job_requirements.role_type:
            text_parts.append(f"Type: {self.job_requirements.role_type}")
        
        if self.job_requirements.role_level:
            text_parts.append(f"Level: {self.job_requirements.role_level}")
        
        # Required skills
        if self.job_requirements.required_skills:
            required_skills_text = ', '.join(self.job_requirements.required_skills)
            text_parts.append(f"Required Skills: {required_skills_text}")
        
        # Preferred skills
        if self.job_requirements.preferred_skills:
            preferred_skills_text = ', '.join(self.job_requirements.preferred_skills)
            text_parts.append(f"Preferred Skills: {preferred_skills_text}")
        
        # Experience requirements
        if self.job_requirements.min_years_experience:
            text_parts.append(f"Minimum Experience: {self.job_requirements.min_years_experience} years")
        
        # Industry preferences
        if self.job_requirements.preferred_industries:
            industries_text = ', '.join(self.job_requirements.preferred_industries)
            text_parts.append(f"Preferred Industries: {industries_text}")
        
        # Company types
        if self.job_requirements.company_types:
            company_types_text = ', '.join(self.job_requirements.company_types)
            text_parts.append(f"Company Types: {company_types_text}")
        
        return " ".join(text_parts) if text_parts else "General position"
    
    def _generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Generate embedding for given text
        """
        
        if not self.model or not text.strip():
            return None
        
        try:
            embedding = self.model.encode(text, convert_to_tensor=False)
            return np.array(embedding)
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    def _calculate_cosine_similarity(self, embedding1: Optional[np.ndarray], 
                                   embedding2: Optional[np.ndarray]) -> float:
        """
        Calculate cosine similarity between two embeddings
        """
        
        if embedding1 is None or embedding2 is None:
            return 0.5  # Neutral similarity if embeddings unavailable
        
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
            return 0.5
    
    def _calculate_profile_similarity(self, candidate: Dict[str, Any]) -> float:
        """
        Calculate similarity for profile section (headline + summary)
        """
        
        profile = candidate.get('profile', {})
        
        # Extract profile text
        profile_text_parts = []
        headline = profile.get('headline', '').strip()
        summary = profile.get('summary', '').strip()
        
        if headline:
            profile_text_parts.append(headline)
        if summary:
            profile_text_parts.append(summary)
        
        profile_text = " ".join(profile_text_parts)
        
        if not profile_text:
            return 0.3  # Low similarity for empty profile
        
        # Generate embeddings
        profile_embedding = self._generate_embedding(profile_text)
        job_embedding = self._generate_embedding(self._create_job_description_text())
        
        return self._calculate_cosine_similarity(profile_embedding, job_embedding)
    
    def _calculate_skills_similarity(self, candidate: Dict[str, Any]) -> float:
        """
        Calculate similarity for skills section
        """
        
        skills = candidate.get('skills', [])
        
        if not skills:
            return 0.2  # Low similarity for no skills
        
        # Extract skill names
        skill_names = [skill.get('name', '') for skill in skills]
        skills_text = ', '.join([skill for skill in skill_names if skill])
        
        if not skills_text:
            return 0.2
        
        # Create job skills text
        job_skills = self.job_requirements.get_all_relevant_skills()
        job_skills_text = ', '.join(job_skills) if job_skills else ""
        
        if not job_skills_text:
            return 0.5  # Neutral if no job skills specified
        
        # Generate embeddings
        candidate_skills_embedding = self._generate_embedding(f"Skills: {skills_text}")
        job_skills_embedding = self._generate_embedding(f"Skills: {job_skills_text}")
        
        return self._calculate_cosine_similarity(candidate_skills_embedding, job_skills_embedding)
    
    def _calculate_experience_similarity(self, candidate: Dict[str, Any]) -> float:
        """
        Calculate similarity for experience/career history
        """
        
        career_history = candidate.get('career_history', [])
        
        if not career_history:
            return 0.3  # Low similarity for no experience
        
        # Extract experience text (job titles and descriptions)
        experience_parts = []
        
        for job in career_history[:3]:  # Most recent 3 jobs
            title = job.get('title', '').strip()
            description = job.get('description', '').strip()
            
            if title:
                experience_parts.append(title)
            if description:
                # Truncate long descriptions
                desc = description[:150] + "..." if len(description) > 150 else description
                experience_parts.append(desc)
        
        experience_text = " ".join(experience_parts)
        
        if not experience_text:
            return 0.3
        
        # Generate embeddings
        experience_embedding = self._generate_embedding(experience_text)
        job_embedding = self._generate_embedding(self._create_job_description_text())
        
        return self._calculate_cosine_similarity(experience_embedding, job_embedding)
    
    def generate_job_embedding(self) -> Optional[np.ndarray]:
        """
        Generate embedding for the current job requirements
        
        Returns:
            Job embedding vector or None if model unavailable
        """
        
        job_text = self._create_job_description_text()
        return self._generate_embedding(job_text)
    
    def find_similar_candidates(self, candidates: List[Dict[str, Any]], 
                               top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find most similar candidates to job requirements
        
        Args:
            candidates: List of candidate dictionaries
            top_k: Number of top candidates to return
            
        Returns:
            List of (candidate_id, similarity_score) tuples
        """
        
        if not self.model:
            return []
        
        job_embedding = self.generate_job_embedding()
        if job_embedding is None:
            return []
        
        candidate_similarities = []
        
        for candidate in candidates:
            candidate_id = candidate.get('candidate_id', '')
            candidate_text = self._extract_candidate_text(candidate)
            candidate_embedding = self._generate_embedding(candidate_text)
            
            if candidate_embedding is not None:
                similarity = self._calculate_cosine_similarity(candidate_embedding, job_embedding)
                candidate_similarities.append((candidate_id, similarity))
        
        # Sort by similarity (descending) and return top k
        candidate_similarities.sort(key=lambda x: x[1], reverse=True)
        return candidate_similarities[:top_k]
    
    def save_embeddings(self, candidates: List[Dict[str, Any]], 
                       output_path: str = 'Data/processed/candidate_embeddings.npy'):
        """
        Generate and save embeddings for all candidates
        
        Args:
            candidates: List of candidate dictionaries
            output_path: Path to save embeddings file
        """
        
        if not self.model:
            print("Embedding model not available. Cannot save embeddings.")
            return
        
        embeddings = []
        candidate_ids = []
        
        print(f"Generating embeddings for {len(candidates)} candidates...")
        
        for i, candidate in enumerate(candidates):
            if i % 1000 == 0:  # Progress indicator
                print(f"Processing candidate {i}/{len(candidates)}")
            
            candidate_id = candidate.get('candidate_id', f'candidate_{i}')
            candidate_text = self._extract_candidate_text(candidate)
            candidate_embedding = self._generate_embedding(candidate_text)
            
            if candidate_embedding is not None:
                embeddings.append(candidate_embedding)
                candidate_ids.append(candidate_id)
        
        if embeddings:
            embeddings_array = np.vstack(embeddings)
            
            # Save embeddings and metadata
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            np.save(output_path, embeddings_array)
            
            # Save candidate IDs
            ids_path = output_path.replace('.npy', '_ids.txt')
            with open(ids_path, 'w') as f:
                for candidate_id in candidate_ids:
                    f.write(f"{candidate_id}\n")
            
            print(f"Saved {len(embeddings)} embeddings to {output_path}")
            print(f"Saved candidate IDs to {ids_path}")
        else:
            print("No embeddings generated. Check model initialization.")
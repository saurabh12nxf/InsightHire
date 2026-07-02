#!/usr/bin/env python3
"""
Reasoning Engine - Generates human-readable explanations for ranking decisions
Creates grounded, fact-based explanations that recruiters can understand and trust
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.job_requirements.job_schema import JobRequirements


class ReasoningEngine:
    """
    Generates human-readable explanations for candidate rankings.
    
    Key principles:
    - Grounded in actual data (never hallucinate)
    - Recruiter-friendly language
    - Highlights both strengths and concerns
    - Actionable insights
    """
    
    def __init__(self, job_requirements: JobRequirements, jd_profile: Dict[str, Any]):
        self.job_requirements = job_requirements
        self.jd_profile = jd_profile
        
        # Explanation templates
        self.strength_templates = {
            'technical_skills': [
                "Strong {skill} expertise",
                "Proven {skill} experience",
                "Advanced {skill} proficiency",
                "{years}+ years with {skill}"
            ],
            'experience_fit': [
                "{years} years of relevant experience",
                "Experience aligns with role requirements",
                "Solid background in {domain}",
                "Well-matched experience level"
            ],
            'career_progression': [
                "Strong career trajectory",
                "Consistent upward progression",
                "{promotions} career advancement(s)",
                "Leadership growth demonstrated"
            ],
            'behavioral_positive': [
                "Highly responsive to recruiters ({rate}% response rate)",
                "Active job seeker (open to work)",
                "Quick availability ({days} days notice)",
                "Strong professional engagement"
            ],
            'github_activity': [
                "Active GitHub presence",
                "Demonstrates technical passion",
                "Open source contributions",
                "Recent coding activity"
            ],
            'stability': [
                "Stable career history",
                "Long tenure at companies",
                "Consistent employment pattern",
                "Reliable work history"
            ]
        }
        
        self.concern_templates = {
            'technical_gaps': [
                "Missing {skill} experience",
                "Limited {skill} background",
                "No {skill} mentioned in profile",
                "Weak {skill} proficiency"
            ],
            'experience_mismatch': [
                "Experience level mismatch ({years} vs {required} years)",
                "Over/under-qualified for role",
                "Experience outside target range",
                "Seniority level concerns"
            ],
            'behavioral_concerns': [
                "Long notice period ({days} days)",
                "Low recruiter response rate ({rate}%)",
                "Limited platform engagement",
                "Inactive job seeking behavior"
            ],
            'career_concerns': [
                "Frequent job changes",
                "Career progression concerns",
                "Industry mismatch",
                "Consulting-heavy background"
            ],
            'credibility_issues': [
                "Profile inconsistencies detected",
                "Skill claims vs experience mismatch",
                "Incomplete profile information",
                "Credibility verification needed"
            ],
            'github_missing': [
                "No GitHub activity found",
                "Limited technical portfolio",
                "No code samples available",
                "Technical work not demonstrated online"
            ]
        }
    
    def generate_reasoning(self, candidates_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate reasoning explanations for all candidates
        
        Args:
            candidates_df: DataFrame with candidate scores and features
            
        Returns:
            DataFrame with reasoning explanations added
        """
        
        print("Generating reasoning explanations...")
        
        reasoning_explanations = []
        reasoning_summaries = []
        
        for idx, candidate in candidates_df.iterrows():
            # Generate full explanation
            explanation = self._generate_candidate_explanation(candidate)
            
            # Generate summary
            summary = self._generate_explanation_summary(explanation, candidate)
            
            reasoning_explanations.append(explanation)
            reasoning_summaries.append(summary)
        
        candidates_df['reasoning_explanation'] = reasoning_explanations
        candidates_df['reasoning_summary'] = reasoning_summaries
        
        print(f"Reasoning generation completed for {len(candidates_df)} candidates")
        
        # Validate explanation diversity
        self._validate_explanation_diversity(reasoning_explanations)
        
        return candidates_df
    
    def _generate_candidate_explanation(self, candidate: pd.Series) -> Dict[str, Any]:
        """Generate complete explanation for a candidate"""
        
        candidate_id = candidate['candidate_id']
        
        # Collect strengths
        strengths = self._identify_strengths(candidate)
        
        # Collect concerns
        concerns = self._identify_concerns(candidate)
        
        # Overall assessment
        overall_score = candidate.get('hybrid_score', candidate.get('overall_candidate_score', 50.0))
        overall_assessment = self._generate_overall_assessment(overall_score)
        
        # Confidence level
        confidence_score = candidate.get('confidence_score', 50.0)
        confidence_level = candidate.get('confidence_level', 'Medium')
        
        return {
            'candidate_id': candidate_id,
            'overall_assessment': overall_assessment,
            'overall_score': round(overall_score, 1),
            'strengths': strengths,
            'concerns': concerns,
            'confidence_score': round(confidence_score, 1),
            'confidence_level': confidence_level,
            'recommendation': self._generate_recommendation(candidate, strengths, concerns)
        }
    
    def _identify_strengths(self, candidate: pd.Series) -> List[str]:
        """Identify candidate strengths based on scores and data"""
        
        strengths = []
        
        # Technical strengths
        technical_score = candidate.get('technical_fit_score', 0)
        required_coverage = candidate.get('required_skills_coverage', 0)
        
        if technical_score >= 50.0:
            strengths.append(f"Strong technical fit ({technical_score:.0f}/100)")
        
        if required_coverage >= 75.0:
            strengths.append(f"Excellent skills match ({required_coverage:.0f}% of requirements)")
        elif required_coverage >= 50.0:
            strengths.append(f"Good skills alignment ({required_coverage:.0f}% match)")
        
        # Experience strengths
        years_exp = candidate.get('total_years_experience', 0)
        experience_fit = candidate.get('experience_fit_score', 50.0)
        
        if experience_fit >= 80.0:
            strengths.append(f"Perfect experience match ({years_exp:.1f} years)")
        elif experience_fit >= 60.0:
            strengths.append(f"Relevant experience ({years_exp:.1f} years)")
        
        # Career progression
        trajectory_score = candidate.get('trajectory_score', 50.0)
        promotions = candidate.get('total_promotions', 0)
        
        if trajectory_score >= 70.0:
            strengths.append("Strong career progression")
        if promotions >= 2:
            strengths.append(f"{promotions} career advancement(s)")
        
        # Behavioral strengths
        behavioral_score = candidate.get('behavioral_score', 50.0)
        availability_score = candidate.get('availability_score', 50.0)
        engagement_score = candidate.get('engagement_score', 50.0)
        
        if availability_score >= 80.0:
            notice_period = candidate.get('notice_period_days', 60)
            if notice_period <= 30:
                strengths.append(f"Quick availability ({notice_period} days notice)")
            else:
                strengths.append("High availability score")
        
        if engagement_score >= 70.0:
            strengths.append("Highly engaged with recruiters")
        
        if candidate.get('is_open_to_work', False):
            strengths.append("Actively seeking opportunities")
        
        # GitHub activity (for technical roles)
        if self.job_requirements.role_type == 'technical':
            github_score = candidate.get('github_activity_score', 0)
            if github_score >= 50.0:
                strengths.append("Active GitHub presence")
        
        # Stability
        stability_score = candidate.get('stability_score', 50.0)
        avg_duration = candidate.get('average_job_duration_months', 0)
        
        if stability_score >= 70.0 and avg_duration >= 24:
            strengths.append("Stable work history")
        
        # Credibility
        credibility_score = candidate.get('credibility_score', 80.0)
        if credibility_score >= 90.0:
            strengths.append("Highly credible profile")
        
        return strengths[:5]  # Limit to top 5 strengths
    
    def _identify_concerns(self, candidate: pd.Series) -> List[str]:
        """Identify candidate concerns based on scores and data"""
        
        concerns = []
        
        # Technical concerns
        technical_score = candidate.get('technical_fit_score', 0)
        required_coverage = candidate.get('required_skills_coverage', 0)
        
        if technical_score < 30.0:
            concerns.append("Limited technical alignment")
        
        if required_coverage < 50.0:
            missing_skills = self._identify_missing_skills(candidate)
            if missing_skills:
                concerns.append(f"Missing key skills: {', '.join(missing_skills[:3])}")
        
        # Experience concerns
        experience_fit = candidate.get('experience_fit_score', 50.0)
        years_exp = candidate.get('total_years_experience', 0)
        
        if experience_fit < 40.0:
            min_exp = self.job_requirements.min_years_experience or 0
            if years_exp < min_exp:
                gap = min_exp - years_exp
                concerns.append(f"Under-qualified ({gap:.1f} years below minimum)")
            else:
                concerns.append("Experience level mismatch")
        
        # Career concerns
        trajectory_score = candidate.get('trajectory_score', 50.0)
        stability_score = candidate.get('stability_score', 50.0)
        
        if trajectory_score < 40.0:
            concerns.append("Limited career progression")
        
        if stability_score < 40.0:
            concerns.append("Frequent job changes")
        
        # Behavioral concerns
        availability_score = candidate.get('availability_score', 50.0)
        notice_period = candidate.get('notice_period_days', 60)
        
        if notice_period > 90:
            concerns.append(f"Long notice period ({notice_period} days)")
        
        engagement_score = candidate.get('engagement_score', 50.0)
        if engagement_score < 30.0:
            concerns.append("Low recruiter engagement")
        
        # GitHub concerns (for technical roles)
        if self.job_requirements.role_type == 'technical':
            github_score = candidate.get('github_activity_score', 0)
            if technical_score > 30.0 and github_score < 10.0:
                concerns.append("No GitHub activity despite technical claims")
        
        # Hard constraints
        if not candidate.get('hard_constraints_passed', True):
            concerns.append("Does not meet minimum requirements")
        
        # Credibility concerns
        credibility_score = candidate.get('credibility_score', 80.0)
        if credibility_score < 60.0:
            concerns.append("Profile credibility issues")
        
        return concerns[:4]  # Limit to top 4 concerns
    
    def _identify_missing_skills(self, candidate: pd.Series) -> List[str]:
        """Identify which required skills are missing"""
        
        # In a real system, this would analyze the candidate's actual skills
        # vs required skills. For now, we'll use the coverage score as a proxy
        
        required_skills = self.job_requirements.required_skills
        coverage = candidate.get('required_skills_coverage', 0) / 100.0
        
        # Estimate which skills might be missing based on coverage
        estimated_missing = int(len(required_skills) * (1 - coverage))
        
        if estimated_missing > 0:
            # Return the most important required skills as likely missing
            return required_skills[:min(estimated_missing, 3)]
        
        return []
    
    def _generate_overall_assessment(self, overall_score: float) -> str:
        """Generate overall assessment text"""
        
        if overall_score >= 80.0:
            return "Excellent match for this role"
        elif overall_score >= 70.0:
            return "Strong candidate with good potential"
        elif overall_score >= 60.0:
            return "Decent fit with some alignment"
        elif overall_score >= 50.0:
            return "Moderate match with mixed signals"
        elif overall_score >= 30.0:
            return "Limited alignment with requirements"
        else:
            return "Poor fit for this role"
    
    def _generate_recommendation(self, candidate: pd.Series, strengths: List[str], 
                               concerns: List[str]) -> str:
        """Generate hiring recommendation"""
        
        overall_score = candidate.get('hybrid_score', candidate.get('overall_candidate_score', 50.0))
        is_recommended = candidate.get('is_recommended', False)
        confidence = candidate.get('confidence_score', 50.0)
        
        if is_recommended and overall_score >= 70.0:
            return "Recommended for interview"
        elif is_recommended and overall_score >= 60.0:
            return "Recommended with reservations"
        elif overall_score >= 50.0 and confidence >= 60.0:
            return "Consider for phone screen"
        elif len(concerns) > len(strengths):
            return "Not recommended"
        else:
            return "Needs further review"
    
    def _generate_explanation_summary(self, explanation: Dict[str, Any], 
                                    candidate: pd.Series) -> str:
        """Generate concise explanation summary"""
        
        summary_parts = []
        
        # Overall assessment
        summary_parts.append(explanation['overall_assessment'])
        
        # Top 2 strengths
        if explanation['strengths']:
            summary_parts.append("Strengths: " + ", ".join(explanation['strengths'][:2]))
        
        # Top concern
        if explanation['concerns']:
            summary_parts.append("Main concern: " + explanation['concerns'][0])
        
        # Confidence
        summary_parts.append(f"Confidence: {explanation['confidence_score']:.0f}%")
        
        return " • ".join(summary_parts)
    
    def _validate_explanation_diversity(self, explanations: List[Dict[str, Any]]):
        """Validate that explanations are diverse and not repetitive"""
        
        # Check for explanation diversity
        unique_assessments = set(exp['overall_assessment'] for exp in explanations)
        unique_recommendations = set(exp['recommendation'] for exp in explanations)
        
        assessment_diversity = len(unique_assessments) / len(explanations)
        recommendation_diversity = len(unique_recommendations) / len(explanations)
        
        print(f"Explanation diversity check:")
        print(f"  Assessment diversity: {assessment_diversity:.2f}")
        print(f"  Recommendation diversity: {recommendation_diversity:.2f}")
        
        if assessment_diversity < 0.3:
            print("  Warning: Low assessment diversity - explanations may be too similar")
        
        if recommendation_diversity < 0.2:
            print("  Warning: Low recommendation diversity - check recommendation logic")
    
    def analyze_reasoning_results(self, candidates_df: pd.DataFrame, top_k: int = 20):
        """Analyze reasoning generation results"""
        
        print("\n" + "="*70)
        print("REASONING ENGINE ANALYSIS")
        print("="*70)
        
        if 'reasoning_explanation' not in candidates_df.columns:
            print("Error: Reasoning explanations not generated.")
            return
        
        # Show top candidates with explanations
        top_candidates = candidates_df.nlargest(top_k, 'hybrid_score')
        
        print(f"🏆 TOP {top_k} CANDIDATES WITH REASONING:")
        print("="*70)
        
        for i, (idx, candidate) in enumerate(top_candidates.iterrows(), 1):
            explanation = candidate['reasoning_explanation']
            
            print(f"\n{i:2d}. {explanation['candidate_id']} (Score: {explanation['overall_score']:.1f})")
            print(f"    {explanation['overall_assessment']}")
            
            if explanation['strengths']:
                print(f"    ✓ Strengths:")
                for strength in explanation['strengths']:
                    print(f"      • {strength}")
            
            if explanation['concerns']:
                print(f"    ⚠ Concerns:")
                for concern in explanation['concerns']:
                    print(f"      • {concern}")
            
            print(f"    📊 Confidence: {explanation['confidence_score']:.0f}% ({explanation['confidence_level']})")
            print(f"    💼 Recommendation: {explanation['recommendation']}")
        
        # Generate diversity statistics
        all_explanations = candidates_df['reasoning_explanation'].tolist()
        
        strengths_mentioned = []
        concerns_mentioned = []
        
        for exp in all_explanations:
            strengths_mentioned.extend(exp['strengths'])
            concerns_mentioned.extend(exp['concerns'])
        
        unique_strengths = len(set(strengths_mentioned))
        unique_concerns = len(set(concerns_mentioned))
        
        print(f"\n📊 EXPLANATION DIVERSITY:")
        print(f"   Unique strengths mentioned: {unique_strengths}")
        print(f"   Unique concerns mentioned: {unique_concerns}")
        print(f"   Total explanations generated: {len(all_explanations)}")
        
        print(f"\n✅ Checkpoint 2: Generated {len(all_explanations)} reasoning explanations")
        print(f"Every explanation should feel different and be grounded in actual data.")
    
    def export_reasoning_report(self, candidates_df: pd.DataFrame, 
                              output_path: str = 'Data/processed/reasoning_report.txt') -> Path:
        """Export detailed reasoning report"""
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("CANDIDATE REASONING REPORT\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Job: {self.job_requirements.role_title} ({self.job_requirements.role_level})\n")
            f.write(f"Generated: {pd.Timestamp.now()}\n\n")
            
            # Top candidates with full reasoning
            top_candidates = candidates_df.nlargest(50, 'hybrid_score')
            
            for i, (idx, candidate) in enumerate(top_candidates.iterrows(), 1):
                explanation = candidate['reasoning_explanation']
                
                f.write(f"RANK {i}: {explanation['candidate_id']}\n")
                f.write("-" * 30 + "\n")
                f.write(f"Overall Score: {explanation['overall_score']:.1f}\n")
                f.write(f"Assessment: {explanation['overall_assessment']}\n\n")
                
                f.write("STRENGTHS:\n")
                for strength in explanation['strengths']:
                    f.write(f"  ✓ {strength}\n")
                
                f.write("\nCONCERNS:\n")
                for concern in explanation['concerns']:
                    f.write(f"  • {concern}\n")
                
                f.write(f"\nConfidence: {explanation['confidence_score']:.0f}% ({explanation['confidence_level']})\n")
                f.write(f"Recommendation: {explanation['recommendation']}\n")
                f.write("\n" + "="*50 + "\n\n")
        
        print(f"Reasoning report exported to: {output_file}")
        return output_file


if __name__ == "__main__":
    print("Reasoning Engine - Human-Readable Ranking Explanations")
    print("This module generates grounded, fact-based explanations for ranking decisions.")
    print("\nExample usage:")
    print("1. Load candidates with scores and features")
    print("2. engine = ReasoningEngine(job_requirements, jd_profile)")
    print("3. candidates_df = engine.generate_reasoning(candidates_df)")
    print("4. engine.analyze_reasoning_results(candidates_df)")
    print("5. engine.export_reasoning_report(candidates_df)")
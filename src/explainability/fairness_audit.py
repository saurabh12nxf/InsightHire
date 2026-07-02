#!/usr/bin/env python3
"""
Fairness Audit - Analyzes ranking system for bias and fairness
Generates comprehensive fairness reports for algorithmic transparency
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.job_requirements.job_schema import JobRequirements


class FairnessAuditor:
    """
    Analyzes ranking system for potential bias and generates fairness reports.
    
    Fairness dimensions analyzed:
    1. Company type bias
    2. Experience level distribution
    3. Skill diversity
    4. Geographic distribution (if available)
    5. Score distribution patterns
    """
    
    def __init__(self, job_requirements: JobRequirements):
        self.job_requirements = job_requirements
        
        # Fairness thresholds
        self.fairness_thresholds = {
            'company_bias_threshold': 0.15,       # Max acceptable company type skew
            'experience_bias_threshold': 0.20,    # Max acceptable experience skew
            'score_variance_threshold': 25.0,     # Max acceptable score variance
            'top_tier_representation': 0.05       # Min representation in top tier
        }
        
    def conduct_fairness_audit(self, candidates_df: pd.DataFrame, 
                             rankings_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Conduct comprehensive fairness audit
        
        Args:
            candidates_df: DataFrame with all candidates and scores
            rankings_df: Optional DataFrame with final rankings
            
        Returns:
            Dictionary with fairness audit results
        """
        
        print("Conducting fairness audit...")
        
        if rankings_df is None:
            # Use top candidates from the main DataFrame
            rankings_df = candidates_df.nlargest(100, 'hybrid_score')
        
        audit_results = {
            'total_candidates': len(candidates_df),
            'top_candidates': len(rankings_df),
            'audit_date': pd.Timestamp.now().isoformat(),
            'job_role': f"{self.job_requirements.role_title} ({self.job_requirements.role_level})"
        }
        
        # 1. Company Type Analysis
        company_analysis = self._analyze_company_bias(candidates_df, rankings_df)
        audit_results['company_bias'] = company_analysis
        
        # 2. Experience Distribution Analysis
        experience_analysis = self._analyze_experience_distribution(candidates_df, rankings_df)
        audit_results['experience_distribution'] = experience_analysis
        
        # 3. Score Distribution Analysis
        score_analysis = self._analyze_score_distributions(candidates_df, rankings_df)
        audit_results['score_distributions'] = score_analysis
        
        # 4. Skills Diversity Analysis
        skills_analysis = self._analyze_skills_diversity(candidates_df, rankings_df)
        audit_results['skills_diversity'] = skills_analysis
        
        # 5. Overall Fairness Assessment
        fairness_assessment = self._assess_overall_fairness(audit_results)
        audit_results['fairness_assessment'] = fairness_assessment
        
        print(f"Fairness audit completed")
        print(f"Overall fairness score: {fairness_assessment['overall_fairness_score']:.1f}/100")
        
        return audit_results
    
    def _analyze_company_bias(self, all_candidates: pd.DataFrame, 
                            top_candidates: pd.DataFrame) -> Dict[str, Any]:
        """Analyze potential company type bias in rankings"""
        
        # Extract company information from career history or current company
        def extract_company_types(df):
            company_types = []
            
            for idx, candidate in df.iterrows():
                # Use product experience ratio as a proxy for company type
                product_ratio = candidate.get('product_experience_ratio', 0.5)
                
                if product_ratio > 0.7:
                    company_types.append('Product')
                elif product_ratio > 0.3:
                    company_types.append('Mixed')
                else:
                    company_types.append('Consulting')
            
            return company_types
        
        all_company_types = extract_company_types(all_candidates)
        top_company_types = extract_company_types(top_candidates)
        
        # Calculate distributions
        all_dist = Counter(all_company_types)
        top_dist = Counter(top_company_types)
        
        # Normalize to percentages
        total_all = len(all_company_types)
        total_top = len(top_company_types)
        
        all_percentages = {k: (v/total_all)*100 for k, v in all_dist.items()}
        top_percentages = {k: (v/total_top)*100 for k, v in top_dist.items()}
        
        # Calculate bias metrics
        bias_metrics = {}
        for company_type in set(all_company_types):
            all_pct = all_percentages.get(company_type, 0)
            top_pct = top_percentages.get(company_type, 0)
            
            # Representation ratio (>1 means over-represented in top)
            ratio = (top_pct / all_pct) if all_pct > 0 else 0
            bias_metrics[company_type] = {
                'all_candidates_pct': all_pct,
                'top_candidates_pct': top_pct,
                'representation_ratio': ratio,
                'bias_score': abs(ratio - 1.0)  # 0 = no bias, >0 = bias
            }
        
        # Overall company bias assessment
        avg_bias = np.mean([metrics['bias_score'] for metrics in bias_metrics.values()])
        
        return {
            'company_distributions': {
                'all_candidates': all_percentages,
                'top_candidates': top_percentages
            },
            'bias_metrics': bias_metrics,
            'average_bias_score': avg_bias,
            'bias_assessment': 'Low' if avg_bias < self.fairness_thresholds['company_bias_threshold'] else 'High',
            'concerns': self._identify_company_bias_concerns(bias_metrics)
        }
    
    def _analyze_experience_distribution(self, all_candidates: pd.DataFrame,
                                       top_candidates: pd.DataFrame) -> Dict[str, Any]:
        """Analyze experience level distribution fairness"""
        
        def categorize_experience(years):
            if years < 2:
                return 'Junior (0-2 years)'
            elif years < 5:
                return 'Mid-level (2-5 years)'
            elif years < 10:
                return 'Senior (5-10 years)'
            else:
                return 'Principal (10+ years)'
        
        # Categorize experience levels
        all_exp_categories = [categorize_experience(exp) for exp in all_candidates['total_years_experience']]
        top_exp_categories = [categorize_experience(exp) for exp in top_candidates['total_years_experience']]
        
        # Calculate distributions
        all_exp_dist = Counter(all_exp_categories)
        top_exp_dist = Counter(top_exp_categories)
        
        total_all = len(all_exp_categories)
        total_top = len(top_exp_categories)
        
        all_exp_pct = {k: (v/total_all)*100 for k, v in all_exp_dist.items()}
        top_exp_pct = {k: (v/total_top)*100 for k, v in top_exp_dist.items()}
        
        # Calculate representation ratios
        representation_ratios = {}
        for category in set(all_exp_categories):
            all_pct = all_exp_pct.get(category, 0)
            top_pct = top_exp_pct.get(category, 0)
            ratio = (top_pct / all_pct) if all_pct > 0 else 0
            representation_ratios[category] = ratio
        
        # Experience bias assessment
        ratios_list = list(representation_ratios.values())
        exp_bias_score = np.std(ratios_list) if ratios_list else 0  # Higher std = more bias
        
        return {
            'experience_distributions': {
                'all_candidates': all_exp_pct,
                'top_candidates': top_exp_pct
            },
            'representation_ratios': representation_ratios,
            'bias_score': exp_bias_score,
            'bias_assessment': 'Fair' if exp_bias_score < self.fairness_thresholds['experience_bias_threshold'] else 'Biased',
            'job_requirements_alignment': self._check_experience_alignment()
        }
    
    def _analyze_score_distributions(self, all_candidates: pd.DataFrame,
                                   top_candidates: pd.DataFrame) -> Dict[str, Any]:
        """Analyze score distribution patterns for fairness"""
        
        score_columns = ['technical_fit_score', 'career_score', 'behavioral_score', 
                        'trajectory_score', 'credibility_score', 'hybrid_score']
        
        distribution_analysis = {}
        
        for score_col in score_columns:
            if score_col in all_candidates.columns:
                all_scores = all_candidates[score_col]
                top_scores = top_candidates[score_col] if score_col in top_candidates.columns else []
                
                distribution_analysis[score_col] = {
                    'all_candidates': {
                        'mean': all_scores.mean(),
                        'median': all_scores.median(),
                        'std': all_scores.std(),
                        'min': all_scores.min(),
                        'max': all_scores.max()
                    }
                }
                
                if len(top_scores) > 0:
                    distribution_analysis[score_col]['top_candidates'] = {
                        'mean': top_scores.mean(),
                        'median': top_scores.median(),
                        'std': top_scores.std(),
                        'min': top_scores.min(),
                        'max': top_scores.max()
                    }
        
        # Assess score variance fairness
        variance_issues = []
        for score_col, stats in distribution_analysis.items():
            if 'all_candidates' in stats:
                variance = stats['all_candidates']['std']
                if variance > self.fairness_thresholds['score_variance_threshold']:
                    variance_issues.append(f"High variance in {score_col}: {variance:.1f}")
        
        return {
            'score_distributions': distribution_analysis,
            'variance_issues': variance_issues,
            'distribution_fairness': 'Good' if len(variance_issues) == 0 else 'Concerning'
        }
    
    def _analyze_skills_diversity(self, all_candidates: pd.DataFrame,
                                top_candidates: pd.DataFrame) -> Dict[str, Any]:
        """Analyze skills diversity in top candidates"""
        
        # Skills coverage analysis
        all_coverage = all_candidates['required_skills_coverage'].describe()
        top_coverage = top_candidates['required_skills_coverage'].describe()
        
        # Skills count analysis
        all_skills_count = all_candidates['total_relevant_skills'].describe()
        top_skills_count = top_candidates['total_relevant_skills'].describe()
        
        # Diversity assessment
        coverage_bias = abs(top_coverage['mean'] - all_coverage['mean'])
        skills_bias = abs(top_skills_count['mean'] - all_skills_count['mean'])
        
        return {
            'skills_coverage': {
                'all_candidates': all_coverage.to_dict(),
                'top_candidates': top_coverage.to_dict(),
                'bias_score': coverage_bias
            },
            'skills_count': {
                'all_candidates': all_skills_count.to_dict(),
                'top_candidates': top_skills_count.to_dict(),
                'bias_score': skills_bias
            },
            'diversity_assessment': self._assess_skills_diversity(coverage_bias, skills_bias)
        }
    
    def _assess_overall_fairness(self, audit_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall fairness based on all analysis components"""
        
        fairness_scores = []
        issues = []
        
        # Company bias assessment
        company_bias = audit_results['company_bias']['average_bias_score']
        if company_bias < self.fairness_thresholds['company_bias_threshold']:
            fairness_scores.append(90)
        else:
            fairness_scores.append(60)
            issues.append(f"Company type bias detected (score: {company_bias:.2f})")
        
        # Experience distribution assessment
        exp_bias = audit_results['experience_distribution']['bias_score']
        if exp_bias < self.fairness_thresholds['experience_bias_threshold']:
            fairness_scores.append(85)
        else:
            fairness_scores.append(50)
            issues.append(f"Experience level bias detected (score: {exp_bias:.2f})")
        
        # Score distribution assessment
        variance_issues = audit_results['score_distributions']['variance_issues']
        if len(variance_issues) == 0:
            fairness_scores.append(80)
        else:
            fairness_scores.append(40)
            issues.extend(variance_issues)
        
        # Skills diversity assessment
        skills_div = audit_results['skills_diversity']['diversity_assessment']
        if skills_div == 'Good':
            fairness_scores.append(85)
        else:
            fairness_scores.append(55)
            issues.append("Skills diversity concerns")
        
        overall_fairness_score = np.mean(fairness_scores)
        
        return {
            'overall_fairness_score': overall_fairness_score,
            'fairness_level': self._categorize_fairness(overall_fairness_score),
            'component_scores': {
                'company_bias': fairness_scores[0] if len(fairness_scores) > 0 else 50,
                'experience_distribution': fairness_scores[1] if len(fairness_scores) > 1 else 50,
                'score_distributions': fairness_scores[2] if len(fairness_scores) > 2 else 50,
                'skills_diversity': fairness_scores[3] if len(fairness_scores) > 3 else 50
            },
            'identified_issues': issues,
            'recommendations': self._generate_fairness_recommendations(issues)
        }
    
    def _identify_company_bias_concerns(self, bias_metrics: Dict[str, Any]) -> List[str]:
        """Identify specific company bias concerns"""
        
        concerns = []
        
        for company_type, metrics in bias_metrics.items():
            ratio = metrics['representation_ratio']
            
            if ratio > 2.0:
                concerns.append(f"{company_type} companies over-represented by {ratio:.1f}x")
            elif ratio < 0.5 and ratio > 0:
                concerns.append(f"{company_type} companies under-represented by {1/ratio:.1f}x")
        
        return concerns
    
    def _check_experience_alignment(self) -> str:
        """Check if experience requirements are reasonable"""
        
        min_exp = self.job_requirements.min_years_experience or 0
        max_exp = self.job_requirements.max_years_experience or 20
        
        if min_exp > 10:
            return "High experience requirements may exclude diverse candidates"
        elif max_exp < 5:
            return "Low experience cap may exclude senior candidates"
        else:
            return "Experience requirements appear reasonable"
    
    def _assess_skills_diversity(self, coverage_bias: float, skills_bias: float) -> str:
        """Assess skills diversity fairness"""
        
        if coverage_bias < 10 and skills_bias < 2:
            return "Good"
        elif coverage_bias < 20 and skills_bias < 5:
            return "Acceptable"
        else:
            return "Concerning"
    
    def _categorize_fairness(self, fairness_score: float) -> str:
        """Categorize overall fairness level"""
        
        if fairness_score >= 80:
            return "Excellent"
        elif fairness_score >= 70:
            return "Good"
        elif fairness_score >= 60:
            return "Acceptable"
        elif fairness_score >= 50:
            return "Concerning"
        else:
            return "Poor"
    
    def _generate_fairness_recommendations(self, issues: List[str]) -> List[str]:
        """Generate recommendations to improve fairness"""
        
        recommendations = []
        
        if any('company' in issue.lower() for issue in issues):
            recommendations.append("Consider adjusting company type weighting to reduce bias")
            recommendations.append("Review if product vs consulting preference is justified")
        
        if any('experience' in issue.lower() for issue in issues):
            recommendations.append("Broaden experience range requirements")
            recommendations.append("Consider experience-adjusted scoring")
        
        if any('variance' in issue.lower() for issue in issues):
            recommendations.append("Review scoring algorithms for potential bias")
            recommendations.append("Consider score normalization techniques")
        
        if any('diversity' in issue.lower() for issue in issues):
            recommendations.append("Ensure diverse skill requirements")
            recommendations.append("Avoid over-weighting specific skills")
        
        if not recommendations:
            recommendations.append("Fairness appears good - continue monitoring")
        
        return recommendations
    
    def generate_fairness_report(self, audit_results: Dict[str, Any],
                               output_path: str = 'Data/processed/fairness_report.md') -> Path:
        """Generate comprehensive fairness report in Markdown format"""
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Fairness Audit Report\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write(f"**Job Role**: {audit_results['job_role']}  \n")
            f.write(f"**Audit Date**: {audit_results['audit_date']}  \n")
            f.write(f"**Total Candidates Analyzed**: {audit_results['total_candidates']:,}  \n")
            f.write(f"**Top Candidates Reviewed**: {audit_results['top_candidates']:,}  \n\n")
            
            fairness = audit_results['fairness_assessment']
            f.write(f"**Overall Fairness Score**: {fairness['overall_fairness_score']:.1f}/100 ({fairness['fairness_level']})  \n\n")
            
            # Company Bias Analysis
            f.write("## Company Type Analysis\n\n")
            company_bias = audit_results['company_bias']
            
            f.write("### Company Distribution\n\n")
            f.write("| Company Type | All Candidates | Top Candidates | Representation Ratio |\n")
            f.write("|--------------|----------------|----------------|---------------------|\n")
            
            for comp_type, metrics in company_bias['bias_metrics'].items():
                all_pct = metrics['all_candidates_pct']
                top_pct = metrics['top_candidates_pct']
                ratio = metrics['representation_ratio']
                f.write(f"| {comp_type} | {all_pct:.1f}% | {top_pct:.1f}% | {ratio:.2f}x |\n")
            
            f.write(f"\n**Assessment**: {company_bias['bias_assessment']} bias detected  \n")
            f.write(f"**Average Bias Score**: {company_bias['average_bias_score']:.3f}  \n\n")
            
            if company_bias['concerns']:
                f.write("**Concerns**:  \n")
                for concern in company_bias['concerns']:
                    f.write(f"- {concern}  \n")
                f.write("\n")
            
            # Experience Distribution
            f.write("## Experience Level Analysis\n\n")
            exp_analysis = audit_results['experience_distribution']
            
            f.write("### Experience Distribution\n\n")
            f.write("| Experience Level | All Candidates | Top Candidates | Representation Ratio |\n")
            f.write("|------------------|----------------|----------------|---------------------|\n")
            
            all_exp = exp_analysis['experience_distributions']['all_candidates']
            top_exp = exp_analysis['experience_distributions']['top_candidates']
            ratios = exp_analysis['representation_ratios']
            
            for exp_level in all_exp.keys():
                all_pct = all_exp[exp_level]
                top_pct = top_exp.get(exp_level, 0)
                ratio = ratios.get(exp_level, 0)
                f.write(f"| {exp_level} | {all_pct:.1f}% | {top_pct:.1f}% | {ratio:.2f}x |\n")
            
            f.write(f"\n**Assessment**: {exp_analysis['bias_assessment']}  \n")
            f.write(f"**Job Requirements Alignment**: {exp_analysis['job_requirements_alignment']}  \n\n")
            
            # Score Distributions
            f.write("## Score Distribution Analysis\n\n")
            score_analysis = audit_results['score_distributions']
            
            f.write("### Score Statistics\n\n")
            f.write("| Score Type | All Candidates (Mean) | Top Candidates (Mean) | Std Deviation |\n")
            f.write("|------------|----------------------|----------------------|---------------|\n")
            
            for score_type, stats in score_analysis['score_distributions'].items():
                all_mean = stats['all_candidates']['mean']
                all_std = stats['all_candidates']['std']
                top_mean = stats.get('top_candidates', {}).get('mean', 'N/A')
                
                top_mean_str = f"{top_mean:.1f}" if top_mean != 'N/A' else 'N/A'
                f.write(f"| {score_type} | {all_mean:.1f} | {top_mean_str} | {all_std:.1f} |\n")
            
            if score_analysis['variance_issues']:
                f.write(f"\n**Variance Issues**:  \n")
                for issue in score_analysis['variance_issues']:
                    f.write(f"- {issue}  \n")
            
            f.write(f"\n**Distribution Fairness**: {score_analysis['distribution_fairness']}  \n\n")
            
            # Skills Diversity
            f.write("## Skills Diversity Analysis\n\n")
            skills_analysis = audit_results['skills_diversity']
            
            coverage_stats = skills_analysis['skills_coverage']
            f.write(f"**Skills Coverage Bias Score**: {coverage_stats['bias_score']:.1f}  \n")
            
            count_stats = skills_analysis['skills_count']
            f.write(f"**Skills Count Bias Score**: {count_stats['bias_score']:.1f}  \n")
            f.write(f"**Diversity Assessment**: {skills_analysis['diversity_assessment']}  \n\n")
            
            # Overall Assessment
            f.write("## Overall Fairness Assessment\n\n")
            
            f.write("### Component Scores\n\n")
            component_scores = fairness['component_scores']
            for component, score in component_scores.items():
                f.write(f"- **{component.replace('_', ' ').title()}**: {score:.1f}/100  \n")
            
            f.write(f"\n### Identified Issues\n\n")
            if fairness['identified_issues']:
                for issue in fairness['identified_issues']:
                    f.write(f"- {issue}  \n")
            else:
                f.write("No significant fairness issues identified.  \n")
            
            f.write(f"\n### Recommendations\n\n")
            for recommendation in fairness['recommendations']:
                f.write(f"- {recommendation}  \n")
            
            # Conclusion
            f.write(f"\n## Conclusion\n\n")
            f.write(f"The ranking system demonstrates **{fairness['fairness_level'].lower()}** fairness ")
            f.write(f"with an overall score of {fairness['overall_fairness_score']:.1f}/100. ")
            
            if fairness['fairness_level'] in ['Excellent', 'Good']:
                f.write("The system appears to be making fair decisions without obvious bias patterns.")
            elif fairness['fairness_level'] == 'Acceptable':
                f.write("The system shows acceptable fairness with minor areas for improvement.")
            else:
                f.write("The system shows concerning bias patterns that should be addressed.")
            
            f.write(f"\n\n*Report generated on {audit_results['audit_date']}*\n")
        
        print(f"Fairness report saved to: {output_file}")
        return output_file


if __name__ == "__main__":
    print("Fairness Auditor - Algorithmic Bias Detection")
    print("This module analyzes ranking systems for fairness and bias.")
    print("\nExample usage:")
    print("1. Load candidates with rankings and scores")
    print("2. auditor = FairnessAuditor(job_requirements)")
    print("3. audit_results = auditor.conduct_fairness_audit(candidates_df)")
    print("4. auditor.generate_fairness_report(audit_results)")
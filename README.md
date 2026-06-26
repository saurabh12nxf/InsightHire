# TalentLens AI

An intelligent candidate discovery and ranking system for AI/ML roles.

## Project Structure

```
talentlens-ai/
├── Data/
│   ├── raw/                    # Original dataset files
│   └── processed/              # Processed data files
├── notebooks/                  # Jupyter notebooks for analysis
├── src/
│   ├── data_loader/           # Data loading utilities
│   ├── feature_engineering/   # Feature extraction and processing
│   ├── embeddings/            # Embedding generation
│   ├── rankings/              # Candidate ranking algorithms
│   ├── reasoning/             # Reasoning and explanation generation
│   └── validation/            # Validation utilities
├── outputs/                   # Generated outputs and results
├── models/                    # Trained models and weights
└── README.md
```

## Phase 1 - Project Setup ✅

**Goal**: Make sure the dataset loads.

**Status**: Complete

### Checkpoint Test
```bash
python load_candidates.py
```

**Expected Output**:
```
Loaded 100000 candidates
✅ Phase 1 complete
```

### What was accomplished:
- ✅ Created complete project structure
- ✅ Implemented candidate data loader
- ✅ Validated dataset structure (100,000 candidates)
- ✅ Verified data integrity and format
- ✅ Set up modular architecture for future phases

## Phase 2 - Dataset Understanding ✅

**Goal**: Understand candidate data deeply through comprehensive EDA.

**Status**: Complete

### What was accomplished:
- ✅ Analyzed 100,000 candidate profiles comprehensively
- ✅ Identified key skill distributions and patterns
- ✅ Analyzed experience levels and role distributions  
- ✅ Discovered company representation insights (0.03% from FAANG)
- ✅ Evaluated behavioral signals for ranking strategy
- ✅ Generated comprehensive dataset report with strategic insights

### Key Findings:
- **GitHub Activity**: Only 35.4% have accounts, 3.9% highly active (key differentiator)
- **Response Rates**: 13.2% are high responders (engagement signal)
- **Quick Availability**: 13.8% available within 30 days (business value)
- **Experience Distribution**: 7.2 years average, mostly mid-level (4-8 years)
- **Role Diversity**: Mixed backgrounds, few direct AI/ML titles

### Deliverables:
- 📊 `notebooks/eda.ipynb` - Complete exploratory analysis
- 📋 `dataset_report.md` - Comprehensive findings and strategic insights

## Getting Started

1. Ensure Python 3.8+ is installed
2. Run the Phase 1 checkpoint: `python load_candidates.py`
3. Verify you see "Loaded 100000 candidates" message
4. Review EDA findings in `dataset_report.md`

## Dataset Overview

The dataset contains 100,000 candidate profiles with:
- **Profile Information**: Name, headline, summary, location, experience
- **Career History**: Previous roles, companies, durations
- **Education**: Degrees, institutions, grades
- **Skills**: Technical skills with proficiency levels
- **Redrob Signals**: Platform engagement metrics

**Sample Candidate Fields**:
- candidate_id (CAND_XXXXXXX format)
- profile (basic info and current role)
- career_history (up to 10 previous positions)
- education (academic background)
- skills (technical competencies)
- redrob_signals (engagement metrics)
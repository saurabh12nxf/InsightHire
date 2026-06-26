# Dataset Analysis Report - InsightHire

## Executive Summary
This report presents comprehensive findings from exploratory data analysis (EDA) of the candidate dataset for the AI/ML role ranking challenge. The analysis reveals key insights about candidate distribution, skills, experience, and behavioral signals that will inform our ranking algorithm design.

## Dataset Overview

### Basic Statistics
- **Total Candidates**: 100,000
- **Data Quality**: Complete dataset with consistent structure
- **Coverage**: Diverse candidates across experience levels, companies, and skill sets
- **Time Period**: Current snapshot of candidate profiles with behavioral signals

## Key Findings

### 1. Skill Distribution Analysis

#### Top 20 Skills (Most Common)
1. **HTML**: 12,246 candidates (12.2%)
2. **Databricks**: 12,244 candidates (12.2%)
3. **Redux**: 12,222 candidates (12.2%)
4. **Terraform**: 12,187 candidates (12.2%)
5. **Angular**: 12,173 candidates (12.2%)
6. **Figma**: 12,157 candidates (12.2%)
7. **Salesforce CRM**: 12,157 candidates (12.2%)
8. **Vue.js**: 12,142 candidates (12.1%)
9. **Sales**: 12,138 candidates (12.1%)
10. **Accounting**: 12,136 candidates (12.1%)
11. **Agile**: 12,135 candidates (12.1%)
12. **Kafka**: 12,114 candidates (12.1%)
13. **Excel**: 12,109 candidates (12.1%)
14. **BigQuery**: 12,108 candidates (12.1%)
15. **CI/CD**: 12,108 candidates (12.1%)
16. **Project Management**: 12,106 candidates (12.1%)
17. **Airflow**: 12,105 candidates (12.1%)
18. **AWS**: 12,104 candidates (12.1%)
19. **Flask**: 12,104 candidates (12.1%)
20. **Scrum**: 12,083 candidates (12.1%)

#### AI/ML Skill Insights
- **Even Distribution**: Most skills appear in ~12% of candidates, indicating synthetic data generation
- **Diverse Skill Set**: Mix of frontend (HTML, Angular, Vue.js), backend (Flask), data (Databricks, BigQuery), and cloud (AWS) technologies
- **Modern Stack**: Current technologies like Terraform, CI/CD, and cloud platforms well-represented

### 2. Experience Distribution

#### Experience Statistics
- **Average Experience**: 7.2 years
- **Median Experience**: 6.8 years
- **Experience Range**: 1.0 - 16.9 years
- **Distribution**: Slightly right-skewed, most candidates in mid-level range

#### Experience Level Breakdown
- **1-3 years (Junior)**: ~25% of candidates
- **4-8 years (Mid-level)**: ~45% of candidates
- **9-12 years (Senior)**: ~20% of candidates  
- **13+ years (Principal/Staff)**: ~10% of candidates

### 3. Current Roles Analysis

#### Top 15 Current Titles
1. **Business Analyst**: 5,833 candidates (5.8%)
2. **HR Manager**: 5,830 candidates (5.8%)
3. **Mechanical Engineer**: 5,791 candidates (5.8%)
4. **Accountant**: 5,764 candidates (5.8%)
5. **Project Manager**: 5,754 candidates (5.8%)
6. **Customer Support**: 5,750 candidates (5.8%)
7. **Operations Manager**: 5,744 candidates (5.7%)
8. **Content Writer**: 5,727 candidates (5.7%)
9. **Sales Executive**: 5,713 candidates (5.7%)
10. **Civil Engineer**: 5,702 candidates (5.7%)
11. **Graphic Designer**: 5,689 candidates (5.7%)
12. **Marketing Manager**: 5,524 candidates (5.5%)
13. **Software Engineer**: 3,450 candidates (3.5%)
14. **Full Stack Developer**: 2,873 candidates (2.9%)
15. **Cloud Engineer**: 2,836 candidates (2.8%)

#### Role Relevance for AI/ML Positions
- **Direct AI/ML Roles**: <10% of candidates have directly relevant titles
- **Technical Roles**: Software Engineer, Full Stack Developer, Cloud Engineer represent minority
- **Career Transition Candidates**: Many from non-technical backgrounds (Business Analyst, HR, etc.)

### 4. Company Distribution Analysis

#### Top 15 Current Companies
1. **Infosys**: 7,590 candidates (7.6%)
2. **Wayne Enterprises**: 7,571 candidates (7.6%) *[Fictional]*
3. **Wipro**: 7,566 candidates (7.6%)
4. **Initech**: 7,528 candidates (7.5%) *[Fictional]*
5. **Pied Piper**: 7,500 candidates (7.5%) *[Fictional]*
6. **Globex Inc**: 7,492 candidates (7.5%) *[Fictional]*
7. **Acme Corp**: 7,490 candidates (7.5%) *[Fictional]*
8. **Dunder Mifflin**: 7,467 candidates (7.5%) *[Fictional]*
9. **TCS**: 7,451 candidates (7.5%)
10. **Hooli**: 7,378 candidates (7.4%) *[Fictional]*
11. **Stark Industries**: 7,323 candidates (7.3%) *[Fictional]*
12. **Swiggy**: 1,288 candidates (1.3%)
13. **Accenture**: 1,274 candidates (1.3%)
14. **Capgemini**: 1,265 candidates (1.3%)
15. **CRED**: 1,257 candidates (1.3%)

#### Tech Giants Analysis
- **Google**: 6 candidates (0.006%)
- **Microsoft**: 5 candidates (0.005%)
- **Amazon**: 5 candidates (0.005%)
- **Apple**: 2 candidates (0.002%)
- **Meta**: 7 candidates (0.007%)
- **Netflix**: 6 candidates (0.006%)
- **Total from FAANG**: 31 candidates (0.03%)

**Key Insight**: Virtually no candidates from top-tier tech companies, indicating need to identify quality through other signals.

### 5. Behavioral Signals Analysis

#### GitHub Activity
- **Candidates with GitHub**: 35,363 (35.4%)
- **Candidates without GitHub**: 64,637 (64.6%)
- **Average GitHub Score**: 29.0/100
- **High GitHub Activity (>50)**: 3,917 candidates (3.9%)

#### Recruiter Response Rates
- **Average Response Rate**: 43.7%
- **High Responders (>70%)**: 13,240 candidates (13.2%)
- **Highest Response Rate**: 95%

#### Availability Analysis
- **Average Notice Period**: 87.4 days
- **Quick Joiners (≤30 days)**: 13,809 candidates (13.8%)
- **Immediate Availability (0 days)**: Available
- **Maximum Notice**: 150 days

#### Profile Quality
- **Average Profile Completeness**: 56.8%
- **High Quality Profiles (>80%)**: 10,613 candidates (10.6%)

## Strategic Insights for Ranking Algorithm

### High-Value Differentiators
1. **GitHub Activity** (35.4% have accounts, 3.9% highly active)
   - Strong technical signal
   - Rare enough to be valuable differentiator
   
2. **Recruiter Response Rate** (13.2% are high responders)
   - Indicates active job seeking
   - Proxy for hiring likelihood

3. **Quick Availability** (13.8% can join ≤30 days)
   - Business impact for urgent hiring needs
   - Competitive advantage signal

4. **Profile Completeness** (10.6% have >80% completion)
   - Attention to detail indicator
   - Professional commitment signal

### Low-Value Differentiators
1. **Company Prestige**: FAANG representation negligible (0.03%)
2. **Direct AI/ML Titles**: Small percentage of candidates
3. **Skill Distribution**: Too evenly distributed to differentiate

### Recommended Ranking Strategy
1. **Technical Skills Assessment**: Focus on skill proficiency levels rather than presence
2. **GitHub Activity Bonus**: High weight for candidates with active GitHub profiles
3. **Behavioral Engagement**: Factor in response rates and profile quality
4. **Experience Sweet Spot**: Target 4-8 years experience range (45% of candidates)
5. **Availability Consideration**: Bonus for shorter notice periods

## Data Quality Assessment

### Strengths
- Complete dataset with 100,000 candidates
- Consistent data structure across all profiles
- Rich behavioral signals beyond traditional resume data
- Diverse candidate pool across multiple dimensions

### Limitations
- Synthetic data patterns (even skill distribution, fictional companies)
- Limited representation from top-tier tech companies
- May not reflect real-world candidate quality distribution

## Conclusions

The dataset provides a solid foundation for building a candidate ranking system. The key to success will be leveraging the rare signals (GitHub activity, high engagement, quick availability) to differentiate candidates, while using traditional signals (skills, experience, titles) for baseline qualification.

The analysis reveals that quality identification must rely on engagement and technical activity signals rather than company prestige or direct role relevance, making this a realistic simulation of modern hiring challenges.

---

**Report Generated**: Phase 2 EDA Completion  
**Next Phase**: Feature Engineering & Ranking Algorithm Development
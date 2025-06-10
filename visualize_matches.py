import re
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import seaborn as sns
from collections import Counter
import os

def create_resume_match_dashboard(matches, output_dir='dashboard'):
    """Create a dashboard of actionable visualizations for job seekers"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a DataFrame from matches
    df = pd.DataFrame(matches)
    
    # 1. Match Strength Gauge
    plt.figure(figsize=(8, 2))
    score = df['Score'].mean()
    plt.barh([0], [score], color='#2563eb')
    plt.barh([0], [100-score], left=score, color='#f3f4f6')
    plt.xlim(0, 100)
    plt.xticks([])
    plt.yticks([])
    plt.title(f'Average Match Strength: {score:.1f}%', pad=20)
    plt.savefig(f'{output_dir}/match_strength.png', bbox_inches='tight')
    plt.close()
    
    # 2. Top Skills to Improve
    # Extract skills from descriptions
    skills = []
    for desc in df['Description']:
        skills.extend(re.findall(r'\b(?:Python|Java|JavaScript|SQL|React|Node|AWS|Azure|DevOps|Agile)\b', desc, re.IGNORECASE))
    skill_counts = Counter(skills)
    
    plt.figure(figsize=(10, 6))
    plt.bar(skill_counts.keys(), skill_counts.values(), color='#10b981')
    plt.title('Top Skills to Improve Your Match')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/skills_to_improve.png')
    plt.close()
    
    # 3. Company Growth Potential
    plt.figure(figsize=(10, 6))
    df['Company Size'] = df['Company'].apply(lambda x: random.choice(['Startup', 'Medium', 'Large']))
    plt.bar(df['Company Size'].value_counts().index, df['Company Size'].value_counts(), 
            color=['#f43f5e', '#fbbf24', '#22c55e'])
    plt.title('Company Size Distribution')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/company_size.png')
    plt.close()
    
    # 4. Location Analysis
    plt.figure(figsize=(10, 6))
    df['Country'] = df['Location'].apply(lambda x: x.split(',')[-1].strip())
    plt.barh(df['Country'].value_counts().index, df['Country'].value_counts().values, 
             color='#10b981')
    plt.title('Job Locations')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/job_locations.png')
    plt.close()
    
    # 5. Salary Range (if available)
    plt.figure(figsize=(10, 6))
    # Create dummy salary ranges for demonstration
    salary_data = [70000, 80000, 90000, 100000, 110000]
    plt.boxplot(salary_data)
    plt.title('Salary Range Distribution')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/salary_ranges.png')
    plt.close()
    
    # 6. Application Timeline
    plt.figure(figsize=(10, 6))
    # Create dummy timeline data
    timeline = pd.DataFrame({
        'Stage': ['Application', 'Screening', 'Interview', 'Offer'],
        'Days': [5, 7, 10, 3]
    })
    plt.bar(timeline['Stage'], timeline['Days'], color='#3b82f6')
    plt.title('Typical Application Timeline')
    plt.ylabel('Average Days')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/application_timeline.png')
    plt.close()
    
    # 7. Skills Gap Analysis
    plt.figure(figsize=(10, 6))
    # Create dummy skills gap data
    skills = ['Python', 'SQL', 'AWS', 'Docker', 'Git', 'Agile']
    current = [80, 70, 50, 30, 40, 60]
    required = [100, 90, 80, 70, 70, 80]
    
    plt.bar(skills, current, color='#f43f5e', alpha=0.7, label='Current')
    plt.bar(skills, required, bottom=current, color='#22c55e', alpha=0.7, label='Required')
    plt.title('Skills Gap Analysis')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/skills_gap.png')
    plt.close()
    
    print(f"Dashboard visualizations created in '{output_dir}' directory")

if __name__ == "__main__":
    # Example usage
    matches = [
        {'Score': 85, 'Company': 'Google', 'Location': 'Mountain View, USA', 'Description': 'Python, SQL, AWS'},
        {'Score': 78, 'Company': 'Facebook', 'Location': 'Menlo Park, USA', 'Description': 'Python, React, Docker'},
        {'Score': 82, 'Company': 'Amazon', 'Location': 'Seattle, USA', 'Description': 'AWS, Python, DevOps'},
        {'Score': 75, 'Company': 'Microsoft', 'Location': 'Redmond, USA', 'Description': 'C#, Azure, SQL'}
    ]
    
    create_resume_match_dashboard(matches)
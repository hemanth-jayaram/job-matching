#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ultra-Simple Resume-Job Matcher
No external dependencies - uses only standard library
"""

import os
import re
import csv
import math
import argparse
import PyPDF2
from collections import defaultdict, Counter
from typing import List, Dict, Set, Tuple, Any

# Common stopwords to filter out
STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
    "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
    'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
    'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
    'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
    'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should',
    "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn',
    "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
    "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",
    'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't",
    'wouldn', "wouldn't"
}

# Common technical skills to boost in matching
TECH_SKILLS = {
    'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'swift', 'kotlin', 'go', 'rust',
    'html', 'css', 'react', 'angular', 'vue', 'node', 'django', 'flask', 'spring', 'laravel',
    'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle', 'sqlite',
    'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'ansible', 'jenkins', 'git',
    'machine learning', 'ai', 'data science', 'deep learning', 'nlp', 'computer vision',
    'agile', 'scrum', 'devops', 'ci/cd', 'tcp/ip', 'http', 'rest', 'graphql', 'soap', 'oop',
    'data structures', 'algorithms', 'linux', 'unix', 'bash', 'powershell', 'excel', 'tableau', 'powerbi'
}

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from a PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ' '.join([page.extract_text() or '' for page in reader.pages])
            return text.strip()
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return ""

def extract_text_from_txt(file_path: str) -> str:
    """Extract text from a plain text file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            return file.read().strip()
    except Exception as e:
        print(f"Error reading text file: {e}")
        return ""

def extract_skills(text: str) -> Set[str]:
    """Extract skills from text by looking for common technical terms."""
    if not text:
        return set()
    
    # Convert to lowercase and find all words
    words = re.findall(r'\b[a-z0-9+#]+\b', text.lower())
    
    # Find n-grams (up to 3 words) that match our skills list
    skills_found = set()
    max_gram = 3
    
    for n in range(1, max_gram + 1):
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            if ngram in TECH_SKILLS:
                skills_found.add(ngram)
    
    return skills_found

def preprocess_text(text: str, is_resume: bool = False) -> Dict[str, float]:
    """
    Preprocess text by tokenizing, lowercasing, and removing non-alphabetic characters.
    Returns a dictionary of terms with their weights.
    """
    if not text:
        return {}
    
    # Convert to lowercase and remove non-alphabetic characters
    words = re.findall(r'\b[a-z0-9]+\b', text.lower())
    
    # Remove stopwords and short words
    words = [w for w in words if w not in STOPWORDS and len(w) > 2]
    
    # Count term frequencies
    term_freq = Counter(words)
    
    # If this is a resume, boost technical terms
    if is_resume:
        # Extract skills and give them higher weight
        skills = extract_skills(text)
        for skill in skills:
            term_freq[skill] += 5  # Boost skills
    
    # Convert to weights (log scale to reduce impact of very frequent terms)
    max_freq = max(term_freq.values()) if term_freq else 1
    weights = {term: 1 + math.log(1 + freq) for term, freq in term_freq.items()}
    
    return weights

def calculate_match_score(
    resume_terms: Dict[str, float], 
    job: Dict[str, str], 
    resume_skills: Set[str]
) -> float:
    """
    Calculate a match score between resume and job posting.
    Returns a score between 0 and 100.
    """
    if not job.get('description'):
        return 0.0
    
    # Extract job components
    job_title = job.get('title', '').lower()
    job_desc = job.get('description', '').lower()
    job_skills = extract_skills(job_title + ' ' + job_desc)
    
    # Calculate base scores for different components
    title_score = calculate_text_similarity(resume_terms, job_title, is_title=True)
    desc_score = calculate_text_similarity(resume_terms, job_desc)
    skill_score = calculate_skill_similarity(resume_skills, job_skills)
    
    # Weighted average of scores (adjust weights as needed)
    weights = {
        'title': 0.5,    # Title match is very important
        'skills': 0.3,   # Skills match is also very important
        'desc': 0.2      # General description match is less important
    }
    
    # Calculate weighted score (0-1 range)
    weighted_score = (
        weights['title'] * title_score +
        weights['skills'] * skill_score +
        weights['desc'] * desc_score
    )
    
    # Apply a non-linear scaling to make higher scores more distinct
    scaled_score = 1 - (1 - weighted_score) ** 2
    
    # Convert to percentage and ensure it's within 0-100 range
    final_score = min(max(scaled_score, 0.0), 1.0) * 100
    
    # Add a small random factor to break ties (0-2%)
    import random
    final_score += random.uniform(0, 2)
    
    return min(final_score, 100.0)  # Cap at 100%

def calculate_text_similarity(resume_terms: Dict[str, float], text: str, is_title: bool = False) -> float:
    """Calculate similarity between resume terms and text."""
    if not text or not resume_terms:
        return 0.0
    
    # Preprocess the text
    text_terms = preprocess_text(text)
    
    # Calculate weighted dot product
    dot_product = 0.0
    resume_norm = math.sqrt(sum(w**2 for w in resume_terms.values()))
    text_norm = math.sqrt(sum(w**2 for w in text_terms.values()))
    
    if resume_norm == 0 or text_norm == 0:
        return 0.0
    
    # Calculate cosine similarity
    common_terms = set(resume_terms) & set(text_terms)
    dot_product = sum(resume_terms[term] * text_terms[term] for term in common_terms)
    
    similarity = dot_product / (resume_norm * text_norm)
    
    # If this is a title match, boost the score
    if is_title:
        similarity = min(1.0, similarity * 1.5)  # Up to 50% boost for title matches
    
    return similarity

def calculate_skill_similarity(resume_skills: Set[str], job_skills: Set[str]) -> float:
    """Calculate similarity between resume skills and job required skills."""
    if not job_skills:
        return 0.0
    if not resume_skills:
        return 0.0
    
    # Calculate Jaccard similarity for skills
    intersection = len(resume_skills & job_skills)
    union = len(resume_skills | job_skills)
    
    return intersection / union if union > 0 else 0.0

def load_company_names(csv_path: str) -> List[str]:
    """Load company names from CSV file."""
    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            return [row['Company Name'].strip() for row in reader if row.get('Company Name')]
    except Exception as e:
        print(f"Error loading company names: {e}")
        return []

def load_job_postings(csv_path: str, company_names: List[str], limit: int = 1000) -> List[Dict]:
    """Load job postings from CSV file with error handling for large files."""
    jobs = []
    try:
        with open(csv_path, 'r', encoding='utf-8', errors='replace') as file:
            reader = csv.DictReader(file)
            for i, row in enumerate(reader):
                if i >= limit:
                    break
                # Add company name based on position in the company_names list
                company = company_names[i % len(company_names)] if company_names else ""
                jobs.append({
                    'id': row.get('Job Id', '').strip(),
                    'title': row.get('Job Title', '').strip(),
                    'description': row.get('Job Description', '').strip(),
                    'experience': row.get('Experience', '').strip(),
                    'qualifications': row.get('Qualifications', '').strip(),
                    'location': row.get('location', '').strip(),
                    'country': row.get('Country', '').strip(),
                    'work_type': row.get('Work Type', '').strip(),
                    'category': row.get('Category', '').strip(),
                    'company': company
                })
    except Exception as e:
        print(f"Error loading job postings: {e}")
    return jobs

def find_best_matches(
    resume_text: str, 
    jobs: List[Dict], 
    top_n: int = 5,
    min_score: float = 20.0  # Increased minimum score threshold
) -> List[Tuple[float, Dict]]:
    """Find the top N job matches for the given resume text."""
    if not resume_text or not jobs:
        return []
    
    # Preprocess resume text and extract skills
    resume_terms = preprocess_text(resume_text, is_resume=True)
    resume_skills = extract_skills(resume_text)
    
    # Calculate scores for all jobs
    scored_jobs = []
    for job in jobs:
        score = calculate_match_score(resume_terms, job, resume_skills)
        if score >= min_score:  # Only include jobs with score above threshold
            scored_jobs.append((score, job))
    
    # Sort by score in descending order and return top N
    scored_jobs.sort(key=lambda x: x[0], reverse=True)
    return scored_jobs[:top_n]

def print_matches(matches: List[Tuple[float, Dict]], output_file: str = None):
    """Print or save the job matches in a readable format."""
    output = []
    
    if not matches:
        output.append("No matching jobs found based on the resume content.")
    else:
        output.append("\n" + "="*80)
        output.append("TOP JOB MATCHES")
        output.append("="*80)
        
        for i, (score, job) in enumerate(matches, 1):
            output.append(f"\n{'='*40}")
            output.append(f"MATCH #{i} (Score: {score:.1f}%)")
            output.append(f"{'='*40}")
            if job.get('company'):
                output.append(f"Company: {job['company']}")
            output.append(f"Title: {job.get('title', 'N/A')}")
            output.append(f"Category: {job.get('category', 'N/A')}")
            output.append(f"Experience: {job.get('experience', 'N/A')}")
            output.append(f"Qualifications: {job.get('qualifications', 'N/A')}")
            
            location = ", ".join(filter(None, [job.get('location', ''), job.get('country', '')]))
            if location:
                output.append(f"Location: {location}")
                
            if job.get('work_type'):
                output.append(f"Work Type: {job['work_type']}")
            
            # Truncate long descriptions
            desc = job.get('description', '')
            if desc:
                if len(desc) > 300:
                    desc = desc[:300] + "..."
                output.append(f"\nDescription: {desc}")
            
            output.append("-" * 40)
    
    # Print to console
    print("\n".join(output))
    
    # Save to file if specified
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8', errors='replace') as f:
                f.write("\n".join(output))
            print(f"\nResults saved to: {output_file}")
        except Exception as e:
            print(f"Error saving results to file: {e}")

def main():
    parser = argparse.ArgumentParser(description='Resume-Job Matcher')
    parser.add_argument('--resume', type=str, required=True, 
                        help='Path to resume file (PDF, TXT)')
    parser.add_argument('--jobs', type=str, default='job_market_data.csv',
                        help='Path to job postings CSV file (default: job_market_data.csv)')
    parser.add_argument('--companies', type=str, default='company_names_149999.csv',
                        help='Path to company names CSV file (default: company_names_149999.csv)')
    parser.add_argument('--output', type=str, default='job_matches.txt',
                        help='Output file for results (default: job_matches.txt)')
    parser.add_argument('--limit', type=int, default=1000,
                        help='Maximum number of jobs to process (default: 1000)')
    parser.add_argument('--top', type=int, default=5,
                        help='Number of top matches to return (default: 5)')
    parser.add_argument('--min-score', type=float, default=20.0,
                        help='Minimum match score to include (default: 20.0)')
    
    args = parser.parse_args()
    
    # Check if resume file exists
    if not os.path.exists(args.resume):
        print(f"Error: Resume file not found: {args.resume}")
        return
    
    # Check if jobs file exists
    if not os.path.exists(args.jobs):
        print(f"Error: Job postings file not found: {args.jobs}")
        return
    
    # Load company names
    print(f"Loading company names from: {args.companies}")
    company_names = load_company_names(args.companies)
    if not company_names:
        print("Warning: No company names loaded. Using empty company names.")
    
    print(f"Processing resume: {args.resume}")
    print(f"Using job postings from: {args.jobs}")
    
    # Extract text from resume
    if args.resume.lower().endswith('.pdf'):
        resume_text = extract_text_from_pdf(args.resume)
    else:
        resume_text = extract_text_from_txt(args.resume)
    
    if not resume_text.strip():
        print("Error: Could not extract text from resume.")
        return
    
    # Load job postings with company names
    print(f"Loading up to {args.limit} job postings...")
    jobs = load_job_postings(args.jobs, company_names, args.limit)
    
    if not jobs:
        print("Error: No job postings found or could not read the file.")
        return
    
    print(f"Loaded {len(jobs)} job postings with company names.")
    print("Finding best matches...")
    
    # Find best matches
    matches = find_best_matches(
        resume_text, 
        jobs, 
        top_n=args.top,
        min_score=args.min_score
    )
    
    # Print and save results
    print_matches(matches, args.output)

if __name__ == "__main__":
    main()


# Satya: AI-Powered Resume Screening and Recommendation Analysis

## Introduction

Satya is an innovative AI-driven solution designed to revolutionize the HR landscape by streamlining the resume screening process and analyzing recommendation networks. This project aims to create a more efficient, reliable, and objective approach to talent acquisition in a world where careers are shaped by networks, connections, and credibility.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Project Structure](#project-structure)
6. [Methodology](#methodology)
7. [Results](#results)
8. [Future Directions](#future-directions)

## Project Overview

Satya employs advanced natural language processing (NLP) techniques, graph theory, and machine learning algorithms to:

1. Screen and score resumes
2. Analyze recommendation networks
3. Detect potential fraud in recommendations
4. Provide comprehensive candidate evaluations

The project is divided into three main parts:

1. Resume Screening and Initial Scoring
2. Network Analysis of Recommendations
3. Final Reliability Score Calculation

## Features

- Extraction and analysis of key resume attributes
- Graph-based analysis of recommendation networks
- Sentiment analysis of recommendation letters
- Fraud detection in recommendations
- Composite scoring system for candidate evaluation
- Visualization of score distributions and network graphs

## Installation

```bash
git clone https://github.com/your-repo/satya.git
cd satya
pip install -r requirements.txt
```

## Usage

This will perform resume screening, recommendation analysis, final score calculation, and dashboard creation.

## Methodology

### Part 1: Resume Screening and Initial Scoring

This stage employs advanced NLP techniques to extract and analyze various measures from resumes:

- Years of Experience: Extracted using regex to identify date patterns.
- Education Level: Categorized using pattern matching algorithms.
- Spell Check Ratio: Implemented using the LanguageTool library.
- Resume Section Score: Semantic analysis to identify key resume sections.
- Brevity Score: Evaluates resume conciseness based on word count and content density.
- Skill Count and Relevance: Uses keyword extraction and semantic similarity analysis.
- Technical Score: Composite score derived from education, experience, and technical skills.
- Managerial Score: Uses sentiment analysis and achievement quantification algorithms.
- Overall Score: Weighted combination of all previous scores.
- Job Match Score: TF-IDF vectorization and cosine similarity with job description.

### Part 2: Network Analysis of Recommendations

This stage constructs and analyzes a directed graph G(V,E) representing the recommendation network:

- PageRank: Identifies influential candidates and recommenders.
- Betweenness Centrality: Flags candidates with potential but not heavily networked.
- In-Degree: Measures the number of incoming recommendations.
- Reciprocal Recommendations: Detects and flags mutual recommendations.
- Cycle Detection: Identifies closed loops in the recommendation network.

Composite Credibility Score (CreditScore) Calculation:


CreditScore = IncomingEdgeWeightSum * ((0.4 * PageRank + 0.3 * Inverse_Betweenness) / (InDegree + 1)) - 0.3 * Reciprocity

### Part 3: Final Reliability Score

- Combines scores from Parts 1 and 2
- Incorporates sentiment analysis of recommendation letters using LLaMA 3.1 model
- Generates a final normalized reliability score for each candidate

## Results

The project outputs:
- CSV files with processed data and scores
- Visualizations of score distributions
- Network graphs showing recommendation patterns
- A final ranked list of candidates based on reliability scores
- An interactive dashboard for HR decision support

## Future Directions

- Enhance recommendation credibility by matching company names and work terms
- Implement continuous learning to improve fraud detection accuracy

# Application/UI for Satya

## Features
- Candidate database browsing with pagination
- Skill-based filtering
- Detailed candidate profile view
- AI-generated resume summaries
- Visual representation of candidate scores
- Chatbot for asking questions about candidates

## Prerequisites
- Python 3.7+
- Streamlit
- Pandas
- Groq API key
- Plotly
- LangChain

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/your-repo/hiring-with-satya.git
   cd hiring-with-satya
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your Groq API key:
   - Create a `.env` file in the project root
   - Add your Groq API key: `GROQ_API_KEY=your_api_key_here`

## Usage
1. Ensure you have the necessary CSV files in the project directory:
   - `streamlit_table.csv`
   - `resume_text.csv`

2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

3. Open your web browser and navigate to the provided local URL (usually `http://localhost:8501`)

## How It Works
1. **Candidate Database**: Browse through candidate profiles with pagination.
2. **Skill Filtering**: Use the multiselect feature to filter candidates based on skills.
3. **Profile View**: Click on "View Profile" to see detailed information about a candidate.
4. **Resume Summary**: AI-generated summary of the candidate's resume.
5. **Score Visualization**: Circular bar charts display various scores for the candidate.
6. **Chatbot**: Ask questions about the candidate and receive AI-powered responses.


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

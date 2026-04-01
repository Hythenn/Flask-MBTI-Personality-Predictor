## Project Overview

Linguistic and behavioral analysis tool developed as a Final Project to categorize users into Myers-Briggs personality types.
System architecture integrates a manual ensemble of three distinct models within a Python Flask web interface for real-time inference.
Application utilizes the [MBTI Type Dataset](https://www.kaggle.com/datasets/datasnaek/mbti-type) to correlate 30-question survey inputs with specific psychological profiles.

## Core Features

- Ensemble Modeling: Combines Linear Regression, Random Forest Classifier, and LinearSVC for robust categorical prediction.
- Interactive Assessment: Features a custom-built 30-question survey interface with progress tracking and slider-based inputs.
- Predictive Analytics: Translates user responses into descriptive personality insights and visualization based on machine learning outputs.

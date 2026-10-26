# Student Placement Predictor & Skill Profiler

A machine learning web app that predicts whether an engineering student is likely to be placed and generates a personalised skill-gap report showing what to improve before campus placements.

Built during my Machine Learning with Python internship (EISystems, Prabandhan IIT Kanpur, Sept 2026).

## Features
- Predicts placement status (Placed / Not Placed) with a confidence score
- Computes 4 skill pillar indices: Technical, Analytical, Soft Skills, Practical & Career Readiness, plus an overall Skill Index
- Radar chart benchmarking the student against the average placed candidate
- Student persona classification (e.g. Tech Maverick, Analytical Specialist, Growing Candidate)
- Skill-gap analysis with strengths, weaknesses and actionable recommendations
- Downloadable evaluation report (.txt) and benchmark roadmap (.csv)
- Ready-made persona presets for quick demo, plus full manual profile entry

## Tech Stack
Python, Pandas, NumPy, scikit-learn, Streamlit, Plotly

## Model
- Dataset: 1,00,000 student records (sourced from Kaggle)
- Models compared: Logistic Regression and Random Forest (80/20 stratified split)
- Logistic Regression performed better and was selected as the deployed model

## Project Structure
- `train_model.py` : cleans data, trains and compares models, saves the best one
- `app.py` : Streamlit web app
- `placement_data.csv` : dataset
- `model.pkl`, `scaler.pkl`, `encoders.pkl`, `feature_names.pkl` : saved model and preprocessing files
- `requirements.txt` : dependencies

## How to Run
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py

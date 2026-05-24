# CausalLift

CausalLift is a modular causal machine learning framework for A/B testing and uplift analysis.

The project focuses on estimating treatment effects, identifying high-impact user segments, and evaluating uplift modeling performance using causal inference techniques.

---

## Features

- Synthetic treatment-control dataset generation
- Classical A/B testing workflow
- Statistical significance testing
- T-Learner uplift modeling
- Individual treatment effect estimation
- Qini / uplift curve evaluation
- Segment-level uplift analysis
- Modular Python architecture

---

## Project Structure

```text
CausalLift/
│
├── data.py
├── ab_test.py
├── model.py
├── evaluation.py
├── analysis.py
├── main.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- SciPy
- Matplotlib

---

## Methodology

### 1. Classical A/B Testing
- Control vs treatment comparison
- Conversion rate analysis
- Statistical significance testing

### 2. T-Learner Uplift Modeling
Separate models are trained for:
- treatment group
- control group

The difference in predicted outcomes estimates individual treatment effects.

### 3. Uplift Evaluation
Users are ranked based on predicted uplift scores and evaluated using:
- uplift ranking
- Qini curve visualization

### 4. Segment Analysis
The framework identifies high-response user segments for targeted experimentation strategies.

---

## Installation

```bash
pip install -r requirements.txt

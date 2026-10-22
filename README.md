# NeedHub™ AI & Data Science Intelligence Platform

> **Empirical Machine Learning & Statistical Analysis for Hyperlocal Campus Peer-to-Peer Marketplace**  
> Developed for **Department of Information Technology, Pimpri Chinchwad College of Engineering (PCCoE), Pune**  
> **Course:** Data Science & Analytics Laboratory  
> **Academic Year:** 2026-27 | **Class:** T.E. Information Technology (Div A, Batch A4)

---

## 👥 Authors & Team
- **Nakshatra Lalwani** (Roll No: `125B1F066`) — *Project Lead, ML Architecture, Predictive Modeling & Web Development*
- **Siya Dubey** (Roll No: `125B1F065`) — *Exploratory Data Analysis, Feature Engineering & Statistical Hypothesis Testing*
- **Pawan Koshti** (Roll No: `125B1F079`) — *Data Acquisition, Preprocessing & Research Documentation*

---

## 🎯 Executive Summary & Problem Statement

College campuses exhibit severe micro-market frictions:
1. **Unstructured Communication:** $66.7\%$ of students report urgent requests for study materials or tools get lost in high-volume WhatsApp groups.
2. **Discovery & Pricing Deficits:** $48.0\%$ face peer discovery barriers and $35.6\%$ lack confidence in fair micro-gig pricing.
3. **Dorm Waste & Idleness:** $61.6\%$ retain unused academic assets (drafters, calculators, kits) that could be monetized.
4. **Peer Trust Advantage:** $66.7\%$ of students explicitly favor peer-to-peer campus solutions over external commercial platforms.

**NeedHub** transforms latent student skills and idle dorm resources into a structured, predictive peer marketplace powered by machine intelligence.

---

## 🚀 Live Interactive Web Platform (`index.html`)

This repository contains a full-stack, single-page interactive showcase dashboard:
- **Live ML Simulator:** Real-time client-side inference executing logistic regression and StandardScaler normalization to predict individual student gig adoption propensity $P(y=1)$ with dynamic SVG gauge animation.
- **Visual EDA Gallery:** 10 high-resolution (300 DPI) analytical visualizations with modal zoom and category filtering.
- **Model Arena:** Full evaluation of 7 supervised classification models and 4 continuous regression models.
- **Persona Stratification:** Multi-dimensional $k$-Means clustering ($k=4$) visualized with an interactive radar chart and 2D PCA mapping.
- **Product Strategy:** Concrete feature blueprints (Smart Pricing Engine, UPI Escrow, Dorm Rental Hub).

### How to Run Locally:
Simply open `index.html` in any modern web browser.

---

## 🤖 Machine Learning Model Benchmarks

### Supervised Classification (Earning Propensity $P(\text{Gig}=1)$)
Target: Student willingness to offer micro-gigs on NeedHub (Binary: $\ge 4/5$).

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Utility |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Random Forest Classifier** | **66.7%** | **0.6875** | **0.9167** | **0.7857** | 0.5139 | **Top Supply Onboarding Champion (Max Recall)** |
| **Support Vector Machine (RBF)** | 63.9% | 0.6667 | 0.9167 | 0.7719 | **0.5903** | Best Discrimination Power |
| **Gradient Boosting Classifier** | 63.9% | 0.6897 | 0.8333 | 0.7547 | 0.5243 | Robust Tree Ensemble |
| **Logistic Regression (L2)** | 58.3% | 0.6923 | 0.7500 | 0.7199 | 0.5104 | Powers Browser Inference Simulator |
| **Decision Tree (CART)** | 58.3% | 0.6800 | 0.7083 | 0.6939 | 0.5208 | Interpretable Rule Branches |
| **Gaussian Naive Bayes** | 55.6% | 0.7222 | 0.5417 | 0.6190 | 0.5625 | Fast Probabilistic Baseline |
| **K-Nearest Neighbors ($k=5$)** | 55.6% | 0.6552 | 0.7917 | 0.7170 | 0.4375 | Metric Space Neighbor Search |

### Continuous Regression (Urgent Campus Need Forecasting)
Target: Continuous campus need score ($0 - 15$).

| Regression Model | MAE | RMSE | $R^2$ | Penalty Type |
| :--- | :---: | :---: | :---: | :--- |
| **Ridge Regression** | **3.143** | **4.011** | 0.047 | $L_2$ Regularized |
| **Linear Regression (OLS)** | 3.176 | 4.032 | 0.037 | Unregularized Ordinary Least Squares |
| **Lasso Regression** | 3.250 | 4.088 | 0.010 | Sparse $L_1$ Feature Selection |
| **Gradient Boosting Regressor** | 3.328 | 4.382 | -0.138 | Non-linear Tree Ensemble |

---

## 👥 Student Persona Clusters ($k=4$)

1. **Cluster 0: The Core Peer Gigger (41.2%)** — High willingness ($4.23/5$), high peer trust ($2.07/3$), dominant in tutoring and homework review.
2. **Cluster 1: Stressed Academic Seeker (17.5%)** — High need intensity ($8.27/15$), high social isolation ($4.81/8$), frequent consumer of peer assistance.
3. **Cluster 2: The Casual Browser (23.2%)** — Low current gig willingness ($3.05/5$), low trust ($1.15/3$), passive observer requiring low-friction onboarding.
4. **Cluster 3: The Power Freelancer (18.1%)** — Highest skills breadth ($4.66/6$), highest gig willingness ($4.69/5$), strong dorm surplus ($2.41/3$).

---

## 📁 Repository Structure

```
Needhub_stat/
├── index.html                           # Standalone interactive web dashboard & live simulator
├── README.md                            # Complete technical documentation
├── NeedHub_App_ML_Report.pdf            # 15-Page formal academic report with 300 DPI figures
├── NeedHub_App_ML_Report.md             # Markdown source of the formal report
├── NeedHub_Data_Science_ML_Analysis.ipynb# Comprehensive Jupyter Notebook with all outputs
├── needhub_cleaned_ml_dataset.csv       # Cleaned & engineered dataset (177 rows, 32 columns)
├── Needhubstat.xlsx                     # Raw empirical campus survey dataset
├── run_needhub_analysis.py              # Python ML training & statistical execution pipeline
├── build_needhub_notebook.py            # Automated Jupyter notebook generation script
├── generate_needhub_pdf.py              # Automated 15-page PDF report generator
└── graphs/                              # 10 High-Resolution 300 DPI Matplotlib/Seaborn Figures
    ├── 01_gig_earning_propensity_distribution.png
    ├── 02_whatsapp_group_inefficacy.png
    ├── 03_campus_skills_supply.png
    ├── 04_peer_trust_advantage_pie.png
    ├── 05_dorm_waste_recommerce_opportunity.png
    ├── 06_social_isolation_companion_deficit.png
    ├── 07_classification_roc_curves.png
    ├── 08_feature_importance_needhub_adoption.png
    ├── 09_regression_need_intensity_fit.png
    └── 10_user_persona_kmeans_pca.png
```

---

## 📜 Academic Integrity & Citation
This project was designed, analyzed, and synthesized by Nakshatra Lalwani, Siya Dubey, and Pawan Koshti at Pimpri Chinchwad College of Engineering (PCCoE), Pune.
All rights reserved © 2026 NeedHub.

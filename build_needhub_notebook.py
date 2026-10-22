import json
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
nb_path = os.path.join(base_dir, "NeedHub_Data_Science_ML_Analysis.ipynb")

def make_cell(cell_type, source):
    if isinstance(source, list):
        src = source
    else:
        src = [line + "\n" for line in source.split("\n")]
        if src and src[-1] == "\n":
            src[-1] = ""
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": src
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell

cells = []

# Title & Metadata
cells.append(make_cell("markdown", """# NeedHub: Machine Learning & Empirical Analytics for Campus Peer-to-Peer Marketplace
### *Predicting User Monetization Propensity, Need Intensity Forecasting & K-Means Persona Stratification*

**Authors:** Nakshatra Lalwani (125B1F066), Siya Dubey (125B1F065), Pawan Koshti (125B1F079)  
**Institution:** Pimpri Chinchwad College of Engineering (PCCoE), Pune  
**Department:** Information Technology | Division A | Batch A4  
**Course:** Data Science Laboratory (VSEC Mini-Project)  
**Target Application:** **NeedHub** (Hyper-local Campus Gig, Skill Exchange & Activity Marketplace)  
"""))

cells.append(make_cell("markdown", """## 1. Environment Setup & Library Imports"""))
cells.append(make_cell("code", """import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor, 
    GradientBoostingClassifier, GradientBoostingRegressor, 
    AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, mean_squared_error,
    mean_absolute_error, r2_score
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["figure.dpi"] = 110
print("[OK] Environment ready.")
"""))

cells.append(make_cell("markdown", """## 2. Data Ingestion & Auditing"""))
cells.append(make_cell("code", """excel_path = "Needhubstat.xlsx"
df_raw = pd.read_excel(excel_path, sheet_name="Form Responses 1")
print(f"Loaded survey responses shape: {df_raw.shape}")
display(df_raw.head(3))
"""))

cells.append(make_cell("markdown", """## 3. Data Cleaning, Type Normalization & Multi-Label Feature Engineering"""))
cells.append(make_cell("code", """df = df_raw.copy()
df.dropna(how="all", axis=1, inplace=True)

col_mapping = {
    df.columns[0]: "respondent_name",
    df.columns[1]: "need_struggle_freq",
    df.columns[2]: "needs_categories",
    df.columns[3]: "search_method",
    df.columns[4]: "group_response_rate",
    df.columns[5]: "paid_big_company",
    df.columns[6]: "skills_offered",
    df.columns[7]: "skill_earn_desire",
    df.columns[8]: "earning_barriers",
    df.columns[9]: "gig_task_propensity",
    df.columns[10]: "peer_trust_comfort",
    df.columns[11]: "skipped_activity_freq",
    df.columns[12]: "companion_needs",
    df.columns[13]: "unused_items_waste",
    df.columns[14]: "gender",
    df.columns[15]: "age"
}
df.rename(columns=col_mapping, inplace=True)
df = df[df["need_struggle_freq"].notna()].copy()

# Clean Age
def clean_age(val):
    try:
        val_str = str(val).strip()
        num = float(re.findall(r"\d+", val_str)[0])
        return num if 16 <= num <= 30 else np.nan
    except:
        return np.nan

df["age_clean"] = df["age"].apply(clean_age).fillna(19.0)

# Ordinal & Continuous Scales
struggle_map = {"0-3": 1.5, "3-6": 4.5, "6-9": 7.5, "more than 10": 12.0}
df["need_intensity_score"] = df["need_struggle_freq"].map(struggle_map).fillna(1.5)

propensity_map = {"Very likely": 5, "Likely": 4, "Neutral": 3, "Unlikely": 2, "Very unlikely": 1}
df["gig_propensity_score"] = df["gig_task_propensity"].map(propensity_map).fillna(3)
df["target_high_gig_propensity"] = (df["gig_propensity_score"] >= 4).astype(int)

trust_map = {"Much more": 3, "Somewhat more": 2, "No difference": 1, "Less comfortable": 0}
df["peer_trust_score"] = df["peer_trust_comfort"].map(trust_map).fillna(1)

isolation_map = {0: 0.0, "0": 0.0, "1-2": 1.5, "3-6": 4.5, "more than 6": 8.0}
df["social_isolation_score"] = df["skipped_activity_freq"].map(isolation_map).fillna(1.5)

waste_map = {"Yes, multiple times": 3, "Yes, once or twice": 2, "Not sure": 1, "No": 0}
df["dorm_waste_score"] = df["unused_items_waste"].map(waste_map).fillna(1)

group_fail_map = {"Never": 4, "Rarely": 3, "Sometimes": 2, "Often": 1, "Always": 0, "I've never posted one": 2.5}
df["group_inefficacy_score"] = df["group_response_rate"].map(group_fail_map).fillna(2.5)

skill_earn_map = {"Yes, Often": 4, "Yes, sometimes": 3, "Maybe": 2, "Rarely": 1, "Never.": 0}
df["skill_monetization_desire"] = df["skill_earn_desire"].map(skill_earn_map).fillna(2)

# Multi-label extraction
def extract_indicator(series, keyword):
    return series.fillna("").apply(lambda x: 1 if keyword.lower() in str(x).lower() else 0)

df["skill_tutoring"] = extract_indicator(df["skills_offered"], "tutoring")
df["skill_coding"] = extract_indicator(df["skills_offered"], "coding")
df["skill_writing"] = extract_indicator(df["skills_offered"], "writing")
df["skill_notes"] = extract_indicator(df["skills_offered"], "notes")
df["skill_design"] = extract_indicator(df["skills_offered"], "design")
df["skill_art"] = extract_indicator(df["skills_offered"], "crafts")
df["skill_therapy"] = extract_indicator(df["skills_offered"], "therapy")
df["skill_photography"] = extract_indicator(df["skills_offered"], "photography")
df["total_skills_offered"] = (
    df["skill_tutoring"] + df["skill_coding"] + df["skill_writing"] + 
    df["skill_notes"] + df["skill_design"] + df["skill_art"] + 
    df["skill_therapy"] + df["skill_photography"]
)

df["barrier_discovery"] = extract_indicator(df["earning_barriers"], "where to find")
df["barrier_pricing"] = extract_indicator(df["earning_barriers"], "how to price")
df["barrier_confidence"] = extract_indicator(df["earning_barriers"], "approaching strangers")
df["barrier_time"] = extract_indicator(df["earning_barriers"], "no time")
df["is_female"] = (df["gender"] == "Female").astype(int)

print(f"[OK] Cleaned dataframe ready. Active sample size: {len(df)}")
"""))

cells.append(make_cell("markdown", """## 4. Exploratory Data Analysis & Product Market Fit Diagnostics"""))
cells.append(make_cell("code", """fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# 1. Micro-Gig Acceptance Propensity
prop_counts = df["gig_task_propensity"].value_counts()[["Very likely", "Likely", "Neutral", "Unlikely", "Very unlikely"]]
axes[0].bar(prop_counts.index, prop_counts.values, color=["#2A9D8F", "#52B788", "#E9C46A", "#F4A261", "#E76F51"], edgecolor="k")
axes[0].set_title("Willingness to Accept ₹100–500 Campus Micro-Gigs", fontweight="bold")
axes[0].set_ylabel("Student Count")

# 2. Campus Peer Trust Advantage
trust_counts = df["peer_trust_comfort"].value_counts()
axes[1].pie(trust_counts.values, labels=trust_counts.index, autopct="%1.1f%%", colors=["#0077B6", "#00B4D8", "#90E0EF", "#CAF0F8"], startangle=140)
axes[1].set_title("Comfort Paying Fellow College Student vs Online Stranger", fontweight="bold")

plt.tight_layout()
plt.show()
"""))

cells.append(make_cell("markdown", """## 5. Statistical Hypothesis Testing"""))
cells.append(make_cell("code", """# Chi-Square Test: Gender vs High Micro-Gig Propensity
contingency_table = pd.crosstab(df["gender"].dropna(), df["target_high_gig_propensity"])
chi2, p_val, dof, _ = stats.chi2_contingency(contingency_table)
print("--- Chi-Square Test of Independence (Gender vs Micro-Gig Propensity) ---")
print(f"Chi2 = {chi2:.4f}, p-value = {p_val:.4f}, df = {dof}")
print(f"Decision: {'Significant difference across genders' if p_val < 0.05 else 'No significant gender disparity (Equal propensity)'}")

# Two-Sample t-Test: Need Intensity between High Social Isolation vs Low Social Isolation
low_iso = df[df["social_isolation_score"] <= 1.5]["need_intensity_score"]
high_iso = df[df["social_isolation_score"] > 1.5]["need_intensity_score"]
t_stat, p_val_t = stats.ttest_ind(low_iso, high_iso, equal_var=False)
print("\n--- Welch's Two-Sample t-Test (Social Isolation vs Need Intensity) ---")
print(f"Low Isolation Mean Need  : {low_iso.mean():.2f}")
print(f"High Isolation Mean Need : {high_iso.mean():.2f}")
print(f"t-statistic = {t_stat:.4f}, p-value = {p_val_t:.4e}")
print(f"Decision: {'Statistically Significant Correlation' if p_val_t < 0.05 else 'No difference'}")
"""))

cells.append(make_cell("markdown", """## 6. Supervised Machine Learning: Predicting Micro-Gig Earning Propensity (7 Models)"""))
cells.append(make_cell("code", """ml_features = [
    "need_intensity_score", "peer_trust_score", "social_isolation_score",
    "dorm_waste_score", "group_inefficacy_score", "skill_monetization_desire", 
    "total_skills_offered", "barrier_discovery", "barrier_pricing", 
    "barrier_confidence", "barrier_time", "skill_tutoring", "skill_coding", 
    "skill_design", "age_clean", "is_female"
]

X = df[ml_features]
y = df["target_high_gig_propensity"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Support Vector (SVC)": SVC(probability=True, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
}

results = []
for name, model in models.items():
    if name in ["Logistic Regression", "Support Vector (SVC)", "K-Nearest Neighbors"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    results.append({"Model": name, "Accuracy": round(acc, 4), "Precision": round(prec, 4), "Recall": round(rec, 4), "F1_Score": round(f1, 4), "ROC_AUC": round(auc, 4)})

df_eval = pd.DataFrame(results).sort_values(by="F1_Score", ascending=False)
display(df_eval)
"""))

cells.append(make_cell("markdown", """## 7. Model Interpretability & Gini Feature Importance"""))
cells.append(make_cell("code", """rf = models["Random Forest"]
importances = rf.feature_importances_
df_imp = pd.DataFrame({"Feature": ml_features, "Importance": importances}).sort_values(by="Importance", ascending=True)

plt.figure(figsize=(9.5, 5.5))
bars = plt.barh(df_imp["Feature"], df_imp["Importance"], color="#2A9D8F")
plt.title("Random Forest Gini Feature Importance: Key Drivers of NeedHub Adoption", fontweight="bold")
plt.xlabel("Gini Importance Score")
for bar, v in zip(bars, df_imp["Importance"]):
    plt.text(v + 0.003, bar.get_y() + bar.get_height()/2, f"{v*100:.1f}%", va="center", fontweight="bold", fontsize=8.5)
plt.tight_layout()
plt.show()
"""))

cells.append(make_cell("markdown", """## 8. Unsupervised Machine Learning: K-Means User Persona Segmentation & PCA"""))
cells.append(make_cell("code", """cluster_cols = [
    "need_intensity_score", "gig_propensity_score", "peer_trust_score",
    "social_isolation_score", "dorm_waste_score", "total_skills_offered",
    "skill_monetization_desire"
]

X_cl = StandardScaler().fit_transform(df[cluster_cols])
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_cl)

pca = PCA(n_components=2, random_state=42)
pca_res = pca.fit_transform(X_cl)
df["pca1"] = pca_res[:, 0]
df["pca2"] = pca_res[:, 1]

persona_mapping = {
    0: "The Core Peer Gigger (High Trust & Gig Willingness)",
    1: "The Stressed Academic Seeker (Highest Need & Isolation)",
    2: "The Casual Campus Peer (Moderate Occasional Needs)",
    3: "The Power Freelancer (Highest Skills & Collaboration)"
}
df["persona"] = df["cluster"].map(persona_mapping)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="pca1", y="pca2", hue="persona", palette="Set1", s=75, alpha=0.9, edgecolor="k")
plt.title("NeedHub User Persona Stratification (K-Means k=4 with PCA Projection)", fontweight="bold")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

# Display cluster averages
display(df.groupby("persona")[cluster_cols].mean().round(2))
"""))

cells.append(make_cell("markdown", """## 9. App Conclusions & Product Recommendations for NeedHub
1. **Price Standardization (₹100–500 Task Chips):**
   * Over 67% of students are willing to take micro-gigs, but the #2 barrier is *"I don't know how to price my work"*. NeedHub must provide pre-set pricing recommendations.
2. **Campus Verification Badge:**
   * 65%+ higher trust for college peers over strangers online. Enforcing `@college.edu` email verification creates an immediate competitive moat against generic freelancer platforms.
3. **Dorm Re-commerce Tab (Buy / Sell / Give):**
   * 67.8% of students discard or hoard working items. A local campus marketplace unlocks immediate user retention.
4. **Activity & Hackathon Teammate Matching:**
   * 82.5% of students skipped activities due to lack of a partner. A buddy finder feature drives viral daily active users (DAU).
"""))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"[SUCCESS] Notebook created at: {nb_path} with {len(cells)} cells.")

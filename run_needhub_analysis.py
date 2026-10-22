import os
import sys
import re
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, mean_squared_error,
    mean_absolute_error, r2_score
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Setup paths
base_dir = os.path.dirname(os.path.abspath(__file__))
graphs_dir = os.path.join(base_dir, "graphs")
os.makedirs(graphs_dir, exist_ok=True)
excel_path = os.path.join(base_dir, "Needhubstat.xlsx")

print("=" * 70)
print("NEEDHUB DATA SCIENCE & MACHINE LEARNING PIPELINE")
print("Target Directory:", base_dir)
print("=" * 70)

# Set visual styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10.5
plt.rcParams["figure.dpi"] = 300

# ==============================================================================
# 1. DATA INGESTION & ROBUST CLEANING
# ==============================================================================
df_raw = pd.read_excel(excel_path, sheet_name="Form Responses 1")
print(f"Raw Survey Records Loaded: {df_raw.shape}")

# Drop empty metadata columns
df = df_raw.copy()
df.dropna(how="all", axis=1, inplace=True)

# Standardize column names
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

# Filter out completely blank rows
df = df[df["need_struggle_freq"].notna()].copy()
print(f"Valid Active Responses: {len(df)}")

# --- Feature Engineering & Numerical Encoding ---

# 1. Clean Age: Parse integers, handle outliers / joke responses, impute median
def clean_age(val):
    try:
        val_str = str(val).strip()
        num = float(re.findall(r"\d+", val_str)[0])
        if 16 <= num <= 30:
            return num
        return np.nan
    except:
        return np.nan

df["age_clean"] = df["age"].apply(clean_age)
df["age_clean"] = df["age_clean"].fillna(df["age_clean"].median())

# 2. Need Struggle Frequency (Q1): Ordinal mapping to numeric estimated annual scale
struggle_map = {
    "0-3": 1.5,
    "3-6": 4.5,
    "6-9": 7.5,
    "more than 10": 12.0
}
df["need_intensity_score"] = df["need_struggle_freq"].map(struggle_map).fillna(1.5)

# 3. Micro-Gig Earning Propensity (Q9 - Target for Classification):
# "If offered ₹100-500 for a task you're good at, how likely are you to take it?"
propensity_map = {
    "Very likely": 5,
    "Likely": 4,
    "Neutral": 3,
    "Unlikely": 2,
    "Very unlikely": 1
}
df["gig_propensity_score"] = df["gig_task_propensity"].map(propensity_map).fillna(3)
# Binary Target: 1 if Likely / Very Likely, 0 otherwise
df["target_high_gig_propensity"] = (df["gig_propensity_score"] >= 4).astype(int)

# 4. Campus Peer Trust Advantage (Q10):
# Comfort paying fellow student vs online stranger
trust_map = {
    "Much more": 3,
    "Somewhat more": 2,
    "No difference": 1,
    "Less comfortable": 0
}
df["peer_trust_score"] = df["peer_trust_comfort"].map(trust_map).fillna(1)

# 5. Social Isolation / Skipped Activities (Q11):
isolation_map = {
    0: 0.0,
    "0": 0.0,
    "1-2": 1.5,
    "3-6": 4.5,
    "more than 6": 8.0
}
df["social_isolation_score"] = df["skipped_activity_freq"].map(isolation_map).fillna(1.5)

# 6. Second-hand Waste / Dorm Re-commerce (Q13):
waste_map = {
    "Yes, multiple times": 3,
    "Yes, once or twice": 2,
    "Not sure": 1,
    "No": 0
}
df["dorm_waste_score"] = df["unused_items_waste"].map(waste_map).fillna(1)

# 7. WhatsApp / Social Group Failure (Q4):
group_fail_map = {
    "Never": 4,
    "Rarely": 3,
    "Sometimes": 2,
    "Often": 1,
    "Always": 0,
    "I've never posted one": 2.5
}
df["group_inefficacy_score"] = df["group_response_rate"].map(group_fail_map).fillna(2.5)

# 8. Paid Big Company Out of Convenience (Q5):
paid_company_map = {
    "Yes , multiple times": 3,
    "Yes, once or twice": 2,
    "No": 1,
    "No, I DON'T BUY STUFF!": 0
}
df["convenience_leakage_score"] = df["paid_big_company"].map(paid_company_map).fillna(1)

# 9. Desire to Earn from Skill (Q7):
skill_earn_map = {
    "Yes, Often": 4,
    "Yes, sometimes": 3,
    "Maybe": 2,
    "Rarely": 1,
    "Never.": 0
}
df["skill_monetization_desire"] = df["skill_earn_desire"].map(skill_earn_map).fillna(2)

# --- Multi-Label Binary Indicator Extraction ---
def extract_indicator(series, keyword):
    return series.fillna("").apply(lambda x: 1 if keyword.lower() in str(x).lower() else 0)

# Skills Offered
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

# Needs Categories
df["need_assignment"] = extract_indicator(df["needs_categories"], "assignment")
df["need_learn_skill"] = extract_indicator(df["needs_categories"], "learning")
df["need_repair"] = extract_indicator(df["needs_categories"], "repairing")
df["need_design"] = extract_indicator(df["needs_categories"], "designed")
df["need_hobby"] = extract_indicator(df["needs_categories"], "hobby")

# Barriers to Earning
df["barrier_discovery"] = extract_indicator(df["earning_barriers"], "where to find")
df["barrier_pricing"] = extract_indicator(df["earning_barriers"], "how to price")
df["barrier_confidence"] = extract_indicator(df["earning_barriers"], "approaching strangers")
df["barrier_time"] = extract_indicator(df["earning_barriers"], "no time")

# Gender Encoding
df["is_female"] = (df["gender"] == "Female").astype(int)
df["is_male"] = (df["gender"] == "Male").astype(int)

print("[OK] Feature engineering completed. Shape:", df.shape)

# ==============================================================================
# 2. PUBLICATION GRAPH GENERATION (10 GRAPHS)
# ==============================================================================

# Graph 1: Gig Earning Propensity Distribution (Q9)
fig, ax = plt.subplots(figsize=(8.5, 5))
prop_counts = df["gig_task_propensity"].value_counts()[["Very likely", "Likely", "Neutral", "Unlikely", "Very unlikely"]]
colors_bar = ["#2A9D8F", "#52B788", "#E9C46A", "#F4A261", "#E76F51"]
bars = ax.bar(prop_counts.index, prop_counts.values, color=colors_bar, edgecolor="black", linewidth=0.8)
ax.set_title("Willingness to Accept ₹100–500 Campus Micro-Gigs on NeedHub", fontweight="bold", fontsize=13)
ax.set_xlabel("Likelihood to Accept Task")
ax.set_ylabel("Number of Students")
for bar in bars:
    h = bar.get_height()
    pct = (h / len(df)) * 100
    ax.text(bar.get_x() + bar.get_width()/2, h + 1.5, f"{h} ({pct:.1f}%)", ha="center", fontweight="bold", fontsize=9.5)
plt.tight_layout()
p1 = os.path.join(graphs_dir, "01_gig_earning_propensity_distribution.png")
plt.savefig(p1, bbox_inches="tight")
plt.close()
print("Saved:", p1)

# Graph 2: WhatsApp / Social Group Inefficacy (Q4)
fig, ax = plt.subplots(figsize=(9, 5))
group_counts = df["group_response_rate"].value_counts()
sns.barplot(x=group_counts.values, y=group_counts.index, hue=group_counts.index, palette="mako", ax=ax, legend=False)
ax.set_title("Inefficacy of WhatsApp/Instagram Groups for Urgent Help", fontweight="bold", fontsize=13)
ax.set_xlabel("Number of Respondents")
ax.set_ylabel("Response Utility Rate")
for i, v in enumerate(group_counts.values):
    ax.text(v + 1, i, f" {v} ({v/len(df)*100:.1f}%)", va="center", fontweight="bold", fontsize=9.5)
plt.tight_layout()
p2 = os.path.join(graphs_dir, "02_whatsapp_group_inefficacy.png")
plt.savefig(p2, bbox_inches="tight")
plt.close()
print("Saved:", p2)

# Graph 3: Skills Offered vs Needs Demanded (Marketplace Balance)
skills_supply = {
    "Tutoring": df["skill_tutoring"].sum(),
    "Coding": df["skill_coding"].sum(),
    "Writing": df["skill_writing"].sum(),
    "Notes": df["skill_notes"].sum(),
    "Design": df["skill_design"].sum(),
    "Therapy": df["skill_therapy"].sum(),
    "Art/Crafts": df["skill_art"].sum(),
    "Photo/Video": df["skill_photography"].sum()
}
df_supply = pd.DataFrame(list(skills_supply.items()), columns=["Skill", "Student_Count"]).sort_values(by="Student_Count", ascending=True)

fig, ax = plt.subplots(figsize=(9, 5.5))
bars = ax.barh(df_supply["Skill"], df_supply["Student_Count"], color="#3A86FF", edgecolor="black", linewidth=0.8)
ax.set_title("Skills Offered by Students on Campus (Supply Side for NeedHub)", fontweight="bold", fontsize=13)
ax.set_xlabel("Number of Students Capable of Offering Skill")
ax.set_ylabel("Skill Domain")
for bar in bars:
    w = bar.get_width()
    ax.text(w + 1.5, bar.get_y() + bar.get_height()/2, f" {int(w)} ({w/len(df)*100:.1f}%)", va="center", fontweight="bold")
plt.tight_layout()
p3 = os.path.join(graphs_dir, "03_campus_skills_supply.png")
plt.savefig(p3, bbox_inches="tight")
plt.close()
print("Saved:", p3)

# Graph 4: Campus Peer Trust Advantage (Q10)
fig, ax = plt.subplots(figsize=(8, 5))
trust_counts = df["peer_trust_comfort"].value_counts()
colors_trust = ["#0077B6", "#00B4D8", "#90E0EF", "#CAF0F8"]
wedges, texts, autotexts = ax.pie(
    trust_counts.values, labels=trust_counts.index, autopct="%1.1f%%",
    colors=colors_trust, startangle=140, textprops=dict(fontweight="bold")
)
ax.set_title("Comfort Paying Fellow College Student vs Online Stranger", fontweight="bold", fontsize=13)
plt.tight_layout()
p4 = os.path.join(graphs_dir, "04_peer_trust_advantage_pie.png")
plt.savefig(p4, bbox_inches="tight")
plt.close()
print("Saved:", p4)

# Graph 5: Dorm Waste & Unused Items (Q13)
fig, ax = plt.subplots(figsize=(8.5, 4.8))
waste_counts = df["unused_items_waste"].value_counts()[["Yes, once or twice", "Yes, multiple times", "No", "Not sure"]]
bars = ax.bar(waste_counts.index, waste_counts.values, color=["#E63946", "#F1FAEE", "#A8DADC", "#457B9D"], edgecolor="black")
ax.set_title("Dormitory Resource Waste: Kept or Discarded Usable Items", fontweight="bold", fontsize=13)
ax.set_ylabel("Number of Students")
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 1.5, f"{h} ({h/len(df)*100:.1f}%)", ha="center", fontweight="bold")
plt.tight_layout()
p5 = os.path.join(graphs_dir, "05_dorm_waste_recommerce_opportunity.png")
plt.savefig(p5, bbox_inches="tight")
plt.close()
print("Saved:", p5)

# Graph 6: Social Isolation & Companion Deficit (Q11)
fig, ax = plt.subplots(figsize=(8.5, 4.8))
iso_counts = df["skipped_activity_freq"].astype(str).value_counts()[["0", "1-2", "3-6", "more than 6"]]
bars = ax.bar(iso_counts.index, iso_counts.values, color="#9D4EDD", edgecolor="black")
ax.set_title("Times Skipped an Activity Due to Lack of Companions (Past Month)", fontweight="bold", fontsize=13)
ax.set_xlabel("Frequency of Skipped Activities")
ax.set_ylabel("Student Count")
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 1.5, f"{h} ({h/len(df)*100:.1f}%)", ha="center", fontweight="bold")
plt.tight_layout()
p6 = os.path.join(graphs_dir, "06_social_isolation_companion_deficit.png")
plt.savefig(p6, bbox_inches="tight")
plt.close()
print("Saved:", p6)

# ==============================================================================
# 3. MACHINE LEARNING MODELING SUITE
# ==============================================================================

# Feature Matrix for Classification
ml_features = [
    "need_intensity_score", "peer_trust_score", "social_isolation_score",
    "dorm_waste_score", "group_inefficacy_score", "convenience_leakage_score",
    "skill_monetization_desire", "total_skills_offered", "barrier_discovery",
    "barrier_pricing", "barrier_confidence", "barrier_time",
    "skill_tutoring", "skill_coding", "skill_design", "age_clean", "is_female"
]

X_c = df[ml_features].copy()
y_c = df["target_high_gig_propensity"].copy()

# Stratified Split (80% Train, 20% Test)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_c, y_c, test_size=0.20, random_state=42, stratify=y_c
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_c)
X_test_scaled = scaler.transform(X_test_c)

# 7 Classification Models
clf_models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Support Vector (SVC)": SVC(probability=True, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
}

benchmarks_clf = []
roc_curves_data = {}

for name, model in clf_models.items():
    if name in ["Logistic Regression", "Support Vector (SVC)", "K-Nearest Neighbors"]:
        model.fit(X_train_scaled, y_train_c)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train_c, y_train_c)
        y_pred = model.predict(X_test_c)
        y_prob = model.predict_proba(X_test_c)[:, 1]

    acc = accuracy_score(y_test_c, y_pred)
    prec = precision_score(y_test_c, y_pred, zero_division=0)
    rec = recall_score(y_test_c, y_pred, zero_division=0)
    f1 = f1_score(y_test_c, y_pred, zero_division=0)
    auc = roc_auc_score(y_test_c, y_prob)

    benchmarks_clf.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1_Score": round(f1, 4),
        "ROC_AUC": round(auc, 4)
    })
    
    fpr, tpr, _ = roc_curve(y_test_c, y_prob)
    roc_curves_data[name] = (fpr, tpr, auc)

df_clf_results = pd.DataFrame(benchmarks_clf).sort_values(by="F1_Score", ascending=False)
print("\n--- CLASSIFICATION BENCHMARK RESULTS ---")
print(df_clf_results.to_string(index=False))

# Graph 7: ROC Curves for Classification Models
fig, ax = plt.subplots(figsize=(9, 6))
for name, (fpr, tpr, auc) in roc_curves_data.items():
    ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", linewidth=1.8)
ax.plot([0, 1], [0, 1], "k--", label="Random Classifier (0.500)")
ax.set_title("ROC-AUC Curves: Predicting High Micro-Gig Propensity", fontweight="bold", fontsize=13)
ax.set_xlabel("False Positive Rate (1 - Specificity)")
ax.set_ylabel("True Positive Rate (Recall)")
ax.legend(loc="lower right")
plt.tight_layout()
p7 = os.path.join(graphs_dir, "07_classification_roc_curves.png")
plt.savefig(p7, bbox_inches="tight")
plt.close()
print("Saved:", p7)

# Graph 8: Feature Importance (Random Forest)
rf_model = clf_models["Random Forest"]
importances = rf_model.feature_importances_
df_imp = pd.DataFrame({"Feature": ml_features, "Importance": importances}).sort_values(by="Importance", ascending=True)

fig, ax = plt.subplots(figsize=(9.5, 6))
bars = ax.barh(df_imp["Feature"], df_imp["Importance"], color="#2A9D8F")
ax.set_title("Random Forest Feature Importance: Key Drivers of NeedHub Adoption", fontweight="bold", fontsize=13)
ax.set_xlabel("Gini Importance Score")
for bar, v in zip(bars, df_imp["Importance"]):
    ax.text(v + 0.003, bar.get_y() + bar.get_height()/2, f"{v*100:.1f}%", va="center", fontweight="bold", fontsize=8.5)
plt.tight_layout()
p8 = os.path.join(graphs_dir, "08_feature_importance_needhub_adoption.png")
plt.savefig(p8, bbox_inches="tight")
plt.close()
print("Saved:", p8)

# --- Regression Modeling: Forecasting Assistance Need Intensity ---
reg_features = [f for f in ml_features if f != "need_intensity_score"]
X_r = df[reg_features].copy()
y_r = df["need_intensity_score"].copy()

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_r, y_r, test_size=0.20, random_state=42
)

reg_models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
    "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
}

benchmarks_reg = []
for name, model in reg_models.items():
    model.fit(X_train_r, y_train_r)
    y_pred_r = model.predict(X_test_r)
    mae = mean_absolute_error(y_test_r, y_pred_r)
    mse = mean_squared_error(y_test_r, y_pred_r)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_r, y_pred_r)
    benchmarks_reg.append({
        "Model": name,
        "MAE": round(mae, 3),
        "MSE": round(mse, 3),
        "RMSE": round(rmse, 3),
        "R2_Score": round(r2, 4)
    })

df_reg_results = pd.DataFrame(benchmarks_reg).sort_values(by="MAE")
print("\n--- REGRESSION BENCHMARK RESULTS ---")
print(df_reg_results.to_string(index=False))

# Graph 9: Regression Actual vs Predicted (Random Forest)
rf_reg = reg_models["Random Forest Regressor"]
y_pred_rf_reg = rf_reg.predict(X_test_r)

fig, ax = plt.subplots(figsize=(8.5, 5.5))
ax.scatter(y_test_r, y_pred_rf_reg, color="#1D3557", alpha=0.6, s=60, edgecolors="k", label="Student Need Score")
min_v = min(y_test_r.min(), y_pred_rf_reg.min())
max_v = max(y_test_r.max(), y_pred_rf_reg.max())
ax.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=2, label="Perfect Fit (y = x)")
ax.set_title("Random Forest Regressor: Forecasting Need Intensity on Campus", fontweight="bold", fontsize=13)
ax.set_xlabel("Actual Need Intensity Score")
ax.set_ylabel("Predicted Need Intensity Score")
ax.legend()
plt.tight_layout()
p9 = os.path.join(graphs_dir, "09_regression_need_intensity_fit.png")
plt.savefig(p9, bbox_inches="tight")
plt.close()
print("Saved:", p9)

# --- Unsupervised Learning: K-Means Persona Clustering & PCA ---
cluster_features = [
    "need_intensity_score", "gig_propensity_score", "peer_trust_score",
    "social_isolation_score", "dorm_waste_score", "total_skills_offered",
    "skill_monetization_desire"
]
X_clust = scaler.fit_transform(df[cluster_features])

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["user_cluster"] = kmeans.fit_predict(X_clust)

pca = PCA(n_components=2, random_state=42)
pca_coords = pca.fit_transform(X_clust)
df["pca1"] = pca_coords[:, 0]
df["pca2"] = pca_coords[:, 1]

# Cluster Profiles
cluster_summary = df.groupby("user_cluster").agg(
    Size=("need_intensity_score", "count"),
    Avg_Need_Intensity=("need_intensity_score", "mean"),
    Avg_Gig_Propensity=("gig_propensity_score", "mean"),
    Avg_Peer_Trust=("peer_trust_score", "mean"),
    Avg_Isolation=("social_isolation_score", "mean"),
    Avg_Dorm_Waste=("dorm_waste_score", "mean"),
    Avg_Skills_Offered=("total_skills_offered", "mean")
).round(2)

print("\n--- K-MEANS USER PERSONA CLUSTER PROFILES ---")
print(cluster_summary.to_string())

# Graph 10: 2D PCA Cluster Map of User Personas
persona_labels = {
    0: "The Disconnected Seeker (High Isolation & Need)",
    1: "The Campus Freelancer (High Skills & Gig Willingness)",
    2: "The Re-Commerce Trader (Dorm Waste & Trust)",
    3: "The Casual Campus Peer (Moderate Needs)"
}
df["persona_name"] = df["user_cluster"].map(persona_labels)

fig, ax = plt.subplots(figsize=(10, 6.5))
palette_clust = ["#E63946", "#2A9D8F", "#E76F51", "#457B9D"]
sns.scatterplot(
    data=df, x="pca1", y="pca2", hue="persona_name",
    palette=palette_clust, s=80, alpha=0.85, edgecolor="k", ax=ax
)
ax.set_title("NeedHub User Persona Stratification (K-Means k=4 with PCA Projection)", fontweight="bold", fontsize=13)
ax.set_xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)")
ax.set_ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)")
ax.legend(title="User Personas", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
p10 = os.path.join(graphs_dir, "10_user_persona_kmeans_pca.png")
plt.savefig(p10, bbox_inches="tight")
plt.close()
print("Saved:", p10)

# Save cleaned analysis dataset
cleaned_csv_path = os.path.join(base_dir, "needhub_cleaned_ml_dataset.csv")
df.to_csv(cleaned_csv_path, index=False)
print("Saved Cleaned Dataset to:", cleaned_csv_path)

print("=" * 70)
print("[SUCCESS] All Machine Learning Models Trained and 10 Graphs Exported!")
print("=" * 70)

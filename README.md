# Rashomon Set Visualizer

An interactive research tool that demonstrates **model multiplicity** in machine learning using the UCI Cleveland Heart Disease dataset. Built to visualize and explain core concepts from Prof. Lesia Semenova's research on the Rashomon Effect at Rutgers University.

---

## What Is the Rashomon Effect?

In 1950, Akira Kurosawa's film *Rashomon* showed the same murder described by four witnesses — each giving a different but equally believable account. Machine learning has the same problem.

Train the same algorithm on the same data with slightly different settings and you get models that are **statistically indistinguishable in overall accuracy** — yet they can disagree on individual patients.

The **Rashomon Set** formalizes this:

```
R(ε) = { all models f : accuracy(f) ≥ L* − ε }
```

Where `L*` is the best achievable accuracy and `ε` is a tolerance threshold. Every model inside this set is "almost as good" as the best. This tool visualizes what that means in practice.

---

## Live Demo

🔗 **[rashomon-set-visualizer.streamlit.app](https://rashomon-visualizer-4v9ilvjpasx6vc43kghcfp.streamlit.app/)**

> First load takes ~2 minutes to train 3,360 models. Every subsequent page click is instant.

---

## Key Findings

| Finding | Result |
|---------|--------|
| **Best accuracy (L\*)** | 0.7978 (5-Fold CV) |
| **Rashomon Set size** at ε=0.02 | 633 of 3,360 models (18.8%) |
| **Depth-3 trees** qualification rate | 100% — a 3-question flowchart matches deep black boxes |
| **Patients with unstable predictions** | 3 of 297 have 30–70% model disagreement |
| **Set growth** from ε=0.02 → ε=0.05 | 633 → ~2,246 models |

---

## Project Structure

```
rashomon-visualizer/
│
├── app.py                  # Main Streamlit application (7 interactive pages)
│
├── rashomon/
│   ├── __init__.py         # Makes rashomon/ a Python package
│   ├── train.py            # Data pipeline + Rashomon Set builder
│   └── analyze.py          # Four analyses on the Rashomon Set
│
├── data/
│   ├── .gitkeep            # Keeps the folder in git
│   └── heart.csv           # UCI Cleveland Heart Disease data (auto-downloaded)
│
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## The Seven Pages

| # | Page | What You'll See |
|---|------|-----------------|
| 1 | **Home** | Full explanation of the Rashomon Effect, 6-step visitor guide, three key findings |
| 2 | **Dataset** | All 297 patients in a scrollable table, feature dictionary, distribution charts |
| 3 | **Build the Set** | Scatter plot of 3,360 models with a live epsilon slider |
| 4 | **Patient View** | Donut chart showing model disagreement for any individual patient |
| 5 | **Features** | Mean ± std feature importance across all 633 Rashomon Set models |
| 6 | **Epsilon** | Area chart showing how set size explodes as tolerance grows |
| 7 | **About** | Technical pipeline, limitations, research paper connections |

---

## Technical Pipeline

### 1. Data

The [UCI Cleveland Heart Disease dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) is downloaded automatically on first run.

- **303 rows** in the original file → **297 patients** after dropping 6 rows with missing values (`?` in `ca` and `thal` columns)
- **13 input features**: age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG, max heart rate, exercise angina, ST depression, ST slope, major vessels, thalassemia
- **Target**: binary (0 = no disease, 1 = disease present)

### 2. Training 3,360 Models

Decision trees are trained across a full hyperparameter grid:

```
7 max_depths   × [2, 3, 4, 5, 6, 7, 8]
4 min_splits   × [2, 5, 10, 20]
4 min_leaves   × [1, 2, 5, 10]
30 seeds       × range(30)
─────────────────────────────────────
3,360 unique models
```

### 3. Evaluation: 5-Fold Cross Validation

Each model is evaluated with **5-Fold CV** rather than a single train/test split.

**Why?** A single train/test split on 297 patients gives only ~60 test patients. Accuracy steps are `1/60 = 0.0167` apart, meaning nearly every model lands at the same step and the Rashomon Set appears to include 100% of all models — a meaningless result. CV averages five test scores and gives much finer accuracy differences.

### 4. Finding True L*

`L*` is computed as the **maximum CV accuracy across all 3,360 models**, not from one hand-picked reference tree. The threshold is `L* − ε`. Every model above the threshold forms the Rashomon Set.

### 5. Honest Per-Patient Predictions

Predictions use `sklearn.cross_val_predict` — each patient is predicted by a model trained **without that patient** (out-of-fold). This avoids data leakage and gives genuine disagreement measurements. A separate `model.fit(X, y)` is called afterward only to populate `feature_importances_`.

### 6. Four Analyses

| Analysis | What It Reveals |
|----------|-----------------|
| **Patient Disagreement** | For each of 297 patients, what % of models predict disease |
| **Feature Importance** | Mean ± std importance per feature across 633 models |
| **Complexity vs Accuracy** | Which depth levels qualify and at what rate |
| **Epsilon Sweep** | Set size at 20 epsilon values from 0.005 to 0.100 |

---

## Installation & Running Locally

### Prerequisites

- Python 3.11+
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/rashomon-visualizer.git
cd rashomon-visualizer

# Create a virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

> **First load:** ~2 minutes to train 3,360 models and run analyses.  
> **Subsequent page clicks:** instant (cached in memory for the session).

---

## Deployment

The app is deployed on **Streamlit Community Cloud** (free):

1. Push the repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect the repo → set main file to `app.py`
4. Deploy

> Note: GitHub Pages will not work — Streamlit is a Python server, not static HTML.

---

## Dataset

**UCI Cleveland Heart Disease** — collected at the Cleveland Clinic Foundation by Robert Detrano, M.D., Ph.D.

| Column | Description |
|--------|-------------|
| `age` | Patient age in years |
| `sex` | 1 = Male, 0 = Female |
| `cp` | Chest pain type (0=typical angina, 1=atypical, 2=non-anginal, 3=asymptomatic) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1=yes, 0=no) |
| `restecg` | Resting ECG results (0=normal, 1=ST-T abnormality, 2=LV hypertrophy) |
| `thalach` | Maximum heart rate achieved during exercise |
| `exang` | Exercise-induced angina (1=yes, 0=no) |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | Slope of peak exercise ST segment (1–3) |
| `ca` | Number of major vessels colored by fluoroscopy (0–3) |
| `thal` | Thalassemia (3=normal, 6=fixed defect, 7=reversible defect) |
| `target` | **Label** — 0 = no disease, 1 = disease present |

---

## Limitations

**Decision trees only.** The real Rashomon Set also includes logistic regression, neural networks, SVMs, and ensembles. True multiplicity is far larger than what this tool demonstrates.

**Small dataset.** 297 patients is a small sample. CV accuracy varies significantly between folds (std ≈ 0.08), meaning Rashomon Set membership has some noise. A larger dataset would give more stable results.

**Few unstable patients.** Only 3 of 297 patients fall in the 30–70% model disagreement zone. This reflects the dominance of `thal` as a predictor in this specific dataset and the fact that 480 of 633 Rashomon Set models are depth-3 trees with similar structure. Expanding the model class would reveal more disagreement.

---

## Related Research

This project directly implements and visualizes concepts from:

- Semenova et al. — **"The Rashomon Set Has It All"** (NeurIPS 2025)  
  *Trustworthiness properties of trees under model multiplicity*

- Semenova et al. — **"The Double-Edged Nature of the Rashomon Set"** (arXiv 2025)  
  *Risks and opportunities of model multiplicity*

- Semenova & Rudin — **"On the Existence of Simpler Machine Learning Models"** (FAccT 2022)  
  *Simple models can match complex ones in accuracy*

- Fisher et al. — **"All Models Are Wrong, But Many Are Useful"** (ICML 2024)  
  *Using the Rashomon Set to find better models*

---

## Tech Stack

| Library | Version | Role |
|---------|---------|------|
| Python | 3.11 | Language |
| scikit-learn | 1.8.0 | Decision trees, cross-validation |
| Streamlit | 1.57.0 | Web application framework |
| Plotly | 6.7.0 | Interactive charts |
| Pandas | 3.0.3 | Data handling |
| NumPy | 2.4.6 | Numerical operations |

---

## Reproducibility

| Parameter | Value |
|-----------|-------|
| Patients | 297 |
| Input features | 13 |
| Models trained | 3,360 |
| Models in Rashomon Set (ε=0.02) | 633 |
| True L* | 0.7978 |
| Epsilon used | 0.02 |
| CV folds | 5 |
| Random seeds | 0–29 |

---

*Built to explore Prof. Lesia Semenova's research on model multiplicity at Rutgers University.*

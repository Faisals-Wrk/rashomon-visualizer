# =============================================================================
# train.py — Rashomon Set Builder
# =============================================================================
# Builds the Rashomon Set for the UCI Heart Disease dataset.
#
# Pipeline:
#   1. Download UCI Cleveland heart disease data (297 patients, 13 features)
#   2. Load and prepare features (X) and labels (y)
#   3. Train 3360 decision trees with varied hyperparameters
#   4. Evaluate each using 5-Fold Cross Validation for reliable accuracy
#   5. Find true L* = best accuracy across all trained models
#   6. Keep models within epsilon of L* → these form the Rashomon Set
#
# The Rashomon Set formalizes a key ML truth:
#   Many equally-accurate models exist, yet they disagree on individuals.
#   L* = best achievable accuracy | Rashomon Set = {f : accuracy(f) >= L* - ε}
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import urllib.request
import os


# -----------------------------------------------------------------------------
# SECTION 1 — DOWNLOAD DATA
# -----------------------------------------------------------------------------

def download_data():
    """
    Downloads the UCI Cleveland Heart Disease dataset and saves it
    as data/heart.csv with proper column headers and binary target.

    Raw file quirks handled here:
      - No header row  → we assign column names manually
      - Target is 0-4  → we convert to binary (0 = no disease, 1 = disease)
      - '?' for missing → we drop those rows (~6 out of 303)
    """

    os.makedirs('data', exist_ok=True)
    path = "data/heart.csv"

    if os.path.exists(path):
        print("Dataset already exists, skipping download.")
        return

    url = ("https://archive.ics.uci.edu/ml/machine-learning-databases"
           "/heart-disease/processed.cleveland.data")

    # Official column names for the 13 features + target
    uci_columns = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
        'restecg', 'thalach', 'exang', 'oldpeak',
        'slope', 'ca', 'thal', 'target'
    ]

    try:
        print("Downloading heart disease dataset from UCI repository...")
        temp_path = path + ".tmp"
        urllib.request.urlretrieve(url, temp_path)

        # Read raw file — no headers, '?' means missing value
        df_raw = pd.read_csv(
            temp_path,
            header=None,
            names=uci_columns,
            na_values='?'
        )

        # Convert target: 0 stays 0, values 1-4 all become 1 (disease present)
        df_raw['target'] = (df_raw['target'] > 0).astype(int)

        # Drop the ~6 rows with missing values in 'ca' and 'thal' columns
        df_raw = df_raw.dropna()

        print(f"  Rows: {len(df_raw)} | "
              f"Disease: {df_raw['target'].sum()} | "
              f"No disease: {(df_raw['target']==0).sum()}")

        df_raw.to_csv(path, index=False)
        os.remove(temp_path)
        print("Download complete.")

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise RuntimeError(
            f"Download failed: {e}\n"
            "Manually download processed.cleveland.data from UCI, "
            "add headers, binarize target, save as data/heart.csv"
        )


# -----------------------------------------------------------------------------
# SECTION 2 — LOAD DATA
# -----------------------------------------------------------------------------

def load_data():
    """
    Reads heart.csv and returns features, labels, and the full dataframe.

    Returns:
        X  (DataFrame): 297 × 13 — patient measurements (model inputs)
        y  (Series):    297 values — 0/1 disease label (what we predict)
        df (DataFrame): full table including target (for display in app)
    """

    download_data()
    df = pd.read_csv('data/heart.csv')

    # X = all columns except 'target' — these are the 13 input features
    X = df.drop('target', axis=1)

    # y = only the 'target' column — this is the ground truth we predict
    y = df['target']

    return X, y, df


# -----------------------------------------------------------------------------
# SECTION 3 — REFERENCE ACCURACY (quick preview)
# -----------------------------------------------------------------------------

def find_reference_accuracy(X, y):
    """
    Trains one decision tree with 5-Fold CV as a quick preview.
    The true L* is computed later inside build_rashomon_set()
    after all 3360 models are evaluated.

    Returns:
        mean_accuracy (float): this model's mean CV accuracy
        std_accuracy  (float): standard deviation across 5 folds
    """

    # A single well-tuned tree — depth 5 prevents overfitting
    reference_model = DecisionTreeClassifier(max_depth=5, random_state=42)

    # cross_val_score splits X,y into 5 folds, trains and tests 5 times,
    # returns an array of 5 accuracy scores
    cv_scores = cross_val_score(reference_model, X, y, cv=5,
                                scoring='accuracy')

    mean_accuracy = cv_scores.mean()
    std_accuracy  = cv_scores.std()

    print(f"  Sample CV scores: {[round(s,4) for s in cv_scores]}")
    print(f"  Sample accuracy:  {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")
    print(f"  (True L* determined after all models are trained)\n")

    return mean_accuracy, std_accuracy


# -----------------------------------------------------------------------------
# SECTION 4 — BUILD THE RASHOMON SET
# -----------------------------------------------------------------------------

def build_rashomon_set(X, y, epsilon=0.05):
    """
    Core function. Trains 3360 decision trees, evaluates each with
    5-Fold CV, finds the true best accuracy (L*), and returns all
    models that fall within epsilon of L* as the Rashomon Set.

    Why 3360 models?
        7 max_depths × 4 min_splits × 4 min_leaves × 30 seeds = 3360

    Why 5-Fold CV instead of a single train/test split?
        With only ~60 test patients, accuracy steps are 1/60 = 0.017 apart.
        Nearly every model lands at the same step → Rashomon Set = 100%.
        CV averages 5 test scores → much finer accuracy differences.

    Why find L* after training all models?
        L* must be the best accuracy in the entire model class, not just
        one hand-picked reference tree. We can only know the true peak
        after seeing all 3360 results.

    Args:
        X (DataFrame): full features table (297 × 13)
        y (Series):    full labels (297 values)
        epsilon (float): tolerance below L*. Default 0.05.
                         Models with accuracy >= L* - epsilon qualify.

    Returns:
        dict with keys: all_models, rashomon_models, reference_accuracy,
                        threshold, epsilon, total_trained, rashomon_size,
                        X, y
    """

    print("Quick preview with one reference model:")
    find_reference_accuracy(X, y)

    # ------------------------------------------------------------------
    # Hyperparameter grid — every combination creates a different tree
    # ------------------------------------------------------------------
    # max_depth:         how many questions the tree can ask (2=simple, 8=complex)
    # min_samples_split: minimum patients in a node before splitting is allowed
    # min_samples_leaf:  minimum patients that must land in each leaf node
    # random_seeds:      controls tie-breaking when splits are equally good
    max_depths   = [2, 3, 4, 5, 6, 7, 8]
    min_splits   = [2, 5, 10, 20]
    min_leaves   = [1, 2, 5, 10]
    random_seeds = range(30)

    total = (len(max_depths) * len(min_splits) *
             len(min_leaves) * len(random_seeds))

    print(f"Training {total} models with 5-Fold CV each...")
    print("(Each model runs 5 training cycles — takes ~1-2 minutes)\n")

    all_models = []
    count = 0

    for depth in max_depths:
        for min_split in min_splits:
            for min_leaf in min_leaves:
                for seed in random_seeds:

                    model = DecisionTreeClassifier(
                        max_depth=depth,
                        min_samples_split=min_split,
                        min_samples_leaf=min_leaf,
                        random_state=seed
                    )

                    # 5-Fold CV: trains 5 times, tests on each held-out fold,
                    # returns array of 5 accuracy scores
                    cv_scores = cross_val_score(
                        model, X, y, cv=5, scoring='accuracy'
                    )

                    all_models.append({
                        'model':             model,
                        'accuracy':          cv_scores.mean(),
                        'cv_scores':         cv_scores.tolist(),
                        'cv_std':            cv_scores.std(),
                        'max_depth':         depth,
                        'min_samples_split': min_split,
                        'min_samples_leaf':  min_leaf,
                        'random_state':      seed,
                        'in_rashomon_set':   False   # updated below
                    })

                    count += 1
                    if count % 500 == 0:
                        print(f"  Trained {count}/{total} models...")

    # ------------------------------------------------------------------
    # Find true L* — the peak accuracy across ALL 3360 models
    # ------------------------------------------------------------------
    # This is the correct definition: L* = loss of the best model
    # in the hypothesis class. We measure from this true peak,
    # not from one arbitrarily chosen reference tree.
    true_L_star = max(m['accuracy'] for m in all_models)
    threshold   = true_L_star - epsilon

    print(f"\nTrue L*:         {true_L_star:.4f}")
    print(f"Epsilon:         {epsilon}")
    print(f"Threshold:       {threshold:.4f}")
    print(f"Qualifying rule: accuracy >= {threshold:.4f}\n")

    # ------------------------------------------------------------------
    # Apply Rashomon Set filter
    # ------------------------------------------------------------------
    for m in all_models:
        m['in_rashomon_set'] = bool(m['accuracy'] >= threshold)

    rashomon_models = [m for m in all_models if m['in_rashomon_set']]

    print(f"Total trained:          {len(all_models)}")
    print(f"In Rashomon Set:        {len(rashomon_models)}")
    print(f"Outside Rashomon Set:   {len(all_models) - len(rashomon_models)}")

    # ------------------------------------------------------------------
    # Generate per-patient predictions for every Rashomon Set model
    # ------------------------------------------------------------------
    # We fit each qualifying model on the FULL dataset and store its
    # predictions for all 297 patients. analyze.py uses these to answer:
    # "For patient John, what fraction of Rashomon Set models predict disease?"
    # A 95% agreement = we're confident. A 50% split = deeply uncertain.
    print("\nGenerating per-patient predictions for Rashomon Set models...")

    for m in rashomon_models:
        m['model'].fit(X, y)
        m['predictions'] = m['model'].predict(X).tolist()

    # Accuracy summary
    in_accs  = [m['accuracy'] for m in rashomon_models]
    out_accs = [m['accuracy'] for m in all_models if not m['in_rashomon_set']]

    print(f"\nRashomon Set accuracy — "
          f"Min: {min(in_accs):.4f}  "
          f"Max: {max(in_accs):.4f}  "
          f"Mean: {sum(in_accs)/len(in_accs):.4f}")
    if out_accs:
        print(f"Outside Set accuracy  — "
              f"Min: {min(out_accs):.4f}  "
              f"Max: {max(out_accs):.4f}")

    return {
        'all_models':         all_models,
        'rashomon_models':    rashomon_models,
        'reference_accuracy': true_L_star,
        'threshold':          threshold,
        'epsilon':            epsilon,
        'total_trained':      len(all_models),
        'rashomon_size':      len(rashomon_models),
        'X':                  X,
        'y':                  y
    }


# -----------------------------------------------------------------------------
# SECTION 5 — MAIN (run this file directly to test)
# -----------------------------------------------------------------------------
# Only executes when you run: python rashomon/train.py
# Skipped when train.py is imported by app.py or analyze.py

if __name__ == "__main__":

    print("=" * 50)
    print("RASHOMON SET BUILDER")
    print("=" * 50)

    print("\nStep 1: Loading data...")
    X, y, df = load_data()
    print(f"Patients: {df.shape[0]} | Features: {df.shape[1]-1}")
    print(f"Disease: {y.sum()} | No disease: {(y==0).sum()}")

    print("\nStep 2: Building Rashomon Set (epsilon=0.02)...")
    results = build_rashomon_set(X, y, epsilon=0.02)

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"True L*:             {results['reference_accuracy']:.4f}")
    print(f"Threshold:           {results['threshold']:.4f}")
    print(f"Total trained:       {results['total_trained']}")
    print(f"Rashomon Set size:   {results['rashomon_size']}")
    print(f"Rashomon fraction:   "
          f"{results['rashomon_size']/results['total_trained']:.1%}")
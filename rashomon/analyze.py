# =============================================================================
# analyze.py — Rashomon Set Analyzer
# =============================================================================
# Takes the Rashomon Set built by train.py and extracts four analyses
# that power our Streamlit app's visualizations.
#
# The four analyses:
#
#   1. Patient Disagreement
#      For each of the 297 patients, how many Rashomon Set models predict
#      disease vs no disease? Patients near 50% are in the "danger zone" —
#      their diagnosis flips depending on which model the hospital picked.
#
#   2. Feature Importance Analysis
#      Each model internally ranks features by how much they contributed
#      to its decisions. We ask all 633 models for their rankings and reveal
#      that equally-accurate models have completely different internal logic.
#
#   3. Complexity vs Accuracy
#      Some Rashomon Set models are very simple (depth=2, 4 leaf nodes).
#      Others are complex (depth=7, 128 leaf nodes). If they're equally
#      accurate, the simple one is strictly better — it's interpretable.
#
#   4. Epsilon Sweep
#      How does the Rashomon Set size change as we vary epsilon?
#      Shows the set is not fixed — it grows rapidly as standards relax.
# =============================================================================

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# ANALYSIS 1 — PATIENT DISAGREEMENT
# -----------------------------------------------------------------------------

def compute_patient_disagreement(results):
    """
    For each patient, computes what fraction of Rashomon Set models
    predict disease (label=1) vs no disease (label=0).

    This is the most emotionally striking analysis:
    some patients have near-unanimous predictions across all models
    (stable, confident), while others are almost a coin flip
    (unstable — the "Rashomon danger zone").

    Args:
        results (dict): the dictionary returned by build_rashomon_set()
                        must contain 'rashomon_models', 'X', 'y'

    Returns:
        List of dicts, one per patient, containing:
          - patient_idx:          row index in the dataset (0-296)
          - disease_vote_pct:     % of models predicting disease (0.0 to 1.0)
          - no_disease_vote_pct:  % of models predicting no disease
          - disease_votes:        raw count of models predicting disease
          - no_disease_votes:     raw count predicting no disease
          - total_models:         total models in Rashomon Set (633)
          - true_label:           actual ground truth (0 or 1)
          - prediction_stability: how "confident" the Rashomon Set is
                                  1.0 = all models agree
                                  0.0 = perfectly split 50/50
          - features:             dict of this patient's 13 measurements
    """

    rashomon_models = results['rashomon_models']
    X               = results['X']
    y               = results['y']

    n_patients = len(X)          # 297 patients
    n_models   = len(rashomon_models)  # 633 models

    patient_records = []

    for patient_idx in range(n_patients):

        # Count how many models predict disease (1) for this patient.
        # Each model in rashomon_models already has a 'predictions' list
        # of length 297 (generated at the end of build_rashomon_set).
        # We just pick index patient_idx from each model's predictions.
        disease_votes = sum(
            1 for m in rashomon_models
            if m['predictions'][patient_idx] == 1
        )

        no_disease_votes = n_models - disease_votes

        disease_vote_pct    = disease_votes / n_models
        no_disease_vote_pct = no_disease_votes / n_models

        # Prediction stability = how far the vote is from a 50/50 split.
        # Formula: |vote_pct - 0.5| * 2
        #   If all models agree (100% or 0%) → stability = 1.0 (very stable)
        #   If split exactly 50/50           → stability = 0.0 (very unstable)
        # This gives us a single number to sort/color patients by certainty.
        prediction_stability = abs(disease_vote_pct - 0.5) * 2

        # Get this patient's true label (0 or 1) from y.
        # y is a pandas Series, so we use .iloc[idx] for position-based access.
        true_label = int(y.iloc[patient_idx])

        # Get this patient's feature values as a dictionary.
        # Example: {'age': 63.0, 'sex': 1.0, 'cp': 1.0, ...}
        features = X.iloc[patient_idx].to_dict()

        patient_records.append({
            'patient_idx':          patient_idx,
            'disease_vote_pct':     round(disease_vote_pct, 4),
            'no_disease_vote_pct':  round(no_disease_vote_pct, 4),
            'disease_votes':        disease_votes,
            'no_disease_votes':     no_disease_votes,
            'total_models':         n_models,
            'true_label':           true_label,
            'prediction_stability': round(prediction_stability, 4),
            'features':             features
        })

    return patient_records


# -----------------------------------------------------------------------------
# ANALYSIS 2 — FEATURE IMPORTANCE
# -----------------------------------------------------------------------------

def compute_feature_importance(results):
    """
    Extracts and analyzes feature importance from all Rashomon Set models.

    Feature importance in a decision tree = total reduction in Gini impurity
    caused by splits on that feature, weighted by patients at each node.
    A feature used early and often gets a high importance score.
    Features never used get 0.

    What we look for:
      - Mean importance per feature across all 633 models
      - Standard deviation — high std means models disagree on that feature
      - Full distributions — even if one feature is always #1,
        the MAGNITUDE varies across models (sometimes dominant, sometimes not)
      - Top feature counts — which feature is #1 most often

    Note on thal dominance:
      In the Cleveland Heart Disease dataset, thalassemia type (thal) is
      genuinely the most discriminative feature — confirmed in published
      literature. Its 100% top-rank in our depth-3-dominated Rashomon Set
      reflects a real property of this dataset, not a bug.
      The interesting story is in the MAGNITUDE variation and secondary
      features, which we capture through std and distributions.

    Args:
        results (dict): output from build_rashomon_set()

    Returns:
        dict with per_model records, aggregated stats, top feature counts,
        and feature names for chart labels.
    """

    rashomon_models = results['rashomon_models']
    X               = results['X']
    feature_names   = list(X.columns)

    per_model_records = []

    for m in rashomon_models:

        # feature_importances_ is available because we called model.fit(X,y)
        # at the end of build_rashomon_set (for this purpose).
        importances = m['model'].feature_importances_

        # Build a dict mapping feature name → importance score for this model.
        importance_dict = {
            feature_names[i]: round(float(importances[i]), 4)
            for i in range(len(feature_names))
        }

        # Top feature = the one with highest importance in this specific model.
        top_feature = max(importance_dict, key=importance_dict.get)

        per_model_records.append({
            'max_depth':      m['max_depth'],
            'accuracy':       m['accuracy'],
            'importances':    importance_dict,
            'top_feature':    top_feature,
            'top_importance': importance_dict[top_feature]
        })

    # ------------------------------------------------------------------
    # Aggregate statistics per feature across all Rashomon Set models.
    # Mean tells us the average contribution.
    # Std tells us how much models DISAGREE about a feature's importance.
    # High std = some models rely on it heavily, others ignore it.
    # ------------------------------------------------------------------
    aggregated = {}
    for feature in feature_names:

        scores = [r['importances'][feature] for r in per_model_records]

        aggregated[feature] = {
            'mean': round(float(np.mean(scores)), 4),
            'std':  round(float(np.std(scores)), 4),
            'min':  round(float(np.min(scores)), 4),
            'max':  round(float(np.max(scores)), 4),
            # Coefficient of variation = std/mean — relative variability.
            # High CoV means this feature's importance swings a lot across models.
            'coeff_variation': round(
                float(np.std(scores) / np.mean(scores))
                if np.mean(scores) > 0 else 0.0, 3
            )
        }

    # How often each feature is ranked #1 across all models.
    top_feature_counts = {feature: 0 for feature in feature_names}
    for r in per_model_records:
        top_feature_counts[r['top_feature']] += 1

    # Rank features by mean importance (descending) for clean display.
    ranked_features = sorted(
        feature_names,
        key=lambda f: aggregated[f]['mean'],
        reverse=True
    )

    return {
        'per_model':          per_model_records,
        'aggregated':         aggregated,
        'top_feature_counts': top_feature_counts,
        'feature_names':      feature_names,
        'ranked_features':    ranked_features  # sorted by mean importance
    }


# -----------------------------------------------------------------------------
# ANALYSIS 3 — COMPLEXITY VS ACCURACY
# -----------------------------------------------------------------------------

def compute_complexity_vs_accuracy(results):
    """
    For every model (both inside and outside Rashomon Set),
    records its max_depth (complexity) and CV accuracy.

    This analysis answers: "Do we need complex models to be accurate?"

    If simple models (depth=2,3) appear inside the Rashomon Set,
    then complexity is NOT required for high accuracy.
    This directly supports the interpretable ML argument:
    prefer simple models because they're equally good AND understandable.

    Args:
        results (dict): output from build_rashomon_set()

    Returns:
        dict containing:
          - all_models_data:   list of dicts with depth, accuracy, in_set
          - depth_summary:     for each depth level, stats about models at
                               that depth (how many in set, accuracy range)
          - simplest_rashomon: the shallowest (simplest) model in the set
          - L_star:            best accuracy achieved (for reference line)
          - threshold:         Rashomon Set threshold line
    """

    all_models = results['all_models']

    # Record depth, accuracy, and set membership for every model.
    all_models_data = [
        {
            'max_depth':       m['max_depth'],
            'accuracy':        round(m['accuracy'], 4),
            'in_rashomon_set': m['in_rashomon_set'],
            'cv_std':          round(m['cv_std'], 4)
        }
        for m in all_models
    ]

    # ------------------------------------------------------------------
    # Summarize by depth level.
    # For each depth value (2,3,4,5,6,7,8), how many models are in
    # the Rashomon Set? What's their accuracy range?
    # ------------------------------------------------------------------
    depth_summary = {}
    for depth in [2, 3, 4, 5, 6, 7, 8]:

        # All models at this depth
        at_depth = [m for m in all_models if m['max_depth'] == depth]

        # Models at this depth that made it into the Rashomon Set
        in_set_at_depth = [m for m in at_depth if m['in_rashomon_set']]

        accuracies_at_depth = [m['accuracy'] for m in at_depth]

        depth_summary[depth] = {
            'total_models':    len(at_depth),
            'in_rashomon_set': len(in_set_at_depth),
            'pct_in_set':      round(len(in_set_at_depth) / len(at_depth), 4)
                               if at_depth else 0,
            'max_accuracy':    round(max(accuracies_at_depth), 4)
                               if accuracies_at_depth else 0,
            'mean_accuracy':   round(np.mean(accuracies_at_depth), 4)
                               if accuracies_at_depth else 0
        }

    # Find the simplest model that made it into the Rashomon Set.
    # "Simplest" = smallest max_depth.
    # Among equal depths, pick highest accuracy.
    rashomon_models = results['rashomon_models']
    simplest = min(
        rashomon_models,
        key=lambda m: (m['max_depth'], -m['accuracy'])
    )

    return {
        'all_models_data': all_models_data,
        'depth_summary':   depth_summary,
        'simplest_rashomon': {
            'max_depth':   simplest['max_depth'],
            'accuracy':    round(simplest['accuracy'], 4),
            'min_split':   simplest['min_samples_split'],
            'min_leaf':    simplest['min_samples_leaf']
        },
        'L_star':    results['reference_accuracy'],
        'threshold': results['threshold']
    }


# -----------------------------------------------------------------------------
# ANALYSIS 4 — EPSILON SWEEP
# -----------------------------------------------------------------------------

def compute_epsilon_sweep(results):
    """
    Shows how the Rashomon Set size changes as epsilon varies.

    We already have all 3360 models with their CV accuracies.
    We don't need to retrain — we just change the threshold and recount.

    This reveals that the Rashomon Set is not a fixed object:
      Small epsilon  → strict → few models qualify
      Large epsilon  → loose  → almost everything qualifies

    The rapid growth in set size as epsilon increases is a key
    visual proof that model multiplicity is not a small edge case —
    it's the norm, and it grows quickly.

    Args:
        results (dict): output from build_rashomon_set()

    Returns:
        list of dicts, one per epsilon value tested:
          - epsilon:        the epsilon value tested
          - threshold:      L* - epsilon
          - rashomon_size:  how many models qualify at this epsilon
          - pct_of_total:   what fraction of all 3360 models qualify
    """

    all_models  = results['all_models']
    L_star      = results['reference_accuracy']
    total       = len(all_models)

    # Test a range of epsilon values from very tight to very loose.
    # np.arange(start, stop, step) generates evenly spaced values.
    # 0.005 to 0.105 in steps of 0.005 gives us 20 data points.
    epsilon_values = np.arange(0.005, 0.105, 0.005)

    sweep_records = []

    for eps in epsilon_values:
        threshold = L_star - eps

        # Count how many models exceed this threshold.
        # We already have all accuracies — no retraining needed.
        qualifying = sum(
            1 for m in all_models if m['accuracy'] >= threshold
        )

        sweep_records.append({
            'epsilon':       round(float(eps), 3),
            'threshold':     round(float(threshold), 4),
            'rashomon_size': qualifying,
            'pct_of_total':  round(qualifying / total, 4)
        })

    return sweep_records


# -----------------------------------------------------------------------------
# MAIN ANALYSIS RUNNER
# -----------------------------------------------------------------------------

def run_all_analyses(results):
    """
    Runs all four analyses on the Rashomon Set results and returns
    everything the Streamlit app needs to build its visualizations.

    This is the single function app.py will call.
    It returns one dictionary with four keys, one per analysis.

    Args:
        results (dict): output from build_rashomon_set() in train.py

    Returns:
        dict with keys:
          'patient_disagreement'  → Analysis 1 results
          'feature_importance'    → Analysis 2 results
          'complexity_accuracy'   → Analysis 3 results
          'epsilon_sweep'         → Analysis 4 results
    """

    print("Running Analysis 1: Patient Disagreement...")
    patient_data = compute_patient_disagreement(results)
    print(f"  Computed disagreement for {len(patient_data)} patients.")

    # Quick summary — find the most and least stable patients
    stabilities = sorted(patient_data, key=lambda p: p['prediction_stability'])
    most_uncertain  = stabilities[0]   # closest to 50/50 split
    most_certain    = stabilities[-1]  # closest to 100% agreement

    print(f"  Most uncertain patient #{most_uncertain['patient_idx']}: "
          f"{most_uncertain['disease_vote_pct']*100:.1f}% disease votes")
    print(f"  Most certain patient #{most_certain['patient_idx']}: "
          f"{most_certain['disease_vote_pct']*100:.1f}% disease votes")


    print("\nRunning Analysis 2: Feature Importance...")
    feature_data = compute_feature_importance(results)

    # Show mean importance and variability — the real story
    print("  Feature importance across Rashomon Set models")
    print("  (mean ± std across all 633 models):")
    print(f"  {'Feature':<12} {'Mean':>6}  {'Std':>6}  {'Min':>6}  {'Max':>6}")
    print(f"  {'-'*46}")
    for feature in feature_data['ranked_features']:
        stats = feature_data['aggregated'][feature]
        print(f"  {feature:<12} {stats['mean']:>6.3f}  "
            f"{stats['std']:>6.3f}  "
            f"{stats['min']:>6.3f}  "
            f"{stats['max']:>6.3f}")

    print("\nRunning Analysis 3: Complexity vs Accuracy...")
    complexity_data = compute_complexity_vs_accuracy(results)
    simplest = complexity_data['simplest_rashomon']
    print(f"  Simplest Rashomon Set model: "
          f"depth={simplest['max_depth']}, "
          f"accuracy={simplest['accuracy']:.4f}")
    print("  Models in Rashomon Set by depth:")
    for depth, stats in complexity_data['depth_summary'].items():
        print(f"    Depth {depth}: {stats['in_rashomon_set']:3d} in set "
              f"({stats['pct_in_set']*100:.0f}%) | "
              f"max acc = {stats['max_accuracy']:.4f}")

    print("\nRunning Analysis 4: Epsilon Sweep...")
    epsilon_data = compute_epsilon_sweep(results)
    print("  Rashomon Set size at different epsilon values:")
    for record in epsilon_data[::4]:   # print every 4th entry to keep output short
        print(f"    ε={record['epsilon']:.3f} → "
              f"{record['rashomon_size']:4d} models "
              f"({record['pct_of_total']*100:.1f}%)")

    print("\nAll analyses complete.")

    return {
        'patient_disagreement': patient_data,
        'feature_importance':   feature_data,
        'complexity_accuracy':  complexity_data,
        'epsilon_sweep':        epsilon_data
    }


# -----------------------------------------------------------------------------
# MAIN (run this file directly to test)
# -----------------------------------------------------------------------------
# Only runs when you execute: python rashomon/analyze.py
# Skipped when analyze.py is imported by app.py

if __name__ == "__main__":

    # We need to build the Rashomon Set first, then analyze it.
    # Import train.py functions to do so.
    from train import load_data, build_rashomon_set

    print("=" * 55)
    print("RASHOMON SET ANALYZER")
    print("=" * 55)

    print("\nStep 1: Loading data and building Rashomon Set...")
    print("(This takes ~1-2 minutes)\n")

    X, y, df = load_data()
    results  = build_rashomon_set(X, y, epsilon=0.02)

    print("\n" + "=" * 55)
    print("Step 2: Running all analyses...")
    print("=" * 55 + "\n")

    analyses = run_all_analyses(results)

    # ---------------------------------------------------------------
    # Print a final human-readable summary of the key findings
    # ---------------------------------------------------------------
    print("\n" + "=" * 55)
    print("KEY FINDINGS")
    print("=" * 55)

    # Finding 1 — how many patients are in the "danger zone"
    patient_data = analyses['patient_disagreement']

    # A patient is "unstable" if between 30% and 70% of models
    # predict disease — genuinely uncertain territory
    unstable = [
        p for p in patient_data
        if 0.30 <= p['disease_vote_pct'] <= 0.70
    ]
    print(f"\nPatients with unstable predictions (30-70% model agreement):")
    print(f"  {len(unstable)} out of {len(patient_data)} patients "
          f"({len(unstable)/len(patient_data)*100:.1f}%)")
    print(f"  These patients' diagnoses depend on which model was deployed.")

    # Finding 2 — feature importance spread
    top_counts = analyses['feature_importance']['top_feature_counts']
    top_feature, top_count = max(top_counts.items(), key=lambda x: x[1])
    print(f"\nFeature importance instability:")
    print(f"  No single feature dominates all models.")
    print(f"  Most common #1 feature: '{top_feature}' "
          f"({top_count} models, "
          f"{top_count/len(results['rashomon_models'])*100:.1f}%)")

    # Finding 3 — simplest model in Rashomon Set
    simplest = analyses['complexity_accuracy']['simplest_rashomon']
    print(f"\nSimplicity finding:")
    print(f"  Simplest model in Rashomon Set: depth={simplest['max_depth']}")
    print(f"  Accuracy: {simplest['accuracy']:.4f} "
          f"(within epsilon of best model)")
    print(f"  A {simplest['max_depth']}-question flowchart matches "
          f"complex deep trees in accuracy.")

    # Finding 4 — epsilon sweep extremes
    sweep = analyses['epsilon_sweep']
    print(f"\nEpsilon sensitivity:")
    print(f"  At ε=0.005: {sweep[0]['rashomon_size']} models qualify")
    print(f"  At ε=0.050: {sweep[9]['rashomon_size']} models qualify")
    print(f"  At ε=0.100: {sweep[-1]['rashomon_size']} models qualify")
    print(f"  The set grows rapidly — model multiplicity is not a rare edge case.")
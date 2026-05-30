"""
Stacking: Meta-learner on OOF predictions
Pattern Recognition Project (Final: 5/31)

기존 OOF/test prob .npy 파일을 활용하여
meta-learner(Logistic Regression, Ridge, LightGBM)로 성능 향상 시도.

사용법:
1. 필요 파일: train.csv, oof_lgbm.npy, oof_catboost.npy, oof_xgboost.npy,
              prob_lgbm.npy, prob_catboost.npy, prob_xgb.npy
2. python stacking.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 1. 데이터 로드
# ============================================================
print("Loading data...")

# OOF predictions (train set에 대한 out-of-fold 예측)
oof_lgbm = np.load("oof_lgbm.npy")
oof_catboost = np.load("oof_catboost.npy")
oof_xgboost = np.load("oof_xgboost.npy")

# Test predictions (multi-seed averaged)
prob_lgbm = np.load("prob_lgbm.npy")
prob_catboost = np.load("prob_catboost.npy")
prob_xgb = np.load("prob_xgb.npy")

# Target
train_df = pd.read_csv("train.csv", na_values=["", " "])
train_df["income"] = train_df["income"].apply(lambda x: 1 if ">50K" in str(x) else 0)
y = train_df["income"].values

# Test IDs
test_df = pd.read_csv("test.csv")
test_ids = test_df["id"]

print(f"  OOF shape: {oof_lgbm.shape[0]}")
print(f"  Test shape: {prob_lgbm.shape[0]}")
print(f"  Target distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
print()


# ============================================================
# 2. Stacking Features 구성
# ============================================================

# Level 1: 3개 모델 OOF predictions
X_stack = np.column_stack([oof_lgbm, oof_catboost, oof_xgboost])
X_test_stack = np.column_stack([prob_lgbm, prob_catboost, prob_xgb])

# Level 1+: 추가 파생 features (모델 간 차이, 평균 등)
X_stack_ext = np.column_stack([
    oof_lgbm, oof_catboost, oof_xgboost,
    oof_lgbm - oof_catboost,           # LGBM vs Cat 차이
    oof_lgbm - oof_xgboost,            # LGBM vs XGB 차이
    oof_catboost - oof_xgboost,        # Cat vs XGB 차이
    (oof_lgbm + oof_catboost + oof_xgboost) / 3,  # 평균
    np.max(np.column_stack([oof_lgbm, oof_catboost, oof_xgboost]), axis=1),  # 최대
    np.min(np.column_stack([oof_lgbm, oof_catboost, oof_xgboost]), axis=1),  # 최소
    np.std(np.column_stack([oof_lgbm, oof_catboost, oof_xgboost]), axis=1),  # 표준편차 (모델 불일치도)
])

X_test_ext = np.column_stack([
    prob_lgbm, prob_catboost, prob_xgb,
    prob_lgbm - prob_catboost,
    prob_lgbm - prob_xgb,
    prob_catboost - prob_xgb,
    (prob_lgbm + prob_catboost + prob_xgb) / 3,
    np.max(np.column_stack([prob_lgbm, prob_catboost, prob_xgb]), axis=1),
    np.min(np.column_stack([prob_lgbm, prob_catboost, prob_xgb]), axis=1),
    np.std(np.column_stack([prob_lgbm, prob_catboost, prob_xgb]), axis=1),
])

feature_names_3 = ["lgbm", "catboost", "xgboost"]
feature_names_ext = feature_names_3 + ["lgbm-cat", "lgbm-xgb", "cat-xgb", "mean", "max", "min", "std"]


# ============================================================
# 3. 다양한 Meta-learner 비교
# ============================================================

def evaluate_meta(name, X_meta, y, X_test_meta, model_fn, feature_names):
    """5-Fold CV로 meta-learner 평가 + test 예측"""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_meta = np.zeros(len(y))
    test_meta = np.zeros(len(X_test_meta))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_meta, y)):
        X_tr, X_val = X_meta[tr_idx], X_meta[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test_meta)

        model = model_fn()
        model.fit(X_tr_s, y_tr)

        if hasattr(model, "predict_proba"):
            oof_meta[val_idx] = model.predict_proba(X_val_s)[:, 1]
            test_meta += model.predict_proba(X_test_s)[:, 1] / 5
        else:
            oof_meta[val_idx] = model.decision_function(X_val_s)
            test_meta += model.decision_function(X_test_s) / 5

    # AUC
    auc = roc_auc_score(y, oof_meta)

    # Threshold tuning
    best_f1, best_thr = 0, 0.5
    for thr in np.arange(0.20, 0.80, 0.001):
        f1 = f1_score(y, (oof_meta >= thr).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    score = (best_f1 + auc) / 2

    return {
        "name": name,
        "f1": best_f1,
        "auc": auc,
        "score": score,
        "threshold": best_thr,
        "oof": oof_meta,
        "test": test_meta,
    }


print("=" * 60)
print("Meta-learner Comparison")
print("=" * 60)
print()

results = []

# --- Baseline: 단순 가중 평균 (blend_comparison 결과 재현) ---
blend_simple = 0.40 * oof_lgbm + 0.49 * oof_catboost + 0.11 * oof_xgboost
auc_simple = roc_auc_score(y, blend_simple)
best_f1_simple, best_thr_simple = 0, 0.5
for thr in np.arange(0.20, 0.80, 0.001):
    f1 = f1_score(y, (blend_simple >= thr).astype(int))
    if f1 > best_f1_simple:
        best_f1_simple = f1
        best_thr_simple = thr
score_simple = (best_f1_simple + auc_simple) / 2

test_blend_simple = 0.40 * prob_lgbm + 0.49 * prob_catboost + 0.11 * prob_xgb
results.append({
    "name": "Baseline (weighted avg 40:49:11)",
    "f1": best_f1_simple, "auc": auc_simple, "score": score_simple,
    "threshold": best_thr_simple, "oof": blend_simple, "test": test_blend_simple,
})

# --- Meta-learner 1: LogisticRegression (3 features) ---
results.append(evaluate_meta(
    "LR (3 features)", X_stack, y, X_test_stack,
    lambda: LogisticRegression(C=1.0, max_iter=5000, random_state=42),
    feature_names_3
))

# --- Meta-learner 2: LogisticRegression (10 extended features) ---
results.append(evaluate_meta(
    "LR (10 ext features)", X_stack_ext, y, X_test_ext,
    lambda: LogisticRegression(C=1.0, max_iter=5000, random_state=42),
    feature_names_ext
))

# --- Meta-learner 3: LR with different C values ---
for C in [0.01, 0.1, 10.0]:
    results.append(evaluate_meta(
        f"LR C={C} (3 feat)", X_stack, y, X_test_stack,
        lambda c=C: LogisticRegression(C=c, max_iter=5000, random_state=42),
        feature_names_3
    ))

# --- Meta-learner 4: LR with class_weight ---
results.append(evaluate_meta(
    "LR balanced (3 feat)", X_stack, y, X_test_stack,
    lambda: LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced", random_state=42),
    feature_names_3
))

results.append(evaluate_meta(
    "LR balanced (10 ext)", X_stack_ext, y, X_test_ext,
    lambda: LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced", random_state=42),
    feature_names_ext
))

# --- Meta-learner 5: Ridge ---
results.append(evaluate_meta(
    "Ridge (3 feat)", X_stack, y, X_test_stack,
    lambda: RidgeClassifier(alpha=1.0, random_state=42),
    feature_names_3
))

# --- LightGBM meta-learner ---
try:
    import lightgbm as lgb_meta

    def lgbm_meta_fn():
        return lgb_meta.LGBMClassifier(
            n_estimators=100, max_depth=3, num_leaves=8,
            learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=-1, n_jobs=-1,
        )

    results.append(evaluate_meta(
        "LightGBM meta (3 feat)", X_stack, y, X_test_stack,
        lgbm_meta_fn, feature_names_3
    ))
    results.append(evaluate_meta(
        "LightGBM meta (10 ext)", X_stack_ext, y, X_test_ext,
        lgbm_meta_fn, feature_names_ext
    ))
except ImportError:
    print("  LightGBM not available for meta-learner")

# --- 결과 출력 ---
print(f"  {'Model':<35s} {'F1':>7s} {'AUC':>7s} {'Score':>7s} {'Thr':>6s}")
print("  " + "-" * 65)
for r in sorted(results, key=lambda x: x["score"], reverse=True):
    marker = " ★" if r["score"] == max(x["score"] for x in results) else ""
    print(f"  {r['name']:<35s} {r['f1']:>7.4f} {r['auc']:>7.4f} {r['score']:>7.4f} {r['threshold']:>6.3f}{marker}")

print()

# ============================================================
# 4. Best 결과로 prediction.csv 생성
# ============================================================
best = max(results, key=lambda x: x["score"])

print("=" * 60)
print(f"★ BEST: {best['name']}")
print("=" * 60)
print(f"  F1:    {best['f1']:.4f}")
print(f"  AUC:   {best['auc']:.4f}")
print(f"  Score: {best['score']:.4f}")
print(f"  Threshold: {best['threshold']:.3f}")
print()

# Improvement over baseline
delta = best["score"] - score_simple
print(f"  vs Baseline (weighted avg): {delta:+.4f}")
print()

# prediction.csv
test_prob = best["test"]

# 만약 decision_function 기반이면 sigmoid 변환
if test_prob.min() < 0 or test_prob.max() > 1:
    test_prob = 1 / (1 + np.exp(-test_prob))

y_cls = (test_prob >= best["threshold"]).astype(int)

prediction = pd.DataFrame({
    "id": test_ids,
    "y_cls": y_cls,
    "y_prob": test_prob,
})

prediction.to_csv("prediction_stacking.csv", index=False)
print(f"prediction_stacking.csv saved! ({len(prediction)} rows)")
print(f"  y_cls distribution: {prediction['y_cls'].value_counts().to_dict()}")
print(f"  y_prob mean: {prediction['y_prob'].mean():.4f}")
print()

# 기존 블렌딩과 비교용
prediction_blend = pd.DataFrame({
    "id": test_ids,
    "y_cls": (test_blend_simple >= best_thr_simple).astype(int),
    "y_prob": test_blend_simple,
})
prediction_blend.to_csv("prediction_blend.csv", index=False)
print(f"prediction_blend.csv saved! (baseline weighted avg)")
print()

# ============================================================
# 5. Multi-seed Stacking (best meta-learner)
# ============================================================
print("=" * 60)
print("Multi-seed Stacking (5 seeds)")
print("=" * 60)

# best가 baseline이면 skip
if "Baseline" in best["name"]:
    print("  Best is already baseline (weighted avg). Skip multi-seed stacking.")
    print("  Use prediction_blend.csv as final submission.")
else:
    # 결정: 3-feature vs 10-feature
    if "ext" in best["name"] or "10" in best["name"]:
        X_meta_final = X_stack_ext
        X_test_meta_final = X_test_ext
    else:
        X_meta_final = X_stack
        X_test_meta_final = X_test_stack

    seeds = [42, 123, 456, 789, 2024]
    test_probs_all = np.zeros(len(X_test_meta_final))

    for seed in seeds:
        skf_seed = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (tr_idx, val_idx) in enumerate(skf_seed.split(X_meta_final, y)):
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_meta_final[tr_idx])
            X_test_s = scaler.transform(X_test_meta_final)

            # best model 재생성
            if "balanced" in best["name"]:
                model = LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced", random_state=seed)
            elif "LightGBM" in best["name"]:
                model = lgb_meta.LGBMClassifier(
                    n_estimators=100, max_depth=3, num_leaves=8,
                    learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                    random_state=seed, verbosity=-1, n_jobs=-1)
            elif "Ridge" in best["name"]:
                model = RidgeClassifier(alpha=1.0, random_state=seed)
            else:
                C_val = 1.0
                for c in [0.01, 0.1, 10.0]:
                    if f"C={c}" in best["name"]:
                        C_val = c
                        break
                model = LogisticRegression(C=C_val, max_iter=5000, random_state=seed)

            model.fit(X_tr_s, y[tr_idx])

            if hasattr(model, "predict_proba"):
                test_probs_all += model.predict_proba(X_test_s)[:, 1] / (5 * len(seeds))
            else:
                test_probs_all += model.decision_function(X_test_s) / (5 * len(seeds))

        print(f"  Seed {seed} done")

    # sigmoid if needed
    if test_probs_all.min() < 0 or test_probs_all.max() > 1:
        test_probs_all = 1 / (1 + np.exp(-test_probs_all))

    y_cls_ms = (test_probs_all >= best["threshold"]).astype(int)
    prediction_ms = pd.DataFrame({
        "id": test_ids,
        "y_cls": y_cls_ms,
        "y_prob": test_probs_all,
    })
    prediction_ms.to_csv("prediction_stacking_multiseed.csv", index=False)
    print(f"\nprediction_stacking_multiseed.csv saved!")
    print(f"  y_cls distribution: {prediction_ms['y_cls'].value_counts().to_dict()}")
    print(f"  y_prob mean: {prediction_ms['y_prob'].mean():.4f}")

print()
print("=" * 60)
print("Summary of prediction files:")
print("=" * 60)
print("  prediction_blend.csv              — 기존 가중 평균 (40:49:11)")
print("  prediction_stacking.csv           — best meta-learner")
print("  prediction_stacking_multiseed.csv — multi-seed stacking")
print()
print("Done!")

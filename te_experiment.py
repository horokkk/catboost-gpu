"""
Target Encoding 확대 실험
- 기존: occupation만 TE
- 실험: occupation + native_country TE
- 기존 LightGBM 최적 파라미터 고정, OOF 성능 비교

사용법: python te_experiment.py
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. 전처리 함수
# ============================================================
def preprocess_adult_income(df):
    df = df.copy()
    cat_missing_cols = ["workclass", "occupation", "native_country"]
    for col in cat_missing_cols:
        df[col] = df[col].replace(np.nan, "Unknown")
    if "education" in df.columns:
        df.drop(columns=["education"], inplace=True)

    df["has_capital_gain"] = (df["capital_gain"] > 0).astype(int)
    df["has_capital_loss"] = (df["capital_loss"] > 0).astype(int)
    df["log_capital_gain"] = np.log1p(df["capital_gain"])
    df["log_capital_loss"] = np.log1p(df["capital_loss"])
    df["net_capital"] = df["capital_gain"] - df["capital_loss"]
    df["is_max_gain"] = (df["capital_gain"] == 99999).astype(int)
    df["net_capital_bin"] = np.sign(df["net_capital"]).astype(int)
    df["capital_per_hour"] = df["net_capital"] / (df["hours_per_week"] + 1)
    df["log_abs_net_capital"] = np.log1p(np.abs(df["net_capital"])) * np.sign(df["net_capital"])
    df["gain_loss_ratio"] = df["log_capital_gain"] / (df["log_capital_loss"] + 1)

    married_categories = ["Married-civ-spouse", "Married-AF-spouse"]
    df["is_married"] = df["marital_status"].isin(married_categories).astype(int)
    df["married_male"] = ((df["is_married"] == 1) & (df["sex"] == "Male")).astype(int)
    df["is_spouse"] = df["relationship"].isin(["Husband", "Wife"]).astype(int)
    df["age_education"] = df["age"] * df["education_num"]
    df["age_squared"] = df["age"] ** 2
    df["age_per_edu"] = df["age"] / (df["education_num"] + 1)
    df["is_high_edu"] = (df["education_num"] >= 13).astype(int)
    df["high_edu_married"] = df["is_high_edu"] * df["is_married"]
    df["overtime"] = (df["hours_per_week"] > 40).astype(int)
    df["edu_hours"] = df["education_num"] * df["hours_per_week"]
    df["is_part_time"] = (df["hours_per_week"] < 35).astype(int)
    df["is_full_time"] = (df["hours_per_week"] == 40).astype(int)
    df["hours_age"] = df["hours_per_week"] * df["age"]
    df["overtime_married"] = df["overtime"] * df["is_married"]
    return df


# ============================================================
# 2. 데이터 로드
# ============================================================
print("Loading data...")
train_df = pd.read_csv("train.csv", na_values=["", " "])
test_df = pd.read_csv("test.csv", na_values=["", " "])

train_df = preprocess_adult_income(train_df)
test_df = preprocess_adult_income(test_df)

train_df["income"] = train_df["income"].apply(lambda x: 1 if ">50K" in str(x) else 0)

train_df.drop(columns=["id"], inplace=True)
test_ids = test_df["id"]
test_df.drop(columns=["id"], inplace=True)

y = train_df["income"]
X_base = train_df.drop(columns=["income"])
X_test_base = test_df.copy()

cat_cols = ["workclass", "marital_status", "occupation", "relationship",
            "race", "sex", "native_country"]

# LightGBM 최적 파라미터 (Optuna trial 187/200)
lgbm_params = {
    "objective": "binary", "metric": "binary_logloss", "verbosity": -1,
    "seed": 42, "n_jobs": -1,
    "scale_pos_weight": 1.615, "learning_rate": 0.0168,
    "num_leaves": 31, "max_depth": 12, "min_child_samples": 22,
    "subsample": 0.972, "colsample_bytree": 0.635,
    "reg_alpha": 0.000162, "reg_lambda": 0.016,
    "min_split_gain": 0.919, "bagging_freq": 7,
    "feature_fraction_bynode": 0.684,
}

smoothing = 10
global_mean = y.mean()


def apply_te(X, X_test, y, te_cols, smoothing=10):
    """K-Fold Target Encoding 적용"""
    X = X.copy()
    X_test = X_test.copy()
    te_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for col in te_cols:
        te_feature = f"{col}_te"
        X[te_feature] = 0.0

        for tr_idx, val_idx in te_skf.split(X, y):
            means = y.iloc[tr_idx].groupby(X[col].iloc[tr_idx]).agg(["mean", "count"])
            smooth_means = (means["count"] * means["mean"] + smoothing * global_mean) / (means["count"] + smoothing)
            X.iloc[val_idx, X.columns.get_loc(te_feature)] = X[col].iloc[val_idx].map(smooth_means).fillna(global_mean)

        full_means = y.groupby(X[col]).agg(["mean", "count"])
        full_smooth = (full_means["count"] * full_means["mean"] + smoothing * global_mean) / (full_means["count"] + smoothing)
        X_test[te_feature] = X_test[col].map(full_smooth).fillna(global_mean)

    return X, X_test


def evaluate_lgbm_te(X, X_test, y, te_cols, label):
    """TE 적용 후 LightGBM OOF 성능 평가"""
    X, X_test = apply_te(X.copy(), X_test.copy(), y, te_cols)

    for col in cat_cols:
        X[col] = X[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_prob = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        dtrain = lgb.Dataset(X.iloc[train_idx], label=y.iloc[train_idx], categorical_feature=cat_cols)
        dval = lgb.Dataset(X.iloc[val_idx], label=y.iloc[val_idx], categorical_feature=cat_cols)
        model = lgb.train(lgbm_params, dtrain, num_boost_round=3000, valid_sets=[dval],
                          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        oof_prob[val_idx] = model.predict(X.iloc[val_idx])

    f1 = f1_score(y, (oof_prob >= 0.5).astype(int))
    auc = roc_auc_score(y, oof_prob)
    score = (f1 + auc) / 2

    # Threshold tuning
    best_f1, best_thr = f1, 0.5
    for thr in np.arange(0.20, 0.80, 0.001):
        f1_t = f1_score(y, (oof_prob >= thr).astype(int))
        if f1_t > best_f1:
            best_f1 = f1_t
            best_thr = thr
    score_tuned = (best_f1 + auc) / 2

    return {
        "label": label,
        "te_cols": te_cols,
        "f1": f1, "auc": auc, "score": score,
        "f1_tuned": best_f1, "score_tuned": score_tuned, "threshold": best_thr,
        "oof": oof_prob,
        "features": X.shape[1],
        "X": X, "X_test": X_test,
    }


# ============================================================
# 3. 실험
# ============================================================
print()
experiments = [
    (["occupation"], "A: occupation만 (현재)"),
    (["occupation", "native_country"], "B: occupation + native_country"),
    (["occupation", "native_country", "workclass"], "C: occupation + native_country + workclass"),
    (["occupation", "native_country", "workclass", "relationship"], "D: occ + country + work + rel"),
]

results = []
for te_cols, label in experiments:
    print(f"Running {label}...")
    r = evaluate_lgbm_te(X_base, X_test_base, y, te_cols, label)
    results.append(r)
    print(f"  Features: {r['features']}, F1: {r['f1']:.4f}, AUC: {r['auc']:.4f}, "
          f"Score: {r['score']:.4f} → tuned: {r['score_tuned']:.4f} (thr={r['threshold']:.3f})")
    print()


# ============================================================
# 4. 결과 비교
# ============================================================
print("=" * 70)
print("Target Encoding 실험 결과")
print("=" * 70)
print(f"  {'Experiment':<40s} {'Feat':>4s} {'F1':>7s} {'AUC':>7s} {'Score':>7s} {'Tuned':>7s}")
print("  " + "-" * 70)

baseline_score = results[0]["score_tuned"]
for r in results:
    delta = r["score_tuned"] - baseline_score
    marker = " ★" if r["score_tuned"] == max(x["score_tuned"] for x in results) else ""
    delta_str = f"({delta:+.4f})" if r != results[0] else "(base)"
    print(f"  {r['label']:<40s} {r['features']:>4d} {r['f1']:>7.4f} {r['auc']:>7.4f} "
          f"{r['score']:>7.4f} {r['score_tuned']:>7.4f} {delta_str}{marker}")

print()

# ============================================================
# 5. Best로 Test 예측 (multi-seed)
# ============================================================
best = max(results, key=lambda x: x["score_tuned"])
print("=" * 70)
print(f"★ BEST: {best['label']}")
print(f"  Score (tuned): {best['score_tuned']:.4f}, Threshold: {best['threshold']:.3f}")
print("=" * 70)

if best["score_tuned"] > baseline_score + 0.0001:
    print("\n기존보다 개선됨! Multi-seed test prediction 생성 중...")

    X_best = best["X"]
    X_test_best = best["X_test"]
    seeds = [42, 123, 456, 789, 2024]
    test_probs = np.zeros(len(X_test_best))

    for seed_i, seed in enumerate(seeds):
        seed_params = lgbm_params.copy()
        seed_params["seed"] = seed
        skf_seed = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (train_idx, val_idx) in enumerate(skf_seed.split(X_best, y)):
            dtrain = lgb.Dataset(X_best.iloc[train_idx], label=y.iloc[train_idx], categorical_feature=cat_cols)
            dval = lgb.Dataset(X_best.iloc[val_idx], label=y.iloc[val_idx], categorical_feature=cat_cols)
            model = lgb.train(seed_params, dtrain, num_boost_round=3000, valid_sets=[dval],
                              callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
            test_probs += model.predict(X_test_best) / (5 * len(seeds))
        print(f"  Seed {seed} done ({seed_i+1}/{len(seeds)})")

    np.save("prob_lgbm_te.npy", test_probs)
    print(f"\nprob_lgbm_te.npy saved!")
    print(f"  mean prob: {test_probs.mean():.4f}")

    # 기존 블렌딩에 적용 (LGBM_te 40% + Cat 49% + XGB 11%)
    prob_cat = np.load("prob_catboost.npy")
    prob_xgb = np.load("prob_xgb.npy")
    blend = 0.40 * test_probs + 0.49 * prob_cat + 0.11 * prob_xgb

    prediction = pd.DataFrame({
        "id": test_ids,
        "y_cls": (blend >= best["threshold"]).astype(int),
        "y_prob": blend,
    })
    prediction.to_csv("prediction_te_blend.csv", index=False)
    print(f"\nprediction_te_blend.csv saved!")
    print(f"  y_cls distribution: {prediction['y_cls'].value_counts().to_dict()}")
    print(f"  y_prob mean: {prediction['y_prob'].mean():.4f}")
else:
    print("\n기존과 차이 없음. TE 확대 불필요.")

print("\nDone!")

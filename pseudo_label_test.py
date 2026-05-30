"""
Pseudo-labeling 실험
- 기존 3-model blend 확률로 고확신 test 샘플에 pseudo-label 부여
- 확장된 데이터로 LightGBM 재학습
- OOF (원본 train만) 성능 비교

사용법: python pseudo_label_test.py
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 1. 전처리
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
X = train_df.drop(columns=["income"])
X_test = test_df.copy()

n_train = len(X)

cat_cols = ["workclass", "marital_status", "occupation", "relationship",
            "race", "sex", "native_country"]

# LightGBM 최적 파라미터 (200 trial 결과)
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


# ============================================================
# 3. Baseline (pseudo-label 없이)
# ============================================================
print("\n" + "=" * 60)
print("Baseline: No pseudo-labeling")
print("=" * 60)

smoothing = 10
global_mean = y.mean()
te_cols = ["occupation", "native_country"]


def train_lgbm_with_te(X_tr_full, y_tr_full, X_test, n_original, label=""):
    """TE 적용 + LightGBM 학습, OOF는 원본 train만 평가"""
    X_tr_full = X_tr_full.copy().reset_index(drop=True)
    y_tr_full = y_tr_full.copy().reset_index(drop=True)
    X_test = X_test.copy()

    # K-Fold Target Encoding
    te_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    g_mean = y_tr_full.mean()

    for te_col in te_cols:
        te_feature = f"{te_col}_te"
        X_tr_full[te_feature] = 0.0

        for tr_idx, val_idx in te_skf.split(X_tr_full, y_tr_full):
            means = y_tr_full.iloc[tr_idx].groupby(X_tr_full[te_col].iloc[tr_idx]).agg(["mean", "count"])
            smooth = (means["count"] * means["mean"] + smoothing * g_mean) / (means["count"] + smoothing)
            X_tr_full.iloc[val_idx, X_tr_full.columns.get_loc(te_feature)] = (
                X_tr_full[te_col].iloc[val_idx].map(smooth).fillna(g_mean)
            )

        full_means = y_tr_full.groupby(X_tr_full[te_col]).agg(["mean", "count"])
        full_smooth = (full_means["count"] * full_means["mean"] + smoothing * g_mean) / (full_means["count"] + smoothing)
        X_test[te_feature] = X_test[te_col].map(full_smooth).fillna(g_mean)

    for col in cat_cols:
        X_tr_full[col] = X_tr_full[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    # 5-Fold CV (OOF는 원본 train만)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_prob = np.zeros(len(X_tr_full))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_tr_full, y_tr_full)):
        dtrain = lgb.Dataset(X_tr_full.iloc[train_idx], label=y_tr_full.iloc[train_idx],
                             categorical_feature=cat_cols)
        dval = lgb.Dataset(X_tr_full.iloc[val_idx], label=y_tr_full.iloc[val_idx],
                           categorical_feature=cat_cols)
        model = lgb.train(lgbm_params, dtrain, num_boost_round=3000, valid_sets=[dval],
                          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        oof_prob[val_idx] = model.predict(X_tr_full.iloc[val_idx])

    # 원본 train만 평가
    oof_orig = oof_prob[:n_original]
    y_orig = y_tr_full.iloc[:n_original]

    f1 = f1_score(y_orig, (oof_orig >= 0.5).astype(int))
    auc = roc_auc_score(y_orig, oof_orig)
    score = (f1 + auc) / 2

    # Threshold tuning
    best_f1, best_thr = f1, 0.5
    for thr in np.arange(0.20, 0.80, 0.005):
        f1_t = f1_score(y_orig, (oof_orig >= thr).astype(int))
        if f1_t > best_f1:
            best_f1 = f1_t
            best_thr = thr
    score_tuned = (best_f1 + auc) / 2

    print(f"  {label}")
    print(f"    F1: {f1:.4f}, AUC: {auc:.4f}, Score: {score:.4f}")
    print(f"    Tuned: F1={best_f1:.4f}, thr={best_thr:.3f}, Score={score_tuned:.4f}")

    # Test 예측 (multi-seed)
    seeds = [42, 123, 456, 789, 2024]
    test_probs = np.zeros(len(X_test))
    for seed in seeds:
        seed_params = lgbm_params.copy()
        seed_params["seed"] = seed
        skf_seed = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (tr_idx, va_idx) in enumerate(skf_seed.split(X_tr_full, y_tr_full)):
            dtrain = lgb.Dataset(X_tr_full.iloc[tr_idx], label=y_tr_full.iloc[tr_idx],
                                 categorical_feature=cat_cols)
            dval = lgb.Dataset(X_tr_full.iloc[va_idx], label=y_tr_full.iloc[va_idx],
                               categorical_feature=cat_cols)
            model = lgb.train(seed_params, dtrain, num_boost_round=3000, valid_sets=[dval],
                              callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
            test_probs += model.predict(X_test) / (5 * len(seeds))

    return score_tuned, best_thr, oof_orig, test_probs


# Baseline
baseline_score, baseline_thr, baseline_oof, baseline_test = train_lgbm_with_te(
    X.copy(), y.copy(), X_test.copy(), n_train, "Baseline (no pseudo-label)"
)


# ============================================================
# 4. Pseudo-labeling 실험
# ============================================================
print("\n" + "=" * 60)
print("Loading blend probabilities...")
print("=" * 60)

prob_lgbm_file = np.load("prob_lgbm.npy")
prob_cat_file = np.load("prob_catboost.npy")
prob_xgb_file = np.load("prob_xgb.npy")
blend = 0.40 * prob_lgbm_file + 0.49 * prob_cat_file + 0.11 * prob_xgb_file

print(f"Blend mean: {blend.mean():.4f}")
print(f"Blend >0.90: {(blend > 0.90).sum()}, <0.10: {(blend < 0.10).sum()}")
print(f"Blend >0.95: {(blend > 0.95).sum()}, <0.05: {(blend < 0.05).sum()}")
print(f"Blend >0.97: {(blend > 0.97).sum()}, <0.03: {(blend < 0.03).sum()}")

results = [{"label": "Baseline", "score": baseline_score, "thr": baseline_thr, "n_pseudo": 0}]

for conf_thr in [0.90, 0.93, 0.95, 0.97]:
    print(f"\n{'=' * 60}")
    print(f"Pseudo-labeling: confidence threshold = {conf_thr}")
    print("=" * 60)

    high_mask = (blend > conf_thr) | (blend < (1 - conf_thr))
    pseudo_labels = (blend[high_mask] >= 0.5).astype(int)
    n_pseudo = high_mask.sum()

    print(f"  Pseudo-labeled samples: {n_pseudo} ({n_pseudo/len(blend)*100:.1f}% of test)")
    print(f"  Pseudo label dist: 0={int((pseudo_labels==0).sum())}, 1={int((pseudo_labels==1).sum())}")

    if n_pseudo == 0:
        print("  No samples above threshold. Skipping.")
        continue

    # 확장된 학습 데이터
    X_pseudo = X_test[high_mask].copy()
    X_expanded = pd.concat([X.copy(), X_pseudo], ignore_index=True)
    y_expanded = pd.concat([y.copy(), pd.Series(pseudo_labels.values)], ignore_index=True)

    score, thr, _, test_probs = train_lgbm_with_te(
        X_expanded, y_expanded, X_test.copy(), n_train,
        f"Pseudo-label (conf={conf_thr}, n={n_pseudo})"
    )

    delta = score - baseline_score
    print(f"    Delta vs baseline: {delta:+.4f}")

    results.append({"label": f"PL conf={conf_thr}", "score": score, "thr": thr,
                     "n_pseudo": n_pseudo, "delta": delta, "test_probs": test_probs})


# ============================================================
# 5. 결과 요약
# ============================================================
print("\n" + "=" * 60)
print("Pseudo-labeling 실험 결과 요약")
print("=" * 60)
print(f"  {'Experiment':<30s} {'N_pseudo':>8s} {'Score':>7s} {'Delta':>8s}")
print("  " + "-" * 55)
for r in results:
    delta_str = f"{r.get('delta', 0):+.4f}" if r.get('delta') is not None else "  base"
    print(f"  {r['label']:<30s} {r['n_pseudo']:>8d} {r['score']:>7.4f} {delta_str:>8s}")

# Best result
best = max(results, key=lambda x: x["score"])
print(f"\n★ Best: {best['label']} (Score: {best['score']:.4f})")

if best["score"] > baseline_score + 0.0001 and "test_probs" in best:
    print("\nPseudo-labeling 개선 확인! prob_lgbm_pseudo.npy 저장...")
    np.save("prob_lgbm_pseudo.npy", best["test_probs"])

    # 기존 블렌딩에 적용
    blend_new = 0.40 * best["test_probs"] + 0.49 * prob_cat_file + 0.11 * prob_xgb_file
    test_df_ids = pd.read_csv("test.csv")["id"]
    prediction = pd.DataFrame({
        "id": test_df_ids,
        "y_cls": (blend_new >= best["thr"]).astype(int),
        "y_prob": blend_new,
    })
    prediction.to_csv("prediction_pseudo.csv", index=False)
    print(f"prediction_pseudo.csv saved!")
    print(f"  y_cls dist: {prediction['y_cls'].value_counts().to_dict()}")
else:
    print("\nPseudo-labeling 개선 없음.")

print("\nDone!")

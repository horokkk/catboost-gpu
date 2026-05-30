"""
Blend Comparison: 2-model vs 3-model
Pattern Recognition Project (Final: 5/31)
Metric: (F1 + AUC) / 2

목적:
  A) LightGBM + CatBoost (2개 블렌딩)
  B) LightGBM + CatBoost + XGBoost (3개 블렌딩)
  → OOF 기반으로 어느 쪽이 더 나은지 비교

사용법:
1. train.csv, test.csv 업로드
2. !pip install lightgbm catboost xgboost matplotlib
3. XGBoost Optuna 결과가 있으면 xgb_params 수정 (Section 3)
4. 실행
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
import matplotlib
matplotlib.use('Agg')  # headless (서버용)
import matplotlib.pyplot as plt
matplotlib.rcParams['font.size'] = 12
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

    married_categories = ["Married-civ-spouse", "Married-AF-spouse"]
    df["is_married"] = df["marital_status"].isin(married_categories).astype(int)
    df["married_male"] = ((df["is_married"] == 1) & (df["sex"] == "Male")).astype(int)
    df["is_spouse"] = df["relationship"].isin(["Husband", "Wife"]).astype(int)
    df["age_education"] = df["age"] * df["education_num"]
    df["age_squared"] = df["age"] ** 2
    df["overtime"] = (df["hours_per_week"] > 40).astype(int)
    df["edu_hours"] = df["education_num"] * df["hours_per_week"]
    df["is_part_time"] = (df["hours_per_week"] < 35).astype(int)
    df["hours_age"] = df["hours_per_week"] * df["age"]
    df["is_max_gain"] = (df["capital_gain"] == 99999).astype(int)
    df["net_capital_bin"] = np.sign(df["net_capital"]).astype(int)

    df["is_full_time"] = (df["hours_per_week"] == 40).astype(int)
    df["capital_per_hour"] = df["net_capital"] / (df["hours_per_week"] + 1)
    df["log_abs_net_capital"] = np.log1p(np.abs(df["net_capital"])) * np.sign(df["net_capital"])
    df["age_per_edu"] = df["age"] / (df["education_num"] + 1)
    df["gain_loss_ratio"] = df["log_capital_gain"] / (df["log_capital_loss"] + 1)
    df["is_high_edu"] = (df["education_num"] >= 13).astype(int)
    df["high_edu_married"] = df["is_high_edu"] * df["is_married"]
    df["overtime_married"] = df["overtime"] * df["is_married"]

    return df

# ============================================================
# 2. 데이터 로드 + 전처리
# ============================================================
train_df = pd.read_csv("train.csv", na_values=["", " "])
test_df = pd.read_csv("test.csv", na_values=["", " "])

train_df = preprocess_adult_income(train_df)
test_df = preprocess_adult_income(test_df)

train_df["income"] = train_df["income"].apply(lambda x: 1 if ">50K" in str(x) else 0)

train_ids = train_df["id"]
test_ids = test_df["id"]
train_df.drop(columns=["id"], inplace=True)
test_df.drop(columns=["id"], inplace=True)

y = train_df["income"]
X = train_df.drop(columns=["income"])
X_test = test_df.copy()

cat_cols = ["workclass", "marital_status", "occupation", "relationship",
            "race", "sex", "native_country"]

print(f"Train: {X.shape}, Test: {X_test.shape}")
print(f"Target: {y.value_counts().to_dict()}")
print()

# ============================================================
# 3. 모델 파라미터
# ============================================================

# --- LightGBM (Optuna trial 187/200, 확정) ---
lgbm_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "verbosity": -1,
    "seed": 42,
    "n_jobs": -1,
    "scale_pos_weight": 1.615,
    "learning_rate": 0.0168,
    "num_leaves": 31,
    "max_depth": 12,
    "min_child_samples": 22,
    "subsample": 0.972,
    "colsample_bytree": 0.635,
    "reg_alpha": 0.000162,
    "reg_lambda": 0.016,
    "min_split_gain": 0.919,
    "bagging_freq": 7,
    "feature_fraction_bynode": 0.684,
}

# --- CatBoost (Optuna trial 32/50, 확정) ---
catboost_params = {
    "iterations": 3000,
    "eval_metric": "Logloss",
    "random_seed": 42,
    "task_type": "CPU",
    "verbose": 0,
    "learning_rate": 0.04502966817638666,
    "depth": 5,
    "l2_leaf_reg": 6.142223638226484,
    "border_count": 214,
    "random_strength": 5.524977061878722,
    "bagging_temperature": 7.700920183783566,
    "scale_pos_weight": 1.6721091166174638,
    "min_data_in_leaf": 17,
    "max_ctr_complexity": 2,
}

# --- XGBoost (Optuna trial 69/100, 확정) ---
xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "seed": 42,
    "nthread": -1,
    "verbosity": 0,
    "learning_rate": 0.04575847924217747,
    "max_depth": 3,
    "subsample": 0.9461480650953578,
    "colsample_bytree": 0.5788686655621644,
    "alpha": 0.932698037041515,
    "lambda": 1.1293430698748572,
    "scale_pos_weight": 1.5434211788366061,
    "min_child_weight": 3,
    "gamma": 0.014390756699769983,
}

# ============================================================
# 4. 모델별 데이터 준비
# ============================================================

# --- LightGBM: category dtype + K-Fold Target Encoding ---
X_lgbm = X.copy()
X_test_lgbm = X_test.copy()

te_col = "occupation"
te_feature = f"{te_col}_te"
X_lgbm[te_feature] = 0.0
te_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
global_mean = y.mean()
smoothing = 10

for tr_idx, val_idx in te_skf.split(X_lgbm, y):
    means = y.iloc[tr_idx].groupby(X_lgbm[te_col].iloc[tr_idx]).agg(["mean", "count"])
    smooth_means = (means["count"] * means["mean"] + smoothing * global_mean) / (means["count"] + smoothing)
    X_lgbm.iloc[val_idx, X_lgbm.columns.get_loc(te_feature)] = (
        X_lgbm[te_col].iloc[val_idx].map(smooth_means).fillna(global_mean)
    )

full_means = y.groupby(X_lgbm[te_col]).agg(["mean", "count"])
full_smooth = (full_means["count"] * full_means["mean"] + smoothing * global_mean) / (full_means["count"] + smoothing)
X_test_lgbm[te_feature] = X_test_lgbm[te_col].map(full_smooth).fillna(global_mean)

for col in cat_cols:
    X_lgbm[col] = X_lgbm[col].astype("category")
    X_test_lgbm[col] = X_test_lgbm[col].astype("category")

# --- XGBoost: category dtype (Target Encoding 없음) ---
X_xgb = X.copy()
X_test_xgb = X_test.copy()
for col in cat_cols:
    X_xgb[col] = X_xgb[col].astype("category")
    X_test_xgb[col] = X_test_xgb[col].astype("category")

# --- CatBoost: string dtype ---
X_cat = X.copy()
X_test_cat = X_test.copy()
for col in cat_cols:
    X_cat[col] = X_cat[col].astype(str)
    X_test_cat[col] = X_test_cat[col].astype(str)
cat_indices = [X_cat.columns.get_loc(c) for c in cat_cols]

print("Data prepared for all 3 models.")
print(f"  LightGBM features: {X_lgbm.shape[1]} (with occupation_te)")
print(f"  XGBoost features:  {X_xgb.shape[1]}")
print(f"  CatBoost features: {X_cat.shape[1]}")
print()

# ============================================================
# 5. 5-Fold CV: OOF Predictions
# ============================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_lgbm = np.zeros(len(y))
oof_catboost = np.zeros(len(y))
oof_xgboost = np.zeros(len(y))

print("=" * 60)
print("5-Fold CV: OOF predictions for 3 models")
print("=" * 60)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # --- LightGBM ---
    dtrain = lgb.Dataset(X_lgbm.iloc[train_idx], label=y_tr, categorical_feature=cat_cols)
    dval = lgb.Dataset(X_lgbm.iloc[val_idx], label=y_val, categorical_feature=cat_cols)
    m_lgbm = lgb.train(
        lgbm_params, dtrain, num_boost_round=3000, valid_sets=[dval],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    oof_lgbm[val_idx] = m_lgbm.predict(X_lgbm.iloc[val_idx])

    # --- CatBoost ---
    tp = Pool(X_cat.iloc[train_idx], label=y_tr, cat_features=cat_indices)
    vp = Pool(X_cat.iloc[val_idx], label=y_val, cat_features=cat_indices)
    m_cat = CatBoostClassifier(**catboost_params)
    m_cat.fit(tp, eval_set=vp, early_stopping_rounds=100, verbose=0)
    oof_catboost[val_idx] = m_cat.predict_proba(X_cat.iloc[val_idx])[:, 1]

    # --- XGBoost ---
    dtrain_xgb = xgb.DMatrix(X_xgb.iloc[train_idx], label=y_tr, enable_categorical=True)
    dval_xgb = xgb.DMatrix(X_xgb.iloc[val_idx], label=y_val, enable_categorical=True)
    m_xgb = xgb.train(
        xgb_params, dtrain_xgb, num_boost_round=3000,
        evals=[(dval_xgb, "val")], early_stopping_rounds=100, verbose_eval=False
    )
    oof_xgboost[val_idx] = m_xgb.predict(dval_xgb)

    print(f"  Fold {fold+1}/5 done")

print()

# ============================================================
# 6. 개별 모델 성능
# ============================================================
def calc_score(probs, y):
    cls = (probs >= 0.5).astype(int)
    f1 = f1_score(y, cls)
    auc = roc_auc_score(y, probs)
    return f1, auc, (f1 + auc) / 2

f1_l, auc_l, score_l = calc_score(oof_lgbm, y)
f1_c, auc_c, score_c = calc_score(oof_catboost, y)
f1_x, auc_x, score_x = calc_score(oof_xgboost, y)

print("=" * 60)
print("Individual Model Scores (OOF)")
print("=" * 60)
print(f"  LightGBM:  F1={f1_l:.4f}  AUC={auc_l:.4f}  Score={score_l:.4f}")
print(f"  CatBoost:  F1={f1_c:.4f}  AUC={auc_c:.4f}  Score={score_c:.4f}")
print(f"  XGBoost:   F1={f1_x:.4f}  AUC={auc_x:.4f}  Score={score_x:.4f}")
print()

# OOF 저장 (재사용용)
np.save("oof_lgbm.npy", oof_lgbm)
np.save("oof_catboost.npy", oof_catboost)
np.save("oof_xgboost.npy", oof_xgboost)
print("OOF predictions saved: oof_lgbm.npy, oof_catboost.npy, oof_xgboost.npy")
print()

# ============================================================
# 7. OOF 상관관계 분석
# ============================================================
print("=" * 60)
print("OOF Prediction Correlation (Pearson)")
print("=" * 60)

corr_df = pd.DataFrame({
    "LightGBM": oof_lgbm,
    "CatBoost": oof_catboost,
    "XGBoost": oof_xgboost
})
corr_matrix = corr_df.corr()
print(corr_matrix.to_string(float_format="%.4f"))
print()
print("  * 상관계수가 낮을수록 블렌딩 이득이 크다.")
print(f"  LGBM ↔ CatBoost: {corr_matrix.loc['LightGBM','CatBoost']:.4f}")
print(f"  LGBM ↔ XGBoost:  {corr_matrix.loc['LightGBM','XGBoost']:.4f}")
print(f"  CatBoost ↔ XGBoost: {corr_matrix.loc['CatBoost','XGBoost']:.4f}")
print()

# ============================================================
# 8. 2-Model Blend: 3쌍 모두 비교
# ============================================================
alphas_fine = np.arange(0.0, 1.005, 0.01)

pairs = [
    ("LGBM+Cat", oof_lgbm, oof_catboost, "LightGBM", "CatBoost"),
    ("LGBM+XGB", oof_lgbm, oof_xgboost, "LightGBM", "XGBoost"),
    ("Cat+XGB",  oof_catboost, oof_xgboost, "CatBoost", "XGBoost"),
]

pair_results = {}  # name -> {alpha, f1, auc, score, scores_curve, f1s_curve, aucs_curve}

for pair_name, oof_a, oof_b, name_a, name_b in pairs:
    print("=" * 60)
    print(f"2-Model Blend: {name_a} + {name_b}")
    print("=" * 60)

    scores_curve = []
    f1s_curve = []
    aucs_curve = []

    for alpha in alphas_fine:
        blend = alpha * oof_a + (1 - alpha) * oof_b
        f1, auc, score = calc_score(blend, y)
        scores_curve.append(score)
        f1s_curve.append(f1)
        aucs_curve.append(auc)

    best_idx = np.argmax(scores_curve)
    best_alpha = alphas_fine[best_idx]

    pair_results[pair_name] = {
        "alpha": best_alpha,
        "name_a": name_a, "name_b": name_b,
        "f1": f1s_curve[best_idx],
        "auc": aucs_curve[best_idx],
        "score": scores_curve[best_idx],
        "scores_curve": scores_curve,
        "f1s_curve": f1s_curve,
        "aucs_curve": aucs_curve,
    }

    print(f"  Best: {name_a} {best_alpha*100:.0f}% + {name_b} {(1-best_alpha)*100:.0f}%")
    print(f"    F1={f1s_curve[best_idx]:.4f}  AUC={aucs_curve[best_idx]:.4f}  Score={scores_curve[best_idx]:.4f}")

    # Top 3
    sorted_idx = np.argsort(scores_curve)[::-1][:3]
    print("  Top 3:")
    for idx in sorted_idx:
        a = alphas_fine[idx]
        print(f"    {name_a} {a*100:4.0f}% + {name_b} {(1-a)*100:4.0f}%  →  {scores_curve[idx]:.4f}")
    print()

# 3쌍 비교 요약
print("=" * 60)
print("2-Model Blend Summary (3 pairs)")
print("=" * 60)
print(f"  {'Pair':<15s} {'Best Ratio':<25s} {'F1':>7s} {'AUC':>7s} {'Score':>7s}")
print("  " + "-" * 60)
for pname, pr in pair_results.items():
    ratio = f"{pr['name_a']} {pr['alpha']*100:.0f}% + {pr['name_b']} {(1-pr['alpha'])*100:.0f}%"
    print(f"  {pname:<15s} {ratio:<25s} {pr['f1']:>7.4f} {pr['auc']:>7.4f} {pr['score']:>7.4f}")
print()

# Best 2-model pair
best_pair_name = max(pair_results, key=lambda k: pair_results[k]["score"])
best_pair = pair_results[best_pair_name]
best_alpha_2 = best_pair["alpha"]
best_score_2 = best_pair["score"]
best_f1_2 = best_pair["f1"]
best_auc_2 = best_pair["auc"]
print(f"  >>> Best 2-model pair: {best_pair_name}")
print()

# ============================================================
# 9. 3-Model Blend: LightGBM + CatBoost + XGBoost
# ============================================================
print("=" * 60)
print("3-Model Blend Optimization (LightGBM + CatBoost + XGBoost)")
print("=" * 60)

best_score_3 = 0
best_weights_3 = (1/3, 1/3, 1/3)
all_3model_results = []

# Coarse grid: step 0.05
for w_l in np.arange(0.0, 1.05, 0.05):
    for w_c in np.arange(0.0, 1.05 - w_l, 0.05):
        w_x = round(1.0 - w_l - w_c, 2)
        if w_x < -0.001:
            continue
        w_x = max(0, w_x)

        blend = w_l * oof_lgbm + w_c * oof_catboost + w_x * oof_xgboost
        f1, auc, score = calc_score(blend, y)
        all_3model_results.append((w_l, w_c, w_x, f1, auc, score))

        if score > best_score_3:
            best_score_3 = score
            best_weights_3 = (w_l, w_c, w_x)

# Fine grid: step 0.01 around best
w_l_best, w_c_best, w_x_best = best_weights_3
for w_l in np.arange(max(0, w_l_best - 0.1), min(1.001, w_l_best + 0.1), 0.01):
    for w_c in np.arange(max(0, w_c_best - 0.1), min(1.001 - w_l, w_c_best + 0.1), 0.01):
        w_x = round(1.0 - w_l - w_c, 2)
        if w_x < -0.001 or w_x > 1.001:
            continue
        w_x = max(0, min(1, w_x))

        blend = w_l * oof_lgbm + w_c * oof_catboost + w_x * oof_xgboost
        f1, auc, score = calc_score(blend, y)
        all_3model_results.append((w_l, w_c, w_x, f1, auc, score))

        if score > best_score_3:
            best_score_3 = score
            best_weights_3 = (w_l, w_c, w_x)

w_l_final, w_c_final, w_x_final = best_weights_3
blend_3_best = w_l_final * oof_lgbm + w_c_final * oof_catboost + w_x_final * oof_xgboost
f1_3, auc_3, score_3 = calc_score(blend_3_best, y)

print(f"  Best: LGBM {w_l_final*100:.0f}% + Cat {w_c_final*100:.0f}% + XGB {w_x_final*100:.0f}%")
print(f"    F1={f1_3:.4f}  AUC={auc_3:.4f}  Score={score_3:.4f}")
print()

# Top 5
results_sorted = sorted(all_3model_results, key=lambda x: x[5], reverse=True)[:5]
print("  Top 5 ratios:")
for wl, wc, wx, f1, auc, sc in results_sorted:
    print(f"    LGBM {wl*100:4.0f}% + Cat {wc*100:4.0f}% + XGB {wx*100:4.0f}%  →  Score={sc:.4f}")
print()

# ============================================================
# 10. 비교 및 권장
# ============================================================
print("=" * 60)
print("★ FINAL COMPARISON ★")
print("=" * 60)
print()

# --- Individual models ---
print(f"  {'Model':<42s} {'F1':>7s} {'AUC':>7s} {'Score':>7s}")
print("  " + "-" * 64)
print(f"  {'LightGBM (single)':<42s} {f1_l:>7.4f} {auc_l:>7.4f} {score_l:>7.4f}")
print(f"  {'CatBoost (single)':<42s} {f1_c:>7.4f} {auc_c:>7.4f} {score_c:>7.4f}")
print(f"  {'XGBoost (single)':<42s} {f1_x:>7.4f} {auc_x:>7.4f} {score_x:>7.4f}")
print()

# --- 2-model (all 3 pairs) ---
print("  [2-Model Blends]")
for pname, pr in pair_results.items():
    ratio = f"{pr['name_a']} {pr['alpha']*100:.0f}% + {pr['name_b']} {(1-pr['alpha'])*100:.0f}%"
    label = f"  {pname}: {ratio}"
    marker = " ◀ best pair" if pname == best_pair_name else ""
    print(f"  {label:<42s} {pr['f1']:>7.4f} {pr['auc']:>7.4f} {pr['score']:>7.4f}{marker}")
print()

# --- 3-model ---
best_3_label = f"3-Model: L{w_l_final*100:.0f}% C{w_c_final*100:.0f}% X{w_x_final*100:.0f}%"
print(f"  {best_3_label:<42s} {f1_3:>7.4f} {auc_3:>7.4f} {score_3:>7.4f}")
print()

# --- 권장 ---
delta = score_3 - best_score_2

if delta > 0.0005:
    recommendation = "3-Model"
    rec_detail = f"3-Model이 best 2-Model보다 +{delta:.4f} 높음 → XGBoost 포함 권장"
elif delta < -0.0005:
    recommendation = "2-Model"
    rec_detail = f"Best 2-Model({best_pair_name})이 3-Model보다 +{-delta:.4f} 높음"
else:
    recommendation = "2-Model"
    rec_detail = f"3-Model과 차이 미미 ({delta:+.4f}) → 단순한 2-Model({best_pair_name}) 권장"

print(f"  >>> 권장: {recommendation}")
print(f"      {rec_detail}")
print()

# ============================================================
# 11. 최종 블렌드 Threshold Tuning
# ============================================================
print("=" * 60)
print("Threshold Tuning (Best Blend OOF)")
print("=" * 60)

if recommendation == "3-Model":
    oof_final = w_l_final * oof_lgbm + w_c_final * oof_catboost + w_x_final * oof_xgboost
    blend_label = f"LGBM {w_l_final*100:.0f}% + Cat {w_c_final*100:.0f}% + XGB {w_x_final*100:.0f}%"
else:
    # Best 2-model pair의 OOF 조합
    bp = best_pair
    if best_pair_name == "LGBM+Cat":
        oof_final = bp["alpha"] * oof_lgbm + (1 - bp["alpha"]) * oof_catboost
    elif best_pair_name == "LGBM+XGB":
        oof_final = bp["alpha"] * oof_lgbm + (1 - bp["alpha"]) * oof_xgboost
    else:  # Cat+XGB
        oof_final = bp["alpha"] * oof_catboost + (1 - bp["alpha"]) * oof_xgboost
    blend_label = f"{bp['name_a']} {bp['alpha']*100:.0f}% + {bp['name_b']} {(1-bp['alpha'])*100:.0f}%"

thresholds = np.arange(0.20, 0.80, 0.005)
auc_fixed = roc_auc_score(y, oof_final)
f1_by_thr = [f1_score(y, (oof_final >= thr).astype(int)) for thr in thresholds]
combined_by_thr = [(f + auc_fixed) / 2 for f in f1_by_thr]

best_thr_idx = np.argmax(combined_by_thr)
best_threshold = thresholds[best_thr_idx]
best_f1_thr = f1_by_thr[best_thr_idx]
best_combined_thr = combined_by_thr[best_thr_idx]

print(f"  Blend: {blend_label}")
print(f"  기본 (thr=0.5): F1={f1_score(y, (oof_final>=0.5).astype(int)):.4f}  AUC={auc_fixed:.4f}  Score={calc_score(oof_final, y)[2]:.4f}")
print(f"  최적 (thr={best_threshold:.3f}): F1={best_f1_thr:.4f}  AUC={auc_fixed:.4f}  Score={best_combined_thr:.4f}")
print()

# ============================================================
# 12. Test Prediction (Multi-seed averaging)
# ============================================================
print("=" * 60)
print("Test Prediction (5 seeds × 5 folds)")
print("=" * 60)

seeds = [42, 123, 456, 789, 2024]
n_folds = 5
n_total = n_folds * len(seeds)

test_probs_lgbm = np.zeros(len(X_test))
test_probs_catboost = np.zeros(len(X_test))
test_probs_xgboost = np.zeros(len(X_test))

# 어떤 모델을 학습할지 결정
if recommendation == "3-Model":
    train_lgbm = train_cat = train_xgb = True
    models_used = "LGBM + CatBoost + XGBoost"
else:
    # Best pair에 포함된 모델만 학습
    train_lgbm = "LGBM" in best_pair_name or "LightGBM" in best_pair_name
    train_cat = "Cat" in best_pair_name
    train_xgb = "XGB" in best_pair_name
    models_used = best_pair_name

print(f"  Models: {models_used}")
print()

for seed_i, seed in enumerate(seeds):
    skf_seed = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(skf_seed.split(X, y)):
        y_tr = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        # --- LightGBM ---
        if train_lgbm:
            lgbm_seed_params = lgbm_params.copy()
            lgbm_seed_params["seed"] = seed

            dtrain = lgb.Dataset(X_lgbm.iloc[train_idx], label=y_tr, categorical_feature=cat_cols)
            dval = lgb.Dataset(X_lgbm.iloc[val_idx], label=y_val, categorical_feature=cat_cols)
            m_l = lgb.train(
                lgbm_seed_params, dtrain, num_boost_round=3000, valid_sets=[dval],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
            )
            test_probs_lgbm += m_l.predict(X_test_lgbm) / n_total

        # --- CatBoost ---
        if train_cat:
            cat_seed_params = catboost_params.copy()
            cat_seed_params["random_seed"] = seed

            tp = Pool(X_cat.iloc[train_idx], label=y_tr, cat_features=cat_indices)
            vp = Pool(X_cat.iloc[val_idx], label=y_val, cat_features=cat_indices)
            m_c = CatBoostClassifier(**cat_seed_params)
            m_c.fit(tp, eval_set=vp, early_stopping_rounds=100, verbose=0)
            test_probs_catboost += m_c.predict_proba(X_test_cat)[:, 1] / n_total

        # --- XGBoost ---
        if train_xgb:
            xgb_seed_params = xgb_params.copy()
            xgb_seed_params["seed"] = seed

            dtrain_x = xgb.DMatrix(X_xgb.iloc[train_idx], label=y_tr, enable_categorical=True)
            dval_x = xgb.DMatrix(X_xgb.iloc[val_idx], label=y_val, enable_categorical=True)
            m_x = xgb.train(
                xgb_seed_params, dtrain_x, num_boost_round=3000,
                evals=[(dval_x, "val")], early_stopping_rounds=100, verbose_eval=False
            )
            dtest_x = xgb.DMatrix(X_test_xgb, enable_categorical=True)
            test_probs_xgboost += m_x.predict(dtest_x) / n_total

    print(f"  Seed {seed} done ({seed_i+1}/{len(seeds)})")

# 블렌딩
if recommendation == "3-Model":
    test_blend = (w_l_final * test_probs_lgbm
                  + w_c_final * test_probs_catboost
                  + w_x_final * test_probs_xgboost)
elif best_pair_name == "LGBM+Cat":
    test_blend = best_alpha_2 * test_probs_lgbm + (1 - best_alpha_2) * test_probs_catboost
elif best_pair_name == "LGBM+XGB":
    test_blend = best_alpha_2 * test_probs_lgbm + (1 - best_alpha_2) * test_probs_xgboost
else:  # Cat+XGB
    test_blend = best_alpha_2 * test_probs_catboost + (1 - best_alpha_2) * test_probs_xgboost

# prediction.csv 저장
prediction = pd.DataFrame({
    "id": test_ids,
    "y_cls": (test_blend >= best_threshold).astype(int),
    "y_prob": test_blend
})

prediction.to_csv("prediction.csv", index=False)
print(f"\nprediction.csv saved! ({len(prediction)} rows)")
print(f"  Threshold: {best_threshold:.3f}")
print(f"  y_cls distribution: {prediction['y_cls'].value_counts().to_dict()}")
print(f"  y_prob mean: {prediction['y_prob'].mean():.4f}")
print()

# 모델별 prob 저장
np.save("prob_lgbm.npy", test_probs_lgbm)
np.save("prob_catboost.npy", test_probs_catboost)
np.save("prob_xgboost.npy", test_probs_xgboost)
print("prob files saved!")
print()

# ============================================================
# 13. 시각화
# ============================================================

# --- Chart 1: OOF Correlation Heatmap ---
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr_matrix.values, cmap="YlOrRd", vmin=0.9, vmax=1.0)
ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])
ax.set_xticklabels(["LightGBM", "CatBoost", "XGBoost"], fontsize=11)
ax.set_yticklabels(["LightGBM", "CatBoost", "XGBoost"], fontsize=11)

for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{corr_matrix.values[i, j]:.4f}",
                ha="center", va="center", fontsize=13, fontweight="bold",
                color="white" if corr_matrix.values[i, j] > 0.97 else "black")

plt.colorbar(im, ax=ax, label="Pearson Correlation")
ax.set_title("OOF Prediction Correlation", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("correlation_heatmap.png saved!")

# --- Chart 2: 2-Model Blend Weight Curves (all 3 pairs) ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

pair_colors = {"LGBM+Cat": "#AB47BC", "LGBM+XGB": "#26A69A", "Cat+XGB": "#FFA726"}

for ax, (pname, pr) in zip(axes, pair_results.items()):
    sc_curve = pr["scores_curve"]
    f1_curve = pr["f1s_curve"]
    color = pair_colors[pname]

    ax.plot(alphas_fine * 100, sc_curve, color=color, linewidth=2, label="(F1+AUC)/2")
    ax.plot(alphas_fine * 100, f1_curve, color="#E53935", linewidth=1.2, alpha=0.5, label="F1")

    best_idx = np.argmax(sc_curve)
    ax.scatter([alphas_fine[best_idx] * 100], [sc_curve[best_idx]], color=color, s=100, zorder=5)
    ax.annotate(f'{pr["name_a"]} {pr["alpha"]*100:.0f}%\n{sc_curve[best_idx]:.4f}',
                xy=(alphas_fine[best_idx] * 100, sc_curve[best_idx]),
                xytext=(alphas_fine[best_idx] * 100 + 5, sc_curve[best_idx] - 0.003),
                fontsize=9, arrowprops=dict(arrowstyle="->", color="gray"))

    ax.set_xlabel(f"{pr['name_a']} Weight (%)", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(f"{pname}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim(-2, 102)
    ax.grid(alpha=0.3)

    # Best pair 강조
    if pname == best_pair_name:
        ax.set_title(f"{pname} ★ BEST", fontsize=13, fontweight="bold", color="red")

plt.suptitle("2-Model Blend: Weight Optimization (3 Pairs)", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("blend_2model_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("blend_2model_curves.png saved!")

# --- Chart 3: 3-Model Weight Heatmap (LGBM vs CatBoost, XGB = 1 - L - C) ---
fig, ax = plt.subplots(figsize=(8, 7))

# Grid data
w_range = np.arange(0.0, 1.05, 0.05)
score_grid = np.full((len(w_range), len(w_range)), np.nan)

for wl_i, wl in enumerate(w_range):
    for wc_i, wc in enumerate(w_range):
        wx = round(1.0 - wl - wc, 2)
        if wx < -0.001 or wx > 1.001:
            continue
        wx = max(0, min(1, wx))
        blend = wl * oof_lgbm + wc * oof_catboost + wx * oof_xgboost
        _, _, sc = calc_score(blend, y)
        score_grid[wc_i, wl_i] = sc  # y=CatBoost, x=LightGBM

im = ax.imshow(score_grid, origin="lower", cmap="viridis", aspect="auto",
               extent=[0, 100, 0, 100])
plt.colorbar(im, ax=ax, label="(F1+AUC)/2")

# Best point
ax.scatter([w_l_final * 100], [w_c_final * 100], color="red", s=150,
           marker="*", zorder=5, edgecolors="white", linewidths=1.5)
ax.annotate(f'Best: L{w_l_final*100:.0f}% C{w_c_final*100:.0f}% X{w_x_final*100:.0f}%\n'
            f'Score={score_3:.4f}',
            xy=(w_l_final * 100, w_c_final * 100),
            xytext=(w_l_final * 100 + 5, w_c_final * 100 + 5),
            fontsize=9, color="white", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="white"))

# Diagonal line (XGBoost = 0)
ax.plot([0, 100], [100, 0], "w--", alpha=0.5, linewidth=1)
ax.text(50, 55, "XGB=0%", color="white", fontsize=9, alpha=0.7, rotation=-45)

ax.set_xlabel("LightGBM Weight (%)", fontsize=13)
ax.set_ylabel("CatBoost Weight (%)", fontsize=13)
ax.set_title("3-Model Blend Weight Heatmap\n(XGBoost = 100% - LGBM% - Cat%)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("blend_3model_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("blend_3model_heatmap.png saved!")

# --- Chart 4: Final Comparison Bar (개별 3개 + 2-model 3쌍 + 3-model) ---
fig, ax = plt.subplots(figsize=(14, 6))

labels = ["LightGBM", "CatBoost", "XGBoost"]
scores_all = [score_l, score_c, score_x]
colors = ["#42A5F5", "#FF7043", "#66BB6A"]

# 2-model pairs
for pname, pr in pair_results.items():
    a = pr["alpha"]
    labels.append(f"{pname}\n({pr['name_a'][:1]}{a*100:.0f}+{pr['name_b'][:1]}{(1-a)*100:.0f})")
    scores_all.append(pr["score"])
    colors.append(pair_colors[pname])

# 3-model
labels.append(f"3-Blend\n(L{w_l_final*100:.0f}+C{w_c_final*100:.0f}+X{w_x_final*100:.0f})")
scores_all.append(score_3)
colors.append("#78909C")

bars = ax.bar(labels, scores_all, color=colors, edgecolor="white", width=0.65)

for bar, score in zip(bars, scores_all):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0002,
            f"{score:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

# Winner highlight
winner_idx = np.argmax(scores_all)
bars[winner_idx].set_edgecolor("gold")
bars[winner_idx].set_linewidth(3)

# Section dividers
ax.axvline(x=2.5, color="gray", linestyle="--", alpha=0.3)
ax.axvline(x=5.5, color="gray", linestyle="--", alpha=0.3)
ax.text(1, min(scores_all) - 0.003, "Single Models", ha="center", fontsize=9, color="gray")
ax.text(4, min(scores_all) - 0.003, "2-Model Blends", ha="center", fontsize=9, color="gray")
ax.text(6, min(scores_all) - 0.003, "3-Model", ha="center", fontsize=9, color="gray")

ax.set_ylabel("(F1 + AUC) / 2", fontsize=13)
ax.set_title("Model & Blend Comparison (OOF Score)", fontsize=14, fontweight="bold")
ax.set_ylim(min(scores_all) - 0.006, max(scores_all) + 0.003)
ax.grid(axis="y", alpha=0.3)
plt.xticks(fontsize=9)
plt.tight_layout()
plt.savefig("model_comparison_chart.png", dpi=150, bbox_inches="tight")
plt.close()
print("model_comparison_chart.png saved!")

# --- Chart 5: Threshold Tuning Curve ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(thresholds, f1_by_thr, color="#E53935", linewidth=2, label="F1 Score")
ax.plot(thresholds, combined_by_thr, color="#2196F3", linewidth=2, label="(F1+AUC)/2")
ax.axhline(y=auc_fixed, color="#4CAF50", linestyle="--", alpha=0.7,
           label=f"AUC = {auc_fixed:.4f} (fixed)")

ax.axvline(x=best_threshold, color="gray", linestyle=":", alpha=0.7)
ax.scatter([best_threshold], [best_combined_thr], color="#2196F3", s=100, zorder=5)
ax.annotate(f'Best: thr={best_threshold:.3f}\n(F1+AUC)/2={best_combined_thr:.4f}',
            xy=(best_threshold, best_combined_thr),
            xytext=(best_threshold + 0.06, best_combined_thr - 0.015),
            fontsize=10, arrowprops=dict(arrowstyle="->", color="gray"))

ax.set_xlabel("Threshold", fontsize=13)
ax.set_ylabel("Score", fontsize=13)
ax.set_title(f"Threshold Tuning: {blend_label}", fontsize=14, fontweight="bold")
ax.legend(loc="lower left", fontsize=11)
ax.set_xlim(0.20, 0.80)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("threshold_tuning_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("threshold_tuning_curve.png saved!")

# ============================================================
# 14. 보고서용 마크다운 요약
# ============================================================
print()
print("=" * 60)
print("보고서용 마크다운")
print("=" * 60)
print()

print("### OOF Prediction Correlation")
print()
print("| | LightGBM | CatBoost | XGBoost |")
print("|------|----------|----------|---------|")
print(f"| LightGBM | 1.0000 | {corr_matrix.loc['LightGBM','CatBoost']:.4f} | {corr_matrix.loc['LightGBM','XGBoost']:.4f} |")
print(f"| CatBoost | {corr_matrix.loc['CatBoost','LightGBM']:.4f} | 1.0000 | {corr_matrix.loc['CatBoost','XGBoost']:.4f} |")
print(f"| XGBoost | {corr_matrix.loc['XGBoost','LightGBM']:.4f} | {corr_matrix.loc['XGBoost','CatBoost']:.4f} | 1.0000 |")
print()

print("### Model & Blend Comparison")
print()
print("| Model | F1 | AUC | (F1+AUC)/2 |")
print("|-------|-----|-----|-----------|")
print(f"| LightGBM | {f1_l:.4f} | {auc_l:.4f} | {score_l:.4f} |")
print(f"| CatBoost | {f1_c:.4f} | {auc_c:.4f} | {score_c:.4f} |")
print(f"| XGBoost | {f1_x:.4f} | {auc_x:.4f} | {score_x:.4f} |")
for pname, pr in pair_results.items():
    a = pr["alpha"]
    label = f"{pr['name_a']} {a*100:.0f}% + {pr['name_b']} {(1-a)*100:.0f}%"
    marker = " **← best 2-model**" if pname == best_pair_name else ""
    print(f"| {label}{marker} | {pr['f1']:.4f} | {pr['auc']:.4f} | {pr['score']:.4f} |")
print(f"| LGBM {w_l_final*100:.0f}% + Cat {w_c_final*100:.0f}% + XGB {w_x_final*100:.0f}% | {f1_3:.4f} | {auc_3:.4f} | {score_3:.4f} |")
print()

print(f">>> 권장: **{recommendation}** ({rec_detail})")
print(f">>> Threshold: {best_threshold:.3f}")
print()

# ============================================================
# 15. 생성된 파일 목록
# ============================================================
print("=" * 60)
print("Generated files:")
print("=" * 60)
print("  [Data]")
print("    prediction.csv       — 최종 제출용")
print("    prob_lgbm.npy        — LightGBM test probs")
print("    prob_catboost.npy    — CatBoost test probs")
print("    prob_xgboost.npy     — XGBoost test probs")
print("    oof_lgbm.npy         — LightGBM OOF probs")
print("    oof_catboost.npy     — CatBoost OOF probs")
print("    oof_xgboost.npy      — XGBoost OOF probs")
print()
print("  [Charts]")
print("    correlation_heatmap.png     — OOF 상관관계")
print("    blend_2model_curves.png     — 2-Model 가중치 곡선 (3쌍)")
print("    blend_3model_heatmap.png    — 3-Model 가중치 히트맵")
print("    model_comparison_chart.png  — 모델별 성능 비교")
print("    threshold_tuning_curve.png  — Threshold tuning")
print()
print("Done!")

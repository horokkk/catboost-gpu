"""
Group Aggregation Features + Rank Averaging 테스트
- Part 1: 기존 OOF에 Rank Averaging 적용 (즉시 결과, 모델 학습 없음)
- Part 2: Group Aggregation Features + LightGBM 재학습 (OOF 평가)
- Part 3: Group Agg + Rank Averaging 결합 블렌딩

사용법: python group_rank_test.py
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from scipy.stats import rankdata
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 전처리 함수
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


def add_group_agg_features(X_train, X_test):
    """Group Aggregation Features"""
    X_train = X_train.copy()
    X_test = X_test.copy()

    group_configs = [
        ("occupation", ["age", "hours_per_week", "education_num"]),
        ("workclass", ["age", "hours_per_week", "education_num"]),
    ]

    for group_col, num_cols in group_configs:
        for num_col in num_cols:
            group_mean = X_train.groupby(group_col)[num_col].mean()
            group_std = X_train.groupby(group_col)[num_col].std().fillna(0)

            mean_feat = f"{num_col}_mean_by_{group_col}"
            std_feat = f"{num_col}_std_by_{group_col}"
            dev_feat = f"{num_col}_dev_{group_col}"

            fallback = X_train[num_col].mean()
            X_train[mean_feat] = X_train[group_col].map(group_mean).fillna(fallback)
            X_test[mean_feat] = X_test[group_col].map(group_mean).fillna(fallback)

            X_train[std_feat] = X_train[group_col].map(group_std).fillna(0)
            X_test[std_feat] = X_test[group_col].map(group_std).fillna(0)

            X_train[dev_feat] = X_train[num_col] - X_train[mean_feat]
            X_test[dev_feat] = X_test[num_col] - X_test[mean_feat]

    return X_train, X_test


# ============================================================
# 데이터 로드
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


# ============================================================
# Part 1: Rank Averaging on existing OOF (즉시 결과)
# ============================================================
print("\n" + "=" * 70)
print("Part 1: Rank Averaging on existing OOF predictions")
print("=" * 70)

try:
    oof_lgbm = np.load("oof_lgbm.npy")
    oof_cat = np.load("oof_catboost.npy")
    oof_xgb = np.load("oof_xgboost.npy")

    # 기존 확률 블렌딩 (baseline)
    weights_list = [
        (0.40, 0.49, 0.11, "L40_C49_X11"),
        (0.53, 0.47, 0.00, "L53_C47 (2-model)"),
    ]

    for w_l, w_c, w_x, label in weights_list:
        # Probability averaging
        blend_prob = w_l * oof_lgbm + w_c * oof_cat + w_x * oof_xgb

        best_f1_prob, best_thr_prob = 0, 0.5
        for thr in np.arange(0.20, 0.80, 0.001):
            f1_t = f1_score(y, (blend_prob >= thr).astype(int))
            if f1_t > best_f1_prob:
                best_f1_prob = f1_t
                best_thr_prob = thr
        auc_prob = roc_auc_score(y, blend_prob)
        score_prob = (best_f1_prob + auc_prob) / 2

        # Rank averaging
        rank_l = rankdata(oof_lgbm) / len(oof_lgbm)
        rank_c = rankdata(oof_cat) / len(oof_cat)
        rank_x = rankdata(oof_xgb) / len(oof_xgb)
        blend_rank = w_l * rank_l + w_c * rank_c + w_x * rank_x

        best_f1_rank, best_thr_rank = 0, 0.5
        for thr in np.arange(0.50, 0.90, 0.001):
            f1_t = f1_score(y, (blend_rank >= thr).astype(int))
            if f1_t > best_f1_rank:
                best_f1_rank = f1_t
                best_thr_rank = thr
        auc_rank = roc_auc_score(y, blend_rank)
        score_rank = (best_f1_rank + auc_rank) / 2

        delta = score_rank - score_prob
        print(f"\n  [{label}]")
        print(f"    Prob Avg:  F1={best_f1_prob:.4f}, AUC={auc_prob:.4f}, Score={score_prob:.4f} (thr={best_thr_prob:.3f})")
        print(f"    Rank Avg:  F1={best_f1_rank:.4f}, AUC={auc_rank:.4f}, Score={score_rank:.4f} (thr={best_thr_rank:.3f})")
        print(f"    Delta: {delta:+.4f} {'★ IMPROVED' if delta > 0 else ''}")

except FileNotFoundError:
    print("  OOF files not found. Skipping Part 1.")


# ============================================================
# Part 2: Group Aggregation + LightGBM (기존 params vs 새 params)
# ============================================================
print("\n" + "=" * 70)
print("Part 2: Group Aggregation Features + LightGBM")
print("=" * 70)

# 기존 최적 파라미터 (Optuna trial 187/200)
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
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def run_lgbm_experiment(X, X_test, y, te_cols, use_group_agg, label):
    """LightGBM 실험 실행"""
    X = X.copy()
    X_test = X_test.copy()

    # Group Aggregation
    if use_group_agg:
        X, X_test = add_group_agg_features(X, X_test)

    # Target Encoding
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

    for col in cat_cols:
        X[col] = X[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    # 5-Fold CV
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

    best_f1, best_thr = f1, 0.5
    for thr in np.arange(0.20, 0.80, 0.001):
        f1_t = f1_score(y, (oof_prob >= thr).astype(int))
        if f1_t > best_f1:
            best_f1 = f1_t
            best_thr = thr
    score_tuned = (best_f1 + auc) / 2

    return {
        "label": label, "f1": f1, "auc": auc, "score": score,
        "f1_tuned": best_f1, "score_tuned": score_tuned, "threshold": best_thr,
        "features": X.shape[1], "oof": oof_prob,
        "X": X, "X_test": X_test,
    }


# 실험들
experiments = [
    (["occupation", "native_country"], False, "A: TE(occ+country) only (baseline)"),
    (["occupation", "native_country"], True,  "B: TE(occ+country) + GroupAgg"),
]

results = []
for te_cols, use_ga, label in experiments:
    print(f"\nRunning {label}...")
    r = run_lgbm_experiment(X_base, X_test_base, y, te_cols, use_ga, label)
    results.append(r)
    print(f"  Features: {r['features']}, F1: {r['f1']:.4f}, AUC: {r['auc']:.4f}, "
          f"Score: {r['score']:.4f} → tuned: {r['score_tuned']:.4f} (thr={r['threshold']:.3f})")

baseline_score = results[0]["score_tuned"]
print(f"\n  Improvement: {results[1]['score_tuned'] - baseline_score:+.4f}")


# ============================================================
# Part 3: Group Agg LightGBM + Rank Averaging 블렌딩
# ============================================================
print("\n" + "=" * 70)
print("Part 3: Group Agg LightGBM OOF + Existing CatBoost/XGB OOF + Rank Avg")
print("=" * 70)

try:
    oof_cat = np.load("oof_catboost.npy")
    oof_xgb = np.load("oof_xgboost.npy")
    oof_lgbm_ga = results[1]["oof"]  # Group Agg LightGBM OOF

    # Probability averaging
    for w_l, w_c, w_x in [(0.40, 0.49, 0.11), (0.45, 0.45, 0.10), (0.50, 0.40, 0.10)]:
        blend_prob = w_l * oof_lgbm_ga + w_c * oof_cat + w_x * oof_xgb
        best_f1_p, best_thr_p = 0, 0.5
        for thr in np.arange(0.20, 0.80, 0.001):
            f1_t = f1_score(y, (blend_prob >= thr).astype(int))
            if f1_t > best_f1_p:
                best_f1_p = f1_t
                best_thr_p = thr
        auc_p = roc_auc_score(y, blend_prob)
        score_p = (best_f1_p + auc_p) / 2
        print(f"  Prob  L{w_l:.0%} C{w_c:.0%} X{w_x:.0%}: Score={score_p:.4f} (thr={best_thr_p:.3f})")

    print()

    # Rank averaging
    rank_l = rankdata(oof_lgbm_ga) / len(oof_lgbm_ga)
    rank_c = rankdata(oof_cat) / len(oof_cat)
    rank_x = rankdata(oof_xgb) / len(oof_xgb)

    for w_l, w_c, w_x in [(0.40, 0.49, 0.11), (0.45, 0.45, 0.10), (0.50, 0.40, 0.10)]:
        blend_rank = w_l * rank_l + w_c * rank_c + w_x * rank_x
        best_f1_r, best_thr_r = 0, 0.5
        for thr in np.arange(0.50, 0.90, 0.001):
            f1_t = f1_score(y, (blend_rank >= thr).astype(int))
            if f1_t > best_f1_r:
                best_f1_r = f1_t
                best_thr_r = thr
        auc_r = roc_auc_score(y, blend_rank)
        score_r = (best_f1_r + auc_r) / 2
        print(f"  Rank  L{w_l:.0%} C{w_c:.0%} X{w_x:.0%}: Score={score_r:.4f} (thr={best_thr_r:.3f})")

    # scipy optimize로 최적 비율 탐색
    print("\n  Scipy Optimize (Rank Averaging)...")
    from scipy.optimize import minimize

    def neg_score_rank(weights):
        w = np.abs(weights)
        w = w / w.sum()
        blend = w[0] * rank_l + w[1] * rank_c + w[2] * rank_x
        best_f1 = 0
        for thr in np.arange(0.55, 0.85, 0.005):
            f1_t = f1_score(y, (blend >= thr).astype(int))
            if f1_t > best_f1:
                best_f1 = f1_t
        auc = roc_auc_score(y, blend)
        return -((best_f1 + auc) / 2)

    result_opt = minimize(neg_score_rank, x0=[0.40, 0.49, 0.11], method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 0.001, 'fatol': 1e-6})
    opt_w = np.abs(result_opt.x)
    opt_w = opt_w / opt_w.sum()
    print(f"  Optimal weights: LGBM {opt_w[0]:.3f}, Cat {opt_w[1]:.3f}, XGB {opt_w[2]:.3f}")
    print(f"  Optimal score: {-result_opt.fun:.4f}")

    # 최종 best로 test prediction 생성
    blend_final = opt_w[0] * rank_l + opt_w[1] * rank_c + opt_w[2] * rank_x
    best_f1_final, best_thr_final = 0, 0.5
    for thr in np.arange(0.50, 0.90, 0.001):
        f1_t = f1_score(y, (blend_final >= thr).astype(int))
        if f1_t > best_f1_final:
            best_f1_final = f1_t
            best_thr_final = thr

    print(f"  Best threshold: {best_thr_final:.3f}")

except FileNotFoundError:
    print("  OOF files not found. Skipping Part 3.")


# ============================================================
# Part 4: Multi-seed test prediction (best 결과가 개선된 경우)
# ============================================================
print("\n" + "=" * 70)
print("Part 4: Multi-seed Test Prediction (Group Agg LightGBM)")
print("=" * 70)

best_ga = results[1]
if best_ga["score_tuned"] > results[0]["score_tuned"]:
    print("Group Agg improved! Generating multi-seed test predictions...")

    X_ga = best_ga["X"]
    X_test_ga = best_ga["X_test"]
    seeds = [42, 123, 456, 789, 2024]
    test_probs = np.zeros(len(X_test_ga))

    for seed_i, seed in enumerate(seeds):
        seed_params = lgbm_params.copy()
        seed_params["seed"] = seed
        skf_seed = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (train_idx, val_idx) in enumerate(skf_seed.split(X_ga, y)):
            dtrain = lgb.Dataset(X_ga.iloc[train_idx], label=y.iloc[train_idx], categorical_feature=cat_cols)
            dval = lgb.Dataset(X_ga.iloc[val_idx], label=y.iloc[val_idx], categorical_feature=cat_cols)
            model = lgb.train(seed_params, dtrain, num_boost_round=3000, valid_sets=[dval],
                              callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
            test_probs += model.predict(X_test_ga) / (5 * len(seeds))
        print(f"  Seed {seed} done ({seed_i+1}/{len(seeds)})")

    np.save("prob_lgbm_ga.npy", test_probs)
    print(f"\nprob_lgbm_ga.npy saved!")
    print(f"  mean prob: {test_probs.mean():.4f}")

    # Rank averaging으로 최종 블렌딩
    try:
        prob_cat = np.load("prob_catboost.npy")
        prob_xgb_test = np.load("prob_xgb.npy")

        rank_l = rankdata(test_probs) / len(test_probs)
        rank_c = rankdata(prob_cat) / len(prob_cat)
        rank_x = rankdata(prob_xgb_test) / len(prob_xgb_test)

        # 기본 비율 + 최적 비율 둘 다 생성
        for w_l, w_c, w_x, thr, lbl in [
            (0.40, 0.49, 0.11, best_thr_final if 'best_thr_final' in dir() else 0.76, "default"),
            (opt_w[0], opt_w[1], opt_w[2], best_thr_final, "optimized") if 'opt_w' in dir() else (0.40, 0.49, 0.11, 0.76, "skip"),
        ]:
            if lbl == "skip":
                continue
            blend = w_l * rank_l + w_c * rank_c + w_x * rank_x
            prediction = pd.DataFrame({
                "id": test_ids,
                "y_cls": (blend >= thr).astype(int),
                "y_prob": 0.40 * test_probs + 0.49 * prob_cat + 0.11 * prob_xgb_test,
            })
            fname = f"prediction_ga_rank_{lbl}.csv"
            prediction.to_csv(fname, index=False)
            print(f"\n{fname} saved!")
            print(f"  y_cls distribution: {prediction['y_cls'].value_counts().to_dict()}")

    except FileNotFoundError:
        print("  Test prob files not found. Skipping final blend.")
else:
    print("Group Agg did not improve. Skipping.")


print("\n" + "=" * 70)
print("All done!")
print("=" * 70)

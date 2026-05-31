"""
CatBoost Optuna 200 trial 하이퍼파라미터 튜닝 (GPU)
- XGBoost 이미 완료 → CatBoost만 단독 실행
- A+K 킬러 피처 포함

사용법: PYTHONPATH=/data/jiyoonkim/pylibs:$PYTHONPATH python3 optuna_catboost_only.py
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
import optuna
from optuna.samplers import TPESampler
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
    df["gain_15024"] = (df["capital_gain"] == 15024).astype(int)
    df["gain_7688"] = (df["capital_gain"] == 7688).astype(int)
    df["gain_7298"] = (df["capital_gain"] == 7298).astype(int)
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
    df["log_age"] = np.log1p(df["age"])
    df["age_cubed"] = df["age"] ** 3
    df["inv_age"] = 1.0 / (df["age"] + 1)
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

cat_cols = ["workclass", "marital_status", "occupation", "relationship",
            "race", "sex", "native_country"]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ============================================================
# 3. CatBoost Optuna (GPU)
# ============================================================
print("\n" + "=" * 60)
print("CatBoost Optuna Tuning (200 trials) - GPU")
print("=" * 60)

X_cb = X.copy()
X_test_cb = X_test.copy()
for col in cat_cols:
    X_cb[col] = X_cb[col].astype(str)
    X_test_cb[col] = X_test_cb[col].astype(str)

cat_indices = [X_cb.columns.get_loc(c) for c in cat_cols]


def evaluate_catboost(params):
    oof_prob = np.zeros(len(y))
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_cb, y)):
        train_pool = Pool(X_cb.iloc[train_idx], label=y.iloc[train_idx], cat_features=cat_indices)
        val_pool = Pool(X_cb.iloc[val_idx], label=y.iloc[val_idx], cat_features=cat_indices)
        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100, verbose=0)
        oof_prob[val_idx] = model.predict_proba(X_cb.iloc[val_idx])[:, 1]
    f1 = f1_score(y, (oof_prob >= 0.5).astype(int))
    auc = roc_auc_score(y, oof_prob)
    return f1, auc, oof_prob


def objective_cb(trial):
    params = {
        "iterations": 2000, "eval_metric": "Logloss", "random_seed": 42, "verbose": 0,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 10.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 5.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
        "max_ctr_complexity": trial.suggest_int("max_ctr_complexity", 1, 4),
    }
    f1, auc, _ = evaluate_catboost(params)
    return (f1 + auc) / 2


optuna.logging.set_verbosity(optuna.logging.WARNING)
study_cb = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
study_cb.optimize(objective_cb, n_trials=200, show_progress_bar=True)

print(f"\n★ CatBoost Best (F1+AUC)/2: {study_cb.best_value:.6f}")
print(f"★ CatBoost Best params: {study_cb.best_params}")

# CatBoost 최종 학습 + multi-seed test 예측
best_params_cb = study_cb.best_params.copy()
best_params_cb.update({"iterations": 3000, "eval_metric": "Logloss", "random_seed": 42, "verbose": 0})

f1_final, auc_final, oof_cb = evaluate_catboost(best_params_cb)

best_thr_cb, best_f1_cb = 0.5, f1_final
for thr in np.arange(0.2, 0.8, 0.005):
    f1_t = f1_score(y, (oof_cb >= thr).astype(int))
    if f1_t > best_f1_cb:
        best_f1_cb = f1_t
        best_thr_cb = thr

print(f"  Tuned: F1={best_f1_cb:.4f}, thr={best_thr_cb:.3f}, Score={(best_f1_cb + auc_final) / 2:.4f}")

seeds = [42, 123, 456, 789, 2024]
test_probs_cb = np.zeros(len(X_test_cb))

for seed_i, seed in enumerate(seeds):
    seed_params = best_params_cb.copy()
    seed_params["random_seed"] = seed
    skf_seed = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(skf_seed.split(X_cb, y)):
        train_pool = Pool(X_cb.iloc[tr_idx], label=y.iloc[tr_idx], cat_features=cat_indices)
        val_pool = Pool(X_cb.iloc[va_idx], label=y.iloc[va_idx], cat_features=cat_indices)
        model = CatBoostClassifier(**seed_params)
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100, verbose=0)
        test_probs_cb += model.predict_proba(X_test_cb)[:, 1] / (5 * len(seeds))
    print(f"  CB Seed {seed} done ({seed_i+1}/{len(seeds)})")

np.save("prob_catboost.npy", test_probs_cb)
print("prob_catboost.npy saved!")


# ============================================================
# 4. 하드코딩용 Best Params 출력
# ============================================================
print("\n" + "=" * 60)
print("하드코딩용 CatBoost Best Params")
print("=" * 60)
print(f"best_params_cb = {{")
for k, v in study_cb.best_params.items():
    print(f'    "{k}": {v},')
print(f"}}")

print("\nDone!")

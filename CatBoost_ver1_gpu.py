"""
CatBoost (GPU) - Adult Income Prediction
Pattern Recognition Project (Final: 5/31)
Metric: (F1 + AUC) / 2

Colab / GPU 서버 사용법:
1. train.csv, test.csv 업로드
2. !pip install catboost optuna
3. GPU 런타임 확인 후 실행
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
# 1. 전처리 함수 (LightGBM_ver2와 동일, target encoding 제외)
# ============================================================
def preprocess_adult_income(df):
    df = df.copy()

    # 결측치 처리: NaN -> 'Unknown'
    cat_missing_cols = ["workclass", "occupation", "native_country"]
    for col in cat_missing_cols:
        df[col] = df[col].replace(np.nan, "Unknown")

    # education 제거 (education_num과 같기 때문)
    if "education" in df.columns:
        df.drop(columns=["education"], inplace=True)

    # capital_gain / capital_loss feature engineering
    df["has_capital_gain"] = (df["capital_gain"] > 0).astype(int)
    df["has_capital_loss"] = (df["capital_loss"] > 0).astype(int)
    df["log_capital_gain"] = np.log1p(df["capital_gain"])
    df["log_capital_loss"] = np.log1p(df["capital_loss"])
    df["net_capital"] = df["capital_gain"] - df["capital_loss"]

    # native_country, race, workclass: 원본 유지 (CatBoost native categorical)

    # Feature Engineering
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

    # 추가 feature engineering (v2)
    df["is_full_time"] = (df["hours_per_week"] == 40).astype(int)
    df["capital_per_hour"] = df["net_capital"] / (df["hours_per_week"] + 1)
    df["log_abs_net_capital"] = np.log1p(np.abs(df["net_capital"])) * np.sign(df["net_capital"])
    df["age_per_edu"] = df["age"] / (df["education_num"] + 1)
    df["gain_loss_ratio"] = df["log_capital_gain"] / (df["log_capital_loss"] + 1)
    df["is_high_edu"] = (df["education_num"] >= 13).astype(int)  # Bachelors+
    df["high_edu_married"] = df["is_high_edu"] * df["is_married"]
    df["overtime_married"] = df["overtime"] * df["is_married"]

    # occupation: 원본 유지 (CatBoost native categorical + ordered target encoding)

    return df

# ============================================================
# 2. 데이터 로드 + 전처리
# ============================================================
train_df = pd.read_csv("train.csv", na_values=["", " "])
test_df = pd.read_csv("test.csv", na_values=["", " "])

train_df = preprocess_adult_income(train_df)
test_df = preprocess_adult_income(test_df)

# 타겟 수치화
train_df["income"] = train_df["income"].apply(lambda x: 1 if ">50K" in str(x) else 0)

# id 분리
train_ids = train_df["id"]
test_ids = test_df["id"]
train_df.drop(columns=["id"], inplace=True)
test_df.drop(columns=["id"], inplace=True)

# X, y 분리
y = train_df["income"]
X = train_df.drop(columns=["income"])
X_test = test_df.copy()

# ============================================================
# 3. CatBoost Categorical 설정
# ============================================================
cat_cols = ["workclass", "marital_status", "occupation", "relationship",
            "race", "sex", "native_country"]

# CatBoost: string 타입으로 유지 (ordered target encoding 내장)
for col in cat_cols:
    X[col] = X[col].astype(str)
    X_test[col] = X_test[col].astype(str)

cat_indices = [X.columns.get_loc(c) for c in cat_cols]

print(f"Train shape: {X.shape}")
print(f"Test shape: {X_test.shape}")
print(f"Target distribution: {y.value_counts().to_dict()}")
print(f"Categorical features: {cat_cols}")
print()

# ============================================================
# 4. 평가 함수
# ============================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_catboost(params, X, y, skf):
    """5-Fold CV로 F1, AUC 측정"""
    oof_prob = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        train_pool = Pool(X_tr, label=y_tr, cat_features=cat_indices)
        val_pool = Pool(X_val, label=y_val, cat_features=cat_indices)

        model = CatBoostClassifier(**params)
        model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=100,
            verbose=0
        )

        oof_prob[val_idx] = model.predict_proba(X_val)[:, 1]

    oof_cls = (oof_prob >= 0.5).astype(int)
    f1 = f1_score(y, oof_cls)
    auc = roc_auc_score(y, oof_prob)
    return f1, auc, oof_prob

# ============================================================
# 5. Baseline 평가
# ============================================================
baseline_params = {
    "iterations": 2000,
    "learning_rate": 0.05,
    "depth": 6,
    "eval_metric": "Logloss",
    "random_seed": 42,
    "task_type": "CPU",
    "auto_class_weights": "Balanced",
}

print("=" * 50)
print("Baseline (default params) - GPU")
print("=" * 50)
f1_base, auc_base, _ = evaluate_catboost(baseline_params, X, y, skf)
print(f"F1:  {f1_base:.4f}")
print(f"AUC: {auc_base:.4f}")
print(f"(F1+AUC)/2: {(f1_base + auc_base) / 2:.4f}")
print()

# ============================================================
# 6. Optuna 하이퍼파라미터 튜닝
# ============================================================
def objective(trial):
    params = {
        "iterations": 2000,
        "eval_metric": "Logloss",
        "random_seed": 42,
        "task_type": "CPU",
        "verbose": 0,
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

    f1, auc, _ = evaluate_catboost(params, X, y, skf)
    return (f1 + auc) / 2


print("=" * 50)
print("Optuna Tuning (50 trials) - GPU")
print("=" * 50)

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=50, show_progress_bar=True)

best_params = study.best_params
best_params.update({
    "iterations": 3000,
    "eval_metric": "Logloss",
    "random_seed": 42,
    "task_type": "CPU",
    "verbose": 0,
})

print(f"\nBest (F1+AUC)/2: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
print()

# ============================================================
# 7. 최종 모델 + Threshold Tuning
# ============================================================
print("=" * 50)
print("Final Model (tuned params)")
print("=" * 50)

f1_final, auc_final, oof_prob = evaluate_catboost(best_params, X, y, skf)
print(f"F1:  {f1_final:.4f}")
print(f"AUC: {auc_final:.4f}")
print(f"(F1+AUC)/2: {(f1_final + auc_final) / 2:.4f}")
print()

# Threshold tuning
thresholds = np.arange(0.2, 0.8, 0.005)
best_thr, best_f1_thr = 0.5, f1_final
for thr in thresholds:
    f1_t = f1_score(y, (oof_prob >= thr).astype(int))
    if f1_t > best_f1_thr:
        best_f1_thr = f1_t
        best_thr = thr

print(f"Optimal threshold: {best_thr:.3f} (F1: {best_f1_thr:.4f})")
auc_final_thr = roc_auc_score(y, oof_prob)
print(f"Final (F1+AUC)/2 with threshold tuning: {(best_f1_thr + auc_final_thr) / 2:.4f}")
print()

# ============================================================
# 8. Test 예측 + prob_catboost.npy 저장
# ============================================================
print("=" * 50)
print("Test Prediction (multi-seed averaging)")
print("=" * 50)

seeds = [42, 123, 456, 789, 2024]
test_probs = np.zeros(len(X_test))

for seed_i, seed in enumerate(seeds):
    seed_params = best_params.copy()
    seed_params["random_seed"] = seed
    skf_seed = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(skf_seed.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        train_pool = Pool(X_tr, label=y_tr, cat_features=cat_indices)
        val_pool = Pool(X_val, label=y_val, cat_features=cat_indices)

        model = CatBoostClassifier(**seed_params)
        model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=100,
            verbose=0
        )

        test_probs += model.predict_proba(X_test)[:, 1] / (5 * len(seeds))

    print(f"  Seed {seed} done ({seed_i+1}/{len(seeds)})")

# prediction_catboost.csv
prediction = pd.DataFrame({
    "id": test_ids,
    "y_cls": (test_probs >= best_thr).astype(int),
    "y_prob": test_probs
})

prediction.to_csv("prediction_catboost.csv", index=False)
print(f"prediction_catboost.csv saved! ({len(prediction)} rows)")
print(f"  y_cls distribution: {prediction['y_cls'].value_counts().to_dict()}")
print(f"  y_prob mean: {prediction['y_prob'].mean():.4f}")

# 블렌딩용
np.save("prob_catboost.npy", test_probs)
print("prob_catboost.npy saved!")
print()

# ============================================================
# 9. Feature Importance (보고서용)
# ============================================================
print("=" * 50)
print("Top 15 Feature Importance")
print("=" * 50)

importance = pd.DataFrame({
    "feature": model.feature_names_,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

for i, row in importance.head(15).iterrows():
    print(f"  {row['feature']:25s} {row['importance']:.2f}")

# ============================================================
# 10. LGBM + CatBoost 블렌딩 (prob_lgbm.npy 있을 경우)
# ============================================================
import os
if os.path.exists("prob_lgbm.npy"):
    print()
    print("=" * 50)
    print("Blending: LGBM + CatBoost")
    print("=" * 50)

    lgbm_probs = np.load("prob_lgbm.npy")

    for alpha in [0.3, 0.4, 0.5, 0.6, 0.7]:
        blend = alpha * lgbm_probs + (1 - alpha) * test_probs
        blend_pred = pd.DataFrame({
            "id": test_ids,
            "y_cls": (blend >= 0.5).astype(int),
            "y_prob": blend
        })
        blend_pred.to_csv(f"prediction_blend_{int(alpha*100)}lgbm_{int((1-alpha)*100)}cat.csv", index=False)
        print(f"  LGBM {int(alpha*100)}% + CatBoost {int((1-alpha)*100)}% -> "
              f"prediction_blend_{int(alpha*100)}lgbm_{int((1-alpha)*100)}cat.csv saved "
              f"(mean prob: {blend.mean():.4f})")

    print()
    print("  * 제출 시 다양한 비율의 csv 중 하나를 prediction.csv로 변경하여 제출")
    print("  * 추천: 50:50 또는 OOF 기반 최적 비율")

print("\nDone!")

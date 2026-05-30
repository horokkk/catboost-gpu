"""
XGBoost (Server) - Adult Income Prediction
Pattern Recognition Project (Final: 5/31)
Metric: (F1 + AUC) / 2

서버 사용법:
1. train.csv, test.csv 같은 디렉토리에 배치
2. pip install xgboost optuna scikit-learn
3. python XGBoost_server.py

※ code_clean.ipynb cell-22 기준 (전처리 함수 동일)
※ LightGBM prob 파일(prob_lgbm.npy)이 있으면 블렌딩도 수행
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. 전처리 함수 (code_clean.ipynb와 동일)
# ============================================================
def preprocess_adult_income(df):
    df = df.copy()

    # 1. 결측치 처리: NaN -> 'Unknown'
    cat_missing_cols = ["workclass", "occupation", "native_country"]
    for col in cat_missing_cols:
        df[col] = df[col].replace(np.nan, "Unknown")

    # 2. education 제거 (education_num과 1:1 매칭되므로 중복)
    if "education" in df.columns:
        df.drop(columns=["education"], inplace=True)

    # 3. Capital 관련 Feature Engineering
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

    # 4. 카테고리 변수: 원본 유지
    # native_country, race, workclass, occupation 모두 원본 그대로 사용

    # 5. 결혼/가족 관련 Feature Engineering
    married_categories = ["Married-civ-spouse", "Married-AF-spouse"]
    df["is_married"] = df["marital_status"].isin(married_categories).astype(int)
    df["married_male"] = ((df["is_married"] == 1) & (df["sex"] == "Male")).astype(int)
    df["is_spouse"] = df["relationship"].isin(["Husband", "Wife"]).astype(int)

    # 6. 나이/교육 관련 Feature Engineering
    df["age_education"] = df["age"] * df["education_num"]
    df["age_squared"] = df["age"] ** 2
    df["age_per_edu"] = df["age"] / (df["education_num"] + 1)
    df["is_high_edu"] = (df["education_num"] >= 13).astype(int)
    df["high_edu_married"] = df["is_high_edu"] * df["is_married"]

    # 7. 근무시간 관련 Feature Engineering
    df["overtime"] = (df["hours_per_week"] > 40).astype(int)
    df["edu_hours"] = df["education_num"] * df["hours_per_week"]
    df["is_part_time"] = (df["hours_per_week"] < 35).astype(int)
    df["is_full_time"] = (df["hours_per_week"] == 40).astype(int)
    df["hours_age"] = df["hours_per_week"] * df["age"]
    df["overtime_married"] = df["overtime"] * df["is_married"]

    return df


# ============================================================
# 2. 데이터 로드 및 전처리
# ============================================================
print("Loading data...")
train_df = pd.read_csv("train.csv", na_values=["", " "])
test_df = pd.read_csv("test.csv", na_values=["", " "])

train_df = preprocess_adult_income(train_df)
test_df = preprocess_adult_income(test_df)

train_df["income"] = train_df["income"].apply(lambda x: 1 if ">50K" in str(x) else 0)

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Target distribution: {train_df['income'].value_counts().to_dict()}")

train_ids = train_df["id"]
test_ids = test_df["id"]
train_df.drop(columns=["id"], inplace=True)
test_df.drop(columns=["id"], inplace=True)

y = train_df["income"]
X = train_df.drop(columns=["income"])
X_test = test_df.copy()

# XGBoost Categorical 타입 지정
cat_cols_xgb = ["workclass", "marital_status", "occupation", "relationship",
                "race", "sex", "native_country"]

for col in cat_cols_xgb:
    X[col] = X[col].astype("category")
    X_test[col] = X_test[col].astype("category")

print(f"Categorical features: {cat_cols_xgb}")


# ============================================================
# 3. 평가 함수
# ============================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_xgb(params, X, y, skf, num_boost_round=2000, early_stop=50, tune_threshold=False):
    oof_prob = np.zeros(len(y))
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
        dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
        model = xgb.train(params, dtrain, num_boost_round=num_boost_round,
                          evals=[(dval, "val")], early_stopping_rounds=early_stop, verbose_eval=False)
        oof_prob[val_idx] = model.predict(dval)
    auc = roc_auc_score(y, oof_prob)
    if tune_threshold:
        thresholds = np.arange(0.3, 0.7, 0.02)
        best_f1 = 0
        for thr in thresholds:
            f1_t = f1_score(y, (oof_prob >= thr).astype(int))
            if f1_t > best_f1:
                best_f1 = f1_t
        return best_f1, auc, oof_prob
    else:
        f1 = f1_score(y, (oof_prob >= 0.5).astype(int))
        return f1, auc, oof_prob


# ============================================================
# 4. Baseline (default params)
# ============================================================
print("\n" + "=" * 50)
print("Baseline (default params)")
print("=" * 50)

baseline_params = {
    "objective": "binary:logistic", "eval_metric": "logloss",
    "seed": 42, "n_jobs": -1, "tree_method": "hist",
    "scale_pos_weight": 3.0, "learning_rate": 0.05,
    "max_depth": 6, "min_child_weight": 1,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "alpha": 0.0, "lambda": 1.0, "gamma": 0.0,
}

f1_base, auc_base, _ = evaluate_xgb(baseline_params, X, y, skf)
print(f"F1:  {f1_base:.4f}")
print(f"AUC: {auc_base:.4f}")
print(f"(F1+AUC)/2: {(f1_base + auc_base) / 2:.4f}")


# ============================================================
# 5. Optuna 하이퍼파라미터 튜닝 (100 trials)
# ============================================================
def objective_xgb(trial):
    params = {
        "objective": "binary:logistic", "eval_metric": "logloss",
        "seed": 42, "n_jobs": -1, "tree_method": "hist",
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.5, 3.5),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "alpha": trial.suggest_float("alpha", 0.0, 10.0),
        "lambda": trial.suggest_float("lambda", 0.0, 10.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }
    f1, auc, _ = evaluate_xgb(params, X, y, skf, tune_threshold=True)
    return (f1 + auc) / 2

print("\n" + "=" * 50)
print("Optuna Tuning (100 trials)")
print("=" * 50)

optuna.logging.set_verbosity(optuna.logging.WARNING)
study_xgb = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))

# Baseline trial 주입
study_xgb.enqueue_trial({
    "scale_pos_weight": 3.0, "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 1,
    "subsample": 0.8, "colsample_bytree": 0.8, "alpha": 0.0, "lambda": 1.0, "gamma": 0.0
})

study_xgb.optimize(objective_xgb, n_trials=100, show_progress_bar=True)

best_params_xgb = study_xgb.best_params
best_params_xgb.update({
    "objective": "binary:logistic", "eval_metric": "logloss",
    "seed": 42, "n_jobs": -1, "tree_method": "hist",
})

print(f"\nBest (F1+AUC)/2: {study_xgb.best_value:.4f}")
print(f"Best params: {study_xgb.best_params}")


# ============================================================
# 6. 최종 모델 (tuned params, 3000 rounds)
# ============================================================
print("\n" + "=" * 50)
print("Final Model (tuned params)")
print("=" * 50)

f1_final, auc_final, oof_prob_xgb = evaluate_xgb(
    best_params_xgb, X, y, skf, num_boost_round=3000, early_stop=100
)
print(f"F1:  {f1_final:.4f}")
print(f"AUC: {auc_final:.4f}")
print(f"(F1+AUC)/2: {(f1_final + auc_final) / 2:.4f}")

# Threshold Tuning
thresholds = np.arange(0.2, 0.8, 0.005)
best_thr_xgb, best_f1_xgb = 0.5, f1_final
for thr in thresholds:
    f1_t = f1_score(y, (oof_prob_xgb >= thr).astype(int))
    if f1_t > best_f1_xgb:
        best_f1_xgb = f1_t
        best_thr_xgb = thr

print(f"\nOptimal threshold: {best_thr_xgb:.3f} (F1: {best_f1_xgb:.4f})")
print(f"Final (F1+AUC)/2 with threshold tuning: {(best_f1_xgb + auc_final) / 2:.4f}")


# ============================================================
# 7. Test 예측 (multi-seed averaging)
# ============================================================
print("\n" + "=" * 50)
print("Test Prediction (multi-seed averaging)")
print("=" * 50)

seeds = [42, 123, 456, 789, 2024]
test_probs_xgb = np.zeros(len(X_test))
dtest = xgb.DMatrix(X_test, enable_categorical=True)

for seed_i, seed in enumerate(seeds):
    seed_params = best_params_xgb.copy()
    seed_params["seed"] = seed
    skf_seed = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(skf_seed.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
        dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
        model = xgb.train(seed_params, dtrain, num_boost_round=3000,
                          evals=[(dval, "val")], early_stopping_rounds=100, verbose_eval=False)
        test_probs_xgb += model.predict(dtest) / (5 * len(seeds))
    print(f"  Seed {seed} done ({seed_i+1}/{len(seeds)})")

# prediction CSV 저장
y_cls = (test_probs_xgb >= best_thr_xgb).astype(int)
prediction = pd.DataFrame({
    "id": test_ids,
    "y_cls": y_cls,
    "y_prob": test_probs_xgb,
})
prediction.to_csv("prediction_xgboost.csv", index=False)
print(f"prediction_xgboost.csv saved! ({len(prediction)} rows)")
print(f"  y_cls distribution: {prediction['y_cls'].value_counts().to_dict()}")
print(f"  y_prob mean: {prediction['y_prob'].mean():.4f}")

# prob npy 저장
np.save("prob_xgb.npy", test_probs_xgb)
print("prob_xgb.npy saved!")


# ============================================================
# 8. Feature Importance
# ============================================================
print("\n" + "=" * 50)
print("Top 15 Feature Importance")
print("=" * 50)

# 전체 데이터로 모델 하나 학습해서 importance 확인
dtrain_full = xgb.DMatrix(X, label=y, enable_categorical=True)
model_full = xgb.train(best_params_xgb, dtrain_full, num_boost_round=3000, verbose_eval=False)
importance = model_full.get_score(importance_type="gain")
imp_df = pd.DataFrame({"feature": importance.keys(), "gain": importance.values()})
imp_df = imp_df.sort_values("gain", ascending=False)

for _, row in imp_df.head(15).iterrows():
    print(f"  {row['feature']:<25s} {row['gain']:.2f}")


# ============================================================
# 9. 블렌딩 (LGBM + XGBoost, 참고용)
# ============================================================
import os
if os.path.exists("prob_lgbm.npy"):
    print("\n" + "=" * 50)
    print("Blending: LGBM + XGBoost (참고용)")
    print("=" * 50)
    prob_lgbm = np.load("prob_lgbm.npy")
    for alpha in [0.3, 0.5, 0.7]:
        blend = alpha * prob_lgbm + (1 - alpha) * test_probs_xgb
        print(f"  LGBM {int(alpha*100)}% + XGB {int((1-alpha)*100)}% -> mean prob: {blend.mean():.4f}")

print("\nDone!")

"""
모델 다양성 블렌딩 테스트
- 기존 GBDT 3개 OOF + RF/LR OOF 추가하여 블렌딩 효과 측정
- RF, LR은 구조가 다르므로 GBDT와 상관관계가 낮을 수 있음
- scipy optimize로 최적 비율 탐색

사용법: PYTHONPATH=/data/jiyoonkim/pylibs:$PYTHONPATH python3 diversity_blend_test.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 전처리
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
X = train_df.drop(columns=["income"])
X_test = test_df.copy()

cat_cols = ["workclass", "marital_status", "occupation", "relationship",
            "race", "sex", "native_country"]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ============================================================
# Part 1: RF / ExtraTrees / LR OOF 생성
# ============================================================
print("\n" + "=" * 70)
print("Part 1: Generating diverse model OOF predictions")
print("=" * 70)

# Label Encoding for RF/ExtraTrees
X_le = X.copy()
X_test_le = X_test.copy()
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    X_le[col] = le.fit_transform(X_le[col])
    X_test_le[col] = le.transform(X_test_le[col])
    le_dict[col] = le

# OHE + Scaling for LR
X_ohe = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=int)
X_test_ohe = pd.get_dummies(X_test, columns=cat_cols, drop_first=True, dtype=int)
# align columns
X_test_ohe = X_test_ohe.reindex(columns=X_ohe.columns, fill_value=0)

# drop capital_gain/capital_loss for LR (log versions exist)
for c in ["capital_gain", "capital_loss"]:
    if c in X_ohe.columns:
        X_ohe.drop(columns=[c], inplace=True)
        X_test_ohe.drop(columns=[c], inplace=True)

scaler = StandardScaler()
num_cols = X_ohe.select_dtypes(include=["int64", "float64"]).columns
X_ohe_scaled = X_ohe.copy()
X_test_ohe_scaled = X_test_ohe.copy()
X_ohe_scaled[num_cols] = scaler.fit_transform(X_ohe[num_cols])
X_test_ohe_scaled[num_cols] = scaler.transform(X_test_ohe[num_cols])


diverse_models = {
    "RF": {
        "model": RandomForestClassifier(
            n_estimators=500, max_depth=15, min_samples_split=10,
            min_samples_leaf=1, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
        "X": X_le, "X_test": X_test_le,
    },
    "ExtraTrees": {
        "model": ExtraTreesClassifier(
            n_estimators=500, max_depth=15, min_samples_split=10,
            min_samples_leaf=1, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
        "X": X_le, "X_test": X_test_le,
    },
    "LR": {
        "model": LogisticRegression(
            max_iter=5000, class_weight="balanced", C=10.0,
            solver="lbfgs", random_state=42
        ),
        "X": X_ohe_scaled, "X_test": X_test_ohe_scaled,
    },
}

oof_diverse = {}
test_diverse = {}

for name, cfg in diverse_models.items():
    print(f"\n  Training {name}...")
    model = cfg["model"]
    X_m = cfg["X"]
    X_t = cfg["X_test"]

    oof_prob = np.zeros(len(y))
    test_prob = np.zeros(len(X_t))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_m, y)):
        model_clone = type(model)(**model.get_params())
        model_clone.fit(X_m.iloc[train_idx], y.iloc[train_idx])
        oof_prob[val_idx] = model_clone.predict_proba(X_m.iloc[val_idx])[:, 1]
        test_prob += model_clone.predict_proba(X_t)[:, 1] / 5

    f1 = f1_score(y, (oof_prob >= 0.5).astype(int))
    auc = roc_auc_score(y, oof_prob)

    # threshold tuning
    best_f1, best_thr = f1, 0.5
    for thr in np.arange(0.20, 0.80, 0.001):
        f1_t = f1_score(y, (oof_prob >= thr).astype(int))
        if f1_t > best_f1:
            best_f1 = f1_t
            best_thr = thr

    score = (best_f1 + auc) / 2
    print(f"    F1={best_f1:.4f}, AUC={auc:.4f}, Score={score:.4f} (thr={best_thr:.3f})")

    oof_diverse[name] = oof_prob
    test_diverse[name] = test_prob
    np.save(f"oof_{name.lower()}.npy", oof_prob)
    np.save(f"prob_{name.lower()}.npy", test_prob)


# ============================================================
# Part 2: 상관관계 분석
# ============================================================
print("\n" + "=" * 70)
print("Part 2: OOF Prediction Correlation")
print("=" * 70)

# Load existing GBDT OOFs
oof_lgbm = np.load("oof_lgbm.npy")
oof_cat = np.load("oof_catboost.npy")
oof_xgb = np.load("oof_xgboost.npy")

all_oofs = {
    "LGBM": oof_lgbm,
    "CatBoost": oof_cat,
    "XGBoost": oof_xgb,
    "RF": oof_diverse["RF"],
    "ExtraTrees": oof_diverse["ExtraTrees"],
    "LR": oof_diverse["LR"],
}

names = list(all_oofs.keys())
print(f"\n  {'':>12s}", end="")
for n in names:
    print(f" {n:>10s}", end="")
print()

for i, n1 in enumerate(names):
    print(f"  {n1:>12s}", end="")
    for j, n2 in enumerate(names):
        corr = np.corrcoef(all_oofs[n1], all_oofs[n2])[0, 1]
        print(f" {corr:>10.4f}", end="")
    print()


# ============================================================
# Part 3: 다양한 블렌딩 조합 테스트
# ============================================================
print("\n" + "=" * 70)
print("Part 3: Blending Combinations")
print("=" * 70)


def eval_blend(weights, oof_list, y):
    """가중 블렌딩 후 최적 threshold로 score 계산"""
    w = np.array(weights)
    w = w / w.sum()
    blend = sum(w[i] * oof_list[i] for i in range(len(w)))
    auc = roc_auc_score(y, blend)
    best_f1, best_thr = 0, 0.5
    for thr in np.arange(0.20, 0.80, 0.001):
        f1_t = f1_score(y, (blend >= thr).astype(int))
        if f1_t > best_f1:
            best_f1 = f1_t
            best_thr = thr
    return (best_f1 + auc) / 2, best_f1, auc, best_thr


def optimize_weights(oof_list, y, n_models):
    """scipy로 최적 비율 탐색"""
    def neg_score(weights):
        score, _, _, _ = eval_blend(weights, oof_list, y)
        return -score

    best_result = None
    best_score = -1

    # 여러 초기값으로 시도
    inits = [
        np.ones(n_models) / n_models,  # uniform
        np.array([0.4, 0.49, 0.11] + [0.0] * (n_models - 3)) if n_models > 3 else np.array([0.4, 0.49, 0.11]),
    ]
    # 각 모델에 집중하는 초기값 추가
    for i in range(n_models):
        x = np.ones(n_models) * 0.05
        x[i] = 0.5
        inits.append(x / x.sum())

    for x0 in inits:
        x0 = x0[:n_models]
        result = minimize(neg_score, x0=x0, method='Nelder-Mead',
                          options={'maxiter': 10000, 'xatol': 0.0001, 'fatol': 1e-7})
        if -result.fun > best_score:
            best_score = -result.fun
            best_result = result

    opt_w = np.abs(best_result.x)
    opt_w = opt_w / opt_w.sum()
    return opt_w, best_score


# 테스트할 조합들
combos = [
    ("3-GBDT (baseline)", ["LGBM", "CatBoost", "XGBoost"]),
    ("3-GBDT + RF", ["LGBM", "CatBoost", "XGBoost", "RF"]),
    ("3-GBDT + ExtraTrees", ["LGBM", "CatBoost", "XGBoost", "ExtraTrees"]),
    ("3-GBDT + LR", ["LGBM", "CatBoost", "XGBoost", "LR"]),
    ("3-GBDT + RF + LR", ["LGBM", "CatBoost", "XGBoost", "RF", "LR"]),
    ("3-GBDT + RF + ET", ["LGBM", "CatBoost", "XGBoost", "RF", "ExtraTrees"]),
    ("3-GBDT + RF + ET + LR", ["LGBM", "CatBoost", "XGBoost", "RF", "ExtraTrees", "LR"]),
    ("LGBM + Cat + RF", ["LGBM", "CatBoost", "RF"]),
    ("LGBM + Cat + LR", ["LGBM", "CatBoost", "LR"]),
]

results = []
for label, model_names in combos:
    oof_list = [all_oofs[n] for n in model_names]
    opt_w, opt_score = optimize_weights(oof_list, y, len(model_names))
    score, f1, auc, thr = eval_blend(opt_w, oof_list, y)

    weight_str = " + ".join(f"{n}:{opt_w[i]:.0%}" for i, n in enumerate(model_names) if opt_w[i] >= 0.01)
    results.append((label, score, f1, auc, thr, opt_w, model_names))
    print(f"\n  {label}")
    print(f"    Score: {score:.4f} (F1={f1:.4f}, AUC={auc:.4f}, thr={thr:.3f})")
    print(f"    Weights: {weight_str}")


# ============================================================
# Part 4: 결과 요약
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

results.sort(key=lambda x: x[1], reverse=True)
baseline_score = [r for r in results if r[0] == "3-GBDT (baseline)"][0][1]

print(f"\n  {'Combination':<30s} {'Score':>7s} {'Delta':>8s} {'F1':>7s} {'AUC':>7s}")
print("  " + "-" * 60)
for label, score, f1, auc, thr, opt_w, model_names in results:
    delta = score - baseline_score
    marker = " ***" if score == results[0][1] else ""
    print(f"  {label:<30s} {score:>7.4f} {delta:>+8.4f} {f1:>7.4f} {auc:>7.4f}{marker}")


# ============================================================
# Part 5: Best로 test prediction 생성
# ============================================================
best = results[0]
best_label, best_score, best_f1, best_auc, best_thr, best_w, best_models = best

print(f"\n{'=' * 70}")
print(f"BEST: {best_label}")
print(f"  Score: {best_score:.4f}, Threshold: {best_thr:.3f}")
print(f"{'=' * 70}")

if best_score > baseline_score + 0.00005:
    print("\nBaseline보다 개선! Test prediction 생성...")

    # test probabilities
    test_prob_map = {
        "LGBM": np.load("prob_lgbm.npy"),
        "CatBoost": np.load("prob_catboost.npy"),
        "XGBoost": np.load("prob_xgb.npy"),
    }
    for name in ["RF", "ExtraTrees", "LR"]:
        if name in best_models:
            test_prob_map[name] = test_diverse[name]

    test_probs = [test_prob_map[n] for n in best_models]
    blend_test = sum(best_w[i] * test_probs[i] for i in range(len(best_w)))

    prediction = pd.DataFrame({
        "id": test_ids,
        "y_cls": (blend_test >= best_thr).astype(int),
        "y_prob": blend_test,
    })
    prediction.to_csv("prediction_diverse.csv", index=False)
    print(f"prediction_diverse.csv saved!")
    print(f"  y_cls distribution: {prediction['y_cls'].value_counts().to_dict()}")
    print(f"  y_prob mean: {prediction['y_prob'].mean():.4f}")

    # 비율 출력 (code_clean.ipynb에 적용할 값)
    print(f"\n  === code_clean.ipynb cell-26에 적용할 값 ===")
    for i, n in enumerate(best_models):
        if best_w[i] >= 0.01:
            print(f"  w_{n.lower()} = {best_w[i]:.4f}")
    print(f"  threshold = {best_thr:.3f}")
else:
    print("\nBaseline과 차이 없음. 3-GBDT 유지.")

print("\nDone!")

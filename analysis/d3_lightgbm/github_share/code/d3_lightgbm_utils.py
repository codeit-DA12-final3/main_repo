
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FEATURE_LABELS = {
    "friends_d3": "D3 친구 수",
    "class_users_d0_3": "D0-D3 같은 반 유저 수",
    "grade_users_d0_3": "D0-D3 같은 학년 유저 수",
    "class_active_users_d0_3": "D0-D3 같은 반 활동 유저 수",
    "grade_active_users_d0_3": "D0-D3 같은 학년 활동 유저 수",
    "received_vote_yn_d0_3": "D0-D3 수신 경험 여부",
    "received_vote_count_d0_3": "D0-D3 수신 Ping 수",
    "received_vote_days_d0_3": "D0-D3 수신 발생일수",
    "first_received_vote_day_d0_3": "첫 수신 발생일",
    "point_use_after_received_vote_yn_d0_3": "수신 후 포인트 사용 여부",
    "point_use_after_received_vote_count_d0_3": "수신 후 포인트 사용 횟수",
    "sent_vote_experience_d0_3": "D0-D3 발신 경험 여부",
    "sent_vote_count_d0_3": "D0-D3 발신 투표 수",
    "sent_vote_complete_count_d0_3": "D0-D3 완료 발신 수",
    "sent_vote_active_days_d0_3": "D0-D3 발신 활동일수",
    "sent_vote_distinct_chosen_users_d0_3": "D0-D3 선택 고유 유저 수",
    "sent_vote_distinct_questions_d0_3": "D0-D3 고유 질문 수",
    "sent_vote_complete_rate_d0_3": "D0-D3 발신 완료율",
}

TARGET_LABELS = {
    "returned": "재방문 여부",
    "paid": "결제 여부",
    "payment_count": "결제 횟수",
    "active_days": "활동일수",
    "bought_hearts": "구매 하트 수",
    "relation_experience": "관계 경험 발생 여부",
    "churn": "완전 이탈 여부",
    "high_active": "고활동 여부",
    "repeat_paid": "반복 결제 여부",
    "high_value_paid": "고구매 여부",
}


TARGET_CANDIDATES = {
    "returned": [
        "returned_d4_28",
        "retention_d4_28",
        "returned_d4_7",
        "retention_d4_7",
    ],
    "paid": [
        "paid_d4_28",
        "paid_flag_d4_28",
        "payment_d4_28",
        "paid_d4_7",
        "paid_flag_d4_7",
        "payment_d4_7",
    ],
    "payment_count": [
        "payment_count_d4_28",
        "payment_cnt_d4_28",
        "payment_count_d4_7",
        "payment_cnt_d4_7",
    ],
    "active_days": [
        "active_days_d4_28",
        "active_days_d4_7",
    ],
    "bought_hearts": [
        "bought_hearts_d4_28",
        "purchased_heart_cnt_d4_28",
        "payment_amount_d4_28",
        "bought_hearts_d4_7",
        "purchased_heart_cnt_d4_7",
        "payment_amount_d4_7",
    ],
}


@dataclass(frozen=True)
class ModelRunResult:
    metrics: pd.DataFrame
    topk: pd.DataFrame
    importance: pd.DataFrame
    confusion: pd.DataFrame


def label_for(name: str) -> str:
    return FEATURE_LABELS.get(name, TARGET_LABELS.get(name, name.replace("_", " ")))


def read_mart(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, low_memory=False)
    if "user_id" in data.columns:
        data["user_id"] = data["user_id"].astype("string")
    return data


def summarize_missing(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    out = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": df.isna().sum().values,
            "missing_rate": df.isna().mean().values,
        }
    )
    out["non_missing_count"] = total - out["missing_count"]
    return out.sort_values(["missing_count", "column"], ascending=[False, True]).reset_index(drop=True)


def summarize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        series = df[col]
        rows.append(
            {
                "column": col,
                "dtype": str(series.dtype),
                "nunique": int(series.nunique(dropna=True)),
                "sample_values": ", ".join(map(str, series.dropna().head(3).tolist())),
            }
        )
    return pd.DataFrame(rows)


def summarize_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    numeric = ensure_numeric_frame(df, columns)
    desc = numeric.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T
    desc = desc.reset_index().rename(columns={"index": "column"})
    desc.insert(1, "label", desc["column"].map(label_for))
    return desc


def save_table(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def to_binary_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(int)
    if pd.api.types.is_numeric_dtype(series):
        return (pd.to_numeric(series, errors="coerce").fillna(0) > 0).astype(int)
    lowered = series.astype("string").str.strip().str.lower()
    return lowered.isin(["1", "true", "t", "yes", "y", "paid"]).astype(int)


def ensure_numeric_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in columns:
        if col not in df.columns:
            continue
        if pd.api.types.is_bool_dtype(df[col]):
            out[col] = df[col].fillna(False).astype(int)
        else:
            out[col] = pd.to_numeric(df[col], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan).fillna(0)


def build_feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    groups = {
        "env": [
            "friends_d3",
            "class_users_d0_3",
            "grade_users_d0_3",
            "class_active_users_d0_3",
            "grade_active_users_d0_3",
        ],
        "receive": [
            "received_vote_yn_d0_3",
            "received_vote_count_d0_3",
            "received_vote_days_d0_3",
            "first_received_vote_day_d0_3",
            "point_use_after_received_vote_yn_d0_3",
            "point_use_after_received_vote_count_d0_3",
        ],
        "send": [
            "sent_vote_experience_d0_3",
            "sent_vote_count_d0_3",
            "sent_vote_complete_count_d0_3",
            "sent_vote_active_days_d0_3",
            "sent_vote_distinct_chosen_users_d0_3",
            "sent_vote_distinct_questions_d0_3",
            "sent_vote_complete_rate_d0_3",
        ],
    }
    return {key: [col for col in cols if col in df.columns] for key, cols in groups.items()}


def resolve_target_columns(df: pd.DataFrame) -> tuple[dict[str, str], pd.DataFrame, str]:
    target_map: dict[str, str] = {}
    rows = []
    for target_name, candidates in TARGET_CANDIDATES.items():
        available = [col for col in candidates if col in df.columns]
        selected = available[0] if available else None
        if selected:
            target_map[target_name] = selected
        rows.append(
            {
                "target_key": target_name,
                "target_label": label_for(target_name),
                "selected_column": selected,
                "available_candidates": ", ".join(available),
                "preferred_window_available": any(col.endswith("d4_28") for col in available),
            }
        )
    selected_cols = list(target_map.values())
    if selected_cols and all("d4_28" in col for col in selected_cols):
        actual_window = "D4-D28"
    elif selected_cols and all("d4_7" in col for col in selected_cols):
        actual_window = "D4-D7"
    else:
        actual_window = "혼합 또는 일부 타깃 미확인"
    return target_map, pd.DataFrame(rows), actual_window


def build_target_overview(df: pd.DataFrame, target_map: dict[str, str]) -> pd.DataFrame:
    rows = []
    for target_key, col in target_map.items():
        series = pd.to_numeric(df[col], errors="coerce")
        row = {
            "target_key": target_key,
            "target_label": label_for(target_key),
            "column": col,
            "row_count": int(series.notna().sum()),
            "missing_count": int(series.isna().sum()),
            "mean": float(series.mean()) if series.notna().any() else np.nan,
            "min": float(series.min()) if series.notna().any() else np.nan,
            "median": float(series.median()) if series.notna().any() else np.nan,
            "max": float(series.max()) if series.notna().any() else np.nan,
            "positive_count": int((series.fillna(0) > 0).sum()),
            "positive_rate": float((series.fillna(0) > 0).mean()),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def relation_experience_mask(df: pd.DataFrame) -> pd.Series:
    receive_col = "received_vote_yn_d0_3"
    send_col = "sent_vote_experience_d0_3"
    receive = to_binary_series(df[receive_col]) if receive_col in df.columns else pd.Series(0, index=df.index)
    send = to_binary_series(df[send_col]) if send_col in df.columns else pd.Series(0, index=df.index)
    return (receive.eq(1) | send.eq(1)).rename("relation_experience_d0_3")


def add_derived_targets(df: pd.DataFrame, target_map: dict[str, str], high_active_threshold: int = 3) -> tuple[pd.DataFrame, dict[str, str]]:
    out = df.copy()
    updated = dict(target_map)
    out["relation_experience_d0_3"] = relation_experience_mask(out).astype(int)
    updated["relation_experience"] = "relation_experience_d0_3"
    if "active_days" in updated:
        active = pd.to_numeric(out[updated["active_days"]], errors="coerce").fillna(0)
        out["churn_d4_plus"] = active.eq(0).astype(int)
        out["high_active_d4_plus"] = active.ge(high_active_threshold).astype(int)
        updated["churn"] = "churn_d4_plus"
        updated["high_active"] = "high_active_d4_plus"
    if "payment_count" in updated:
        count = pd.to_numeric(out[updated["payment_count"]], errors="coerce").fillna(0)
        out["repeat_paid_d4_plus"] = count.ge(2).astype(int)
        updated["repeat_paid"] = "repeat_paid_d4_plus"
    if "bought_hearts" in updated:
        hearts = pd.to_numeric(out[updated["bought_hearts"]], errors="coerce").fillna(0)
        paid_mask = to_binary_series(out[updated["paid"]]).eq(1) if "paid" in updated else hearts.gt(0)
        positive_hearts = hearts.loc[paid_mask & hearts.gt(0)]
        threshold = positive_hearts.quantile(0.75) if len(positive_hearts) else np.nan
        if pd.notna(threshold):
            out["high_value_paid_d4_plus"] = hearts.ge(threshold).astype(int)
            updated["high_value_paid"] = "high_value_paid_d4_plus"
    return out, updated


def top_k_lift_table(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    ks: tuple[float, ...] = (0.01, 0.03, 0.05, 0.10),
) -> pd.DataFrame:
    y = pd.Series(y_true).astype(int).reset_index(drop=True)
    score = pd.Series(y_score).astype(float).reset_index(drop=True)
    base_rate = y.mean()
    rows = []
    order = score.sort_values(ascending=False).index
    total_positive = int(y.sum())
    for k in ks:
        n = max(1, int(np.ceil(len(y) * k)))
        idx = order[:n]
        top_positive = int(y.iloc[idx].sum())
        top_rate = float(y.iloc[idx].mean())
        rows.append(
            {
                "top_k": f"Top {int(k * 100)}%",
                "top_k_rate": k,
                "top_n": n,
                "base_positive_rate": float(base_rate),
                "top_positive_rate": top_rate,
                "lift": float(top_rate / base_rate) if base_rate > 0 else np.nan,
                "captured_positive_count": top_positive,
                "capture_rate": float(top_positive / total_positive) if total_positive > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def classification_metric_row(y_true: pd.Series, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

    y = pd.Series(y_true).astype(int)
    pred = (pd.Series(y_score) >= threshold).astype(int)
    row = {
        "row_count": int(len(y)),
        "positive_count": int(y.sum()),
        "positive_rate": float(y.mean()),
        "threshold": threshold,
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": np.nan,
        "pr_auc": np.nan,
    }
    if y.nunique() == 2:
        row["roc_auc"] = float(roc_auc_score(y, y_score))
        row["pr_auc"] = float(average_precision_score(y, y_score))
    return row


def confusion_matrix_rows(y_true: pd.Series, y_score: np.ndarray, threshold: float = 0.5) -> pd.DataFrame:
    from sklearn.metrics import confusion_matrix

    y = pd.Series(y_true).astype(int)
    pred = (pd.Series(y_score) >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return pd.DataFrame(
        [
            {"actual": "실제 0", "predicted": "예측 0", "count": int(cm[0, 0])},
            {"actual": "실제 0", "predicted": "예측 1", "count": int(cm[0, 1])},
            {"actual": "실제 1", "predicted": "예측 0", "count": int(cm[1, 0])},
            {"actual": "실제 1", "predicted": "예측 1", "count": int(cm[1, 1])},
        ]
    )


def regression_metric_row(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y = pd.Series(y_true).astype(float)
    pred = pd.Series(y_pred).astype(float)
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    return {
        "row_count": int(len(y)),
        "target_mean": float(y.mean()),
        "target_median": float(y.median()),
        "target_positive_rate": float((y > 0).mean()),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": rmse,
        "r2": float(r2_score(y, pred)),
    }


def _split_data(X: pd.DataFrame, y: pd.Series, task: str, random_state: int, test_size: float):
    from sklearn.model_selection import train_test_split

    stratify = y if task == "classification" and y.nunique() == 2 and y.value_counts().min() >= 2 else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)


def _empty_result() -> ModelRunResult:
    return ModelRunResult(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())


def run_binary_model(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    population_mask: pd.Series,
    population_name: str,
    feature_set_name: str,
    target_name: str,
    random_state: int = 42,
    test_size: float = 0.3,
    run_baseline: bool = True,
) -> ModelRunResult:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model_rows = []
    topk_rows = []
    importance_rows = []
    confusion_rows = []

    cols = ["user_id", target_col] + feature_cols
    data = df.loc[population_mask, [c for c in cols if c in df.columns]].copy()
    if data.empty or target_col not in data.columns:
        return _empty_result()

    y = to_binary_series(data[target_col])
    X = ensure_numeric_frame(data, feature_cols)
    keep = y.notna()
    y = y.loc[keep]
    X = X.loc[keep]
    if len(y) < 100 or y.nunique() < 2:
        return _empty_result()

    X_train, X_test, y_train, y_test = _split_data(X, y, "classification", random_state, test_size)

    models: list[tuple[str, Any]] = []
    if run_baseline:
        baseline = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=500, class_weight="balanced", n_jobs=-1, random_state=random_state),
        )
        models.append(("Baseline Logistic", baseline))

    from lightgbm import LGBMClassifier

    pos = max(int(y_train.sum()), 1)
    neg = max(int(len(y_train) - y_train.sum()), 1)
    lgbm = LGBMClassifier(
        n_estimators=250,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=100,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        n_jobs=-1,
        scale_pos_weight=neg / pos,
        verbose=-1,
    )
    models.append(("LightGBM", lgbm))

    for model_type, model in models:
        base_info = {
            "task": "classification",
            "target_name": target_name,
            "target_col": target_col,
            "population": population_name,
            "feature_set": feature_set_name,
            "feature_count": len(feature_cols),
            "model_type": model_type,
            "train_row_count": int(len(y_train)),
            "test_row_count": int(len(y_test)),
        }
        try:
            model.fit(X_train, y_train)
            if hasattr(model, "predict_proba"):
                score = model.predict_proba(X_test)[:, 1]
            else:
                score = model.decision_function(X_test)
        except Exception as exc:
            error_row = dict(base_info)
            error_row.update({"error": f"{type(exc).__name__}: {exc}"})
            model_rows.append(error_row)
            continue
        row = classification_metric_row(y_test, score)
        row.update(base_info)
        model_rows.append(row)

        topk = top_k_lift_table(y_test, score)
        topk["target_name"] = target_name
        topk["target_col"] = target_col
        topk["population"] = population_name
        topk["feature_set"] = feature_set_name
        topk["model_type"] = model_type
        topk_rows.append(topk)

        cm = confusion_matrix_rows(y_test, score)
        cm["target_name"] = target_name
        cm["population"] = population_name
        cm["feature_set"] = feature_set_name
        cm["model_type"] = model_type
        confusion_rows.append(cm)

        if model_type == "LightGBM":
            importance = pd.DataFrame(
                {
                    "feature": feature_cols,
                    "importance": model.feature_importances_,
                }
            )
            importance["feature_label"] = importance["feature"].map(label_for)
            importance["target_name"] = target_name
            importance["target_col"] = target_col
            importance["population"] = population_name
            importance["feature_set"] = feature_set_name
            importance["model_type"] = model_type
            importance_rows.append(importance.sort_values("importance", ascending=False))

    return ModelRunResult(
        pd.DataFrame(model_rows),
        pd.concat(topk_rows, ignore_index=True) if topk_rows else pd.DataFrame(),
        pd.concat(importance_rows, ignore_index=True) if importance_rows else pd.DataFrame(),
        pd.concat(confusion_rows, ignore_index=True) if confusion_rows else pd.DataFrame(),
    )


def run_regression_model(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    population_mask: pd.Series,
    population_name: str,
    feature_set_name: str,
    target_name: str,
    random_state: int = 42,
    test_size: float = 0.3,
    run_baseline: bool = True,
) -> ModelRunResult:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import PoissonRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model_rows = []
    importance_rows = []
    cols = ["user_id", target_col] + feature_cols
    data = df.loc[population_mask, [c for c in cols if c in df.columns]].copy()
    if data.empty or target_col not in data.columns:
        return _empty_result()

    y = pd.to_numeric(data[target_col], errors="coerce").fillna(0).clip(lower=0)
    X = ensure_numeric_frame(data, feature_cols)
    if len(y) < 100 or y.nunique() < 2:
        return _empty_result()

    X_train, X_test, y_train, y_test = _split_data(X, y, "regression", random_state, test_size)

    models: list[tuple[str, Any]] = []
    if run_baseline:
        baseline = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            PoissonRegressor(alpha=1e-4, max_iter=300),
        )
        models.append(("Baseline Poisson", baseline))

    from lightgbm import LGBMRegressor

    lgbm = LGBMRegressor(
        objective="poisson",
        n_estimators=250,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=100,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
    models.append(("LightGBM", lgbm))

    for model_type, model in models:
        base_info = {
            "task": "regression",
            "target_name": target_name,
            "target_col": target_col,
            "population": population_name,
            "feature_set": feature_set_name,
            "feature_count": len(feature_cols),
            "model_type": model_type,
            "train_row_count": int(len(y_train)),
            "test_row_count": int(len(y_test)),
        }
        try:
            model.fit(X_train, y_train)
            pred = np.asarray(model.predict(X_test)).clip(min=0)
        except Exception as exc:
            error_row = dict(base_info)
            error_row.update({"error": f"{type(exc).__name__}: {exc}"})
            model_rows.append(error_row)
            continue
        row = regression_metric_row(y_test, pred)
        row.update(base_info)
        model_rows.append(row)

        if model_type == "LightGBM":
            importance = pd.DataFrame(
                {
                    "feature": feature_cols,
                    "importance": model.feature_importances_,
                }
            )
            importance["feature_label"] = importance["feature"].map(label_for)
            importance["target_name"] = target_name
            importance["target_col"] = target_col
            importance["population"] = population_name
            importance["feature_set"] = feature_set_name
            importance["model_type"] = model_type
            importance_rows.append(importance.sort_values("importance", ascending=False))

    return ModelRunResult(
        pd.DataFrame(model_rows),
        pd.DataFrame(),
        pd.concat(importance_rows, ignore_index=True) if importance_rows else pd.DataFrame(),
        pd.DataFrame(),
    )


def combine_results(results: list[ModelRunResult]) -> ModelRunResult:
    return ModelRunResult(
        pd.concat([r.metrics for r in results if not r.metrics.empty], ignore_index=True)
        if any(not r.metrics.empty for r in results)
        else pd.DataFrame(),
        pd.concat([r.topk for r in results if not r.topk.empty], ignore_index=True)
        if any(not r.topk.empty for r in results)
        else pd.DataFrame(),
        pd.concat([r.importance for r in results if not r.importance.empty], ignore_index=True)
        if any(not r.importance.empty for r in results)
        else pd.DataFrame(),
        pd.concat([r.confusion for r in results if not r.confusion.empty], ignore_index=True)
        if any(not r.confusion.empty for r in results)
        else pd.DataFrame(),
    )


def build_spearman_overview(df: pd.DataFrame, feature_cols: list[str], target_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = ensure_numeric_frame(df, feature_cols + target_cols)
    corr = numeric.corr(method="spearman")
    rows = []
    for target_col in target_cols:
        if target_col not in corr.columns:
            continue
        for feature in feature_cols:
            if feature in corr.index:
                rows.append(
                    {
                        "target_col": target_col,
                        "target_label": label_for(target_col),
                        "feature": feature,
                        "feature_label": label_for(feature),
                        "spearman_corr": corr.loc[feature, target_col],
                        "abs_corr": abs(corr.loc[feature, target_col]),
                    }
                )
    return (
        pd.DataFrame(rows).sort_values(["target_col", "abs_corr"], ascending=[True, False]).reset_index(drop=True),
        corr,
    )


def make_feature_definition(groups: dict[str, list[str]]) -> pd.DataFrame:
    axis_labels = {"env": "관계망 환경축", "send": "발신 행동축", "receive": "수신 반응축"}
    rows = []
    for axis, cols in groups.items():
        for col in cols:
            rows.append({"axis": axis, "axis_label": axis_labels.get(axis, axis), "feature": col, "feature_label": label_for(col)})
    return pd.DataFrame(rows)


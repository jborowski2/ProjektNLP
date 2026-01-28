from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from typing import cast
import joblib

from data_loading import load_event_type_training_frame
from model_wrappers import LabelEncodedClassifier


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    model: str
    vectorizer: str
    class_weight_balanced: bool
    oversample: bool


def build_vectorizer(kind: str, *, random_state: int):
    if kind == "word":
        return TfidfVectorizer(
            ngram_range=(1, 2),
            lowercase=True,
            min_df=2,
            max_df=0.95,
        )
    if kind == "word_char":
        return FeatureUnion(
            [
                (
                    "word",
                    cast(
                        TransformerMixin,
                        TfidfVectorizer(
                            ngram_range=(1, 2),
                            lowercase=True,
                            min_df=2,
                            max_df=0.95,
                        ),
                    ),
                ),
                (
                    "char",
                    cast(
                        TransformerMixin,
                        TfidfVectorizer(
                            analyzer="char_wb",
                            ngram_range=(3, 5),
                            lowercase=True,
                            min_df=2,
                            max_df=0.95,
                        ),
                    ),
                ),
            ]
        )

    # For models that don't like sparse / very high-dimensional features,
    # we reduce dimensionality and standardize.
    if kind == "word_svd":
        return Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        lowercase=True,
                        min_df=2,
                        max_df=0.95,
                    ),
                ),
                ("svd", TruncatedSVD(n_components=300, random_state=random_state)),
                ("scaler", StandardScaler()),
            ]
        )

    if kind == "word_char_svd":
        return Pipeline(
            [
                (
                    "tfidf",
                    FeatureUnion(
                        [
                            (
                                "word",
                                cast(
                                    TransformerMixin,
                                    TfidfVectorizer(
                                        ngram_range=(1, 2),
                                        lowercase=True,
                                        min_df=2,
                                        max_df=0.95,
                                    ),
                                ),
                            ),
                            (
                                "char",
                                cast(
                                    TransformerMixin,
                                    TfidfVectorizer(
                                        analyzer="char_wb",
                                        ngram_range=(3, 5),
                                        lowercase=True,
                                        min_df=2,
                                        max_df=0.95,
                                    ),
                                ),
                            ),
                        ]
                    ),
                ),
                ("svd", TruncatedSVD(n_components=300, random_state=random_state)),
                ("scaler", StandardScaler()),
            ]
        )

    raise ValueError(f"Unknown vectorizer kind: {kind}")


def build_model(
    model_name: str,
    *,
    class_weight_balanced: bool,
    random_state: int,
    bagging_n_estimators: int = 5,
    bagging_max_samples: float = 0.7,
    bagging_n_jobs: int = -1,
):
    if model_name in {"logreg", "logreg_l2", "tuned_logreg_l2"}:
        C = 1.0
        if model_name == "tuned_logreg_l2":
            C = 2.0
        return LogisticRegression(
            max_iter=8000,
            solver="saga",
            C=C,
            class_weight="balanced" if class_weight_balanced else None,
            random_state=random_state,
        )
    if model_name in {"linearsvc", "tuned_linearsvc"}:
        C = 1.0
        if model_name == "tuned_linearsvc":
            C = 2.0
        return LinearSVC(
            C=C,
            class_weight="balanced" if class_weight_balanced else None,
            random_state=random_state,
        )
    if model_name == "mnb":
        # MultinomialNB doesn't support class_weight; imbalance can be handled via oversampling.
        return MultinomialNB()

    if model_name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=random_state)

    if model_name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(200,),
            max_iter=400,
            random_state=random_state,
        )

    if model_name == "bagging_logreg":
        base = LogisticRegression(
            max_iter=2500,
            solver="saga",
            class_weight="balanced" if class_weight_balanced else None,
            random_state=random_state,
        )
        return BaggingClassifier(
            estimator=base,
            n_estimators=int(bagging_n_estimators),
            max_samples=float(bagging_max_samples),
            n_jobs=int(bagging_n_jobs),
            random_state=random_state,
        )

    if model_name in {"xgboost", "tuned_xgboost"}:
        try:
            from xgboost import XGBClassifier  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "xgboost is not installed. Install it with: pip install xgboost"
            ) from exc

        params: dict[str, Any] = {
            "n_estimators": 250,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
            "random_state": random_state,
            "n_jobs": 0,
            "objective": "multi:softprob",
        }
        if model_name == "tuned_xgboost":
            params.update(
                {
                    "n_estimators": 500,
                    "max_depth": 5,
                    "learning_rate": 0.05,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_lambda": 2.0,
                }
            )
        return LabelEncodedClassifier(XGBClassifier(**params))

    raise ValueError(f"Unknown model: {model_name}")


def _parameter_count(model) -> int:
    # Linear / probabilistic
    if hasattr(model, "coef_") and hasattr(model, "intercept_"):
        return int(np.asarray(model.coef_).size + np.asarray(model.intercept_).size)

    if isinstance(model, MultinomialNB):
        return int(np.asarray(model.feature_log_prob_).size + np.asarray(model.class_log_prior_).size)

    if isinstance(model, MLPClassifier):
        return int(
            sum(np.asarray(w).size for w in model.coefs_) + sum(np.asarray(b).size for b in model.intercepts_)
        )

    if isinstance(model, GradientBoostingClassifier):
        total = 0
        for row in model.estimators_.ravel():
            total += int(getattr(row.tree_, "node_count", 0))
        return total

    if isinstance(model, BaggingClassifier):
        total = 0
        for est in getattr(model, "estimators_", []):
            if hasattr(est, "coef_") and hasattr(est, "intercept_"):
                total += int(np.asarray(est.coef_).size + np.asarray(est.intercept_).size)
        return total

    # XGBoost: count total nodes across all trees
    try:
        booster = model.get_booster()  # type: ignore[attr-defined]
        df = booster.trees_to_dataframe()
        return int(len(df))
    except Exception:
        return 0


def _aic_bic(model, X, y_true: list[str]) -> tuple[float, float]:
    if not hasattr(model, "predict_proba"):
        return float("nan"), float("nan")

    proba = model.predict_proba(X)
    proba = np.asarray(proba)
    if proba.ndim != 2:
        return float("nan"), float("nan")

    classes = getattr(model, "classes_", None)
    if classes is None:
        return float("nan"), float("nan")
    classes = list(classes)
    index = {str(c): i for i, c in enumerate(classes)}

    eps = 1e-15
    ll = 0.0
    for i, y in enumerate(y_true):
        j = index.get(str(y))
        if j is None:
            continue
        p = float(proba[i, j])
        p = min(max(p, eps), 1.0)
        ll += float(np.log(p))

    n = int(len(y_true))
    k = int(_parameter_count(model))
    aic = 2.0 * k - 2.0 * ll
    bic = float(np.log(max(n, 1))) * k - 2.0 * ll
    return float(aic), float(bic)


def oversample_to_max(X_train, y_train, *, random_state: int):
    # Random oversampling to the max class count; works with sparse matrices.
    y_arr = np.asarray(y_train)
    labels, counts = np.unique(y_arr, return_counts=True)
    if len(counts) == 0:
        return X_train, y_train
    target = int(np.max(counts))
    if target <= 1:
        return X_train, y_train

    rng = np.random.default_rng(random_state)
    all_indices = [np.arange(len(y_arr))]

    for label, count in zip(labels, counts, strict=False):
        missing = target - int(count)
        if missing <= 0:
            continue
        idxs = np.flatnonzero(y_arr == label)
        sampled = rng.choice(idxs, size=missing, replace=True)
        all_indices.append(sampled)

    indices = np.concatenate(all_indices)
    rng.shuffle(indices)
    return X_train[indices], y_arr[indices].tolist()


def run_one(
    config: ExperimentConfig,
    sentences: list[str],
    labels: list[str],
    *,
    test_size: float,
    seed: int,
    bagging_n_estimators: int = 5,
    bagging_max_samples: float = 0.7,
    bagging_n_jobs: int = -1,
):
    stratify_labels = labels
    try:
        s_train, s_test, y_train, y_test = train_test_split(
            sentences,
            labels,
            test_size=test_size,
            stratify=stratify_labels,
            random_state=seed,
        )
    except ValueError:
        s_train, s_test, y_train, y_test = train_test_split(
            sentences,
            labels,
            test_size=test_size,
            stratify=None,
            random_state=seed,
        )

    vec = build_vectorizer(config.vectorizer, random_state=seed)
    X_train = vec.fit_transform(s_train)
    X_test = vec.transform(s_test)

    if config.oversample:
        X_train, y_train = oversample_to_max(X_train, y_train, random_state=seed)

    model = build_model(
        config.model,
        class_weight_balanced=config.class_weight_balanced,
        random_state=seed,
        bagging_n_estimators=bagging_n_estimators,
        bagging_max_samples=bagging_max_samples,
        bagging_n_jobs=bagging_n_jobs,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
    }

    aic, bic = _aic_bic(model, X_test, y_test)
    metrics["aic"] = float(aic)
    metrics["bic"] = float(bic)

    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Run event type classification experiments.")
    parser.add_argument("--headlines", default="datasets/id_and_headline_first_sentence (1).csv")
    parser.add_argument("--tagged", default="datasets/tagged.csv")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--out", default="results/experiments.csv")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional: limit number of examples (random sample) for faster runs.",
    )
    parser.add_argument("--bagging-estimators", type=int, default=5)
    parser.add_argument("--bagging-max-samples", type=float, default=0.7)
    parser.add_argument("--bagging-jobs", type=int, default=-1)
    parser.add_argument("--save-best-model", action="store_true")
    parser.add_argument("--save-all-models", action="store_true")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument(
        "--summary-out",
        default="",
        help="Optional path for a leaderboard CSV (mean over seeds). Defaults to '<out>_summary.csv'.",
    )
    parser.add_argument(
        "--include",
        default="",
        help="Comma-separated list of experiment names to run (matches the 'name' column).",
    )
    parser.add_argument(
        "--exclude",
        default="",
        help="Comma-separated list of experiment names to skip (matches the 'name' column).",
    )
    parser.add_argument("--no-xgboost", action="store_true", help="Skip XGBoost models.")
    parser.add_argument(
        "--preset",
        default="full",
        choices=["full", "previous", "both"],
        help=(
            "Which set of experiments to run: "
            "'full' = current model suite, "
            "'previous' = earlier TF-IDF baselines (legacy), "
            "'both' = union of both sets."
        ),
    )

    args = parser.parse_args()

    df = load_event_type_training_frame(headlines_csv_path=args.headlines, tagged_csv_path=args.tagged)

    if args.limit and args.limit > 0 and args.limit < len(df):
        df = df.sample(n=int(args.limit), random_state=42).reset_index(drop=True)

    sentences = df["sentence"].astype(str).tolist()
    labels = df["label"].astype(str).tolist()

    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]

    include = {s.strip() for s in str(args.include).split(",") if s.strip()}
    exclude = {s.strip() for s in str(args.exclude).split(",") if s.strip()}

    # Current model suite.
    full_configs: list[ExperimentConfig] = [
        ExperimentConfig("LinearSVC", "linearsvc", "word_char", True, True),
        ExperimentConfig("Tuned_LinearSVC", "tuned_linearsvc", "word_char", True, True),
        ExperimentConfig("LogReg_L2", "logreg_l2", "word_char", True, True),
        ExperimentConfig("Tuned_LogReg_L2", "tuned_logreg_l2", "word_char", True, True),
        ExperimentConfig("MultinomialNB", "mnb", "word", False, True),
        ExperimentConfig("Bagging_LogReg", "bagging_logreg", "word_char", True, True),
        ExperimentConfig("GradientBoosting", "gradient_boosting", "word_char_svd", False, False),
        ExperimentConfig("MLP", "mlp", "word_char_svd", False, False),
    ]

    # Earlier (legacy) baselines used previously in this project.
    previous_configs: list[ExperimentConfig] = [
        ExperimentConfig("lr_word_balanced_over", "logreg", "word", True, True),
        ExperimentConfig("lr_word_balanced", "logreg", "word", True, False),
        ExperimentConfig("lr_wordchar_balanced_over", "logreg", "word_char", True, True),
        ExperimentConfig("svm_word_balanced_over", "linearsvc", "word", True, True),
        ExperimentConfig("svm_wordchar_balanced_over", "linearsvc", "word_char", True, True),
        ExperimentConfig("mnb_word_over", "mnb", "word", False, True),
    ]

    if args.preset == "full":
        configs = full_configs
    elif args.preset == "previous":
        configs = previous_configs
    else:
        configs = [*full_configs, *previous_configs]

    # Optional XGBoost (requires `xgboost` installed).
    if args.preset in {"full", "both"} and (not args.no_xgboost):
        try:
            _ = build_model("xgboost", class_weight_balanced=False, random_state=seeds[0] if seeds else 42)
            configs.extend(
                [
                    ExperimentConfig("XGBoost", "xgboost", "word", False, False),
                    ExperimentConfig("Tuned_XGBoost", "tuned_xgboost", "word", False, False),
                ]
            )
        except Exception:
            pass

    if include:
        configs = [c for c in configs if c.name in include]
    if exclude:
        configs = [c for c in configs if c.name not in exclude]

    configs_by_name = {c.name: c for c in configs}

    def save_model_for_config(cfg: ExperimentConfig, *, tag: str) -> Path:
        vec = build_vectorizer(cfg.vectorizer, random_state=seeds[0] if seeds else 42)
        X = vec.fit_transform(sentences)
        y = labels
        if cfg.oversample:
            X, y = oversample_to_max(X, y, random_state=seeds[0] if seeds else 42)
        model = build_model(
            cfg.model,
            class_weight_balanced=cfg.class_weight_balanced,
            random_state=seeds[0] if seeds else 42,
            bagging_n_estimators=args.bagging_estimators,
            bagging_max_samples=args.bagging_max_samples,
            bagging_n_jobs=args.bagging_jobs,
        )
        model.fit(X, y)

        models_dir = Path(args.models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = models_dir / f"{tag}_{cfg.name}_{ts}.joblib"

        payload = {
            "vectorizer": vec,
            "classifier": model,
            "is_trained": True,
            "keyword_overrides": [],
            "meta": {
                "source": "experiments.py",
                "config": asdict(cfg),
                "saved_at": datetime.now().isoformat(timespec="seconds"),
                "tag": tag,
            },
        }
        joblib.dump(payload, model_path)
        return model_path

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for cfg in configs:
        for seed in seeds:
            print(f"Running: {cfg.name} (seed={seed})")
            metrics = run_one(
                cfg,
                sentences,
                labels,
                test_size=args.test_size,
                seed=seed,
                bagging_n_estimators=args.bagging_estimators,
                bagging_max_samples=args.bagging_max_samples,
                bagging_n_jobs=args.bagging_jobs,
            )
            rows.append(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "seed": seed,
                    **asdict(cfg),
                    **metrics,
                }
            )

    # Write CSV
    fieldnames = list(rows[0].keys()) if rows else []
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Print best by f1_macro (mean over seeds)
    by_name: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_name.setdefault(r["name"], []).append(r)

    summary = []
    for name, rs in by_name.items():
        summary.append(
            {
                "name": name,
                "n": len(rs),
                "f1_macro_mean": float(np.mean([x["f1_macro"] for x in rs])),
                "f1_weighted_mean": float(np.mean([x["f1_weighted"] for x in rs])),
                "accuracy_mean": float(np.mean([x["accuracy"] for x in rs])),
                "aic_mean": float(np.nanmean([x.get("aic", float("nan")) for x in rs])),
                "bic_mean": float(np.nanmean([x.get("bic", float("nan")) for x in rs])),
            }
        )

    summary.sort(key=lambda x: x["f1_macro_mean"], reverse=True)
    best = summary[0] if summary else None

    print("\n=== EXPERIMENT SUMMARY (mean over seeds) ===")
    for s in summary:
        aic = s.get("aic_mean", float("nan"))
        bic = s.get("bic_mean", float("nan"))
        aic_bic = ""
        if not (np.isnan(aic) or np.isnan(bic)):
            aic_bic = f" aic={aic:.1f} bic={bic:.1f}"
        print(
            f"{s['name']}: f1_macro={s['f1_macro_mean']:.3f} f1_weighted={s['f1_weighted_mean']:.3f} acc={s['accuracy_mean']:.3f}{aic_bic} (n={s['n']})"
        )

    # Leaderboard CSV (similar to the screenshot table)
    if rows:
        summary_out = Path(args.summary_out) if str(args.summary_out).strip() else out_path.with_name(out_path.stem + "_summary.csv")
        summary_out.parent.mkdir(parents=True, exist_ok=True)

        leaderboard_rows: list[dict[str, Any]] = []
        for s in summary:
            leaderboard_rows.append(
                {
                    "Model": s["name"],
                    "Accuracy": s["accuracy_mean"],
                    "F1_weighted": s["f1_weighted_mean"],
                    "AIC": s["aic_mean"],
                    "BIC": s["bic_mean"],
                }
            )

        leaderboard_rows.sort(key=lambda r: float(r["F1_weighted"]), reverse=True)
        with summary_out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["Model", "Accuracy", "F1_weighted", "AIC", "BIC"])
            writer.writeheader()
            writer.writerows(leaderboard_rows)

        print(f"Saved leaderboard: {summary_out}")

    if best:
        print("\nBest by macro-F1:")
        print(json.dumps(best, ensure_ascii=False, indent=2))
        print(f"\nSaved: {out_path}")

        if args.save_best_model:
            cfg = configs_by_name[best["name"]]
            model_path = save_model_for_config(cfg, tag="best")
            print(f"Saved best model to: {model_path}")

        if args.save_all_models:
            for cfg in configs:
                p = save_model_for_config(cfg, tag="all")
                print(f"Saved model: {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Reusable analysis CLI for video game sales machine-learning experiments.

The tool cleans a VideoGames-style dataset, reruns the main project models, and
writes charts, structured metrics, cluster tables, and a Markdown report. It is
designed to be run from the repository root:

    python src/video_games_analysis.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "VideoGames.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
SALES_COLUMNS = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]
SCORE_COLUMNS = ["Critic_Score", "User_Score"]
RANDOM_STATE = 42
REQUIRED_COLUMNS = [
    "Name",
    "Platform",
    "Year_of_Release",
    "Genre",
    *SALES_COLUMNS,
    *SCORE_COLUMNS,
]


@dataclass(frozen=True)
class AnalysisConfig:
    data_path: Path
    output_dir: Path
    hit_threshold: float
    k_values: tuple[int, ...]
    selected_k: int
    example_genre: str
    example_platform: str
    example_critic_score: float
    example_user_score: float
    top_coefficients: int


def parse_k_values(value: str) -> tuple[int, ...]:
    try:
        k_values = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("K values must be comma-separated integers.") from exc

    if not k_values or any(k < 2 for k in k_values):
        raise argparse.ArgumentTypeError("K values must include at least one integer greater than 1.")
    return k_values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a reusable video game sales ML analysis pipeline."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to VideoGames.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where figures and summaries will be written.",
    )
    parser.add_argument(
        "--hit-threshold",
        type=float,
        default=0.6,
        help="Global sales threshold, in millions of units, for the logistic hit model.",
    )
    parser.add_argument(
        "--k-values",
        type=parse_k_values,
        default=(2, 3),
        help="Comma-separated K-Means cluster counts to evaluate. Example: 2,3,4",
    )
    parser.add_argument(
        "--selected-k",
        type=int,
        default=2,
        help="K-Means cluster count used for the selected cluster summary.",
    )
    parser.add_argument(
        "--example-genre",
        default="Action",
        help="Genre for the KNN example prediction.",
    )
    parser.add_argument(
        "--example-platform",
        default="PS4",
        help="Platform for the KNN example prediction.",
    )
    parser.add_argument(
        "--example-critic-score",
        type=float,
        default=85,
        help="Critic score for the KNN example prediction.",
    )
    parser.add_argument(
        "--example-user-score",
        type=float,
        default=8.2,
        help="User score for the KNN example prediction.",
    )
    parser.add_argument(
        "--top-coefficients",
        type=int,
        default=12,
        help="Number of largest linear-model coefficients to store.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> AnalysisConfig:
    if args.selected_k not in args.k_values:
        raise ValueError("--selected-k must be included in --k-values.")
    return AnalysisConfig(
        data_path=args.data,
        output_dir=args.output_dir,
        hit_threshold=args.hit_threshold,
        k_values=args.k_values,
        selected_k=args.selected_k,
        example_genre=args.example_genre,
        example_platform=args.example_platform,
        example_critic_score=args.example_critic_score,
        example_user_score=args.example_user_score,
        top_coefficients=args.top_coefficients,
    )


def validate_schema(data: pd.DataFrame) -> None:
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing_columns:
        joined = ", ".join(missing_columns)
        raise ValueError(f"Dataset is missing required columns: {joined}")


def load_data(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    data = pd.read_csv(data_path)
    validate_schema(data)
    numeric_columns = SALES_COLUMNS + SCORE_COLUMNS + ["Year_of_Release"]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def clean_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    required_columns = SALES_COLUMNS + SCORE_COLUMNS + [
        "Name",
        "Platform",
        "Genre",
        "Year_of_Release",
    ]
    data = raw_data.dropna(subset=required_columns).drop_duplicates().copy()
    sales_are_valid = (data[SALES_COLUMNS] >= 0).all(axis=1) & (data["Global_Sales"] > 0)
    data = data.loc[sales_are_valid].copy()
    data["Year_of_Release"] = data["Year_of_Release"].astype(int)
    data["Decade"] = (data["Year_of_Release"] // 10) * 10
    return data.reset_index(drop=True)


def save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def adjusted_r2(r2: float, n_observations: int, n_features: int) -> float:
    if n_observations <= n_features + 1:
        return float("nan")
    return 1 - (1 - r2) * (n_observations - 1) / (n_observations - n_features - 1)


def plot_descriptive_charts(data: pd.DataFrame, figure_dir: Path) -> None:
    numeric_columns = SALES_COLUMNS + SCORE_COLUMNS

    plt.figure(figsize=(10, 8))
    sns.heatmap(data[numeric_columns].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix for Sales and Review Scores")
    save_figure(figure_dir / "correlation_matrix.png")

    decade_sales = data.groupby("Decade", as_index=False)["Global_Sales"].mean()
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=decade_sales, x="Decade", y="Global_Sales", marker="o")
    plt.title("Average Global Sales by Decade")
    plt.xlabel("Decade")
    plt.ylabel("Average Global Sales (Millions)")
    save_figure(figure_dir / "average_global_sales_by_decade.png")

    genre_sales = (
        data.groupby("Genre", as_index=False)["Global_Sales"]
        .mean()
        .sort_values("Global_Sales", ascending=False)
    )
    plt.figure(figsize=(11, 6))
    sns.barplot(data=genre_sales, x="Global_Sales", y="Genre", color="#4C78A8")
    plt.title("Average Global Sales by Genre")
    plt.xlabel("Average Global Sales (Millions)")
    plt.ylabel("Genre")
    save_figure(figure_dir / "average_global_sales_by_genre.png")

    plt.figure(figsize=(9, 5))
    sns.histplot(data["Global_Sales"], bins=50, kde=True, color="#F58518")
    plt.title("Distribution of Global Sales")
    plt.xlabel("Global Sales (Millions)")
    save_figure(figure_dir / "global_sales_distribution.png")


def run_global_sales_linear_model(data: pd.DataFrame, figure_dir: Path) -> dict[str, float]:
    features = pd.get_dummies(
        data[["Critic_Score", "User_Score", "Genre"]],
        columns=["Genre"],
        drop_first=True,
        dtype=float,
    )
    target = np.log1p(data["Global_Sales"])
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.3, random_state=4349
    )

    model = LinearRegression().fit(x_train, y_train)
    y_pred = model.predict(x_test)
    train_r2 = model.score(x_train, y_train)
    test_r2 = model.score(x_test, y_test)

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, alpha=0.65, color="#4C78A8")
    limits = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(limits, limits, linestyle="--", color="#222222")
    plt.xlabel("Actual Log Global Sales")
    plt.ylabel("Predicted Log Global Sales")
    plt.title("Linear Regression: Predicted vs Actual Global Sales")
    plt.grid(True, alpha=0.3)
    save_figure(figure_dir / "linear_global_sales_predicted_vs_actual.png")

    return {
        "train_r2": float(train_r2),
        "train_adjusted_r2": float(adjusted_r2(train_r2, *x_train.shape)),
        "test_r2": float(test_r2),
        "test_rmse_log_sales": float(np.sqrt(mean_squared_error(y_test, y_pred))),
    }


def run_user_score_linear_model(data: pd.DataFrame) -> dict[str, float]:
    features = data[["Year_of_Release"]].copy()
    target = np.log1p(data["User_Score"])
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.3, random_state=502
    )

    model = LinearRegression().fit(x_train, y_train)
    y_pred = model.predict(x_test)
    train_r2 = model.score(x_train, y_train)
    test_r2 = model.score(x_test, y_test)

    return {
        "train_r2": float(train_r2),
        "train_adjusted_r2": float(adjusted_r2(train_r2, *x_train.shape)),
        "test_r2": float(test_r2),
        "test_rmse_log_user_score": float(np.sqrt(mean_squared_error(y_test, y_pred))),
    }


def run_critic_score_linear_model(
    data: pd.DataFrame, figure_dir: Path, top_coefficients: int
) -> dict[str, Any]:
    features = pd.get_dummies(
        data[["User_Score", "Global_Sales", "Platform", "Genre", "Year_of_Release"]],
        columns=["Platform", "Genre"],
        drop_first=True,
        dtype=float,
    )
    target = data["Critic_Score"]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.4, random_state=123
    )

    model = LinearRegression().fit(x_train, y_train)
    y_pred = model.predict(x_test)
    train_r2 = model.score(x_train, y_train)
    test_r2 = model.score(x_test, y_test)
    coefficients = (
        pd.Series(model.coef_, index=features.columns)
        .sort_values(key=np.abs, ascending=False)
        .head(top_coefficients)
        .round(4)
        .to_dict()
    )

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, alpha=0.65, color="#54A24B")
    limits = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(limits, limits, linestyle="--", color="#222222")
    plt.xlabel("Actual Critic Score")
    plt.ylabel("Predicted Critic Score")
    plt.title("Linear Regression: Predicted vs Actual Critic Score")
    plt.grid(True, alpha=0.3)
    save_figure(figure_dir / "linear_critic_score_predicted_vs_actual.png")

    return {
        "train_r2": float(train_r2),
        "train_adjusted_r2": float(adjusted_r2(train_r2, *x_train.shape)),
        "test_r2": float(test_r2),
        "test_rmse_critic_score": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "largest_coefficients": coefficients,
    }


def run_logistic_hit_model(
    data: pd.DataFrame, figure_dir: Path, hit_threshold: float
) -> dict[str, Any]:
    model_data = data[
        ["Genre", "Platform", "Critic_Score", "User_Score", "Global_Sales"]
    ].dropna()
    model_data = model_data.copy()
    model_data["hit"] = (model_data["Global_Sales"] >= hit_threshold).astype(int)

    genre_means = model_data.groupby("Genre").agg(
        genre_avg_critic=("Critic_Score", "mean"),
        genre_avg_user=("User_Score", "mean"),
    )
    model_data = model_data.join(genre_means, on="Genre")

    features = pd.get_dummies(
        model_data[["Platform", "Genre", "genre_avg_critic", "genre_avg_user"]],
        columns=["Platform", "Genre"],
        drop_first=True,
        dtype=float,
    )
    target = model_data["hit"]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    numeric_columns = ["genre_avg_critic", "genre_avg_user"]
    scaler = StandardScaler()
    x_train = x_train.copy()
    x_test = x_test.copy()
    x_train.loc[:, numeric_columns] = scaler.fit_transform(x_train[numeric_columns])
    x_test.loc[:, numeric_columns] = scaler.transform(x_test[numeric_columns])

    model = LogisticRegression(max_iter=500, class_weight="balanced")
    model.fit(x_train, y_train)
    y_proba = model.predict_proba(x_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted No Hit", "Predicted Hit"],
        yticklabels=["Actual No Hit", "Actual Hit"],
    )
    plt.title("Logistic Regression Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    save_figure(figure_dir / "logistic_confusion_matrix.png")

    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title("Logistic Regression ROC Curve")
    save_figure(figure_dir / "logistic_roc_curve.png")

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return {
        "hit_threshold_millions": hit_threshold,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def run_knn_sales_model(
    data: pd.DataFrame,
    example_genre: str,
    example_platform: str,
    example_critic_score: float,
    example_user_score: float,
) -> dict[str, Any]:
    model_data = data[
        [
            "Name",
            "Genre",
            "Platform",
            "Critic_Score",
            "User_Score",
            "Year_of_Release",
            "Global_Sales",
        ]
    ].dropna()
    model_data = model_data.reset_index(drop=True)

    features = pd.get_dummies(
        model_data[["Genre", "Platform", "Critic_Score", "User_Score"]],
        columns=["Genre", "Platform"],
        drop_first=True,
        dtype=float,
    )
    target = np.log1p(model_data["Global_Sales"])
    numeric_columns = ["Critic_Score", "User_Score"]

    x_train, x_test, y_train, y_test, idx_train, _idx_test = train_test_split(
        features, target, model_data.index, test_size=0.3, random_state=RANDOM_STATE
    )
    scaler = StandardScaler()
    x_train = x_train.copy()
    x_test = x_test.copy()
    x_train.loc[:, numeric_columns] = scaler.fit_transform(x_train[numeric_columns])
    x_test.loc[:, numeric_columns] = scaler.transform(x_test[numeric_columns])

    model = KNeighborsRegressor(n_neighbors=60, weights="distance")
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    new_game = pd.DataFrame(
        {
            "Genre": [example_genre],
            "Platform": [example_platform],
            "Critic_Score": [example_critic_score],
            "User_Score": [example_user_score],
        }
    )
    new_features = pd.get_dummies(
        new_game, columns=["Genre", "Platform"], drop_first=True, dtype=float
    )
    new_features = new_features.reindex(columns=features.columns, fill_value=0).astype(float)
    new_features.loc[:, numeric_columns] = scaler.transform(new_features[numeric_columns])
    predicted_log_sales = model.predict(new_features)[0]

    distances, neighbor_indexes = model.kneighbors(
        new_features, n_neighbors=5, return_distance=True
    )
    neighbors: list[dict[str, Any]] = []
    for rank, (row_index, distance) in enumerate(
        zip(idx_train[neighbor_indexes[0]], distances[0]), start=1
    ):
        row = model_data.loc[row_index]
        neighbors.append(
            {
                "rank": rank,
                "name": row["Name"],
                "global_sales_millions": float(row["Global_Sales"]),
                "critic_score": float(row["Critic_Score"]),
                "user_score": float(row["User_Score"]),
                "distance": float(distance),
            }
        )

    return {
        "test_mae_log_sales": float(mean_absolute_error(y_test, y_pred)),
        "test_r2": float(r2_score(y_test, y_pred)),
        "example_game": {
            "genre": example_genre,
            "platform": example_platform,
            "critic_score": example_critic_score,
            "user_score": example_user_score,
        },
        "example_prediction_global_sales_millions": float(np.expm1(predicted_log_sales)),
        "nearest_neighbors_for_example": neighbors,
    }


def genre_decade_profiles(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.groupby(["Genre", "Decade"])
        .agg(
            avg_NA=("NA_Sales", "mean"),
            avg_EU=("EU_Sales", "mean"),
            avg_JP=("JP_Sales", "mean"),
            avg_Other=("Other_Sales", "mean"),
            avg_critic=("Critic_Score", "mean"),
            avg_user=("User_Score", "mean"),
            n_games=("Genre", "size"),
        )
        .reset_index()
    )


def run_kmeans_genre_decade(
    data: pd.DataFrame, output_dir: Path, k_values: tuple[int, ...], selected_k: int
) -> dict[str, Any]:
    figure_dir = output_dir / "figures"
    profiles = genre_decade_profiles(data)
    feature_columns = ["avg_NA", "avg_EU", "avg_JP", "avg_Other", "avg_critic", "avg_user"]
    features = profiles[feature_columns]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    wcss: list[float] = []
    elbow_k_values = range(1, 11)
    for k in elbow_k_values:
        model = KMeans(n_clusters=k, random_state=0, n_init=10)
        model.fit(scaled_features)
        wcss.append(float(model.inertia_))

    plt.figure(figsize=(8, 5))
    plt.plot(list(elbow_k_values), wcss, marker="o", color="#4C78A8")
    plt.title("Elbow Method for Genre-Decade K-Means")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Within-Cluster Sum of Squares")
    plt.grid(True, alpha=0.3)
    save_figure(figure_dir / "kmeans_elbow_method.png")

    scores: dict[str, float] = {}
    clustered_outputs: dict[int, pd.DataFrame] = {}
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = model.fit_predict(scaled_features)
        scores[f"k_{k}"] = float(silhouette_score(scaled_features, labels))
        clustered = profiles.copy()
        clustered["cluster"] = labels
        clustered_outputs[k] = clustered
        clustered.to_csv(output_dir / f"genre_decade_clusters_k{k}.csv", index=False)

    final_clusters = clustered_outputs[selected_k]
    cluster_summary = (
        final_clusters.groupby("cluster")
        .agg(
            pairs=("cluster", "size"),
            mean_NA=("avg_NA", "mean"),
            mean_EU=("avg_EU", "mean"),
            mean_JP=("avg_JP", "mean"),
            mean_Other=("avg_Other", "mean"),
            mean_critic_score=("avg_critic", "mean"),
            mean_user_score=("avg_user", "mean"),
        )
        .round(3)
    )
    cluster_summary.to_csv(output_dir / f"cluster_summary_k{selected_k}.csv")

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        scaled_features[:, 0],
        scaled_features[:, 4],
        c=final_clusters["cluster"],
        cmap="viridis",
        s=np.clip(final_clusters["n_games"] * 12, 45, 260),
        alpha=0.82,
        edgecolors="white",
        linewidth=0.6,
    )
    plt.xlabel("Scaled Average North America Sales")
    plt.ylabel("Scaled Average Critic Score")
    plt.title("K-Means Clusters by Genre and Decade")
    plt.legend(*scatter.legend_elements(), title="Cluster", loc="best")
    plt.grid(True, alpha=0.25)
    save_figure(figure_dir / "kmeans_genre_decade_clusters.png")

    heatmap_data = cluster_summary[
        ["mean_NA", "mean_EU", "mean_JP", "mean_Other", "mean_critic_score", "mean_user_score"]
    ]
    plt.figure(figsize=(9, 4))
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("K-Means Cluster-Level Means")
    plt.ylabel("Cluster")
    save_figure(figure_dir / "kmeans_cluster_summary.png")

    return {
        "silhouette_scores": scores,
        "selected_k": selected_k,
        "selected_cluster_summary": cluster_summary.reset_index().to_dict(orient="records"),
        "wcss_by_k": dict(zip([str(k) for k in elbow_k_values], wcss)),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    headers = list(rows[0].keys())
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = []
    for row in rows:
        values = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        body_lines.append("| " + " | ".join(values) + " |")
    return "\n".join([header_line, separator, *body_lines])


def write_markdown_report(output_dir: Path, metrics: dict[str, Any]) -> Path:
    dataset = metrics["dataset"]
    global_sales = metrics["linear_global_sales"]
    critic_score = metrics["linear_critic_score"]
    logistic = metrics["logistic_hit_model"]
    knn = metrics["knn_sales_model"]
    kmeans = metrics["kmeans_genre_decade"]
    selected_k = kmeans["selected_k"]
    selected_silhouette = kmeans["silhouette_scores"][f"k_{selected_k}"]
    example_game = knn["example_game"]

    cluster_table = markdown_table(kmeans["selected_cluster_summary"])
    report = f"""# Video Game Sales Analysis Report

This report was generated by `src/video_games_analysis.py`.

## Dataset

- Raw rows: {dataset["raw_rows"]}
- Clean modeling rows: {dataset["clean_rows"]}
- Year range: {dataset["year_min"]}-{dataset["year_max"]}
- Genre count: {len(dataset["genres"])}
- Platform count: {dataset["platform_count"]}

## Model Summary

| Model | Main metric | Value |
| --- | --- | --- |
| Linear regression, global sales | Test R2 | {global_sales["test_r2"]:.3f} |
| Linear regression, critic score | Test R2 | {critic_score["test_r2"]:.3f} |
| Logistic hit classifier | ROC-AUC | {logistic["roc_auc"]:.3f} |
| KNN sales regressor | Test R2 | {knn["test_r2"]:.3f} |
| K-Means genre-decade | Selected k | {selected_k} |
| K-Means genre-decade | Silhouette | {selected_silhouette:.3f} |

## Visuals

![Correlation matrix](figures/correlation_matrix.png)

![Average global sales by decade](figures/average_global_sales_by_decade.png)

![K-Means clusters by genre and decade](figures/kmeans_genre_decade_clusters.png)

![Logistic regression ROC curve](figures/logistic_roc_curve.png)

## Selected K-Means Cluster Summary

{cluster_table}

## Example KNN Prediction

The example game configuration was `{example_game["genre"]}` on
`{example_game["platform"]}` with critic score
`{example_game["critic_score"]}` and user score
`{example_game["user_score"]}`.

Predicted global sales: `{knn["example_prediction_global_sales_millions"]:.3f}`
million units.

## Output Files

- `model_metrics.json`: structured metrics for downstream use.
- `cluster_summary_k{selected_k}.csv`: selected cluster-level summary.
- `genre_decade_clusters_k*.csv`: cluster assignment tables for evaluated k values.
- `figures/`: generated visual outputs.
"""
    path = output_dir / "analysis_report.md"
    path.write_text(report, encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    config = build_config(args)
    output_dir = config.output_dir
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    config_payload = asdict(config)
    config_payload["data_path"] = str(config.data_path)
    config_payload["output_dir"] = str(config.output_dir)

    raw_data = load_data(config.data_path)
    data = clean_data(raw_data)
    plot_descriptive_charts(data, figure_dir)

    metrics: dict[str, Any] = {
        "config": config_payload,
        "dataset": {
            "source": str(config.data_path),
            "raw_rows": int(raw_data.shape[0]),
            "raw_columns": int(raw_data.shape[1]),
            "clean_rows": int(data.shape[0]),
            "clean_columns": int(data.shape[1]),
            "year_min": int(data["Year_of_Release"].min()),
            "year_max": int(data["Year_of_Release"].max()),
            "genres": sorted(data["Genre"].unique().tolist()),
            "platform_count": int(data["Platform"].nunique()),
        },
        "linear_global_sales": run_global_sales_linear_model(data, figure_dir),
        "linear_user_score": run_user_score_linear_model(data),
        "linear_critic_score": run_critic_score_linear_model(
            data, figure_dir, config.top_coefficients
        ),
        "logistic_hit_model": run_logistic_hit_model(
            data, figure_dir, config.hit_threshold
        ),
        "knn_sales_model": run_knn_sales_model(
            data,
            config.example_genre,
            config.example_platform,
            config.example_critic_score,
            config.example_user_score,
        ),
        "kmeans_genre_decade": run_kmeans_genre_decade(
            data, output_dir, config.k_values, config.selected_k
        ),
    }

    write_json(output_dir / "model_metrics.json", metrics)
    report_path = write_markdown_report(output_dir, metrics)
    print(f"Analysis complete. Outputs written to: {output_dir.resolve()}")
    print(f"Markdown report written to: {report_path.resolve()}")
    selected_score_key = f"k_{config.selected_k}"
    print(
        "Selected K-Means silhouette score: "
        f"{metrics['kmeans_genre_decade']['silhouette_scores'][selected_score_key]:.3f}"
    )
    print(
        "Global-sales linear regression test R2: "
        f"{metrics['linear_global_sales']['test_r2']:.3f}"
    )
    print(
        "Logistic hit model ROC-AUC: "
        f"{metrics['logistic_hit_model']['roc_auc']:.3f}"
    )


if __name__ == "__main__":
    main()

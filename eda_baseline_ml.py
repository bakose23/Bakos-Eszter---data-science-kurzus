from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data.csv"
TARGET_COL = "value"
RANDOM_STATE = 7
TEST_SIZE = 0.25
DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
FEATURES = ["facility_code", "day_of_week", "time_band", "tb_sin", "tb_cos"]


class DayHourMeanRegressor(RegressorMixin, BaseEstimator):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DayHourMeanRegressor":
        self.global_mean_ = float(np.mean(y))
        grouped = pd.DataFrame(
            {
                "day_of_week": X["day_of_week"].to_numpy(),
                "time_band": X["time_band"].to_numpy(),
                "target": np.asarray(y),
            }
        )
        self.group_means_ = grouped.groupby(["day_of_week", "time_band"])["target"].mean()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        keys = pd.MultiIndex.from_arrays([X["day_of_week"], X["time_band"]])
        predictions = self.group_means_.reindex(keys).to_numpy(dtype=float)
        return np.where(np.isnan(predictions), self.global_mean_, predictions)


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["row_id", "day_name_long", "data_uploaded"])
    df["facility_code"] = df["facility_code"].astype(str)
    df["time_band"] = df["time_band"].astype(float)
    df["tb_sin"] = np.sin(2 * np.pi * df["time_band"] / 24)
    df["tb_cos"] = np.cos(2 * np.pi * df["time_band"] / 24)
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["facility_code", "day_of_week"]),
            ("num", StandardScaler(), ["time_band", "tb_sin", "tb_cos"]),
        ]
    )


def make_pipeline(model: object, preprocess: bool = True) -> Pipeline:
    preprocessor: object = build_preprocessor() if preprocess else "passthrough"
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def run_eda(df: pd.DataFrame) -> None:
    print("\n=== 1. EDA ===")
    print("\nDescriptive statistics")
    print(df.describe().to_string())
    print("\nMissing-value count by column")
    print(df.isna().sum().to_string())
    print(f"Total missing values: {int(df.isna().sum().sum())}")

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].hist(df[TARGET_COL].dropna(), bins=30, color="steelblue", edgecolor="white")
    axes[0, 0].set(title="Distribution of ED busyness", xlabel="Value", ylabel="Count")

    df.groupby("time_band")[TARGET_COL].mean().sort_index().plot(
        ax=axes[0, 1], marker="o", color="darkorange"
    )
    axes[0, 1].set(title="Mean value by time band", xlabel="Time band", ylabel="Mean value")

    day_means = df.groupby("day_name")[TARGET_COL].mean().reindex(DAY_ORDER)
    axes[1, 0].bar(day_means.index, day_means.values, color="seagreen")
    axes[1, 0].set(title="Mean value by day", xlabel="Day", ylabel="Mean value")

    facility_means = df.groupby("facility_code")[TARGET_COL].mean().sort_values()
    axes[1, 1].bar(facility_means.index, facility_means.values, color="slateblue")
    axes[1, 1].set(title="Mean value by facility", xlabel="Facility code", ylabel="Mean value")
    axes[1, 1].tick_params(axis="x", labelrotation=90)

    figure.tight_layout()
    eda_path = BASE_DIR / "eda.png"
    figure.savefig(eda_path, dpi=200)
    plt.close(figure)

    heatmap = df.pivot_table(
        index="day_of_week", columns="time_band", values=TARGET_COL, aggfunc="mean"
    ).sort_index().sort_index(axis=1)
    figure, axis = plt.subplots(figsize=(12, 6))
    image = axis.imshow(heatmap.to_numpy(), aspect="auto", cmap="viridis")
    axis.set_xticks(np.arange(len(heatmap.columns)), labels=heatmap.columns)
    axis.set_yticks(np.arange(len(heatmap.index)), labels=heatmap.index)
    axis.set(title="Mean ED busyness by day and time", xlabel="Time band", ylabel="Day of week")
    figure.colorbar(image, ax=axis, label="Mean value")
    figure.tight_layout()
    heatmap_path = BASE_DIR / "heatmap.png"
    figure.savefig(heatmap_path, dpi=200)
    plt.close(figure)

    print(f"Saved EDA plots: {eda_path} and {heatmap_path}")


def build_models() -> dict[str, Pipeline]:
    return {
        "DummyRegressor(strategy='mean')": make_pipeline(DummyRegressor(strategy="mean")),
        "Day-hour group mean": make_pipeline(DayHourMeanRegressor(), preprocess=False),
        "LinearRegression": make_pipeline(LinearRegression()),
        "MLPRegressor(hidden_layer_sizes=(16,16), max_iter=500, random_state=7)": make_pipeline(
            MLPRegressor(
                hidden_layer_sizes=(16, 16),
                max_iter=500,
                random_state=RANDOM_STATE,
            )
        ),
        "SVR(kernel='rbf')": make_pipeline(SVR(kernel="rbf")),
    }


def evaluate_models(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES]
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for model_index, (name, pipeline) in enumerate(build_models().items()):
        if model_index == 0:
            print("\n=== 2. Baseline ===")
        elif model_index == 3:
            print("\n=== 3. Machine learning ===")

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", ConvergenceWarning)
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
            cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="r2")

        seen_warnings = set()
        for warning in caught_warnings:
            if isinstance(warning.message, ConvergenceWarning):
                message = str(warning.message)
                if message not in seen_warnings:
                    print(f"ConvergenceWarning for {name}: {message}")
                    seen_warnings.add(message)

        rows.append(
            {
                "model": name,
                "test_R2": r2_score(y_test, predictions),
                "test_RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
                "test_MAE": mean_absolute_error(y_test, predictions),
                "CV_R2_mean": cv_scores.mean(),
                "CV_R2_std": cv_scores.std(),
            }
        )

    return pd.DataFrame(rows).sort_values("test_R2", ascending=False).reset_index(drop=True)


def main() -> None:
    df = load_dataset()
    run_eda(df)
    results = evaluate_models(df)

    print("\nModel evaluation results")
    print(results.to_string(index=False))

    results_path = BASE_DIR / "results.csv"
    results.to_csv(results_path, index=False)

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.barh(results["model"], results["test_R2"], color="steelblue")
    axis.set(xlabel="Test R²", ylabel="Model", title="Model comparison by test R²")
    axis.invert_yaxis()
    figure.tight_layout()
    results_plot_path = BASE_DIR / "results.png"
    figure.savefig(results_plot_path, dpi=200)
    plt.close(figure)

    print(f"\nSaved results table: {results_path}")
    print(f"Saved test R² chart: {results_plot_path}")


if __name__ == "__main__":
    main()
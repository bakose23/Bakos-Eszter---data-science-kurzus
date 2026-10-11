from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

DATA_PATH = Path(r"C:\Users\eszti\Downloads\archive\data.csv")
TARGET_COL = "value"
RANDOM_STATE = 7
TEST_SIZE = 0.25


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["row_id", "day_name", "day_name_long", "data_uploaded"])
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


def build_models() -> dict:
    return {
        "MLPRegressor(hidden_layer_sizes=(16,16), max_iter=500, random_state=7)": MLPRegressor(
            hidden_layer_sizes=(16, 16),
            max_iter=500,
            random_state=RANDOM_STATE,
        ),
        "SVR(kernel='rbf')": SVR(kernel="rbf"),
    }


def evaluate_models() -> pd.DataFrame:
    df = load_dataset()
    X = df[["facility_code", "day_of_week", "time_band", "tb_sin", "tb_cos"]]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    rows = []
    for name, model in build_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", model),
            ]
        )

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", ConvergenceWarning)
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)

        for warning in caught_warnings:
            if isinstance(warning.message, ConvergenceWarning):
                print(f"Warning for {name}: {warning.message}")

        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="r2")

        rows.append(
            {
                "model": name,
                "R2": r2,
                "RMSE": rmse,
                "MAE": mae,
                "CV_R2_mean": cv_scores.mean(),
                "CV_R2_std": cv_scores.std(),
            }
        )

    return pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)


def main() -> None:
    results = evaluate_models()

    print("\nModel comparison results")
    print(results[["model", "R2", "RMSE", "MAE", "CV_R2_mean", "CV_R2_std"]].to_string(index=False))

    plt.figure(figsize=(10, 6))
    plt.barh(results["model"], results["R2"], color="steelblue")
    plt.xlabel("Test R²")
    plt.ylabel("Model")
    plt.title("Model comparison by test R²")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(Path(__file__).resolve().parent / "results.png", dpi=200)
    plt.close()

    print(f"\nSaved chart to: {Path(__file__).resolve().parent / 'results.png'}")


if __name__ == "__main__":
    main()

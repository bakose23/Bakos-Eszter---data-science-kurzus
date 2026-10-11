from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

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


def get_preprocessor() -> ColumnTransformer:
    categorical_features = ["facility_code", "day_of_week"]
    numeric_features = ["time_band", "tb_sin", "tb_cos"]

    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", StandardScaler(), numeric_features),
        ]
    )


def get_models() -> dict:
    return {
        "LinearRegression": LinearRegression(),
        "SVR_rbf": SVR(kernel="rbf"),
        "DecisionTreeRegressor(max_depth=4)": DecisionTreeRegressor(max_depth=4, random_state=RANDOM_STATE),
        "RandomForestRegressor(n_estimators=200, max_depth=6)": RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            random_state=RANDOM_STATE,
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "MLPRegressor(hidden_layer_sizes=(16,16), max_iter=500)": MLPRegressor(
            hidden_layer_sizes=(16, 16),
            max_iter=500,
            random_state=RANDOM_STATE,
        ),
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

    results = []
    for name, model in get_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", get_preprocessor()),
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

        results.append(
            {
                "model": name,
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "R2": r2,
            }
        )

    return pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)


def main() -> None:
    print(f"Loading dataset from: {DATA_PATH}")
    results = evaluate_models().sort_values("R2", ascending=False).reset_index(drop=True)
    print("\nModel comparison results")
    print(results.to_string(index=False))

    plt.figure(figsize=(10, 6))
    plt.barh(results["model"], results["R2"], color="steelblue")
    plt.xlabel("R²")
    plt.ylabel("Model")
    plt.title("Model comparison by R²")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(Path(__file__).resolve().parent / "model_comparison_r2.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


BASE = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront",
    "view", "condition", "grade", "sqft_above", "sqft_basement", "yr_built",
    "lat", "long", "sqft_living15", "sqft_lot15", "t_days",
    "ln_living", "ln_lot", "ln_living15", "ln_lot15",
]
GEO = ["dist_seattle", "dist_bellevue", "dist_min", "rot1", "rot2"]
RATIO = [
    "living_ratio", "lot_ratio", "bath_per_bed", "basement_share",
    "age", "renovated", "years_since_work",
]
COMPS = ["comps_ppsf", "comps_price"]
LINCOLS = [column for column in BASE if column not in {
    "sqft_living", "sqft_lot", "sqft_living15", "sqft_lot15",
}]

DEV_END: pd.Timestamp | None = None
FOLDS: list[tuple[np.ndarray, np.ndarray]] = []
USE_XGBOOST = XGBRegressor is not None
_COMPS_FEATURE_CACHE: dict[tuple[tuple[int, ...], tuple[int, ...], int], pd.DataFrame] = {}


@dataclass(frozen=True)
class Spec:
    name: str
    kind: str
    features: tuple[str, ...]
    target: str = "level"
    zip_mode: str = "onehot"

    def with_features(self, features: list[str] | tuple[str, ...]) -> "Spec":
        return replace(self, features=tuple(features))

    def with_target(self, target: str) -> "Spec":
        return replace(self, target=target)


def _dev_end_and_folds(frame: pd.DataFrame) -> None:
    global DEV_END, FOLDS
    DEV_END = pd.Timestamp(frame["date"].iloc[int(0.80 * len(frame))])
    dev = frame.loc[frame["date"] < DEV_END]
    dates = np.sort(dev["date"].unique())
    starts = [dates[int(fraction * len(dates))] for fraction in (0.50, 0.65, 0.80)]
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for number, start in enumerate(starts):
        stop = starts[number + 1] if number + 1 < len(starts) else dates[-1] + np.timedelta64(1, "D")
        val = dev.loc[(dev["date"] >= start) & (dev["date"] < stop)]
        earlier = dev.loc[dev["date"] < start]
        train = earlier.loc[~earlier["prop"].isin(val["prop"])]
        if not set(train["prop"]).isdisjoint(val["prop"]):
            raise AssertionError("Az ingatlanok átfednek a tanító- és validációs blokkok között.")
        folds.append((train.index.to_numpy(), val.index.to_numpy()))
    FOLDS = folds


def load() -> pd.DataFrame:
    """Betölti, megtisztítja és a dátumrend szerint rendezi az eladásokat."""
    global USE_XGBOOST
    try:
        frame = fetch_openml(data_id=42092, as_frame=True).frame.copy()
    except Exception as openml_error:
        csv_path = Path("kc_house_data.csv")
        if not csv_path.is_file():
            csv_path = Path("lesson 5") / "kc_house_data.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(
                "Az OpenML nem érhető el, és a kc_house_data.csv sem található."
            ) from openml_error
        frame = pd.read_csv(csv_path).drop(columns="id")

    source_rows = len(frame)
    original_max_bedrooms = int(frame["bedrooms"].max())
    missing_count = int(frame.isna().sum().sum())
    renovation_zero_rate = float((frame["yr_renovated"] == 0).mean())
    source_price_mean = float(frame["price"].mean())
    source_price_median = float(frame["price"].median())
    source_price_max = float(frame["price"].max())
    source_over_million_rate = float((frame["price"] > 1_000_000).mean())
    source_zip_count = int(frame["zipcode"].nunique())
    frame["date"] = pd.to_datetime(
        frame["date"].astype(str), format="%Y%m%dT%H%M%S", errors="raise"
    )
    frame["zipcode"] = frame["zipcode"].astype(str).str.zfill(5)
    frame = frame.loc[frame["bedrooms"] < 10].copy()
    frame = frame.sort_values("date", kind="stable").reset_index(drop=True)
    frame["t_days"] = (frame["date"] - frame["date"].min()).dt.days
    frame["age"] = frame["date"].dt.year - frame["yr_built"]
    frame["renovated"] = (frame["yr_renovated"] > 0).astype(int)
    last_work = frame["yr_renovated"].where(frame["renovated"].eq(1), frame["yr_built"])
    frame["years_since_work"] = frame["date"].dt.year - last_work

    def distance_km(latitude: np.ndarray, longitude: np.ndarray, city: tuple[float, float]) -> np.ndarray:
        lat1 = np.radians(latitude)
        lat2 = np.radians(city[0])
        delta_lat = lat2 - lat1
        delta_lon = np.radians(city[1] - longitude)
        a = np.sin(delta_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
        return 12742 * np.arcsin(np.sqrt(a))

    latitude = frame["lat"].to_numpy(dtype=float)
    longitude = frame["long"].to_numpy(dtype=float)
    frame["dist_seattle"] = distance_km(latitude, longitude, (47.6062, -122.3321))
    frame["dist_bellevue"] = distance_km(latitude, longitude, (47.6101, -122.2015))
    frame["dist_min"] = frame[["dist_seattle", "dist_bellevue"]].min(axis=1)
    x = longitude * np.cos(np.deg2rad(47.6))
    y = latitude
    frame["rot1"] = (x + y) / np.sqrt(2)
    frame["rot2"] = (x - y) / np.sqrt(2)
    frame["living_ratio"] = frame["sqft_living"] / frame["sqft_living15"].replace(0, np.nan)
    frame["lot_ratio"] = frame["sqft_lot"] / frame["sqft_lot15"].replace(0, np.nan)
    frame["bath_per_bed"] = frame["bathrooms"] / frame["bedrooms"].clip(lower=1)
    frame["basement_share"] = frame["sqft_basement"] / frame["sqft_living"].replace(0, np.nan)
    frame[["living_ratio", "lot_ratio", "basement_share"]] = frame[
        ["living_ratio", "lot_ratio", "basement_share"]
    ].replace([np.inf, -np.inf], np.nan).fillna(0)
    frame["ln_living"] = np.log1p(frame["sqft_living"].clip(lower=0))
    frame["ln_lot"] = np.log1p(frame["sqft_lot"].clip(lower=0))
    frame["ln_living15"] = np.log1p(frame["sqft_living15"].clip(lower=0))
    frame["ln_lot15"] = np.log1p(frame["sqft_lot15"].clip(lower=0))
    frame["prop"], _ = pd.factorize(pd.MultiIndex.from_frame(frame[
        ["lat", "long", "sqft_living", "sqft_lot", "yr_built", "zipcode"]
    ]))
    frame.attrs.update({
        "source_rows": source_rows,
        "original_max_bedrooms": original_max_bedrooms,
        "source_missing_count": missing_count,
        "renovation_zero_rate": renovation_zero_rate,
        "source_price_mean": source_price_mean,
        "source_price_median": source_price_median,
        "source_price_max": source_price_max,
        "source_over_million_rate": source_over_million_rate,
        "source_zip_count": source_zip_count,
    })
    _dev_end_and_folds(frame)
    USE_XGBOOST = XGBRegressor is not None
    return frame


def lr_naive() -> Spec:
    return Spec("lr_naive", "ridge", tuple(BASE) + ("zipcode",), zip_mode="numeric")


def lr_fair(features: list[str] | tuple[str, ...] | None = None) -> Spec:
    return Spec("lr_fair", "ridge", tuple(features or LINCOLS), zip_mode="onehot")


def tree(features: list[str] | tuple[str, ...] | None = None) -> Spec:
    return Spec("tree", "tree", tuple(features or BASE))


def forest(features: list[str] | tuple[str, ...] | None = None) -> Spec:
    return Spec("forest", "forest", tuple(features or BASE))


def xgb(features: list[str] | tuple[str, ...] | None = None, target: str = "level") -> Spec:
    return Spec("xgb" if USE_XGBOOST else "HistGradientBoosting", "xgb", tuple(features or BASE), target)


def metrics(y: np.ndarray | pd.Series, p: np.ndarray | pd.Series) -> dict[str, float]:
    actual = np.asarray(y, dtype=float)
    predicted = np.asarray(p, dtype=float)
    return {
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "mae": float(mean_absolute_error(actual, predicted)),
        "wape": float(np.abs(actual - predicted).sum() / np.abs(actual).sum()),
        "rmsle": float(mean_squared_log_error(np.maximum(actual, 0), np.maximum(predicted, 0)) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
        "ratio": float(predicted.sum() / actual.sum()),
    }


def fold_metric(oof_frame: pd.DataFrame, key: str) -> list[float]:
    if key not in {"rmse", "mae", "wape", "rmsle", "r2", "ratio"}:
        raise ValueError(f"Ismeretlen metrika: {key}")
    return [
        metrics(group["y"], group["p"])[key]
        for _, group in oof_frame.groupby("fold", sort=True)
    ]


def _coordinates(frame: pd.DataFrame, reference_lat: float | None = None) -> np.ndarray:
    mean_lat = float(frame["lat"].mean()) if reference_lat is None else reference_lat
    return np.column_stack([
        frame["lat"].to_numpy(dtype=float) * 110.574,
        frame["long"].to_numpy(dtype=float) * 111.320 * np.cos(np.deg2rad(mean_lat)),
    ])


def comps_feature(train: pd.DataFrame, query: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    """A tanítóból számolt comps; a lekérdezett ingatlan saját kulcsú sorait kizárja."""
    if train.empty:
        raise ValueError("A comps tanítóhalmaza nem lehet üres.")
    if k < 1:
        raise ValueError("A k paraméternek legalább 1-nek kell lennie.")
    cache_key = (tuple(train.index), tuple(query.index), k)
    if cache_key in _COMPS_FEATURE_CACHE:
        return _COMPS_FEATURE_CACHE[cache_key].copy()
    reference_lat = float(train["lat"].mean())
    neighbors = NearestNeighbors(algorithm="auto", n_jobs=1).fit(_coordinates(train, reference_lat))
    train_prop = train["prop"].to_numpy()
    query_prop = query["prop"].to_numpy()
    per_sqft = train["price"].to_numpy(dtype=float) / train["sqft_living"].clip(lower=1).to_numpy()
    count = min(len(train), k + 1)
    indices = None
    while True:
        _, indices = neighbors.kneighbors(_coordinates(query, reference_lat), n_neighbors=count)
        enough = np.array([
            np.sum(train_prop[row] != query_prop[i]) >= min(k, len(train) - 1)
            for i, row in enumerate(indices)
        ])
        if enough.all() or count == len(train):
            break
        count = min(len(train), max(count + k, count * 2))
    values = np.empty(len(query), dtype=float)
    for i, row in enumerate(indices):
        eligible = row[train_prop[row] != query_prop[i]][:k]
        values[i] = per_sqft[eligible].mean() if len(eligible) else float(per_sqft.mean())
    result = pd.DataFrame(
        {"comps_ppsf": values, "comps_price": values * query["sqft_living"].to_numpy()},
        index=query.index,
    )
    if len(_COMPS_FEATURE_CACHE) >= 64:
        _COMPS_FEATURE_CACHE.clear()
    _COMPS_FEATURE_CACHE[cache_key] = result.copy()
    return result


def comps_leaky(frame: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    """Szándékosan hibás: minden eladást és a saját sort is beleszámolja."""
    reference_lat = float(frame["lat"].mean())
    neighbors = NearestNeighbors(algorithm="auto", n_jobs=1).fit(_coordinates(frame, reference_lat))
    _, indices = neighbors.kneighbors(_coordinates(frame, reference_lat), n_neighbors=min(k, len(frame)))
    per_sqft = frame["price"].to_numpy(dtype=float) / frame["sqft_living"].clip(lower=1).to_numpy()
    values = per_sqft[indices].mean(axis=1)
    return pd.DataFrame(
        {"comps_ppsf": values, "comps_price": values * frame["sqft_living"].to_numpy()},
        index=frame.index,
    )


def _make_model(spec: Spec, seed: int):
    if spec.kind == "ridge":
        return Ridge(alpha=10.0)
    if spec.kind == "tree":
        return DecisionTreeRegressor(max_depth=14, min_samples_leaf=15, random_state=seed)
    if spec.kind == "forest":
        return RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2, max_features=0.5, n_jobs=4,
            random_state=seed,
        )
    if XGBRegressor is not None:
        return XGBRegressor(
            n_estimators=180, max_depth=4, max_bin=64, learning_rate=0.06,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=5.0,
            objective="reg:squarederror", tree_method="hist", n_jobs=4,
            random_state=seed, verbosity=0,
        )
    return HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
        l2_regularization=3.0, random_state=seed,
    )


def _features(spec: Spec, train: pd.DataFrame, query: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_features = [column for column in spec.features if column not in COMPS]
    train_x = train.loc[:, base_features].copy()
    query_x = query.loc[:, base_features].copy()
    if spec.zip_mode == "onehot":
        train_x["zipcode"] = train["zipcode"].astype(str)
        query_x["zipcode"] = query["zipcode"].astype(str)
    if "comps_ppsf" in spec.features or spec.target == "logratio":
        train_comps = comps_feature(train, train)
        query_comps = comps_feature(train, query)
        train_x = train_x.join(train_comps)
        query_x = query_x.join(query_comps)
    return train_x, query_x


def _fit_predict(spec: Spec, train: pd.DataFrame, query: pd.DataFrame, seed: int) -> np.ndarray:
    if spec.kind in {"mean", "zip_median"}:
        if spec.kind == "mean":
            return np.full(len(query), float(train["price"].mean()))
        medians = train.groupby("zipcode")["price"].median()
        return query["zipcode"].map(medians).fillna(float(train["price"].median())).to_numpy()

    train_x, query_x = _features(spec, train, query)
    numeric = [column for column in train_x.columns if column != "zipcode"]
    if spec.zip_mode == "numeric":
        train_x["zipcode"] = pd.to_numeric(train_x["zipcode"])
        query_x["zipcode"] = pd.to_numeric(query_x["zipcode"])
        numeric.append("zipcode")
        categories = []
    else:
        categories = ["zipcode"] if "zipcode" in train_x else []

    scale = StandardScaler() if spec.kind == "ridge" else "passthrough"
    transformers = []
    if numeric:
        transformers.append(("numeric", scale, numeric))
    if categories:
        transformers.append(("zipcode", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categories))
    prep = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)
    model = _make_model(spec, seed)

    if spec.target == "raw":
        target = train["price"].to_numpy(dtype=float)
        inverse = lambda values: values
    elif spec.target == "ppsf":
        target = train["price"].to_numpy(dtype=float) / train["sqft_living"].clip(lower=1).to_numpy()
        inverse = lambda values: values * query["sqft_living"].to_numpy()
    elif spec.target == "logratio":
        comps_price = train_x["comps_price"].to_numpy(dtype=float)
        query_comps_price = query_x["comps_price"].to_numpy(dtype=float)
        target = np.log(train["price"].to_numpy(dtype=float) / np.maximum(comps_price, 1.0))
        inverse = lambda values: np.exp(np.clip(values, -5, 5)) * query_comps_price
    elif spec.target == "level":
        target = np.log1p(train["price"].to_numpy(dtype=float))
        inverse = np.expm1
    else:
        raise ValueError(f"Ismeretlen céltranszformáció: {spec.target}")

    pipe = Pipeline([("preprocess", prep), ("model", model)])
    pipe.fit(train_x, target)
    return np.maximum(inverse(pipe.predict(query_x)), 0.0)


def fit_predict(spec: Spec, train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int = 0) -> np.ndarray:
    return _fit_predict(spec, train_df, test_df, seed)


def quantile_predict(
    spec: Spec, train_df: pd.DataFrame, test_df: pd.DataFrame, alpha: float, seed: int = 0
) -> np.ndarray:
    train_x, test_x = _features(spec, train_df, test_df)
    numeric = [column for column in train_x if column != "zipcode"]
    categories = ["zipcode"] if "zipcode" in train_x else []
    transformers = []
    if numeric:
        transformers.append(("numeric", "passthrough", numeric))
    if categories:
        transformers.append(("zipcode", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categories))
    prep = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)
    if XGBRegressor is not None:
        model = XGBRegressor(
            n_estimators=180, max_depth=4, max_bin=64, learning_rate=0.06,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=5.0,
            objective="reg:quantileerror", quantile_alpha=alpha,
            tree_method="hist", n_jobs=4, random_state=seed, verbosity=0,
        )
    else:
        model = HistGradientBoostingRegressor(
            loss="quantile", quantile=alpha, max_iter=500, max_leaf_nodes=31,
            learning_rate=0.05, l2_regularization=3.0, random_state=seed,
        )
    pipe = Pipeline([("preprocess", prep), ("model", model)])
    pipe.fit(train_x, np.log1p(train_df["price"].to_numpy(dtype=float)))
    return np.maximum(np.expm1(pipe.predict(test_x)), 0.0)


def oof(spec: Spec | str, df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    if isinstance(spec, str):
        if spec == "mean":
            spec = Spec("mean", "mean", ())
        elif spec in {"zip_median", "naive"}:
            spec = Spec("zip_median", "zip_median", ())
        else:
            raise ValueError(f"Ismeretlen modell: {spec}")
    pieces = []
    for fold_number, (train_idx, val_idx) in enumerate(FOLDS, start=1):
        train_df = df.loc[train_idx]
        val_df = df.loc[val_idx]
        if not set(train_df["prop"]).isdisjoint(val_df["prop"]):
            raise AssertionError(f"A {fold_number}. foldban ingatlan-szivárgás történt.")
        prediction = _fit_predict(spec, train_df, val_df, seed)
        pieces.append(pd.DataFrame({
            "idx": val_idx, "y": val_df["price"].to_numpy(),
            "p": prediction, "fold": fold_number,
        }))
    return pd.concat(pieces, ignore_index=True)


def group_bootstrap(
    frame: pd.DataFrame,
    stat_fn: Callable[[pd.DataFrame], float],
    B: int = 2000,
    seed: int = 0,
) -> np.ndarray:
    """Egész ingatlancsoportokat mintáz visszatevéssel."""
    rng = np.random.default_rng(seed)
    codes, keys = pd.factorize(frame["prop"], sort=False)
    groups = [np.flatnonzero(codes == code) for code in range(len(keys))]
    values = np.empty(B, dtype=float)
    for draw in range(B):
        sampled = rng.integers(0, len(groups), size=len(groups))
        rows = np.concatenate([groups[index] for index in sampled])
        sample = frame.iloc[rows]
        values[draw] = stat_fn(sample)
    return values

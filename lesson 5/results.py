from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import house as H


GROUPS = {
    "GEO": H.GEO,
    "RATIO": H.RATIO,
    "COMPS": H.COMPS,
    "dist_seattle": ["dist_seattle"],
    "dist_bellevue": ["dist_bellevue"],
    "rot1": ["rot1"],
    "rot2": ["rot2"],
}
SEEDS = (0, 1, 2)


def _dev_test(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dev = frame.loc[frame["date"] < H.DEV_END].copy()
    test = frame.loc[frame["date"] >= H.DEV_END].copy()
    train = dev.loc[~dev["prop"].isin(test["prop"])].copy()
    if not set(train["prop"]).isdisjoint(test["prop"]):
        raise AssertionError("A teszt ingatlana megjelent a végső tanítóhalmazban.")
    return dev, test, train


def _combined(features: list[str] | tuple[str, ...]) -> list[str]:
    return list(dict.fromkeys([*H.BASE, *features]))


def _mean_oof(predictions: list[pd.DataFrame]) -> pd.DataFrame:
    merged = predictions[0][["idx", "y", "fold", "p"]].rename(columns={"p": "p0"})
    for i, prediction in enumerate(predictions[1:], start=1):
        merged = merged.merge(
            prediction[["idx", "p"]].rename(columns={"p": f"p{i}"}),
            on="idx", how="inner", validate="one_to_one",
        )
    merged["p"] = merged[[f"p{i}" for i in range(len(predictions))]].mean(axis=1)
    return merged[["idx", "y", "p", "fold"]].sort_values("idx").reset_index(drop=True)


def _paired_rows(first: pd.DataFrame, second: pd.DataFrame) -> pd.DataFrame:
    return first[["idx", "y", "fold", "p"]].rename(columns={"p": "base"}).merge(
        second[["idx", "p"]].rename(columns={"p": "new"}),
        on="idx", validate="one_to_one",
    )


def _fold_metrics(oof: pd.DataFrame) -> dict[str, list[float]]:
    return {name: H.fold_metric(oof, name) for name in ("rmse", "mae", "wape", "rmsle", "r2", "ratio")}


def _metric_row(name: str, oof: pd.DataFrame) -> dict[str, Any]:
    score = H.metrics(oof["y"], oof["p"])
    return {"modell": name, **score, "fold_rmse": H.fold_metric(oof, "rmse")}


def _jsonable(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_list()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _compute_diagnostics(
    frame: pd.DataFrame, oof: pd.DataFrame
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    data = frame.loc[oof["idx"].to_numpy()].copy().reset_index(drop=True)
    data["y"] = oof["y"].to_numpy()
    data["p"] = oof["p"].to_numpy()
    data["residual"] = data["y"] - data["p"]
    data["log_residual"] = np.log(np.maximum(data["y"], 1)) - np.log(np.maximum(data["p"], 1))
    data["decile"] = pd.qcut(data["p"], 10, duplicates="drop")
    deciles = data.groupby("decile", observed=True).agg(
        p_mean=("p", "mean"), y_mean=("y", "mean"),
        residual_sd=("residual", "std"), residual_mean=("residual", "mean"),
        n=("y", "size"),
    ).reset_index()
    data["price_band"] = pd.cut(data["y"], [0, 250_000, 500_000, 1_000_000, 2_000_000, np.inf], right=False)
    data["age_band"] = pd.cut(data["age"], [-1, 20, 40, 60, 80, np.inf], right=False)

    def summarize(grouped: Any) -> pd.DataFrame:
        return grouped.agg(
            n=("residual", "size"), mean_residual=("residual", "mean"),
            rmse=("residual", lambda values: float(np.sqrt(np.mean(np.square(values))))),
        ).reset_index()

    by_price = summarize(data.groupby("price_band", observed=True))
    by_age = summarize(data.groupby("age_band", observed=True))
    ten_zips = data["zipcode"].value_counts().head(10).index
    by_zip = summarize(data.loc[data["zipcode"].isin(ten_zips)].groupby("zipcode", observed=True))
    data["actual_decile"] = pd.qcut(data["y"], 10, duplicates="drop")
    cheapest = data.groupby("actual_decile", observed=True).apply(
        lambda group: H.metrics(group["y"], group["p"])["rmse"], include_groups=False
    )

    from sklearn.neighbors import NearestNeighbors

    coords = np.column_stack([
        data["lat"].to_numpy() * 110.574,
        data["long"].to_numpy() * 111.320 * np.cos(np.deg2rad(47.6)),
    ])
    n_neighbors = min(11, len(data))
    neighbor_model = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=1).fit(coords)
    neighbor_indices = neighbor_model.kneighbors(coords, return_distance=False)
    properties = data["prop"].to_numpy()
    log_residual = data["log_residual"].to_numpy()
    local_residual = np.empty(len(data), dtype=float)
    for row, indices in enumerate(neighbor_indices):
        eligible = indices[(indices != row) & (properties[indices] != properties[row])][:10]
        local_residual[row] = log_residual[eligible].mean() if len(eligible) else np.nan
    spatial_corr = float(np.corrcoef(log_residual, local_residual)[0, 1])

    fold_bounds = {}
    for fold_number, (train_idx, _) in enumerate(H.FOLDS, start=1):
        train = frame.loc[train_idx]
        train_ppsf = train["price"].to_numpy() / train["sqft_living"].clip(lower=1).to_numpy()
        fold_bounds[fold_number] = (
            float(train["price"].min()),
            float(np.quantile(train_ppsf, 0.001)),
            float(np.quantile(train_ppsf, 0.999)),
        )
    row_folds = oof["fold"].to_numpy()
    sq_ft = data["sqft_living"].to_numpy()
    predictions = data["p"].to_numpy()
    ppsf = predictions / np.maximum(sq_ft, 1)
    impossible = np.array([
        predictions[i] < fold_bounds[row_folds[i]][0]
        or ppsf[i] < fold_bounds[row_folds[i]][1]
        or ppsf[i] > fold_bounds[row_folds[i]][2]
        for i in range(len(data))
    ])
    funnel = deciles[["p_mean", "residual_sd"]].copy()
    first_sd = float(funnel["residual_sd"].iloc[0])
    last_sd = float(funnel["residual_sd"].iloc[-1])
    ratio = last_sd / max(first_sd, 1)
    comments = [
        f"A decilisek átlagos becslése ${deciles['p_mean'].iloc[0]:,.0f}–${deciles['p_mean'].iloc[-1]:,.0f}, az átlagos tényleges ár ${deciles['y_mean'].iloc[0]:,.0f}–${deciles['y_mean'].iloc[-1]:,.0f}. A kalibrációban a legfelső decilis átlagos reziduuma ${deciles['residual_mean'].iloc[-1]:,.0f}; ezért ezt a sávot külön ellenőrizném.",
        f"A reziduum-szórás az első decilis ${first_sd:,.0f}, az utolsó ${last_sd:,.0f} (arány {ratio:.2f}). {'A növekvő tölcsér miatt a drága szegmens hibáját külön kezelném.' if ratio > 1.2 else 'A szélső decilisek között nincs erős tölcsér, ezért nem illesztenék külön szórásmodellt.'}",
        f"A legközelebbi, más ingatlanhoz tartozó OOF szomszédok log-reziduumával a korreláció {spatial_corr:.3f}. {'A térbeli együttmozgás miatt helyi földrajzi jellemzőt foldon belül tesztelnék.' if spatial_corr > 0.15 else 'A gyenge korreláció nem indokol külön térbeli korrekciót.'}",
        f"A legnagyobb abszolút átlagos reziduum a vizsgált ár-, kor- és ZIP-szegmensekben ${max(by_price['mean_residual'].abs().max(), by_age['mean_residual'].abs().max(), by_zip['mean_residual'].abs().max()):,.0f}. Ezt a csoportot célzottan mintáznám, de csak a fejlesztési foldokon.",
        f"A legolcsóbb decilis RMSE-je ${float(cheapest.iloc[0]):,.0f}, a legdrágábbé ${float(cheapest.iloc[-1]):,.0f}. A foldonkénti árszint- és ár/négyzetláb-határokon kívüli jóslat {int(impossible.sum())}; a legkisebb modelljóslat ${float(np.min(predictions)):,.0f}, ezért a szélső árakat mintavétellel vizsgálnám.",
    ]
    summary = {
        "spatial_corr": spatial_corr,
        "funnel_first_sd": first_sd,
        "funnel_last_sd": last_sd,
        "funnel_ratio": ratio,
        "top_decile_mean_residual": float(deciles["residual_mean"].iloc[-1]),
        "cheapest_rmse": float(cheapest.iloc[0]),
        "most_expensive_rmse": float(cheapest.iloc[-1]),
        "impossible_count": int(impossible.sum()),
        "minimum_prediction": float(predictions.min()),
        "minimum_allowed_price": float(min(bounds[0] for bounds in fold_bounds.values())),
        "comments": comments,
    }
    return summary, {
        "deciles": deciles, "funnel": funnel, "by_price": by_price,
        "by_age": by_age, "by_zip": by_zip,
        "plot_data": data[["lat", "long", "age", "y", "p", "residual"]].copy(),
    }


def _bootstrap_rmse_difference(paired: pd.DataFrame, seed: int) -> np.ndarray:
    actual = paired["y"].to_numpy()
    base_error = actual - paired["base"].to_numpy()
    new_error = actual - paired["new"].to_numpy()
    stats = pd.DataFrame({
        "prop": paired["prop"].to_numpy(),
        "n": np.ones(len(paired), dtype=float),
        "base_sse": base_error**2,
        "new_sse": new_error**2,
    }).groupby("prop", sort=False).sum()
    counts = _cluster_counts(len(stats), 2000, seed)
    sampled_n = counts @ stats["n"].to_numpy()
    base_rmse = np.sqrt((counts @ stats["base_sse"].to_numpy()) / sampled_n)
    new_rmse = np.sqrt((counts @ stats["new_sse"].to_numpy()) / sampled_n)
    return base_rmse - new_rmse


def _cluster_counts(cluster_count: int, B: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multinomial(
        cluster_count, np.full(cluster_count, 1 / cluster_count), size=B
    )


def _bootstrap_final_stats(
    frame: pd.DataFrame,
    comparisons: dict[str, str],
    B: int = 2000,
    seed: int = 2026,
) -> dict[str, Any]:
    actual = frame["y"].to_numpy()
    winner = frame["winner"].to_numpy()
    summary_data = {
        "prop": frame["prop"].to_numpy(),
        "n": np.ones(len(frame), dtype=float),
        "y_abs": np.abs(actual),
        "winner_sse": (actual - winner) ** 2,
        "winner_abs_error": np.abs(actual - winner),
    }
    for prediction_column in comparisons.values():
        prediction = frame[prediction_column].to_numpy()
        summary_data[f"{prediction_column}_sse"] = (actual - prediction) ** 2
        summary_data[f"{prediction_column}_abs_error"] = np.abs(actual - prediction)
    summaries = pd.DataFrame(summary_data).groupby("prop", sort=False).sum()
    counts = _cluster_counts(len(summaries), B, seed)
    sampled_n = counts @ summaries["n"].to_numpy()
    denominator = counts @ summaries["y_abs"].to_numpy()
    winner_rmse = np.sqrt((counts @ summaries["winner_sse"].to_numpy()) / sampled_n)
    winner_wape = (
        counts @ summaries["winner_abs_error"].to_numpy()
    ) / denominator
    result: dict[str, Any] = {
        "rmse": np.quantile(winner_rmse, [0.025, 0.975]).tolist(),
        "wape": np.quantile(winner_wape, [0.025, 0.975]).tolist(),
        "comparisons": {},
    }
    for label, prediction_column in comparisons.items():
        comparison_rmse = np.sqrt(
            (counts @ summaries[f"{prediction_column}_sse"].to_numpy()) / sampled_n
        )
        comparison_wape = (
            counts @ summaries[f"{prediction_column}_abs_error"].to_numpy()
        ) / denominator
        result["comparisons"][label] = {
            "rmse_difference": np.quantile(
                comparison_rmse - winner_rmse, [0.025, 0.975]
            ).tolist(),
            "wape_difference": np.quantile(
                comparison_wape - winner_wape, [0.025, 0.975]
            ).tolist(),
        }
    return result


def _drift(
    train: pd.DataFrame, features: list[str], model: H.Spec, seed: int
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    train_dates = np.sort(train["date"].unique())
    cutoff = train_dates[int(0.50 * len(train_dates))]
    later = train.loc[train["date"] >= cutoff]
    earlier = train.loc[train["date"] < cutoff]
    earlier = earlier.loc[~earlier["prop"].isin(later["prop"])]
    spec = model.with_features(features)
    prediction = H.fit_predict(spec, earlier, later, seed)
    return H.metrics(later["price"], prediction), earlier, later


def _quantile_final(
    train: pd.DataFrame, test: pd.DataFrame, spec: H.Spec, seed: int
) -> dict[str, Any]:
    low = H.quantile_predict(spec, train, test, 0.05, seed)
    high = H.quantile_predict(spec, train, test, 0.95, seed)
    lower = np.minimum(low, high)
    upper = np.maximum(low, high)
    center = (lower + upper) / 2
    actual = test["price"].to_numpy()
    covered = (actual >= lower) & (actual <= upper)
    median_prediction = float(np.median(center))
    output = {
        "coverage": float(covered.mean()),
        "width": float(np.mean(upper - lower)),
        "median_predicted_price": median_prediction,
        "by_band": {},
    }
    for label, mask in (
        ("olcsóbb", center <= median_prediction),
        ("drágább", center > median_prediction),
    ):
        output["by_band"][label] = {
            "n": int(mask.sum()),
            "coverage": float(covered[mask].mean()),
            "width": float(np.mean(upper[mask] - lower[mask])),
        }
    return output


def compute() -> tuple[dict[str, Any], dict[str, Any]]:
    frame = H.load()
    dev, test, final_train = _dev_test(frame)
    R: dict[str, Any] = {}
    O: dict[str, Any] = {}
    cache: dict[tuple[tuple[str, ...], str, str, int], pd.DataFrame] = {}

    def cached(spec: H.Spec, seed: int = 0) -> pd.DataFrame:
        key = (spec.features, spec.kind, spec.target, seed)
        if key not in cache:
            cache[key] = H.oof(spec, frame, seed)
        return cache[key]

    baseline = {
        "átlag": "mean",
        "ZIP-medián": "zip_median",
        "lr_naive": H.lr_naive(),
        "lr_fair": H.lr_fair(),
        "tree": H.tree(),
        "forest": H.forest(),
        "xgb": H.xgb(),
    }
    ladder: dict[str, pd.DataFrame] = {}
    for name, spec in baseline.items():
        ladder[name] = H.oof(spec, frame, 0) if isinstance(spec, str) else cached(spec)
    O["ladder"] = ladder
    R["ladder"] = pd.DataFrame([
        _metric_row(name, oof_frame) for name, oof_frame in ladder.items()
    ]).sort_values("rmse").reset_index(drop=True)
    R["data"] = {
        "rows": int(len(frame)),
        "source_rows": int(frame.attrs["source_rows"]),
        "removed_bedroom_rows": int(frame.attrs["source_rows"] - len(frame)),
        "missing_values": int(frame.attrs["source_missing_count"]),
        "original_max_bedrooms": int(frame.attrs["original_max_bedrooms"]),
        "renovation_zero_rate": float(frame.attrs["renovation_zero_rate"]),
        "mean_price": float(frame.attrs["source_price_mean"]),
        "median_price": float(frame.attrs["source_price_median"]),
        "max_price": float(frame.attrs["source_price_max"]),
        "over_million_rate": float(frame.attrs["source_over_million_rate"]),
        "zip_count": int(frame.attrs["source_zip_count"]),
        "dev_rows": int(len(dev)),
        "test_rows": int(len(test)),
        "dev_end": H.DEV_END.strftime("%Y-%m-%d"),
        "date_min": frame["date"].min().strftime("%Y-%m-%d"),
        "date_max": frame["date"].max().strftime("%Y-%m-%d"),
        "repeat_rows": int(frame.duplicated(["lat", "long", "sqft_living", "sqft_lot", "yr_built", "zipcode"], keep=False).sum()),
        "repeat_properties": int(frame.groupby("prop").size().gt(1).sum()),
        "test_purged_rows": int(len(dev) - len(final_train)),
        "engine": "XGBoost" if H.USE_XGBOOST else "HistGradientBoosting",
        "folds": [
            {"train": int(len(train_idx)), "validation": int(len(val_idx))}
            for train_idx, val_idx in H.FOLDS
        ],
    }

    chronological = O["ladder"]["xgb"]
    chronological_rows = set(chronological["idx"])
    kf = KFold(n_splits=3, shuffle=True, random_state=0)
    random_results = {}
    for purge in (True, False):
        rows = []
        for train_pos, val_pos in kf.split(dev):
            train = dev.iloc[train_pos]
            val = dev.iloc[val_pos]
            if purge:
                train = train.loc[~train["prop"].isin(val["prop"])]
            prediction = H.fit_predict(H.xgb(), train, val, seed=0)
            rows.append(pd.DataFrame({"idx": val.index, "y": val["price"], "p": prediction}))
        random_oof = pd.concat(rows, ignore_index=True)
        same_rows = random_oof.loc[random_oof["idx"].isin(chronological_rows)]
        random_results["property_purged" if purge else "repeat_allowed"] = {
            **H.metrics(same_rows["y"], same_rows["p"]),
            "rows": len(same_rows),
        }
    common = chronological.loc[chronological["idx"].isin(chronological_rows)]
    R["split_comparison"] = {
        "chronological": {**H.metrics(common["y"], common["p"]), "rows": len(common)},
        **random_results,
    }

    # Zip/geography comparison reuses the 0-seed ladder output.
    geo_spec = H.xgb(_combined(H.GEO))
    zip_seed_results = []
    zip_oof: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for seed in range(5):
        base_oof = O["ladder"]["xgb"] if seed == 0 else cached(H.xgb(), seed)
        geo_oof = cached(geo_spec, seed)
        zip_oof[seed] = (base_oof, geo_oof)
        zip_seed_results.append(
            H.metrics(base_oof["y"], base_oof["p"])["rmse"]
            - H.metrics(geo_oof["y"], geo_oof["p"])["rmse"]
        )
    R["zipcode"] = {
        "lr_naive_rmse": H.metrics(O["ladder"]["lr_naive"]["y"], O["ladder"]["lr_naive"]["p"])["rmse"],
        "lr_fair_rmse": H.metrics(O["ladder"]["lr_fair"]["y"], O["ladder"]["lr_fair"]["p"])["rmse"],
        "seed_differences": zip_seed_results,
        "mean": float(np.mean(zip_seed_results)),
        "sd": float(np.std(zip_seed_results, ddof=1)),
    }

    best_name = R["ladder"].iloc[0]["modell"]
    best_oof = O["ladder"][best_name]
    diagnostics, diagnostic_tables = _compute_diagnostics(frame, best_oof)
    R["diagnostics"] = diagnostics
    O["diagnostics"] = diagnostic_tables

    best_spec = baseline[best_name]
    if isinstance(best_spec, str):
        best_spec = H.xgb()
    raw_oof = cached(best_spec.with_target("raw"))
    log_metrics = H.metrics(best_oof["y"], best_oof["p"])
    raw_metrics = H.metrics(raw_oof["y"], raw_oof["p"])
    R["log_target"] = {"log1p": log_metrics, "raw": raw_metrics}
    O["raw_target"] = raw_oof

    leaky_k3 = H.comps_leaky(dev, k=3)
    leaky_frame = dev.copy()
    leaky_frame["leaky_k3_ppsf"] = leaky_k3["comps_ppsf"]
    leaky_frame["leaky_k3_price"] = leaky_k3["comps_price"]
    zip_ppsf = (dev["price"] / dev["sqft_living"].clip(lower=1)).groupby(
        dev["zipcode"]
    ).transform("mean")
    leaky_frame["leaky_zip_ppsf"] = zip_ppsf.to_numpy()
    leaky_frame["leaky_zip_price"] = (
        leaky_frame["leaky_zip_ppsf"] * leaky_frame["sqft_living"]
    )
    correct_comps_oof = cached(H.xgb(_combined(H.COMPS)))
    leaky_oofs = {
        "xgb BASE": O["ladder"]["xgb"],
        "BASE + leaky k=3 comps": H.oof(
            H.xgb(_combined(["leaky_k3_ppsf", "leaky_k3_price"])), leaky_frame, 0
        ),
        "BASE + leaky ZIP átlag": H.oof(
            H.xgb(_combined(["leaky_zip_ppsf", "leaky_zip_price"])), leaky_frame, 0
        ),
        "BASE + helyes fold-comps": correct_comps_oof,
    }
    R["leaky"] = pd.DataFrame([
        _metric_row(name, predictions) for name, predictions in leaky_oofs.items()
    ])
    O["leaky"] = leaky_oofs

    base_seeds = [O["ladder"]["xgb"], cached(H.xgb(), 1), cached(H.xgb(), 2)]
    ablation_oofs: dict[str, list[pd.DataFrame]] = {"BASE": base_seeds}
    for group_name, features in GROUPS.items():
        spec = H.xgb(_combined(features))
        ablation_oofs[group_name] = [cached(spec, seed) for seed in SEEDS]
    # Reuse the already computed 0–2 geography outputs for the five-seed comparison.
    ablation_oofs["GEO"] = [zip_oof[seed][1] for seed in SEEDS[:3]]
    ablation_rows = []
    proven: list[str] = []
    paired_oofs: dict[str, pd.DataFrame] = {}
    for group_name, outputs in ablation_oofs.items():
        if group_name == "BASE":
            continue
        base_average = _mean_oof(base_seeds)
        new_average = _mean_oof(outputs)
        paired = _paired_rows(base_average, new_average)
        paired["prop"] = frame.loc[paired["idx"].to_numpy(), "prop"].to_numpy()
        paired_oofs[group_name] = paired
        fold_records = []
        fold_gains_rmse, fold_gains_wape = [], []
        for fold_number in sorted(paired["fold"].unique()):
            seed_rmse, seed_wape = [], []
            for seed_index in range(3):
                base_pred = base_seeds[seed_index].loc[
                    base_seeds[seed_index]["fold"] == fold_number
                ].sort_values("idx")
                new_pred = outputs[seed_index].loc[
                    outputs[seed_index]["fold"] == fold_number
                ].sort_values("idx")
                diff = _paired_rows(base_pred, new_pred)
                seed_rmse.append(
                    H.metrics(diff["y"], diff["base"])["rmse"]
                    - H.metrics(diff["y"], diff["new"])["rmse"]
                )
                seed_wape.append(
                    H.metrics(diff["y"], diff["base"])["wape"]
                    - H.metrics(diff["y"], diff["new"])["wape"]
                )
            mean_rmse = float(np.mean(seed_rmse))
            mean_wape = float(np.mean(seed_wape))
            fold_gains_rmse.append(mean_rmse)
            fold_gains_wape.append(mean_wape)
            fold_records.append({
                "fold": int(fold_number),
                "rmse_gain": mean_rmse,
                "rmse_sd": float(np.std(seed_rmse, ddof=1)),
                "wape_gain": mean_wape,
                "wape_sd": float(np.std(seed_wape, ddof=1)),
            })
        between_fold_sd = float(np.std(fold_gains_rmse, ddof=1))
        mean_gain = float(np.mean(fold_gains_rmse))
        proved = bool(all(value > 0 for value in fold_gains_rmse) and mean_gain > between_fold_sd)
        if proved:
            proven.append(group_name)
        boot = _bootstrap_rmse_difference(paired, 100 + len(ablation_rows))
        ablation_rows.append({
            "változat": group_name,
            "foldok": fold_records,
            "rmse_gain_mean": mean_gain,
            "between_fold_sd": between_fold_sd,
            "wape_gain_mean": float(np.mean(fold_gains_wape)),
            "proven": proved,
            "rmse_ci": np.quantile(boot, [0.025, 0.975]).tolist(),
            "pooled_rmse_gain": float(
                H.metrics(paired["y"], paired["base"])["rmse"]
                - H.metrics(paired["y"], paired["new"])["rmse"]
            ),
        })
    R["feedback"] = {"ablation": ablation_rows, "proven": proven}
    O["ablation"] = {name: _mean_oof(predictions) for name, predictions in ablation_oofs.items()}
    R["feedback"]["correct_comps"] = {
        **H.metrics(O["ablation"]["COMPS"]["y"], O["ablation"]["COMPS"]["p"]),
        "fold_rmse": H.fold_metric(O["ablation"]["COMPS"], "rmse"),
    }
    fair_new_spec = H.lr_fair(_combined([*H.GEO, *H.RATIO, *H.COMPS]))
    fair_new_oof = cached(fair_new_spec)
    fair_base_oof = O["ladder"]["lr_fair"]
    fair_gain = []
    for fold_number in sorted(fair_base_oof["fold"].unique()):
        left = fair_base_oof.loc[fair_base_oof["fold"] == fold_number].sort_values("idx")
        right = fair_new_oof.loc[fair_new_oof["fold"] == fold_number].sort_values("idx")
        paired = _paired_rows(left, right)
        fair_gain.append({
            "fold": int(fold_number),
            "rmse_gain": H.metrics(paired["y"], paired["base"])["rmse"]
            - H.metrics(paired["y"], paired["new"])["rmse"],
            "wape_gain": H.metrics(paired["y"], paired["base"])["wape"]
            - H.metrics(paired["y"], paired["new"])["wape"],
        })
    R["feedback"]["lr_fair_new"] = fair_gain
    O["fair_new"] = fair_new_oof

    base_oof_average = _mean_oof(base_seeds)
    base_fold_rmse = H.fold_metric(base_oof_average, "rmse")
    xgb_group_candidates = [{
        "név": "xgb BASE",
        "spec": H.xgb(),
        "fold_rmse": float(np.mean(base_fold_rmse)),
        "folds": base_fold_rmse,
    }]
    for item in ablation_rows:
        if item["proven"]:
            group = item["változat"]
            features = GROUPS[group]
            predictions = ablation_oofs[group]
            average = _mean_oof(predictions)
            fold_rmse = H.fold_metric(average, "rmse")
            xgb_group_candidates.append({
                "név": "xgb BASE" if group == "BASE" else f"xgb + {group}",
                "spec": H.xgb(_combined(features)),
                "fold_rmse": float(np.mean(fold_rmse)),
                "folds": fold_rmse,
            })
    best_xgb_candidate = min(xgb_group_candidates, key=lambda item: item["fold_rmse"])

    drift_specs = [
        ("xgb BASE", H.xgb(), H.BASE),
        ("xgb BASE + legjobb jellemzők", best_xgb_candidate["spec"], list(best_xgb_candidate["spec"].features)),
        ("lr_fair", H.lr_fair(), list(H.LINCOLS)),
        ("lr_fair + új jellemzők", fair_new_spec, list(fair_new_spec.features)),
    ]
    drift_rows = []
    for name, spec, features in drift_specs:
        score, drift_train, drift_val = _drift(dev, features, spec, 0)
        drift_rows.append({"modell": name, **score, "train_rows": len(drift_train), "validation_rows": len(drift_val)})
    R["drift"] = drift_rows

    reform_rows = []
    reform_oofs = {}
    for target in ("level", "ppsf", "logratio"):
        spec = H.xgb(target=target)
        predictions = cached(spec)
        reform_oofs[target] = predictions
        reform_rows.append({"cél": target, **H.metrics(predictions["y"], predictions["p"])})
    R["reformulation"] = {"cv": reform_rows, "drift": []}
    for target in ("level", "ppsf", "logratio"):
        spec = H.xgb(target=target)
        score, _, _ = _drift(dev, list(spec.features), spec, 0)
        R["reformulation"]["drift"].append({"cél": target, **score})
    O["reformulation"] = reform_oofs

    final_candidates = list(xgb_group_candidates)
    fair_base = O["ladder"]["lr_fair"]
    final_candidates.extend([
        {"név": "lr_fair", "spec": H.lr_fair(), "fold_rmse": float(np.mean(H.fold_metric(fair_base, "rmse"))),
         "folds": H.fold_metric(fair_base, "rmse")},
        {"név": "lr_fair + új jellemzők", "spec": fair_new_spec,
         "fold_rmse": float(np.mean(H.fold_metric(fair_new_oof, "rmse"))),
         "folds": H.fold_metric(fair_new_oof, "rmse")},
    ])
    chosen = min(final_candidates, key=lambda item: item["fold_rmse"])
    R["final_rule"] = {
        "rule": "A jelöltek közül a legkisebb fejlesztési fold-átlagos RMSE választ; tesztadat nem vesz részt a döntésben.",
        "metric_reason": "Az RMSE az elsődleges és modellválasztási mutató dollárban, mert a nagy hibákat erősen bünteti; a WAPE, MAE, RMSLE, R² és az összegarány másodlagos ellenőrző mutatók.",
        "chosen": chosen["név"],
        "chosen_fold_rmse": chosen["fold_rmse"],
        "candidates": [
            {"name": item["név"], "mean_fold_rmse": item["fold_rmse"], "fold_rmse": item["folds"]}
            for item in sorted(final_candidates, key=lambda item: item["fold_rmse"])
        ],
    }

    # A teszt első és utolsó használata: a négy pontbecslés, intervallumok és kvantilis sáv.
    if set(test["prop"]) & set(final_train["prop"]):
        raise AssertionError("A tesztingatlan átszivárgott a végső tanításba.")
    final_specs = {
        "ZIP-medián": H.Spec("ZIP-medián", "zip_median", ()),
        "lr_fair": H.lr_fair(),
        "xgb BASE": H.xgb(),
    }
    final_predictions = {
        name: H.fit_predict(spec, final_train, test, 0)
        for name, spec in final_specs.items()
    }
    if chosen["név"] in final_predictions:
        final_predictions["Kiválasztott"] = final_predictions[chosen["név"]]
    else:
        final_predictions["Kiválasztott"] = H.fit_predict(
            chosen["spec"], final_train, test, 0
        )
    R["final"] = {
        "metrics": [
            {"modell": name, **H.metrics(test["price"], prediction)}
            for name, prediction in final_predictions.items()
        ],
        "bootstrap": {},
    }
    bootstrap_frame = pd.DataFrame({
        "prop": test["prop"].to_numpy(),
        "y": test["price"].to_numpy(),
        "winner": final_predictions["Kiválasztott"],
        "naive": final_predictions["ZIP-medián"],
        "lr_fair": final_predictions["lr_fair"],
        "xgb_base": final_predictions["xgb BASE"],
    })
    R["final"]["bootstrap"] = _bootstrap_final_stats(
        bootstrap_frame,
        {"ZIP-medián": "naive", "lr_fair": "lr_fair", "xgb BASE": "xgb_base"},
        B=2000,
        seed=2026,
    )
    R["uncertainty"] = _quantile_final(final_train, test, H.xgb(), 0)
    O["final_predictions"] = pd.DataFrame({
        "y": test["price"].to_numpy(),
        "p": final_predictions["Kiválasztott"],
        "prop": test["prop"].to_numpy(),
    })
    O["frame"] = frame
    return R, O


if __name__ == "__main__":
    results, _ = compute()
    (Path(__file__).parent / "results.json").write_text(
        json.dumps(_jsonable(results), ensure_ascii=False, indent=2), encoding="utf-8"
    )

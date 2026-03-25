
import io
import math
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import curve_fit


st.set_page_config(page_title="Paid Search Budget Forecaster", layout="wide")


REQUIRED_FIELDS = [
    ("date", "Date"),
    ("campaign_key", "Campaign ID or Campaign Name"),
    ("campaign_name", "Campaign Name"),
    ("cost", "Spend / Cost"),
    ("conversions", "Conversions"),
]

OPTIONAL_FIELDS = [
    ("clicks", "Clicks"),
    ("impressions", "Impressions"),
    ("impression_share", "Impression Share"),
    ("conversion_value", "Conversion Value / Revenue"),
    ("category", "Category (if already mapped)"),
]


@dataclass
class CategoryModel:
    category: str
    a: float
    b: float
    fit_status: str
    data_points: int
    recent_avg_spend: float

    def predict(self, spend, seasonality_factor=1.0):
        spend = np.asarray(spend, dtype=float)
        spend = np.maximum(spend, 0)
        preds = hill_curve(spend, self.a, self.b) * float(seasonality_factor)
        return np.maximum(preds, 0.0)


def hill_curve(spend, a, b):
    spend = np.asarray(spend, dtype=float)
    return a * spend / (b + spend + 1e-9)


def read_uploaded_csv(uploaded_file):
    raw_bytes = uploaded_file.getvalue()
    return pd.read_csv(io.BytesIO(raw_bytes))


def normalize_impression_share(series):
    if series is None:
        return pd.Series(dtype=float)
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return s
    if s.max(skipna=True) > 1.5:
        s = s / 100.0
    return s.clip(lower=0.0, upper=1.0)


def to_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def monday_of_week(date_series):
    return date_series - pd.to_timedelta(date_series.dt.weekday, unit="D")


def standardize_dataset(df, mapping):
    clean = pd.DataFrame()

    clean["date"] = pd.to_datetime(df[mapping["date"]], errors="coerce")
    clean["campaign_key"] = df[mapping["campaign_key"]].astype(str).str.strip()
    clean["campaign_name"] = df[mapping["campaign_name"]].astype(str).str.strip()
    clean["cost"] = to_numeric(df[mapping["cost"]]).fillna(0.0)
    clean["conversions"] = to_numeric(df[mapping["conversions"]]).fillna(0.0)

    for field in ["clicks", "impressions", "conversion_value"]:
        source = mapping.get(field)
        if source and source != "(not used)":
            clean[field] = to_numeric(df[source]).fillna(0.0)
        else:
            clean[field] = 0.0

    is_col = mapping.get("impression_share")
    if is_col and is_col != "(not used)":
        clean["impression_share"] = normalize_impression_share(df[is_col])
    else:
        clean["impression_share"] = np.nan

    cat_col = mapping.get("category")
    if cat_col and cat_col != "(not used)":
        clean["category_from_file"] = df[cat_col].astype(str).str.strip().replace({"": "Unassigned"})
    else:
        clean["category_from_file"] = "Unassigned"

    clean = clean.replace([np.inf, -np.inf], np.nan)
    clean = clean.dropna(subset=["date", "campaign_key", "campaign_name"])
    clean = clean[(clean["cost"] >= 0) & (clean["conversions"] >= 0)]
    clean["week_start"] = monday_of_week(clean["date"])
    clean["ctr"] = np.where(clean["impressions"] > 0, clean["clicks"] / clean["impressions"], np.nan)
    clean["cvr"] = np.where(clean["clicks"] > 0, clean["conversions"] / clean["clicks"], np.nan)
    clean["cpa"] = np.where(clean["conversions"] > 0, clean["cost"] / clean["conversions"], np.nan)
    return clean


def initialize_campaign_mapping(clean_df):
    unique_campaigns = (
        clean_df[["campaign_key", "campaign_name", "category_from_file"]]
        .drop_duplicates()
        .sort_values(["campaign_name", "campaign_key"])
        .reset_index(drop=True)
    )
    mapping_df = unique_campaigns.rename(columns={"category_from_file": "category"}).copy()
    mapping_df["active"] = True
    mapping_df["category"] = mapping_df["category"].replace({"nan": "Unassigned"}).fillna("Unassigned")
    return mapping_df


def ensure_campaign_mapping(clean_df):
    campaign_signature = tuple(
        clean_df[["campaign_key", "campaign_name"]]
        .drop_duplicates()
        .sort_values(["campaign_name", "campaign_key"])
        .itertuples(index=False, name=None)
    )

    if st.session_state.get("campaign_signature") != campaign_signature:
        st.session_state["campaign_signature"] = campaign_signature
        st.session_state["campaign_mapping"] = initialize_campaign_mapping(clean_df)

    edited = st.data_editor(
        st.session_state["campaign_mapping"],
        hide_index=True,
        width="stretch",
        key="campaign_mapping_editor",
        disabled=["campaign_key", "campaign_name"],
    )
    st.session_state["campaign_mapping"] = edited
    return edited


def apply_campaign_mapping(clean_df, campaign_mapping):
    mapped = clean_df.merge(
        campaign_mapping[["campaign_key", "category", "active"]],
        on="campaign_key",
        how="left",
    )
    mapped["category"] = mapped["category"].fillna("Unassigned").replace({"": "Unassigned"})
    mapped["active"] = mapped["active"].fillna(True)
    mapped = mapped[mapped["active"] == True].copy()
    return mapped


def aggregate_weekly_by_category(mapped_df):
    agg_dict = {
        "cost": "sum",
        "conversions": "sum",
        "clicks": "sum",
        "impressions": "sum",
        "conversion_value": "sum",
    }
    if "impression_share" in mapped_df.columns:
        agg_dict["impression_share"] = "mean"

    weekly = (
        mapped_df.groupby(["week_start", "category"], as_index=False)
        .agg(agg_dict)
        .rename(columns={"cost": "spend"})
    )
    weekly["cpa"] = np.where(weekly["conversions"] > 0, weekly["spend"] / weekly["conversions"], np.nan)
    weekly["roas"] = np.where(weekly["spend"] > 0, weekly["conversion_value"] / weekly["spend"], np.nan)
    return weekly.sort_values(["week_start", "category"]).reset_index(drop=True)


def default_max_budget(recent_avg_spend, recent_impression_share):
    if pd.isna(recent_avg_spend):
        recent_avg_spend = 0.0
    if pd.isna(recent_impression_share):
        multiplier = 2.0
    else:
        multiplier = np.clip(1.1 + (1.0 - recent_impression_share) * 2.0, 1.15, 3.0)
    suggested = max(50.0, recent_avg_spend * multiplier)
    return float(suggested)


def build_category_constraints(weekly_df):
    if weekly_df.empty:
        return pd.DataFrame(
            columns=[
                "category",
                "active",
                "recent_avg_weekly_spend",
                "recent_avg_impression_share",
                "min_weekly_budget",
                "max_weekly_budget",
            ]
        )

    latest_week = weekly_df["week_start"].max()
    recent_cutoff = latest_week - pd.Timedelta(weeks=7)
    recent = weekly_df[weekly_df["week_start"] >= recent_cutoff].copy()

    summary = (
        recent.groupby("category", as_index=False)
        .agg(
            recent_avg_weekly_spend=("spend", "mean"),
            recent_avg_impression_share=("impression_share", "mean"),
        )
        .sort_values("category")
        .reset_index(drop=True)
    )
    summary["active"] = True
    summary["min_weekly_budget"] = 0.0
    summary["max_weekly_budget"] = summary.apply(
        lambda row: default_max_budget(row["recent_avg_weekly_spend"], row["recent_avg_impression_share"]),
        axis=1,
    )
    return summary


def ensure_category_constraints(weekly_df):
    signature = tuple(sorted(weekly_df["category"].dropna().unique().tolist()))
    if st.session_state.get("constraint_signature") != signature:
        st.session_state["constraint_signature"] = signature
        st.session_state["category_constraints"] = build_category_constraints(weekly_df)

    constraints = st.data_editor(
        st.session_state["category_constraints"],
        hide_index=True,
        width="stretch",
        key="category_constraints_editor",
        disabled=["recent_avg_weekly_spend", "recent_avg_impression_share"],
    )
    constraints["min_weekly_budget"] = pd.to_numeric(constraints["min_weekly_budget"], errors="coerce").fillna(0.0)
    constraints["max_weekly_budget"] = pd.to_numeric(constraints["max_weekly_budget"], errors="coerce").fillna(0.0)
    constraints["max_weekly_budget"] = np.maximum(constraints["max_weekly_budget"], constraints["min_weekly_budget"])
    st.session_state["category_constraints"] = constraints
    return constraints


def fit_single_category_model(cat_df, category_name):
    cat_df = cat_df.copy().sort_values("week_start")
    cat_df = cat_df[(cat_df["spend"] >= 0) & (cat_df["conversions"] >= 0)]
    cat_df = cat_df[["spend", "conversions"]].dropna()

    recent_avg_spend = float(cat_df["spend"].tail(8).mean()) if not cat_df.empty else 0.0

    if len(cat_df) < 6 or cat_df["spend"].nunique() < 4:
        return heuristic_model(cat_df, category_name, recent_avg_spend, fit_status="heuristic: limited history")

    x = cat_df["spend"].astype(float).values
    y = cat_df["conversions"].astype(float).values

    x_positive = x[x > 0]
    median_x = float(np.median(x_positive)) if len(x_positive) else 1.0
    max_y = max(float(np.nanmax(y)), 1.0)
    mean_efficiency = float(np.nanmean(np.where(x > 0, y / np.maximum(x, 1e-9), np.nan)))
    if np.isnan(mean_efficiency) or mean_efficiency <= 0:
        mean_efficiency = 0.01

    initial_b = max(median_x, 1.0)
    initial_a = max(max_y * 1.5, mean_efficiency * initial_b * 2.0, 1.0)

    upper_a = max(max_y * 10.0, initial_a * 5.0, 10.0)
    upper_b = max(float(np.nanmax(x)) * 10.0, initial_b * 10.0, 100.0)

    try:
        params, _ = curve_fit(
            hill_curve,
            x,
            y,
            p0=[initial_a, initial_b],
            bounds=([0.01, 0.01], [upper_a, upper_b]),
            maxfev=30000,
        )
        a, b = float(params[0]), float(params[1])

        preds = hill_curve(x, a, b)
        if np.any(np.isnan(preds)) or np.nanmean(np.abs(preds - y)) > max(5.0, np.nanmean(y) * 3):
            return heuristic_model(cat_df, category_name, recent_avg_spend, fit_status="heuristic: unstable fit")

        return CategoryModel(
            category=category_name,
            a=a,
            b=b,
            fit_status="curve_fit",
            data_points=len(cat_df),
            recent_avg_spend=recent_avg_spend,
        )
    except Exception:
        return heuristic_model(cat_df, category_name, recent_avg_spend, fit_status="heuristic: fit failed")


def heuristic_model(cat_df, category_name, recent_avg_spend, fit_status):
    if cat_df.empty:
        a = 1.0
        b = 100.0
        points = 0
    else:
        positive = cat_df[cat_df["spend"] > 0].copy()
        points = len(cat_df)
        if positive.empty:
            a = max(float(cat_df["conversions"].max()), 1.0)
            b = 100.0
        else:
            median_spend = max(float(positive["spend"].median()), 1.0)
            median_efficiency = float((positive["conversions"] / positive["spend"]).replace([np.inf, -np.inf], np.nan).median())
            if np.isnan(median_efficiency) or median_efficiency <= 0:
                median_efficiency = 0.01
            a = max(float(positive["conversions"].max()) * 1.5, median_efficiency * median_spend * 2.0, 1.0)
            b = median_spend
    return CategoryModel(
        category=category_name,
        a=float(a),
        b=float(b),
        fit_status=fit_status,
        data_points=points,
        recent_avg_spend=float(recent_avg_spend),
    )


def fit_all_models(weekly_df, category_constraints):
    active_categories = category_constraints[category_constraints["active"] == True]["category"].tolist()
    models = {}
    diagnostics = []
    for category in active_categories:
        cat_df = weekly_df[weekly_df["category"] == category].copy()
        model = fit_single_category_model(cat_df, category)
        models[category] = model
        diagnostics.append(
            {
                "category": category,
                "fit_status": model.fit_status,
                "data_points": model.data_points,
                "curve_a": round(model.a, 4),
                "curve_b": round(model.b, 4),
                "recent_avg_weekly_spend": round(model.recent_avg_spend, 2),
            }
        )
    return models, pd.DataFrame(diagnostics)


def compute_month_seasonality(weekly_df):
    base = {m: 1.0 for m in range(1, 13)}
    if weekly_df.empty:
        return base

    total = (
        weekly_df.groupby("week_start", as_index=False)
        .agg(spend=("spend", "sum"), conversions=("conversions", "sum"))
        .sort_values("week_start")
    )
    total = total[total["spend"] > 0].copy()
    if len(total) < 8:
        return base

    total["month"] = pd.to_datetime(total["week_start"]).dt.month
    total["efficiency"] = total["conversions"] / total["spend"].replace(0, np.nan)
    overall_eff = total["efficiency"].median()

    if pd.isna(overall_eff) or overall_eff <= 0:
        return base

    seasonality = {}
    for month in range(1, 13):
        sample = total.loc[total["month"] == month, "efficiency"]
        if len(sample) < 2:
            seasonality[month] = 1.0
        else:
            factor = float(np.clip(sample.median() / overall_eff, 0.7, 1.3))
            seasonality[month] = factor
    return seasonality


def incremental_allocate(total_budget, categories, models, constraint_rows, seasonality_factor):
    mins = np.array([float(constraint_rows[c]["min_weekly_budget"]) for c in categories], dtype=float)
    maxs = np.array([float(constraint_rows[c]["max_weekly_budget"]) for c in categories], dtype=float)

    if total_budget < mins.sum() - 1e-6:
        raise ValueError(f"Budget is below the required category minimums. Minimum needed: {mins.sum():,.2f}")

    if total_budget > maxs.sum() + 1e-6:
        raise ValueError(f"Budget is above the allowed category maximums. Maximum allowed: {maxs.sum():,.2f}")

    spend = mins.copy()
    remaining = float(total_budget - mins.sum())

    if remaining <= 1e-6:
        return spend

    step = max(total_budget / 250.0, 10.0)

    safety = 0
    while remaining > 1e-6 and safety < 100000:
        safety += 1
        best_idx = None
        best_gain = -1.0

        for idx, category in enumerate(categories):
            if spend[idx] >= maxs[idx] - 1e-6:
                continue

            delta = min(step, remaining, maxs[idx] - spend[idx])
            current_pred = float(models[category].predict(spend[idx], seasonality_factor))
            new_pred = float(models[category].predict(spend[idx] + delta, seasonality_factor))
            marginal_gain = (new_pred - current_pred) / max(delta, 1e-9)

            if marginal_gain > best_gain:
                best_gain = marginal_gain
                best_idx = idx

        if best_idx is None:
            break

        delta = min(step, remaining, maxs[best_idx] - spend[best_idx])
        spend[best_idx] += delta
        remaining -= delta

    if remaining > 1e-3:
        raise ValueError("Could not allocate the full budget within the category constraints.")

    return spend


def predicted_total_conversions(total_budget, categories, models, constraint_rows, seasonality_factor):
    spend = incremental_allocate(total_budget, categories, models, constraint_rows, seasonality_factor)
    total_conversions = 0.0
    for idx, category in enumerate(categories):
        total_conversions += float(models[category].predict(spend[idx], seasonality_factor))
    return total_conversions, spend


def solve_budget_for_goal(goal_conversions, categories, models, constraint_rows, seasonality_factor):
    mins = np.array([float(constraint_rows[c]["min_weekly_budget"]) for c in categories], dtype=float)
    maxs = np.array([float(constraint_rows[c]["max_weekly_budget"]) for c in categories], dtype=float)

    low = float(mins.sum())
    high = float(maxs.sum())

    max_possible, _ = predicted_total_conversions(high, categories, models, constraint_rows, seasonality_factor)
    if goal_conversions > max_possible:
        raise ValueError(
            f"Your weekly goal is higher than this version of the model thinks is possible inside the current max budgets. "
            f"Estimated max weekly conversions: {max_possible:,.2f}"
        )

    best_budget = high
    best_spend = None

    for _ in range(30):
        mid = (low + high) / 2.0
        convs, spend = predicted_total_conversions(mid, categories, models, constraint_rows, seasonality_factor)
        if convs >= goal_conversions:
            best_budget = mid
            best_spend = spend
            high = mid
        else:
            low = mid

    if best_spend is None:
        _, best_spend = predicted_total_conversions(best_budget, categories, models, constraint_rows, seasonality_factor)

    return best_budget, best_spend


def build_forecast(
    weekly_df,
    category_constraints,
    models,
    horizon_weeks,
    input_mode,
    weekly_budget=None,
    weekly_goal=None,
):
    categories = category_constraints[category_constraints["active"] == True]["category"].tolist()
    constraint_rows = category_constraints.set_index("category").to_dict(orient="index")
    month_factors = compute_month_seasonality(weekly_df)

    start_week = weekly_df["week_start"].max() + pd.Timedelta(weeks=1)

    rows = []
    summary_rows = []

    for week_index in range(horizon_weeks):
        forecast_week = start_week + pd.Timedelta(weeks=week_index)
        factor = month_factors.get(int(forecast_week.month), 1.0)

        if input_mode == "I know my weekly budget":
            if weekly_budget is None:
                raise ValueError("Weekly budget is required.")
            spend_vector = incremental_allocate(weekly_budget, categories, models, constraint_rows, factor)
            chosen_budget = float(weekly_budget)
        else:
            if weekly_goal is None:
                raise ValueError("Weekly goal is required.")
            chosen_budget, spend_vector = solve_budget_for_goal(weekly_goal, categories, models, constraint_rows, factor)

        week_total_conversions = 0.0
        for idx, category in enumerate(categories):
            predicted_conversions = float(models[category].predict(spend_vector[idx], factor))
            week_total_conversions += predicted_conversions
            rows.append(
                {
                    "week_start": forecast_week.date().isoformat(),
                    "category": category,
                    "recommended_budget": round(float(spend_vector[idx]), 2),
                    "predicted_conversions": round(predicted_conversions, 2),
                    "seasonality_factor": round(float(factor), 3),
                }
            )

        summary_rows.append(
            {
                "week_start": forecast_week.date().isoformat(),
                "recommended_total_budget": round(chosen_budget, 2),
                "predicted_total_conversions": round(week_total_conversions, 2),
                "seasonality_factor": round(float(factor), 3),
            }
        )

    detail_df = pd.DataFrame(rows)
    summary_df = pd.DataFrame(summary_rows)
    return detail_df, summary_df


def csv_download_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


st.title("Paid Search Budget Forecaster")
st.caption(
    "MVP version: upload a CSV, map your columns, assign campaign categories, and get weekly budget recommendations."
)

with st.expander("What this starter app assumes", expanded=False):
    st.markdown(
        """
        - You have historical paid search data at daily or weekly level.
        - You care about optimizing **conversions** from **spend**.
        - Categories are user-defined (brand, nonbrand, competitor, etc.).
        - This version is decision support, not autopilot. Always sanity-check the output.
        """
    )

uploaded_file = st.file_uploader("Upload your paid search CSV", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV to begin. Use the sample file included with this project if you want to test the app first.")
    st.stop()

raw_df = read_uploaded_csv(uploaded_file)

st.subheader("1) Map your columns")
st.write("Choose the columns from your file that match each required field.")

columns = ["(not used)"] + raw_df.columns.tolist()

mapping = {}
for field_key, label in REQUIRED_FIELDS:
    default_index = 0
    for idx, col_name in enumerate(columns):
        if col_name.lower() in {field_key, label.lower(), label.lower().replace(" / ", "_")}:
            default_index = idx
            break
    mapping[field_key] = st.selectbox(label, options=columns[1:], key=f"map_{field_key}")

for field_key, label in OPTIONAL_FIELDS:
    mapping[field_key] = st.selectbox(label, options=columns, index=0, key=f"map_{field_key}")

missing_required = [label for field_key, label in REQUIRED_FIELDS if mapping.get(field_key) in [None, "(not used)"]]
if missing_required:
    st.error("Map all required fields before continuing.")
    st.stop()

clean_df = standardize_dataset(raw_df, mapping)

if clean_df.empty:
    st.error("After cleaning, there are no usable rows left. Check your date and numeric columns.")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Usable rows", f"{len(clean_df):,}")
col2.metric("Campaigns", f"{clean_df['campaign_key'].nunique():,}")
col3.metric("Date range", f"{clean_df['date'].min().date()} → {clean_df['date'].max().date()}")

with st.expander("Preview cleaned data", expanded=False):
    st.dataframe(clean_df.head(50), use_container_width=True)

st.subheader("2) Assign categories to campaigns")
st.write("Edit the category column below. Keep the names consistent, like Brand, Nonbrand, Competitor, Shopping.")

campaign_mapping = ensure_campaign_mapping(clean_df)
if campaign_mapping["category"].eq("Unassigned").any():
    st.warning("Some campaigns are still marked as Unassigned. That is okay for testing, but not ideal for a real forecast.")

mapped_df = apply_campaign_mapping(clean_df, campaign_mapping)
weekly_df = aggregate_weekly_by_category(mapped_df)

if weekly_df.empty:
    st.error("No active data left after campaign mapping.")
    st.stop()

st.subheader("3) Set category budget rules")
st.write("These defaults are just starting points. Edit them. The max budget suggestion loosely uses recent spend and impression share headroom.")

category_constraints = ensure_category_constraints(weekly_df)

active_categories = category_constraints[category_constraints["active"] == True]["category"].tolist()
if not active_categories:
    st.error("Turn on at least one category in the category rules table.")
    st.stop()

models, diagnostics_df = fit_all_models(weekly_df, category_constraints)

st.subheader("4) Choose your forecasting mode")
input_mode = st.radio(
    "How do you want to plan?",
    options=["I know my weekly budget", "I have a weekly conversion goal"],
    horizontal=True,
)

horizon_weeks = int(st.number_input("Forecast horizon (weeks)", min_value=1, max_value=26, value=8, step=1))

weekly_budget = None
weekly_goal = None

if input_mode == "I know my weekly budget":
    recent_total_spend = float(
        weekly_df.groupby("week_start")["spend"].sum().tail(8).mean()
    ) if not weekly_df.empty else 1000.0
    weekly_budget = float(
        st.number_input(
            "Weekly total budget",
            min_value=0.0,
            value=round(max(recent_total_spend, 100.0), 2),
            step=100.0,
        )
    )
else:
    recent_total_conversions = float(
        weekly_df.groupby("week_start")["conversions"].sum().tail(8).mean()
    ) if not weekly_df.empty else 50.0
    weekly_goal = float(
        st.number_input(
            "Weekly conversion goal",
            min_value=0.0,
            value=round(max(recent_total_conversions, 1.0), 2),
            step=5.0,
        )
    )

try:
    detail_forecast, summary_forecast = build_forecast(
        weekly_df=weekly_df,
        category_constraints=category_constraints,
        models=models,
        horizon_weeks=horizon_weeks,
        input_mode=input_mode,
        weekly_budget=weekly_budget,
        weekly_goal=weekly_goal,
    )
except ValueError as exc:
    st.error(str(exc))
    st.stop()

st.subheader("Forecast summary")
st.dataframe(summary_forecast, use_container_width=True)

summary_chart = summary_forecast.set_index("week_start")[["recommended_total_budget", "predicted_total_conversions"]]
st.line_chart(summary_chart)

st.subheader("Recommended budgets by week and category")
st.dataframe(detail_forecast, use_container_width=True)

pivot_budget = detail_forecast.pivot(index="week_start", columns="category", values="recommended_budget")
st.bar_chart(pivot_budget)

with st.expander("Model diagnostics", expanded=False):
    st.dataframe(diagnostics_df, use_container_width=True)
    st.write("Weekly data used for modeling")
    st.dataframe(weekly_df, use_container_width=True)

st.download_button(
    "Download weekly forecast CSV",
    data=csv_download_bytes(detail_forecast),
    file_name="weekly_budget_forecast.csv",
    mime="text/csv",
)

st.download_button(
    "Download forecast summary CSV",
    data=csv_download_bytes(summary_forecast),
    file_name="forecast_summary.csv",
    mime="text/csv",
)

st.download_button(
    "Download cleaned weekly history CSV",
    data=csv_download_bytes(weekly_df),
    file_name="cleaned_weekly_history.csv",
    mime="text/csv",
)

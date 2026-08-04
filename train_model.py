"""
Cleans and joins the raw shortage/drug data, engineers features, trains a
logistic regression risk model using a time-based validation split, then
applies it to current data to score every drug's shortage risk.

INPUT (must already exist in this folder):
    shortage_discontinuation_reports.csv   -- from pull_shortage_data.py
    drug_product.csv                       -- from pull_drug_product_data.py
    active_ingredient.csv                  -- from pull_drug_product_data.py

OUTPUT:
    combined_drug_shortage_dataset.csv     -- cleaned, joined dataset
    drug_risk_scores.csv                   -- current risk score per drug (used by app.py)
    shortage_risk_model.joblib             -- the trained model
"""

import ast

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

CUTOFF = pd.Timestamp("2024-01-01", tz="UTC")


def safe_extract(val, key):
    """Pulls a value out of the API's nested dict-as-string columns."""
    if pd.isna(val):
        return None
    try:
        return ast.literal_eval(val).get(key)
    except (ValueError, SyntaxError):
        return None


def build_combined_dataset():
    """Cleans the shortage data and joins it with manufacturer/ingredient data."""
    shortages = pd.read_csv("shortage_discontinuation_reports.csv", low_memory=False)
    drug_product = pd.read_csv("drug_product.csv", low_memory=False)
    active_ingredient = pd.read_csv("active_ingredient.csv", low_memory=False)

    shortages["report_type"] = shortages["type"].apply(lambda v: safe_extract(v, "label"))
    shortages["shortage_reason_label"] = shortages["shortage_reason"].apply(
        lambda v: safe_extract(v, "en_reason")
    )

    # Manufacturer count per active ingredient, across the FULL drug database
    ai_dp = active_ingredient.merge(
        drug_product[["drug_code", "company_name"]], on="drug_code", how="left"
    )
    mfr_count = (
        ai_dp.groupby("ingredient_name")["company_name"]
        .nunique()
        .reset_index()
        .rename(columns={"company_name": "manufacturer_count"})
    )

    primary_ingredient = (
        active_ingredient.sort_values("drug_code")
        .groupby("drug_code")
        .first()
        .reset_index()[["drug_code", "ingredient_name"]]
        .rename(columns={"ingredient_name": "primary_ingredient"})
    )

    dp_enriched = drug_product.merge(primary_ingredient, on="drug_code", how="left")
    dp_enriched = dp_enriched.merge(
        mfr_count, left_on="primary_ingredient", right_on="ingredient_name", how="left"
    )
    dp_enriched = dp_enriched[
        ["drug_identification_number", "class_name", "primary_ingredient", "manufacturer_count"]
    ].rename(columns={"drug_identification_number": "din"})

    # Normalize DIN to a clean integer string on both sides before joining --
    # one side loaded as float, the other as int, which silently broke this
    # join the first time around.
    shortages["din_clean"] = pd.to_numeric(shortages["din"], errors="coerce").astype("Int64").astype(str)
    dp_enriched["din_clean"] = pd.to_numeric(dp_enriched["din"], errors="coerce").astype("Int64").astype(str)
    dp_enriched = dp_enriched.drop_duplicates(subset="din_clean", keep="first")

    final = shortages.merge(
        dp_enriched[["din_clean", "class_name", "primary_ingredient", "manufacturer_count"]],
        on="din_clean",
        how="left",
    )

    keep_cols = [
        "id", "din", "en_drug_brand_name", "company_name", "atc_number", "atc_description",
        "report_type", "status", "shortage_reason_label", "anticipated_start_date",
        "actual_start_date", "actual_end_date", "resolved", "avoided", "tier_3",
        "primary_ingredient", "manufacturer_count", "class_name",
    ]
    return final[keep_cols].rename(columns={"class_name": "drug_class"})


def build_features(data, reference_date, top_classes):
    """Turns raw shortage rows into one row per drug, with predictive features."""
    d = data.copy()
    d["class_bucket"] = d["atc_description"].apply(lambda x: x if x in top_classes else "OTHER")
    agg = d.groupby("din").agg(
        prior_shortage_count=("report_type", lambda x: (x == "shortage").sum()),
        prior_discontinuation_count=("report_type", lambda x: (x == "discontinuance").sum()),
        prior_tier3_count=("tier_3", "sum"),
        manufacturer_count=("manufacturer_count", "first"),
        most_recent_date=("start_date", "max"),
        resolved_rate=("resolved", "mean"),
        class_bucket=("class_bucket", "first"),
        brand_name=("en_drug_brand_name", "first"),
        atc_description=("atc_description", "first"),
    ).reset_index()

    agg["days_since_last_shortage"] = (reference_date - agg["most_recent_date"]).dt.days
    agg["manufacturer_count"] = agg["manufacturer_count"].fillna(agg["manufacturer_count"].median())
    agg["resolved_rate"] = agg["resolved_rate"].fillna(0)
    agg["days_since_last_shortage"] = agg["days_since_last_shortage"].fillna(
        agg["days_since_last_shortage"].median()
    )
    return agg


def main():
    df = build_combined_dataset()
    df.to_csv("combined_drug_shortage_dataset.csv", index=False)

    df["start_date"] = pd.to_datetime(df["actual_start_date"], errors="coerce", utc=True)
    df["start_date"] = df["start_date"].fillna(
        pd.to_datetime(df["anticipated_start_date"], errors="coerce", utc=True)
    )

    top_classes = df[df["start_date"] < CUTOFF]["atc_description"].value_counts().head(10).index.tolist()

    # --- Train on pre-2024 data, label using 2024-2025 outcomes (no leakage) ---
    pre = df[df["start_date"] < CUTOFF]
    post = df[df["start_date"] >= CUTOFF]

    train_agg = build_features(pre, CUTOFF, top_classes)
    train_agg["label"] = train_agg["din"].isin(post["din"].unique()).astype(int)

    feature_cols = [
        "prior_shortage_count", "prior_discontinuation_count", "prior_tier3_count",
        "manufacturer_count", "days_since_last_shortage", "resolved_rate",
    ]
    class_dummies_train = pd.get_dummies(train_agg["class_bucket"], prefix="class")
    X_train = pd.concat([train_agg[feature_cols], class_dummies_train], axis=1).fillna(0)
    y_train = train_agg["label"].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)

    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    joblib.dump(model, "shortage_risk_model.joblib")
    joblib.dump(scaler, "shortage_risk_scaler.joblib")

    # --- Apply the validated model to ALL current data to get live risk scores ---
    now = df["start_date"].max()
    current_agg = build_features(df, now, top_classes)
    class_dummies_current = pd.get_dummies(current_agg["class_bucket"], prefix="class")
    X_current = pd.concat([current_agg[feature_cols], class_dummies_current], axis=1)
    X_current = X_current.reindex(columns=X_train.columns, fill_value=0).fillna(0)

    X_current_scaled = scaler.transform(X_current.values)
    current_agg["risk_score"] = model.predict_proba(X_current_scaled)[:, 1]

    output = current_agg[
        ["din", "brand_name", "atc_description", "prior_shortage_count",
         "manufacturer_count", "days_since_last_shortage", "risk_score"]
    ].sort_values("risk_score", ascending=False)

    output.to_csv("drug_risk_scores.csv", index=False)
    print(f"Scored {len(output)} drugs. Saved to drug_risk_scores.csv")


if __name__ == "__main__":
    main()

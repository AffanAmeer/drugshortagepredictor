# Builds the training dataset and generates drug risk scores.

import ast

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

CUTOFF = pd.Timestamp("2024-01-01", tz="UTC")


def safe_extract(value, key):
    if pd.isna(value):
        return None

    try:
        return ast.literal_eval(value).get(key)
    except (ValueError, SyntaxError):
        return None


def build_combined_dataset():
    shortages = pd.read_csv("shortage_discontinuation_reports.csv", low_memory=False)
    drug_product = pd.read_csv("drug_product.csv", low_memory=False)
    active_ingredient = pd.read_csv("active_ingredient.csv", low_memory=False)

    shortages["report_type"] = shortages["type"].apply(
        lambda x: safe_extract(x, "label")
    )

    shortages["shortage_reason_label"] = shortages["shortage_reason"].apply(
        lambda x: safe_extract(x, "en_reason")
    )

    ingredient_data = active_ingredient.merge(
        drug_product[["drug_code", "company_name"]],
        on="drug_code",
        how="left",
    )

    manufacturer_counts = (
        ingredient_data.groupby("ingredient_name")["company_name"]
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

    drug_info = drug_product.merge(
        primary_ingredient,
        on="drug_code",
        how="left",
    )

    drug_info = drug_info.merge(
        manufacturer_counts,
        left_on="primary_ingredient",
        right_on="ingredient_name",
        how="left",
    )

    drug_info = drug_info[
        [
            "drug_identification_number",
            "class_name",
            "primary_ingredient",
            "manufacturer_count",
        ]
    ].rename(columns={"drug_identification_number": "din"})

    # Make sure DINs use the same format before merging.
    shortages["din_clean"] = (
        pd.to_numeric(shortages["din"], errors="coerce")
        .astype("Int64")
        .astype(str)
    )

    drug_info["din_clean"] = (
        pd.to_numeric(drug_info["din"], errors="coerce")
        .astype("Int64")
        .astype(str)
    )

    drug_info = drug_info.drop_duplicates("din_clean")

    combined = shortages.merge(
        drug_info[
            [
                "din_clean",
                "class_name",
                "primary_ingredient",
                "manufacturer_count",
            ]
        ],
        on="din_clean",
        how="left",
    )

    keep_cols = [
        "id",
        "din",
        "en_drug_brand_name",
        "company_name",
        "atc_number",
        "atc_description",
        "report_type",
        "status",
        "shortage_reason_label",
        "anticipated_start_date",
        "actual_start_date",
        "actual_end_date",
        "resolved",
        "avoided",
        "tier_3",
        "primary_ingredient",
        "manufacturer_count",
        "class_name",
    ]

    return combined[keep_cols].rename(columns={"class_name": "drug_class"})


def build_features(data, reference_date, top_classes):
    df = data.copy()

    df["class_bucket"] = df["atc_description"].apply(
        lambda x: x if x in top_classes else "OTHER"
    )

    features = (
        df.groupby("din")
        .agg(
            prior_shortage_count=("report_type", lambda x: (x == "shortage").sum()),
            prior_discontinuation_count=(
                "report_type",
                lambda x: (x == "discontinuance").sum(),
            ),
            prior_tier3_count=("tier_3", "sum"),
            manufacturer_count=("manufacturer_count", "first"),
            most_recent_date=("start_date", "max"),
            resolved_rate=("resolved", "mean"),
            class_bucket=("class_bucket", "first"),
            brand_name=("en_drug_brand_name", "first"),
            atc_description=("atc_description", "first"),
        )
        .reset_index()
    )

    features["days_since_last_shortage"] = (
        reference_date - features["most_recent_date"]
    ).dt.days

    features["manufacturer_count"] = features["manufacturer_count"].fillna(
        features["manufacturer_count"].median()
    )

    features["resolved_rate"] = features["resolved_rate"].fillna(0)

    features["days_since_last_shortage"] = (
        features["days_since_last_shortage"].fillna(
            features["days_since_last_shortage"].median()
        )
    )

    return features


def main():
    df = build_combined_dataset()
    df.to_csv("combined_drug_shortage_dataset.csv", index=False)

    df["start_date"] = pd.to_datetime(
        df["actual_start_date"],
        errors="coerce",
        utc=True,
    )

    df["start_date"] = df["start_date"].fillna(
        pd.to_datetime(
            df["anticipated_start_date"],
            errors="coerce",
            utc=True,
        )
    )

    top_classes = (
        df[df["start_date"] < CUTOFF]["atc_description"]
        .value_counts()
        .head(10)
        .index.tolist()
    )

    pre = df[df["start_date"] < CUTOFF]
    post = df[df["start_date"] >= CUTOFF]

    train = build_features(pre, CUTOFF, top_classes)
    train["label"] = train["din"].isin(post["din"].unique()).astype(int)

    features = [
        "prior_shortage_count",
        "prior_discontinuation_count",
        "prior_tier3_count",
        "manufacturer_count",
        "days_since_last_shortage",
        "resolved_rate",
    ]

    train_classes = pd.get_dummies(
        train["class_bucket"],
        prefix="class",
    )

    X_train = pd.concat(
        [train[features], train_classes],
        axis=1,
    ).fillna(0)

    y_train = train["label"].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.values)

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )

    model.fit(X_train, y_train)

    joblib.dump(model, "shortage_risk_model.joblib")
    joblib.dump(scaler, "shortage_risk_scaler.joblib")

    current = build_features(
        df,
        df["start_date"].max(),
        top_classes,
    )

    current_classes = pd.get_dummies(
        current["class_bucket"],
        prefix="class",
    )

    X_current = pd.concat(
        [current[features], current_classes],
        axis=1,
    )

    # Keep the same columns used during training.
    X_current = (
        X_current.reindex(columns=X_train.columns, fill_value=0)
        .fillna(0)
    )

    X_current = scaler.transform(X_current.values)

    current["risk_score"] = model.predict_proba(X_current)[:, 1]

    results = current[
        [
            "din",
            "brand_name",
            "atc_description",
            "prior_shortage_count",
            "manufacturer_count",
            "days_since_last_shortage",
            "risk_score",
        ]
    ].sort_values("risk_score", ascending=False)

    results.to_csv("drug_risk_scores.csv", index=False)

    print(f"Saved risk scores for {len(results)} drugs.")


if __name__ == "__main__":
    main()

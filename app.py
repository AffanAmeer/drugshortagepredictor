import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Drug Shortage Early-Warning Tool",
    layout="wide"
)

st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: "Times New Roman", Times, serif;
    }
    h1, h2, h3 {
        font-family: "Times New Roman", Times, serif;
        border-bottom: 1px solid #333333;
        padding-bottom: 8px;
    }
    [data-testid="stMetric"] {
        background-color: #f7f5f0;
        border: 1px solid #d9d5c9;
        padding: 12px;
        border-radius: 2px;
    }
    [data-testid="stSidebar"] {
        background-color: #f7f5f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Canadian Drug Shortage Early-Warning Tool")
st.write(
    "Predicts which drugs are at elevated risk of a shortage based on historical "
    "Health Canada shortage data. Results update whenever the underlying dataset "
    "is refreshed."
)

df = pd.read_csv("drug_risk_scores.csv")
df["risk_percent"] = (df["risk_score"] * 100).round(1)

# Filters
st.sidebar.header("Filters")

search = st.sidebar.text_input("Search by drug name")

therapeutic_classes = ["All"] + sorted(
    df["atc_description"].dropna().unique()
)
selected_class = st.sidebar.selectbox(
    "Therapeutic class",
    therapeutic_classes,
)

min_risk = st.sidebar.slider(
    "Minimum risk score (%)",
    0,
    100,
    0,
)

filtered = df.copy()

if search:
    filtered = filtered[
        filtered["brand_name"].str.contains(
            search,
            case=False,
            na=False,
        )
    ]

if selected_class != "All":
    filtered = filtered[
        filtered["atc_description"] == selected_class
    ]

filtered = filtered[
    filtered["risk_percent"] >= min_risk
]

left, middle, right = st.columns(3)

left.metric("Drugs shown", len(filtered))

if len(filtered):
    avg_risk = f"{filtered['risk_percent'].mean():.1f}%"
else:
    avg_risk = "—"
middle.metric("Average risk score", avg_risk)

right.metric(
    "High risk (>70%)",
    (filtered["risk_percent"] > 70).sum(),
)

st.subheader("Highest-Risk Drugs Right Now")

top = (
    filtered.sort_values("risk_score", ascending=False)
    .head(15)
)

if top.empty:
    st.info("No drugs match the current filters.")
else:
    st.bar_chart(
        top.set_index("brand_name")["risk_percent"],
        color="#7a1f2b",
    )

st.subheader("Full Results")

results = (
    filtered.sort_values("risk_score", ascending=False)[
        [
            "brand_name",
            "atc_description",
            "prior_shortage_count",
            "manufacturer_count",
            "days_since_last_shortage",
            "risk_percent",
        ]
    ]
    .rename(
        columns={
            "brand_name": "Drug",
            "atc_description": "Therapeutic Class",
            "prior_shortage_count": "Past Shortages",
            "manufacturer_count": "# Manufacturers",
            "days_since_last_shortage": "Days Since Last Shortage",
            "risk_percent": "Risk Score (%)",
        }
    )
)

st.dataframe(
    results,
    use_container_width=True,
    height=500,
)

st.caption(
    "Risk score is generated from a logistic regression model trained on "
    "historical shortage data (2016-2023) and evaluated using 2024-2025 data. "
    "It should not replace official Health Canada guidance."
)

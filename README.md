# Canadian Drug Shortage Early-Warning Tool

A tool that predicts which prescription drugs in Canada are at elevated risk of a future shortage, built entirely on real data from public Health Canada.

**Live demo:** https://drugshortagepredictor-7mzxc4scdsbx32m8edvrzv.streamlit.app/

## The Problem

Drug shortages are a chronic, ongoing issue in Canada, around 2,000-3,000 new shortage or discontinuation reports are filed every year, affecting everyday medications like antidepressants and cholesterol drugs, not just rare specialty treatments. Patients and pharmacists typically only find out a drug is short once it's already unavailable. This project asks: **can shortage risk be predicted in advance, using a drug's own history?**

## Data Sources

- **[Health Product Shortages Canada](https://healthproductshortages.ca)** - 29,378 historical shortage and discontinuation reports (2016-2025), pulled via their public REST API
- **[Health Canada Drug Product Database](https://health-products.canada.ca/api/drug/)** - 58,180 marketed drug records and 120,659 active ingredient records, used to determine manufacturer counts and therapeutic classes

## Approach

1. **Data collection** - Pulled both datasets from Health Canada's public APIs (see `pull_shortage_data.py` and `pull_drug_product_data.py`) and handled authentication, pagination, and rate limits.
2. **Cleaning & joining** - Cleaned the data, fixed formatting issues (including a float vs. integer DIN mismatch that caused the first merge to fail), and combined both datasets into a single table using the Drug Identification Number (DIN).
3. **Exploratory analysis** - Explored the data to identify patterns that could help predict shortages, including previous shortages, therapeutic class, manufacturer count, and seasonal trends.
4. **Feature engineering** - Created features for each drug, including previous shortage count, previous Tier 3 (severe) shortages, manufacturer count, days since the last shortage, resolution rate, and therapeutic class.
5. **Modeling** - Trained a logistic regression model using data before January 1, 2024, and predicted whether a drug would have a new shortage in 2024–2025. A time-based split was used to avoid using future information during training.
6. **Deployment** - Used the trained model to generate risk scores for 9,060 drugs and displayed the results in an interactive Streamlit dashboard.
## Key Findings

- **Manufacturer count is a weak predictor.** The intuitive assumption that drugs made by only one company shortage often did not hold up. Most shortages involve drugs with 5+ manufacturers already.
- **Shortage history is a strong predictor.** 67% of drugs that have ever shortaged do so more than once. Recency and frequency of past shortages were the strongest signals in the model.
- **Seasonality is weak.** Shortage reports are fairly evenly distributed across the year, with no strong seasonal pattern worth building into the model.
- Certain therapeutic classes: antidepressants, antipsychotics, and lipid-modifying agents are disproportionately represented in shortage reports.

## Model Performance

Evaluated on a held-out set of drugs, using features from before 2024 to predict shortages in 2024-2025:

| Metric | Score |
|---|---|
| AUC | 0.767 |
| Recall | 0.74 |
| Precision | 0.56 |
| Accuracy | — see `model_feature_importance.csv` |

## Tech Stack

Python, pandas, scikit-learn (logistic regression), Streamlit, REST APIs (requests library)

## Running Locally

```
pip install requests pandas scikit-learn streamlit joblib
python pull_shortage_data.py       # requires a free healthproductshortages.ca account
python pull_drug_product_data.py   # no account needed
streamlit run app.py
```

## Limitations

This tool is a prototype built on historical, publicly reported data, it's not a live feed and should not replace official Health Canada guidance. Risk scores reflect statistical patterns in past reporting, not guaranteed future outcomes.

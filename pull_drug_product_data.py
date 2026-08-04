import requests
import pandas as pd

BASE_URL = "https://health-products.canada.ca/api/drug"


def fetch_endpoint(endpoint, params=None):
    params = params or {}
    params.update({"lang": "en", "type": "json"})
    print(f"Requesting {endpoint} ...")
    resp = requests.get(f"{BASE_URL}/{endpoint}/", params=params)
    resp.raise_for_status()
    data = resp.json()
    print(f"  -> got {len(data)} records")
    return data


if __name__ == "__main__":
    print("Pulling drug product data (this endpoint is large, may take a minute)...\n")
    products = fetch_endpoint("drugproduct")
    df_products = pd.DataFrame(products)
    df_products.to_csv("drug_product.csv", index=False)
    print(f"Saved {len(df_products)} drug product records to drug_product.csv\n")

    print("Pulling active ingredient data...\n")
    ingredients = fetch_endpoint("activeingredient")
    df_ingredients = pd.DataFrame(ingredients)
    df_ingredients.to_csv("active_ingredient.csv", index=False)
    print(f"Saved {len(df_ingredients)} active ingredient records to active_ingredient.csv\n")

    print("Done. Both CSV files are now in this same folder.")

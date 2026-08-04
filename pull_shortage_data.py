"""
Pulls ALL shortage + discontinuation reports from healthproductshortages.ca
via their public API, bypassing the 10,000-record UI export limit by
paging through results properly.

SETUP:
1. pip install requests pandas
2. Fill in EMAIL and PASSWORD below with your account credentials
   (never share these with anyone, including in a chat/Claude conversation)
3. Run: python pull_shortage_data.py

OUTPUT:
A file called shortage_discontinuation_reports.csv with every record.
"""

import requests
import pandas as pd
import time

BASE_URL = "https://healthproductshortages.ca/api/v1"

EMAIL = "aaffan2008@gmail.com"
PASSWORD = "Affan0424___"


def login(email, password):
    """Logs in and returns the auth token needed for all other requests."""
    # NOTE: the API docs specify form-urlencoded parameter encoding, not JSON,
    # so we send this as form data (the `data=` argument) rather than `json=`.
    resp = requests.post(f"{BASE_URL}/login", data={"email": email, "password": password})

    print(f"Status code: {resp.status_code}")
    print(f"Raw response: {resp.text}")

    if resp.status_code == 403:
        raise Exception("Account not verified yet — check your email for the verification link.")
    if resp.status_code == 400:
        raise Exception("Bad request — see the raw response printed above for the real reason.")
    resp.raise_for_status()
    token = resp.headers.get("auth-token")
    if not token:
        raise Exception(f"No auth-token found in response headers. Headers were: {resp.headers}")
    return token


def test_single_request(token):
    """
    Run this first (see bottom of file) to see the raw shape of one page,
    since the exact pagination parameter names aren't 100% confirmed from
    the docs alone. This prints the response so we can check field names
    before looping through everything.
    """
    headers = {"auth-token": token}
    params = {"orderby": "id", "order": "asc"}
    resp = requests.get(f"{BASE_URL}/search", headers=headers, params=params)
    resp.raise_for_status()
    result = resp.json()
    print("Keys in response:", list(result.keys()))
    print("Total records available:", result.get("total"))
    print("Limit per page (default):", result.get("limit"))
    print("Total pages:", result.get("total_pages"))
    print("Sample record:", result.get("data", [None])[0])
    return result


def fetch_all_reports(token, limit=100):
    """Loops through every page of results and combines them into one list."""
    headers = {"auth-token": token}
    all_data = []
    offset = 0
    page = 1

    while True:
        params = {
            "orderby": "id",
            "order": "asc",
            "limit": limit,
            "offset": offset,
        }
        resp = requests.get(f"{BASE_URL}/search", headers=headers, params=params)
        resp.raise_for_status()
        result = resp.json()

        data = result.get("data", [])
        all_data.extend(data)
        total = result.get("total", 0)
        total_pages = result.get("total_pages", 1)

        print(f"Page {page}/{total_pages} — {len(all_data)}/{total} records collected so far")

        remaining = result.get("remaining", 0)
        if remaining <= 0 or not data:
            break

        offset += limit
        page += 1
        time.sleep(0.3)  # stay well under the 1000 requests/hour rate limit

    return all_data


if __name__ == "__main__":
    print("Logging in...")
    token = login(EMAIL, PASSWORD)
    print("Login successful.\n")

    print("Testing a single request first to confirm the response format...")
    test_single_request(token)
    print("\nIf that looked right, comment out the line below this and")
    print("uncomment the fetch_all_reports block to pull everything.\n")

    # --- Full pull, now that the test above confirmed the format works ---
    reports = fetch_all_reports(token)
    df = pd.DataFrame(reports)
    df.to_csv("shortage_discontinuation_reports.csv", index=False)
    print(f"\nDone. Saved {len(df)} records to shortage_discontinuation_reports.csv")
